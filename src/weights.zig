// TigerStyle two-pass ModelWeights loader.
// Pass 1 computes the exact 256-byte-aligned VRAM requirement.
// Pass 2 allocates ONE contiguous VRAMArena (1 cudaMalloc instead of ~600) and packs every tensor,
// asserting exact fill (`arena.offset_bytes == arena.capacity_bytes`) on completion.

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");
const safetensors = @import("safetensors.zig");
const tiger = @import("tiger.zig");
const vram_arena = @import("vram_arena.zig");

pub const LayerWeights = struct {
    is_global: bool = false,
    has_controllers: bool = false,

    attn_norm: ?*anyopaque = null,
    q_in_norm: ?*anyopaque = null,
    k_in_norm: ?*anyopaque = null,
    v_in_norm: ?*anyopaque = null,
    mlp_norm: ?*anyopaque = null,

    w_qrgs: cuda.CLinearWeight = .{},
    w_kv: cuda.CLinearWeight = .{},
    w_k: cuda.CLinearWeight = .{},
    w_v: cuda.CLinearWeight = .{},
    w_o: cuda.CLinearWeight = .{},

    k_sconv_w: ?*anyopaque = null,
    v_sconv_w: ?*anyopaque = null,
    k_sconv_proj: ?*anyopaque = null,
    v_sconv_proj: ?*anyopaque = null,
    attn_sconv_w: ?*anyopaque = null,
    mlp_sconv_w: ?*anyopaque = null,

    q_clip_scale: ?*anyopaque = null,
    rel_clip_scale: ?*anyopaque = null,
    rel_logits_proj: ?*anyopaque = null,

    ctrl: [4]cuda.CControllerWeights = .{ .{}, .{}, .{}, .{} },

    router_w: ?*anyopaque = null,
    router_nvfp4: cuda.CLinearWeight = .{},
    router_bias: ?[*]f32 = null,
    w13: cuda.CExpertBank = .{},
    w2: cuda.CExpertBank = .{},
};

pub const AblationWeights = struct {
    active: bool = false,
    has_detectors: bool = false,
    attn_dirs: ?[*]f32 = null, // [48 * 2304] unit directions r_v
    mlp_dirs: ?[*]f32 = null, // [48 * 2304] unit directions r_v
    attn_det: ?[*]f32 = null, // [48 * 2304] Fisher-LDA detectors d_v (optional)
    mlp_det: ?[*]f32 = null, // [48 * 2304] Fisher-LDA detectors d_v (optional)
    attn_params: ?[*]f32 = null, // [48 * 4] {weight, offset, floor, threshold}
    mlp_params: ?[*]f32 = null, // [48 * 4] {weight, offset, floor, threshold}
    attn_nonzero: [constants.visit_count]bool = @splat(false),
    mlp_nonzero: [constants.visit_count]bool = @splat(false),
};

pub const ModelWeights = struct {
    mode: constants.QuantMode = .nvfp4,
    arena: vram_arena.VRAMArena = .{},
    pinned_host_embed: ?*anyopaque = null,

    embed: ?*anyopaque = null,
    embed_norm: ?*anyopaque = null,
    final_norm: ?*anyopaque = null,
    unembed: cuda.CLinearWeight = .{},

    attn_queries: [constants.layer_count - 1]?[*]f32 = @splat(null),
    mlp_queries: [constants.layer_count]?[*]f32 = @splat(null),
    output_query: ?[*]f32 = null,
    cross_pass_queries: ?[*]f32 = null,
    cross_pass_kv_queries: ?[*]f32 = null,

    layers: [constants.layer_count]LayerWeights = @splat(.{}),
    dspark: cuda.CDsparkWeights = .{},
    ablation: AblationWeights = .{},

    pub fn deinit(self: *ModelWeights) void {
        self.arena.deinit();
        if (self.pinned_host_embed) |host_ptr| {
            tiger.assert_cuda(cuda.squid_cuda_free_host(host_ptr));
            self.pinned_host_embed = null;
        }
        self.* = .{};
    }

    pub fn load(
        self: *ModelWeights,
        model_directory: []const u8,
        quant_mode: constants.QuantMode,
        load_mtp: bool,
        embed_on_host: bool,
        ablation_path: []const u8,
    ) bool {
        tiger.assert(model_directory.len > 0);
        tiger.assert(self.arena.base_ptr == null);

        const start_ns: u64 = tiger.clock_monotonic_ns();
        self.mode = quant_mode;

        var main_path_buf: [1024]u8 = undefined;
        const main_path_z = tiger.format_path_z(&main_path_buf, model_directory, "/model-00001-of-00001.safetensors");
        var sf_main: safetensors.SafetensorsFile = .{};
        if (!sf_main.open_file(main_path_z)) {
            tiger.print_stderr("[weights] Failed to open {s}/model-00001-of-00001.safetensors\n", .{model_directory});
            return false;
        }
        defer sf_main.close_file();

        var mtp_path_buf: [1024]u8 = undefined;
        const mtp_path_z = tiger.format_path_z(&mtp_path_buf, model_directory, "/mtp.safetensors");
        var sf_mtp: safetensors.SafetensorsFile = .{};
        const has_mtp: bool = load_mtp and sf_mtp.open_file(mtp_path_z);
        defer if (has_mtp) sf_mtp.close_file();

        var abl_path_buf: [1024]u8 = undefined;
        var sf_abl: safetensors.SafetensorsFile = .{};
        var has_ablation: bool = false;
        var has_ablation_detectors: bool = false;
        if (ablation_path.len > 0) {
            tiger.assert(ablation_path.len < abl_path_buf.len);
            @memcpy(abl_path_buf[0..ablation_path.len], ablation_path);
            abl_path_buf[ablation_path.len] = 0;
            const abl_path_z: [*:0]const u8 = @ptrCast(abl_path_buf[0..ablation_path.len :0]);
            if (!sf_abl.open_file(abl_path_z)) {
                tiger.print_stderr("[weights] Failed to open ablation safetensors: {s}\n", .{ablation_path});
                return false;
            }
            has_ablation = true;
            const plain_layout: bool = sf_abl.lookup_optional("attn_w") != null;
            has_ablation_detectors = !plain_layout and sf_abl.lookup_optional("attn_det") != null;
        }
        defer if (has_ablation) sf_abl.close_file();

        const dense_kind: constants.WeightKind = switch (quant_mode) {
            .exact => .bf16,
            .fp8 => .fp8,
            .nvfp4 => .nvfp4,
        };
        const env_mtp_bf16 = tiger.getenv("SQUID_MTP_BF16");
        const force_mtp_bf16: bool = (env_mtp_bf16 != null and env_mtp_bf16.?[0] == '1');
        const mtp_kind: constants.WeightKind = if (quant_mode == .exact or force_mtp_bf16) .bf16 else .nvfp4;

        // Check optional pruned draft vocab file before Pass 1 so VRAM byte accounting is exact!
        var draft_vocab_host: []i32 = &.{};
        var draft_vocab_count: u32 = constants.vocab_unpadded_count;
        var has_pruned_draft_vocab: bool = false;
        if (has_mtp) {
            if (try_load_draft_vocab(model_directory, quant_mode)) |loaded_slice| {
                draft_vocab_host = loaded_slice;
                draft_vocab_count = @intCast(loaded_slice.len);
                has_pruned_draft_vocab = true;
            }
        }
        defer if (draft_vocab_host.len > 0) tiger.free_host_slice(i32, draft_vocab_host);

        // --- PASS 1: Compute Exact 256-Byte-Aligned VRAM Arena Capacity ---
        const total_vram_bytes: u64 = compute_total_vram_bytes(
            quant_mode,
            dense_kind,
            mtp_kind,
            embed_on_host,
            has_mtp,
            has_pruned_draft_vocab,
            draft_vocab_count,
            has_ablation,
            has_ablation_detectors,
        );
        self.arena = vram_arena.VRAMArena.init(total_vram_bytes);

        // Temporary staging buffers on GPU for BF16 upload & quantization (freed before returning)
        const stage_bytes: u64 = @as(u64, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
        var d_stage_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_stage_opt, @intCast(stage_bytes)));
        const d_stage: *anyopaque = d_stage_opt.?;
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_stage));

        var d_amax_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_amax_opt, @sizeOf(f32)));
        const d_amax: *f32 = @ptrCast(@alignCast(d_amax_opt.?));
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_amax_opt));

        // --- PASS 2: Upload & Pack Directly Into Contiguous VRAMArena ---

        // Embeddings & top-level norms
        if (embed_on_host) {
            const ei = sf_main.lookup("model.llm.embed.weight");
            const embed_bytes: usize = @as(usize, constants.vocab_padded_count) * constants.hidden_dim * 2;
            tiger.assert(ei.byte_length == embed_bytes);
            var h_ptr: ?*anyopaque = null;
            tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_ptr, embed_bytes, 1));
            tiger.assert(h_ptr != null);
            self.pinned_host_embed = h_ptr;
            const dst_slice = @as([*]u8, @ptrCast(h_ptr.?))[0..embed_bytes];
            @memcpy(dst_slice, sf_main.data_slice(ei));
            var dev_ptr: ?*anyopaque = null;
            tiger.assert_cuda(cuda.squid_cuda_host_get_device_pointer(&dev_ptr, h_ptr.?));
            self.embed = dev_ptr;
            tiger.print_stdout(
                "[weights] Embed table pinned in zero-copy host RAM (saved {d:.2} GiB VRAM)\n",
                .{@as(f64, @floatFromInt(embed_bytes)) / (1024.0 * 1024.0 * 1024.0)},
            );
        } else {
            self.embed = self.upload_bf16(&sf_main, "model.llm.embed.weight", @as(u64, constants.vocab_padded_count) * constants.hidden_dim);
        }
        self.embed_norm = self.upload_bf16(&sf_main, "model.llm.embed_norm.weight", constants.hidden_dim);
        self.final_norm = self.upload_bf16(&sf_main, "model.llm.norm.weight", constants.hidden_dim);

        // Unembed (only first vocab_unpadded_count = 200,058 rows are usable)
        {
            const ui = sf_main.lookup("model.llm.unembed.weight");
            const usable_bytes: usize = @as(usize, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
            tiger.assert(ui.byte_length >= usable_bytes);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_stage, sf_main.data_ptr(ui), usable_bytes));
            self.unembed = self.pack_linear(d_stage, constants.vocab_unpadded_count, constants.hidden_dim, dense_kind, d_amax);
        }

        // AttnRes & cross-pass queries
        var name_buf: [128]u8 = undefined;
        var layer_idx: u32 = 1;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const elems: u64 = if (constants.is_global_layer(layer_idx)) 3 * constants.hidden_dim else constants.hidden_dim;
            const tname = std.fmt.bufPrint(&name_buf, "model.llm.attn_res.attn_queries.{d}", .{layer_idx - 1}) catch unreachable;
            self.attn_queries[layer_idx - 1] = self.upload_f32(&sf_main, tname, elems);
        }
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const tname = std.fmt.bufPrint(&name_buf, "model.llm.attn_res.mlp_queries.{d}", .{layer_idx}) catch unreachable;
            self.mlp_queries[layer_idx] = self.upload_f32(&sf_main, tname, constants.hidden_dim);
        }
        self.output_query = self.upload_f32(&sf_main, "model.llm.attn_res.output_query", constants.hidden_dim);
        self.cross_pass_queries = self.upload_f32(&sf_main, "model.llm.middle_cycle_cross_pass_queries", 24 * constants.hidden_dim);
        self.cross_pass_kv_queries = self.upload_f32(&sf_main, "model.llm.middle_cycle_cross_pass_kv_queries", 24 * constants.hidden_dim);

        // 24 physical layers
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const lw: *LayerWeights = &self.layers[layer_idx];
            lw.is_global = constants.is_global_layer(layer_idx);

            if (lw.is_global) {
                lw.q_in_norm = self.upload_layer_bf16(&sf_main, layer_idx, "q_input_norm.weight", constants.hidden_dim);
                lw.k_in_norm = self.upload_layer_bf16(&sf_main, layer_idx, "k_input_norm.weight", constants.hidden_dim);
                lw.v_in_norm = self.upload_layer_bf16(&sf_main, layer_idx, "v_input_norm.weight", constants.hidden_dim);
            } else {
                lw.attn_norm = self.upload_layer_bf16(&sf_main, layer_idx, "attn_norm.weight", constants.hidden_dim);
            }
            lw.mlp_norm = self.upload_layer_bf16(&sf_main, layer_idx, "mlp_norm.weight", constants.hidden_dim);

            // Concatenate [wq(3072), wr(384), gate(3072), sink(24)] -> 6552 x 2304
            {
                const stage_bytes_ptr: [*]u8 = @ptrCast(d_stage);
                var offset: usize = 0;
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.wq_du.weight", stage_bytes_ptr + offset, @as(u64, constants.query_dim) * constants.hidden_dim * 2);
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.wr_du.weight", stage_bytes_ptr + offset, @as(u64, constants.r_dim) * constants.hidden_dim * 2);
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.gate_attn.weight", stage_bytes_ptr + offset, @as(u64, constants.query_dim) * constants.hidden_dim * 2);
                offset += stage_layer_raw(&sf_main, layer_idx, "attn.attn_sink_proj.weight", stage_bytes_ptr + offset, @as(u64, constants.head_count) * constants.hidden_dim * 2);
                tiger.assert(offset == @as(usize, constants.qrgs_out_dim) * constants.hidden_dim * 2);
                lw.w_qrgs = self.pack_linear(d_stage, constants.qrgs_out_dim, constants.hidden_dim, dense_kind, d_amax);
            }

            // Concatenate [wk(1024), wv(1024)] -> 2048 x 2304, then slice w_k & w_v
            {
                const stage_bytes_ptr: [*]u8 = @ptrCast(d_stage);
                const kv_bytes: u64 = @as(u64, constants.kv_dim) * constants.hidden_dim * 2;
                _ = stage_layer_raw(&sf_main, layer_idx, "attn.wk_dv.weight", stage_bytes_ptr, kv_bytes);
                _ = stage_layer_raw(&sf_main, layer_idx, "attn.wv_dv.weight", stage_bytes_ptr + @as(usize, @intCast(kv_bytes)), kv_bytes);
                lw.w_kv = self.pack_linear(d_stage, 2 * constants.kv_dim, constants.hidden_dim, dense_kind, d_amax);

                lw.w_k = lw.w_kv;
                lw.w_k.rows = constants.kv_dim;
                lw.w_v = lw.w_kv;
                lw.w_v.rows = constants.kv_dim;
                const base_data: [*]u8 = @ptrCast(lw.w_kv.data.?);
                switch (dense_kind) {
                    .bf16 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * constants.hidden_dim * 2);
                    },
                    .fp8 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * constants.hidden_dim);
                        lw.w_v.row_scales = lw.w_kv.row_scales.? + constants.kv_dim;
                    },
                    .nvfp4 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * (constants.hidden_dim / 2));
                        lw.w_v.scales = lw.w_kv.scales.? + @as(usize, constants.kv_dim) * (constants.hidden_dim / 16);
                    },
                }
            }

            // w_o [2304, 3072]
            {
                _ = stage_layer_raw(&sf_main, layer_idx, "attn.wo_ud.weight", @ptrCast(d_stage), @as(u64, constants.hidden_dim) * constants.query_dim * 2);
                lw.w_o = self.pack_linear(d_stage, constants.hidden_dim, constants.query_dim, dense_kind, d_amax);
            }

            // Short convolutions & clip/rel projections
            if (lw.is_global) {
                lw.k_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.k_sconv.static_w", constants.conv_tap_count * constants.kv_dim);
                lw.v_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.v_sconv.static_w", constants.conv_tap_count * constants.kv_dim);
                lw.k_sconv_proj = self.upload_layer_bf16(&sf_main, layer_idx, "attn.k_sconv.weight_proj", constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim);
                lw.v_sconv_proj = self.upload_layer_bf16(&sf_main, layer_idx, "attn.v_sconv.weight_proj", constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim);
            } else {
                lw.k_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.k_sconv.weight", constants.kv_dim * constants.conv_tap_count);
                lw.v_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn.v_sconv.weight", constants.kv_dim * constants.conv_tap_count);
            }
            lw.attn_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "attn_sconv.weight", constants.hidden_dim * constants.conv_tap_count);
            lw.mlp_sconv_w = self.upload_layer_bf16(&sf_main, layer_idx, "mlp_sconv.weight", constants.hidden_dim * constants.conv_tap_count);
            lw.q_clip_scale = self.upload_layer_bf16(&sf_main, layer_idx, "attn.q_l2_clip.scale", constants.head_count * constants.head_dim);
            lw.rel_clip_scale = self.upload_layer_bf16(&sf_main, layer_idx, "attn.rel_l2_clip.scale", constants.head_count * constants.rel_dim);
            lw.rel_logits_proj = self.upload_layer_bf16(&sf_main, layer_idx, "attn.rel_logits_proj.proj", constants.rel_dim * constants.rel_extent);

            // Cross-pass controllers (layers 6..17)
            if (layer_idx >= constants.loop_start_layer and layer_idx < constants.loop_end_layer) {
                lw.has_controllers = true;
                const stems: [4][]const u8 = .{ "attn", "k", "v", "mlp" };
                var s_idx: u32 = 0;
                while (s_idx < 4) : (s_idx += 1) {
                    if (quant_mode == .nvfp4) {
                        var tb: [128]u8 = undefined;
                        const dn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.down_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        upload_raw(&sf_main, dn, d_stage, @as(u64, constants.gate_rank) * 2 * constants.hidden_dim * 2);
                        lw.ctrl[s_idx].down_nvfp4 = self.pack_linear(d_stage, constants.gate_rank, 2 * constants.hidden_dim, .nvfp4, d_amax);

                        const fn_name = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        upload_raw(&sf_main, fn_name, d_stage, @as(u64, constants.hidden_dim) * constants.gate_rank * 2);
                        lw.ctrl[s_idx].forget_nvfp4 = self.pack_linear(d_stage, constants.hidden_dim, constants.gate_rank, .nvfp4, d_amax);

                        const gn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        upload_raw(&sf_main, gn, d_stage, @as(u64, constants.hidden_dim) * constants.gate_rank * 2);
                        lw.ctrl[s_idx].gain_nvfp4 = self.pack_linear(d_stage, constants.hidden_dim, constants.gate_rank, .nvfp4, d_amax);
                    } else {
                        var tb: [128]u8 = undefined;
                        const dn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.down_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        lw.ctrl[s_idx].down_w = self.upload_bf16(&sf_main, dn, @as(u64, constants.gate_rank) * 2 * constants.hidden_dim);

                        const fn_name = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        lw.ctrl[s_idx].forget_w = self.upload_bf16(&sf_main, fn_name, @as(u64, constants.hidden_dim) * constants.gate_rank);

                        const gn = std.fmt.bufPrint(&tb, "model.llm.layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                        lw.ctrl[s_idx].gain_w = self.upload_bf16(&sf_main, gn, @as(u64, constants.hidden_dim) * constants.gate_rank);
                    }
                }
            }

            // Router
            if (quant_mode == .nvfp4) {
                _ = stage_layer_raw(&sf_main, layer_idx, "mlp.gate.weight", @ptrCast(d_stage), @as(u64, constants.expert_count) * constants.hidden_dim * 2);
                lw.router_nvfp4 = self.pack_linear(d_stage, constants.expert_count, constants.hidden_dim, .nvfp4, d_amax);
            } else {
                lw.router_w = self.upload_layer_bf16(&sf_main, layer_idx, "mlp.gate.weight", @as(u64, constants.expert_count) * constants.hidden_dim);
            }
            const rb_name = std.fmt.bufPrint(&name_buf, "model.llm.layers.{d}.mlp.gate.bias", .{layer_idx}) catch unreachable;
            lw.router_bias = self.upload_f32(&sf_main, rb_name, constants.expert_count);

            // MoE Expert Banks (w13 and w2)
            lw.w13 = self.load_expert_bank(&sf_main, layer_idx, "w13", 2 * constants.intermediate_dim, constants.hidden_dim, quant_mode, d_stage);
            lw.w2 = self.load_expert_bank(&sf_main, layer_idx, "w2", constants.hidden_dim, constants.intermediate_dim, quant_mode, d_stage);
        }

        // Load DSpark MTP weights if available
        if (has_mtp) {
            self.load_dspark(
                &sf_main,
                &sf_mtp,
                mtp_kind,
                d_stage,
                d_amax,
                has_pruned_draft_vocab,
                draft_vocab_host,
                draft_vocab_count,
            );
        }

        if (has_ablation) {
            self.load_ablation(&sf_abl, has_ablation_detectors);
        }

        tiger.assert_cuda(cuda.squid_cuda_device_sync());
        // Paired TigerStyle invariant: Pass 1 predicted byte total == Pass 2 exact arena usage!
        self.arena.assert_exact_fill();

        const elapsed_sec: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - start_ns)) / 1_000_000_000.0;
        const vram_gib: f64 = @as(f64, @floatFromInt(self.arena.capacity_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[weights] Loaded mode={s} in {d:.2}s | VRAM={d:.2} GiB (1 arena, {d} tensors) | MTP={s} | Ablation={s}\n",
            .{
                quant_mode.name(),
                elapsed_sec,
                vram_gib,
                self.arena.allocation_count,
                if (self.dspark.loaded != 0) "on" else "off",
                if (!self.ablation.active) "off" else if (self.ablation.has_detectors) "gated-lda" else "plain",
            },
        );
        return true;
    }

    pub fn load_gguf(
        self: *ModelWeights,
        gguf_path: []const u8,
        draft_gguf_path: []const u8,
        meta_dir: []const u8,
        quant_mode: constants.QuantMode,
        load_mtp: bool,
        embed_on_host: bool,
        ablation_path: []const u8,
    ) bool {
        tiger.assert(gguf_path.len > 0);
        tiger.assert(meta_dir.len > 0);
        tiger.assert(self.arena.base_ptr == null);

        const start_ns: u64 = tiger.clock_monotonic_ns();
        self.mode = quant_mode;

        var main_pz: [1024]u8 = undefined;
        var gf_main: gguf.GgufFile = .{};
        if (!gf_main.open_file(to_cstr(&main_pz, gguf_path))) {
            tiger.print_stderr("[weights] Failed to open GGUF: {s}\n", .{gguf_path});
            return false;
        }
        defer gf_main.close_file();

        var draft_pz: [1024]u8 = undefined;
        var gf_draft: gguf.GgufFile = .{};
        const has_mtp: bool = load_mtp and draft_gguf_path.len > 0 and gf_draft.open_file(to_cstr(&draft_pz, draft_gguf_path));
        defer if (has_mtp) gf_draft.close_file();

        var abl_pz: [1024]u8 = undefined;
        var sf_abl: safetensors.SafetensorsFile = .{};
        var has_ablation: bool = false;
        var has_ablation_detectors: bool = false;
        if (ablation_path.len > 0) {
            if (!sf_abl.open_file(to_cstr(&abl_pz, ablation_path))) {
                tiger.print_stderr("[weights] Failed to open ablation safetensors: {s}\n", .{ablation_path});
                return false;
            }
            has_ablation = true;
            const plain_layout: bool = sf_abl.lookup_optional("attn_w") != null;
            has_ablation_detectors = !plain_layout and sf_abl.lookup_optional("attn_det") != null;
        }
        defer if (has_ablation) sf_abl.close_file();

        const dense_kind: constants.WeightKind = switch (quant_mode) {
            .exact => .bf16,
            .fp8 => .fp8,
            .nvfp4 => .nvfp4,
        };
        const env_mtp_bf16 = tiger.getenv("SQUID_MTP_BF16");
        const force_mtp_bf16: bool = (env_mtp_bf16 != null and env_mtp_bf16.?[0] == '1');
        const mtp_kind: constants.WeightKind = if (quant_mode == .exact or force_mtp_bf16) .bf16 else .nvfp4;

        var draft_vocab_host: []i32 = &.{};
        var draft_vocab_count: u32 = constants.vocab_unpadded_count;
        var has_pruned_draft_vocab: bool = false;
        if (has_mtp) {
            if (try_load_draft_vocab(meta_dir, quant_mode)) |loaded_slice| {
                draft_vocab_host = loaded_slice;
                draft_vocab_count = @intCast(loaded_slice.len);
                has_pruned_draft_vocab = true;
            }
        }
        defer if (draft_vocab_host.len > 0) tiger.free_host_slice(i32, draft_vocab_host);

        // --- PASS 1: Exact 256-Byte-Aligned VRAM Capacity ---
        const total_vram_bytes: u64 = compute_total_vram_bytes_gguf(
            &gf_main,
            quant_mode,
            dense_kind,
            mtp_kind,
            embed_on_host,
            has_mtp,
            has_pruned_draft_vocab,
            draft_vocab_count,
            has_ablation,
            has_ablation_detectors,
        );
        self.arena = vram_arena.VRAMArena.init(total_vram_bytes);

        const h_raw = tiger.alloc_host_zeroed(u8, 32 * 1024 * 1024);
        defer tiger.free_host_slice(u8, h_raw);
        const h_aux = tiger.alloc_host_zeroed(u8, 16 * 1024 * 1024);
        defer tiger.free_host_slice(u8, h_aux);

        var d_raw_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_raw_opt, 32 * 1024 * 1024));
        const d_raw: [*]u8 = @ptrCast(d_raw_opt.?);
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_raw_opt));

        var d_stage_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_stage_opt, 88 * 1024 * 1024));
        const d_stage: *anyopaque = d_stage_opt.?;
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_stage));

        var d_amax_opt: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&d_amax_opt, @sizeOf(f32)));
        const d_amax: *f32 = @ptrCast(@alignCast(d_amax_opt.?));
        defer tiger.assert_cuda(cuda.squid_cuda_free(d_amax_opt));

        // --- PASS 2: Stream, Dequant, & Pack Directly Into Contiguous VRAMArena ---

        // Embeddings (`token_embd.weight`: [2304, 201024] IQ4_XS -> BF16)
        {
            const ei = gf_main.require("token_embd.weight");
            tiger.assert(ei.dims[0] == constants.hidden_dim and ei.dims[1] == constants.vocab_padded_count);
            if (embed_on_host) {
                const embed_bytes: usize = @as(usize, constants.vocab_padded_count) * constants.hidden_dim * 2;
                var h_ptr: ?*anyopaque = null;
                tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_ptr, embed_bytes, 1));
                tiger.assert(h_ptr != null);
                self.pinned_host_embed = h_ptr;
                const h_embed_dst: [*]u8 = @ptrCast(h_ptr.?);
                const rb = gguf.row_bytes(ei.ggml_type, constants.hidden_dim);
                const row_bf16_bytes: u64 = @as(u64, constants.hidden_dim) * 2;
                const chunk_max: u64 = 8192;
                var r0: u64 = 0;
                while (r0 < constants.vocab_padded_count) {
                    const cnt = tiger.min_u64(chunk_max, constants.vocab_padded_count - r0);
                    gf_main.read_range(ei.abs_offset + r0 * rb, h_raw[0..@intCast(cnt * rb)]);
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_raw, h_raw.ptr, @intCast(cnt * rb)));
                    cuda.squid_dequant_gguf_to_bf16(ei.ggml_type, d_raw, d_stage, @intCast(cnt), constants.hidden_dim);
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h(h_embed_dst + @as(usize, @intCast(r0 * row_bf16_bytes)), d_stage, @intCast(cnt * row_bf16_bytes)));
                    r0 += cnt;
                }
                var dev_ptr: ?*anyopaque = null;
                tiger.assert_cuda(cuda.squid_cuda_host_get_device_pointer(&dev_ptr, h_ptr.?));
                self.embed = dev_ptr;
            } else {
                const dst = self.arena.alloc_raw(@as(u64, constants.vocab_padded_count) * constants.hidden_dim * 2);
                stream_gguf_to_bf16_dev(&gf_main, ei, constants.vocab_padded_count, constants.hidden_dim, h_raw, d_raw, dst);
                self.embed = dst;
            }
        }

        self.embed_norm = self.upload_gguf_bf16(&gf_main, "token_embd_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
        self.final_norm = self.upload_gguf_bf16(&gf_main, "output_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);

        // Unembed (`output.weight` rows [0..200058)) + DSpark (reuse decoded BF16 unembed before freeing!)
        {
            const ui = gf_main.require("output.weight");
            tiger.assert(ui.dims[0] == constants.hidden_dim and ui.dims[1] >= constants.vocab_unpadded_count);
            const unembed_bf16_bytes: u64 = @as(u64, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
            var d_unembed_bf16_opt: ?*anyopaque = null;
            tiger.assert_cuda(cuda.squid_cuda_malloc(&d_unembed_bf16_opt, @intCast(unembed_bf16_bytes)));
            const d_unembed_bf16: *anyopaque = d_unembed_bf16_opt.?;
            defer tiger.assert_cuda(cuda.squid_cuda_free(d_unembed_bf16));

            stream_gguf_to_bf16_dev(&gf_main, ui, constants.vocab_unpadded_count, constants.hidden_dim, h_raw, d_raw, d_unembed_bf16);
            self.unembed = self.pack_linear(d_unembed_bf16, constants.vocab_unpadded_count, constants.hidden_dim, dense_kind, d_amax);

            if (has_mtp) {
                self.load_dspark_gguf(
                    &gf_draft,
                    d_unembed_bf16,
                    mtp_kind,
                    h_raw,
                    d_raw,
                    d_stage,
                    d_amax,
                    has_pruned_draft_vocab,
                    draft_vocab_host,
                    draft_vocab_count,
                );
            }
        }

        // AttnRes & cross-pass queries (F32)
        var name_buf: [128]u8 = undefined;
        var layer_idx: u32 = 1;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const elems: u64 = if (constants.is_global_layer(layer_idx)) 3 * constants.hidden_dim else constants.hidden_dim;
            const tname = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_res_query.weight", .{layer_idx}) catch unreachable;
            self.attn_queries[layer_idx - 1] = self.upload_gguf_f32(&gf_main, tname, elems, h_raw);
        }
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const tname = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_res_query.weight", .{layer_idx}) catch unreachable;
            self.mlp_queries[layer_idx] = self.upload_gguf_f32(&gf_main, tname, constants.hidden_dim, h_raw);
        }
        self.output_query = self.upload_gguf_f32(&gf_main, "output_res_query.weight", constants.hidden_dim, h_raw);
        if (gf_main.find("loop_res_query.weight") != null) {
            self.cross_pass_queries = self.upload_gguf_f32(&gf_main, "loop_res_query.weight", 24 * constants.hidden_dim, h_raw);
            self.cross_pass_kv_queries = self.upload_gguf_f32(&gf_main, "loop_kv_res_query.weight", 24 * constants.hidden_dim, h_raw);
        } else {
            const cp_q = self.arena.alloc_many(f32, 24 * constants.hidden_dim);
            const cp_kv = self.arena.alloc_many(f32, 24 * constants.hidden_dim);
            self.cross_pass_queries = cp_q;
            self.cross_pass_kv_queries = cp_kv;
            const row_bytes: usize = constants.hidden_dim * 4;
            var li: u32 = constants.loop_start_layer;
            while (li < constants.loop_end_layer) : (li += 1) {
                const mq_name = std.fmt.bufPrint(&name_buf, "blk.{d}.loop_msg_query.weight", .{li}) catch unreachable;
                const mq_info = gf_main.require(mq_name);
                tiger.assert(mq_info.ggml_type == 0 and mq_info.n_dims == 2 and mq_info.dims[0] == constants.hidden_dim and mq_info.dims[1] == 4);
                gf_main.read_range(mq_info.abs_offset, h_raw[0 .. 4 * row_bytes]);
                const k: usize = @as(usize, li - constants.loop_start_layer) * 2;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(cp_q + k * constants.hidden_dim), h_raw[0..row_bytes].ptr, row_bytes));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(cp_kv + k * constants.hidden_dim), h_raw[row_bytes .. 2 * row_bytes].ptr, row_bytes));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(cp_q + (k + 1) * constants.hidden_dim), h_raw[2 * row_bytes .. 3 * row_bytes].ptr, row_bytes));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(cp_kv + (k + 1) * constants.hidden_dim), h_raw[3 * row_bytes .. 4 * row_bytes].ptr, row_bytes));
            }
        }

        // 24 physical layers
        layer_idx = 0;
        while (layer_idx < constants.layer_count) : (layer_idx += 1) {
            const lw: *LayerWeights = &self.layers[layer_idx];
            lw.is_global = constants.is_global_layer(layer_idx);

            if (lw.is_global) {
                lw.q_in_norm = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_q_inp_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
                lw.k_in_norm = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_k_inp_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
                lw.v_in_norm = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_v_inp_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
            } else {
                lw.attn_norm = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
            }
            lw.mlp_norm = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "ffn_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);

            // w_qrgs [6552, 2304]
            {
                const sb: [*]u8 = @ptrCast(d_stage);
                var off: usize = 0;
                off += stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_q.weight", constants.hidden_dim, constants.query_dim, h_raw, d_raw, @ptrCast(sb + off));
                off += stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_rel.weight", constants.hidden_dim, constants.r_dim, h_raw, d_raw, @ptrCast(sb + off));
                off += stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_gate.weight", constants.hidden_dim, constants.query_dim, h_raw, d_raw, @ptrCast(sb + off));
                off += stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_sink_proj.weight", constants.hidden_dim, constants.head_count, h_raw, d_raw, @ptrCast(sb + off));
                tiger.assert(off == @as(usize, constants.qrgs_out_dim) * constants.hidden_dim * 2);
                lw.w_qrgs = self.pack_linear(d_stage, constants.qrgs_out_dim, constants.hidden_dim, dense_kind, d_amax);
            }

            // w_kv [2048, 2304] + w_k/w_v views
            {
                const sb: [*]u8 = @ptrCast(d_stage);
                const kv_bytes: usize = @as(usize, constants.kv_dim) * constants.hidden_dim * 2;
                const b0 = stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_k.weight", constants.hidden_dim, constants.kv_dim, h_raw, d_raw, @ptrCast(sb));
                const b1 = stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_v.weight", constants.hidden_dim, constants.kv_dim, h_raw, d_raw, @ptrCast(sb + kv_bytes));
                tiger.assert(b0 == kv_bytes and b1 == kv_bytes);
                lw.w_kv = self.pack_linear(d_stage, 2 * constants.kv_dim, constants.hidden_dim, dense_kind, d_amax);

                lw.w_k = lw.w_kv;
                lw.w_k.rows = constants.kv_dim;
                lw.w_v = lw.w_kv;
                lw.w_v.rows = constants.kv_dim;
                const base_data: [*]u8 = @ptrCast(lw.w_kv.data.?);
                switch (dense_kind) {
                    .bf16 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * constants.hidden_dim * 2);
                    },
                    .fp8 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * constants.hidden_dim);
                        lw.w_v.row_scales = lw.w_kv.row_scales.? + constants.kv_dim;
                    },
                    .nvfp4 => {
                        lw.w_v.data = @ptrCast(base_data + @as(usize, constants.kv_dim) * (constants.hidden_dim / 2));
                        lw.w_v.scales = lw.w_kv.scales.? + @as(usize, constants.kv_dim) * (constants.hidden_dim / 16);
                    },
                }
            }

            // w_o [2304, 3072]
            {
                _ = stage_gguf_layer_bf16(&gf_main, layer_idx, "attn_output.weight", constants.query_dim, constants.hidden_dim, h_raw, d_raw, d_stage);
                lw.w_o = self.pack_linear(d_stage, constants.hidden_dim, constants.query_dim, dense_kind, d_amax);
            }

            // Short convolutions & clip/rel projections
            if (lw.is_global) {
                lw.k_sconv_w = self.upload_gguf_global_sconv_bf16(&gf_main, layer_idx, "attn_k_conv.weight", h_raw, d_raw);
                lw.v_sconv_w = self.upload_gguf_global_sconv_bf16(&gf_main, layer_idx, "attn_v_conv.weight", h_raw, d_raw);
                lw.k_sconv_proj = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_k_conv_proj.weight", constants.hidden_dim, constants.kv_head_count * constants.conv_tap_count, h_raw, d_raw);
                lw.v_sconv_proj = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_v_conv_proj.weight", constants.hidden_dim, constants.kv_head_count * constants.conv_tap_count, h_raw, d_raw);
            } else {
                lw.k_sconv_w = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_k_conv.weight", constants.conv_tap_count, constants.kv_dim, h_raw, d_raw);
                lw.v_sconv_w = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_v_conv.weight", constants.conv_tap_count, constants.kv_dim, h_raw, d_raw);
            }
            lw.attn_sconv_w = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_post_conv.weight", constants.conv_tap_count, constants.hidden_dim, h_raw, d_raw);
            lw.mlp_sconv_w = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "ffn_post_conv.weight", constants.conv_tap_count, constants.hidden_dim, h_raw, d_raw);
            lw.q_clip_scale = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_q_clip.weight", constants.head_dim, constants.head_count, h_raw, d_raw);
            lw.rel_clip_scale = self.upload_gguf_layer_bf16(&gf_main, layer_idx, "attn_rel_clip.weight", constants.rel_dim, constants.head_count, h_raw, d_raw);
            lw.rel_logits_proj = self.upload_gguf_transpose_bf16(&gf_main, layer_idx, "attn_rel_bias.weight", constants.rel_extent, constants.rel_dim, h_raw, d_raw);

            // Cross-pass controllers (layers 6..17)
            if (layer_idx >= constants.loop_start_layer and layer_idx < constants.loop_end_layer) {
                lw.has_controllers = true;
                const use_new_ctrl_names: bool = (gf_main.find("blk.6.loop_q_ctrl_down.weight") != null);
                const stems: [4][]const u8 = if (use_new_ctrl_names) .{ "q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl" } else .{ "attn", "k", "v", "mlp" };
                var s_idx: u32 = 0;
                while (s_idx < 4) : (s_idx += 1) {
                    const c_kind: constants.WeightKind = if (quant_mode == .nvfp4) .nvfp4 else .bf16;
                    const dn = std.fmt.bufPrint(&name_buf, "blk.{d}.loop_{s}_down.weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                    const dw_lin = self.load_gguf_linear_direct(&gf_main, dn, constants.gate_rank, 2 * constants.hidden_dim, c_kind, h_raw, d_raw, d_stage, d_amax);
                    const fn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.loop_{s}_forget.weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                    const fw_lin = self.load_gguf_linear_direct(&gf_main, fn_name, constants.hidden_dim, constants.gate_rank, c_kind, h_raw, d_raw, d_stage, d_amax);
                    const gn = std.fmt.bufPrint(&name_buf, "blk.{d}.loop_{s}_gain.weight", .{ layer_idx, stems[s_idx] }) catch unreachable;
                    const gw_lin = self.load_gguf_linear_direct(&gf_main, gn, constants.hidden_dim, constants.gate_rank, c_kind, h_raw, d_raw, d_stage, d_amax);

                    if (quant_mode == .nvfp4) {
                        lw.ctrl[s_idx].down_nvfp4 = dw_lin;
                        lw.ctrl[s_idx].forget_nvfp4 = fw_lin;
                        lw.ctrl[s_idx].gain_nvfp4 = gw_lin;
                    } else {
                        lw.ctrl[s_idx].down_w = dw_lin.data;
                        lw.ctrl[s_idx].forget_w = fw_lin.data;
                        lw.ctrl[s_idx].gain_w = gw_lin.data;
                    }
                }
            }

            // Router
            const rn = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_inp.weight", .{layer_idx}) catch unreachable;
            if (quant_mode == .nvfp4) {
                lw.router_nvfp4 = self.load_gguf_linear_direct(&gf_main, rn, constants.expert_count, constants.hidden_dim, .nvfp4, h_raw, d_raw, d_stage, d_amax);
            } else {
                lw.router_w = self.upload_gguf_bf16(&gf_main, rn, constants.hidden_dim, constants.expert_count, h_raw, d_raw);
            }
            const rbn = std.fmt.bufPrint(&name_buf, "blk.{d}.exp_probs_b.bias", .{layer_idx}) catch unreachable;
            lw.router_bias = self.upload_gguf_f32(&gf_main, rbn, constants.expert_count, h_raw);

            // MoE Expert Banks (kept lossless in GGUF quant blocks!)
            lw.w13 = self.load_gguf_expert_w13(&gf_main, layer_idx, h_raw, h_aux);
            lw.w2 = self.load_gguf_expert_w2(&gf_main, layer_idx, h_raw);
        }

        if (has_ablation) {
            self.load_ablation(&sf_abl, has_ablation_detectors);
        }

        tiger.assert_cuda(cuda.squid_cuda_device_sync());
        self.arena.assert_exact_fill();

        const elapsed_sec: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - start_ns)) / 1_000_000_000.0;
        const vram_gib: f64 = @as(f64, @floatFromInt(self.arena.capacity_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[weights] Loaded GGUF mode={s} in {d:.2}s | VRAM={d:.2} GiB (1 arena, {d} tensors) | MTP={s} | Ablation={s}\n",
            .{
                quant_mode.name(),
                elapsed_sec,
                vram_gib,
                self.arena.allocation_count,
                if (self.dspark.loaded != 0) "on" else "off",
                if (!self.ablation.active) "off" else if (self.ablation.has_detectors) "gated-lda" else "plain",
            },
        );
        return true;
    }

    fn upload_gguf_bf16(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        name: []const u8,
        cols: u64,
        rows: u64,
        h_raw: []u8,
        d_raw: [*]u8,
    ) *anyopaque {
        const info = gf.require(name);
        tiger.assert(info.dims[0] == cols and (info.n_dims == 1 or info.dims[1] == rows));
        const dst = self.arena.alloc_raw(rows * cols * 2);
        stream_gguf_to_bf16_dev(gf, info, rows, cols, h_raw, d_raw, dst);
        return dst;
    }

    fn upload_gguf_layer_bf16(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        layer_idx: u32,
        suffix: []const u8,
        cols: u64,
        rows: u64,
        h_raw: []u8,
        d_raw: [*]u8,
    ) *anyopaque {
        var nb: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, "blk.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
        return self.upload_gguf_bf16(gf, name, cols, rows, h_raw, d_raw);
    }

    fn upload_gguf_f32(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        name: []const u8,
        elems: u64,
        h_raw: []u8,
    ) [*]f32 {
        const info = gf.require(name);
        tiger.assert(info.ggml_type == 0);
        tiger.assert(info.byte_size == elems * 4);
        const dst = self.arena.alloc_many(f32, elems);
        gf.read_range(info.abs_offset, h_raw[0..@intCast(info.byte_size)]);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(dst), h_raw.ptr, @intCast(info.byte_size)));
        return dst;
    }

    fn upload_gguf_global_sconv_bf16(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        layer_idx: u32,
        suffix: []const u8,
        h_raw: []u8,
        d_raw: [*]u8,
    ) *anyopaque {
        var nb: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, "blk.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
        const info = gf.require(name);
        tiger.assert(info.dims[0] == constants.conv_tap_count and info.dims[1] == constants.kv_dim and info.ggml_type == 0);
        gf.read_range(info.abs_offset, h_raw[0..@intCast(info.byte_size)]);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_raw, h_raw.ptr, @intCast(info.byte_size)));
        const dst = self.arena.alloc_raw(@as(u64, constants.conv_tap_count) * constants.kv_dim * 2);
        cuda.squid_global_sconv_f32_to_bf16(@ptrCast(@alignCast(d_raw)), dst, constants.kv_dim);
        return dst;
    }

    fn upload_gguf_transpose_bf16(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        layer_idx: u32,
        suffix: []const u8,
        src_rows: u32,
        src_cols: u32,
        h_raw: []u8,
        d_raw: [*]u8,
    ) *anyopaque {
        var nb: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, "blk.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
        const info = gf.require(name);
        tiger.assert(info.dims[0] == src_cols and info.dims[1] == src_rows and info.ggml_type == 0);
        gf.read_range(info.abs_offset, h_raw[0..@intCast(info.byte_size)]);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_raw, h_raw.ptr, @intCast(info.byte_size)));
        const dst = self.arena.alloc_raw(@as(u64, src_rows) * src_cols * 2);
        cuda.squid_transpose_f32_to_bf16(@ptrCast(@alignCast(d_raw)), dst, @intCast(src_rows), @intCast(src_cols));
        return dst;
    }

    fn load_gguf_linear_direct(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        name: []const u8,
        rows: u32,
        cols: u32,
        kind: constants.WeightKind,
        h_raw: []u8,
        d_raw: [*]u8,
        d_stage: *anyopaque,
        d_amax: *f32,
    ) cuda.CLinearWeight {
        const info = gf.require(name);
        tiger.assert(info.dims[0] == cols and info.dims[1] == rows);
        stream_gguf_to_bf16_dev(gf, info, rows, cols, h_raw, d_raw, d_stage);
        return self.pack_linear(d_stage, rows, cols, kind, d_amax);
    }

    fn load_gguf_expert_w13(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        layer_idx: u32,
        h_raw: []u8,
        h_aux: []u8,
    ) cuda.CExpertBank {
        var nb: [128]u8 = undefined;
        const wn = std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_up_exps.weight", .{layer_idx}) catch unreachable;
        const info = gf.require(wn);
        const rows_per_exp: usize = 2 * constants.intermediate_dim;
        tiger.assert(info.n_dims == 3 and info.dims[0] == constants.hidden_dim and info.dims[1] == rows_per_exp and info.dims[2] == constants.expert_count);

        const rb: usize = @intCast(gguf.row_bytes(info.ggml_type, constants.hidden_dim));
        const exp_bytes: usize = rows_per_exp * rb;
        const data_ptr = self.arena.alloc_many(u8, info.byte_size);

        var e0: usize = 0;
        const max_chunk_e: usize = @min(h_raw.len / exp_bytes, h_aux.len / exp_bytes);
        tiger.assert(max_chunk_e >= 1);
        while (e0 < constants.expert_count) {
            const cnt_e = @min(max_chunk_e, constants.expert_count - e0);
            const cbytes = cnt_e * exp_bytes;
            gf.read_range(info.abs_offset + @as(u64, e0) * exp_bytes, h_raw[0..cbytes]);
            var er: usize = 0;
            while (er < cnt_e) : (er += 1) {
                const src_e = h_raw[er * exp_bytes .. (er + 1) * exp_bytes];
                const dst_e = h_aux[er * exp_bytes .. (er + 1) * exp_bytes];
                var i: usize = 0;
                while (i < constants.intermediate_dim) : (i += 1) {
                    @memcpy(dst_e[(2 * i) * rb ..][0..rb], src_e[i * rb ..][0..rb]);
                    @memcpy(dst_e[(2 * i + 1) * rb ..][0..rb], src_e[(constants.intermediate_dim + i) * rb ..][0..rb]);
                }
            }
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(data_ptr + e0 * exp_bytes, h_aux.ptr, cbytes));
            e0 += cnt_e;
        }

        var es_ptr: ?[*]f32 = null;
        const sn = std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_up_exps.scale", .{layer_idx}) catch unreachable;
        if (gf.find(sn) != null) {
            es_ptr = self.upload_gguf_f32(gf, sn, constants.expert_count, h_raw);
        }

        return .{
            .is_nvfp4 = 0,
            .gguf_type = @intCast(info.ggml_type),
            .n_experts = constants.expert_count,
            .rows = @intCast(rows_per_exp),
            .cols = constants.hidden_dim,
            .data = data_ptr,
            .scales = null,
            .expert_scale = es_ptr,
        };
    }

    fn load_gguf_expert_w2(
        self: *ModelWeights,
        gf: *const gguf.GgufFile,
        layer_idx: u32,
        h_raw: []u8,
    ) cuda.CExpertBank {
        var nb: [128]u8 = undefined;
        const wn = std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.weight", .{layer_idx}) catch unreachable;
        const info = gf.require(wn);
        tiger.assert(info.n_dims == 3 and info.dims[0] == constants.intermediate_dim and info.dims[1] == constants.hidden_dim and info.dims[2] == constants.expert_count);

        const data_ptr = self.arena.alloc_many(u8, info.byte_size);
        var off: u64 = 0;
        while (off < info.byte_size) {
            const chunk: usize = @intCast(tiger.min_u64(h_raw.len, info.byte_size - off));
            gf.read_range(info.abs_offset + off, h_raw[0..chunk]);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(data_ptr + @as(usize, @intCast(off)), h_raw.ptr, chunk));
            off += chunk;
        }

        var es_ptr: ?[*]f32 = null;
        const sn = std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.scale", .{layer_idx}) catch unreachable;
        if (gf.find(sn) != null) {
            es_ptr = self.upload_gguf_f32(gf, sn, constants.expert_count, h_raw);
        }

        return .{
            .is_nvfp4 = 0,
            .gguf_type = @intCast(info.ggml_type),
            .n_experts = constants.expert_count,
            .rows = constants.hidden_dim,
            .cols = constants.intermediate_dim,
            .data = data_ptr,
            .scales = null,
            .expert_scale = es_ptr,
        };
    }

    fn load_dspark_gguf(
        self: *ModelWeights,
        gf_draft: *const gguf.GgufFile,
        d_unembed_bf16: *const anyopaque,
        mtp_kind: constants.WeightKind,
        h_raw: []u8,
        d_raw: [*]u8,
        d_stage: *anyopaque,
        d_amax: *f32,
        has_pruned_draft_vocab: bool,
        draft_vocab_host: []const i32,
        draft_vocab_count: u32,
    ) void {
        // dspark_fuse.weight [4, 4] F32 -> row-wise softmax F32 [4, 4]
        {
            const fi = gf_draft.require("dspark_fuse.weight");
            tiger.assert(fi.ggml_type == 0 and fi.byte_size == 16 * 4);
            gf_draft.read_range(fi.abs_offset, h_raw[0 .. 16 * 4]);
            var alpha: [16]f32 = undefined;
            var row: u32 = 0;
            while (row < 4) : (row += 1) {
                var vals: [4]f64 = undefined;
                var max_val: f64 = -1e30;
                var col: u32 = 0;
                while (col < 4) : (col += 1) {
                    const bits = std.mem.readInt(u32, h_raw[(row * 4 + col) * 4 ..][0..4], .little);
                    const vf: f32 = @bitCast(bits);
                    vals[col] = @floatCast(vf);
                    if (vals[col] > max_val) max_val = vals[col];
                }
                var sum_exp: f64 = 0.0;
                col = 0;
                while (col < 4) : (col += 1) {
                    vals[col] = @exp(vals[col] - max_val);
                    sum_exp += vals[col];
                }
                col = 0;
                while (col < 4) : (col += 1) {
                    alpha[row * 4 + col] = @floatCast(vals[col] / sum_exp);
                }
            }
            self.dspark.fuse_alpha = self.arena.alloc_many(f32, 16);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.dspark.fuse_alpha.?), @ptrCast(&alpha), @sizeOf(@TypeOf(alpha))));
        }

        self.dspark.mask_embed = self.upload_gguf_bf16(gf_draft, "mask_embd.weight", constants.hidden_dim, 1, h_raw, d_raw);
        self.dspark.norm_w = self.upload_gguf_bf16(gf_draft, "output_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
        self.dspark.markov_dflare_norm = self.upload_gguf_bf16(gf_draft, "dspark_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
        self.dspark.markov_w1 = self.upload_gguf_bf16(gf_draft, "markov_w1.weight", constants.mtp_markov_rank, constants.vocab_padded_count, h_raw, d_raw);
        self.dspark.rnn_bias = self.upload_gguf_bf16(gf_draft, "joint_proj.bias", 3 * constants.mtp_markov_rank, 1, h_raw, d_raw);
        self.dspark.hidden_out_w = self.upload_gguf_bf16(gf_draft, "dspark_hidden_out.weight", constants.mtp_markov_rank, constants.hidden_dim, h_raw, d_raw);
        self.dspark.hidden_out_b = self.upload_gguf_bf16(gf_draft, "dspark_hidden_out.bias", constants.hidden_dim, 1, h_raw, d_raw);

        // Split joint_proj.weight [768, 5120] -> w_sp [768, 512], w_h [768, 2304], w_f [768, 2304]
        {
            const k_in: u64 = 2 * constants.mtp_markov_rank + 2 * constants.hidden_dim;
            const jpi = gf_draft.require("joint_proj.weight");
            tiger.assert(jpi.dims[0] == k_in and jpi.dims[1] == 3 * constants.mtp_markov_rank);
            stream_gguf_to_bf16_dev(gf_draft, jpi, 3 * constants.mtp_markov_rank, k_in, h_raw, d_raw, d_stage);
            self.dspark.rnn_w_sp = self.arena.alloc_raw(768 * 512 * 2);
            self.dspark.rnn_w_h = self.arena.alloc_raw(768 * @as(u64, constants.hidden_dim) * 2);
            self.dspark.rnn_w_f = self.arena.alloc_raw(768 * @as(u64, constants.hidden_dim) * 2);
            cuda.squid_split_dspark_rnn(d_stage, self.dspark.rnn_w_sp.?, self.dspark.rnn_w_h.?, self.dspark.rnn_w_f.?);
        }

        // Folded Markov head
        {
            var d_src_u_opt: ?*anyopaque = null;
            var src_u: *const anyopaque = d_unembed_bf16;
            if (has_pruned_draft_vocab) {
                tiger.assert(draft_vocab_host.len == draft_vocab_count);
                self.dspark.draft_vocab_size = @intCast(draft_vocab_count);
                self.dspark.draft_vocab_ids = self.arena.alloc_many(i32, draft_vocab_count);
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(
                    @ptrCast(self.dspark.draft_vocab_ids.?),
                    @ptrCast(draft_vocab_host.ptr),
                    draft_vocab_host.len * @sizeOf(i32),
                ));

                self.dspark.inv_draft_vocab_ids = self.arena.alloc_many(i32, constants.vocab_unpadded_count);
                const h_inv = tiger.alloc_host_zeroed(i32, constants.vocab_unpadded_count);
                defer tiger.free_host_slice(i32, h_inv);
                @memset(h_inv, -1);
                for (draft_vocab_host, 0..) |tok_id, idx| {
                    tiger.assert(tok_id >= 0 and tok_id < constants.vocab_unpadded_count);
                    h_inv[@intCast(tok_id)] = @intCast(idx);
                }
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(
                    @ptrCast(self.dspark.inv_draft_vocab_ids.?),
                    @ptrCast(h_inv.ptr),
                    h_inv.len * @sizeOf(i32),
                ));

                const sub_bytes: u64 = @as(u64, draft_vocab_count) * constants.hidden_dim * 2;
                tiger.assert_cuda(cuda.squid_cuda_malloc(&d_src_u_opt, @intCast(sub_bytes)));
                src_u = d_src_u_opt.?;
                cuda.squid_gather_bf16_rows(d_unembed_bf16, self.dspark.draft_vocab_ids.?, d_src_u_opt.?, @intCast(draft_vocab_count), constants.hidden_dim);
                self.dspark.unembed_draft = self.pack_linear(src_u, draft_vocab_count, constants.hidden_dim, mtp_kind, d_amax);
            } else {
                self.dspark.draft_vocab_size = constants.vocab_unpadded_count;
                self.dspark.draft_vocab_ids = null;
                self.dspark.inv_draft_vocab_ids = null;
                self.dspark.unembed_draft = self.unembed;
            }
            defer if (d_src_u_opt) |p_u| tiger.assert_cuda(cuda.squid_cuda_free(p_u));

            self.dspark.fold_b = self.arena.alloc_many(f32, draft_vocab_count);
            var d_fold_bf16_opt: ?*anyopaque = null;
            const fold_bytes: u64 = @as(u64, draft_vocab_count) * constants.mtp_markov_rank * 2;
            tiger.assert_cuda(cuda.squid_cuda_malloc(&d_fold_bf16_opt, @intCast(fold_bytes)));
            const d_fold_bf16: *anyopaque = d_fold_bf16_opt.?;
            defer tiger.assert_cuda(cuda.squid_cuda_free(d_fold_bf16));

            cuda.squid_compute_dspark_fold(
                src_u,
                self.dspark.hidden_out_w.?,
                self.dspark.hidden_out_b.?,
                self.dspark.fold_b.?,
                d_fold_bf16,
                @intCast(draft_vocab_count),
                constants.hidden_dim,
                constants.mtp_markov_rank,
            );
            self.dspark.fold_w = self.pack_linear(d_fold_bf16, draft_vocab_count, constants.mtp_markov_rank, mtp_kind, d_amax);
            tiger.assert_cuda(cuda.squid_cuda_device_sync());
            tiger.print_stdout(
                "[dspark] GGUF draft vocab size={d} ({s})\n",
                .{ draft_vocab_count, if (has_pruned_draft_vocab) "pruned" else "full" },
            );
        }

        // 3 DSpark layers
        var l: u32 = 0;
        while (l < constants.mtp_layer_count) : (l += 1) {
            const dl: *cuda.CDsparkLayerWeights = &self.dspark.layers[l];
            dl.is_local = if (constants.is_mtp_local_layer(l)) 1 else 0;
            var nbuf: [128]u8 = undefined;
            const sb: [*]u8 = @ptrCast(d_stage);

            dl.input_norm = self.upload_gguf_layer_bf16(gf_draft, l, "attn_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
            dl.post_norm = self.upload_gguf_layer_bf16(gf_draft, l, "ffn_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
            dl.dflare_norm = self.upload_gguf_layer_bf16(gf_draft, l, "dflare_norm.weight", constants.hidden_dim, 1, h_raw, d_raw);
            dl.q_norm = self.upload_gguf_layer_bf16(gf_draft, l, "attn_q_norm.weight", constants.head_dim, 1, h_raw, d_raw);
            dl.k_norm = self.upload_gguf_layer_bf16(gf_draft, l, "attn_k_norm.weight", constants.head_dim, 1, h_raw, d_raw);
            dl.k_norm_ctx = self.upload_gguf_layer_bf16(gf_draft, l, "attn_k_ctx_norm.weight", constants.head_dim, 1, h_raw, d_raw);

            // w_qkv [5120, 2304]
            const qb: usize = @as(usize, constants.query_dim) * constants.hidden_dim * 2;
            const kb: usize = @as(usize, constants.kv_dim) * constants.hidden_dim * 2;
            _ = stage_gguf_layer_bf16(gf_draft, l, "attn_q.weight", constants.hidden_dim, constants.query_dim, h_raw, d_raw, @ptrCast(sb));
            _ = stage_gguf_layer_bf16(gf_draft, l, "attn_k.weight", constants.hidden_dim, constants.kv_dim, h_raw, d_raw, @ptrCast(sb + qb));
            _ = stage_gguf_layer_bf16(gf_draft, l, "attn_v.weight", constants.hidden_dim, constants.kv_dim, h_raw, d_raw, @ptrCast(sb + qb + kb));
            dl.w_qkv = self.pack_linear(d_stage, constants.dspark_qkv_out_dim, constants.hidden_dim, mtp_kind, d_amax);

            // w_o [2304, 3072]
            _ = stage_gguf_layer_bf16(gf_draft, l, "attn_output.weight", constants.query_dim, constants.hidden_dim, h_raw, d_raw, d_stage);
            dl.w_o = self.pack_linear(d_stage, constants.hidden_dim, constants.query_dim, mtp_kind, d_amax);

            // w_gu [18432, 2304]
            const gb: usize = @as(usize, constants.mtp_intermediate_dim) * constants.hidden_dim * 2;
            _ = stage_gguf_layer_bf16(gf_draft, l, "ffn_gate.weight", constants.hidden_dim, constants.mtp_intermediate_dim, h_raw, d_raw, @ptrCast(sb));
            _ = stage_gguf_layer_bf16(gf_draft, l, "ffn_up.weight", constants.hidden_dim, constants.mtp_intermediate_dim, h_raw, d_raw, @ptrCast(sb + gb));
            dl.w_gu = self.pack_linear(d_stage, 2 * constants.mtp_intermediate_dim, constants.hidden_dim, mtp_kind, d_amax);

            // w_down [2304, 9216]
            const dn = std.fmt.bufPrint(&nbuf, "blk.{d}.ffn_down.weight", .{l}) catch unreachable;
            dl.w_down = self.load_gguf_linear_direct(gf_draft, dn, constants.hidden_dim, constants.mtp_intermediate_dim, mtp_kind, h_raw, d_raw, d_stage, d_amax);

            // w_kv_ctx [2048, 2304]
            _ = stage_gguf_layer_bf16(gf_draft, l, "attn_k_ctx.weight", constants.hidden_dim, constants.kv_dim, h_raw, d_raw, @ptrCast(sb));
            _ = stage_gguf_layer_bf16(gf_draft, l, "attn_v_ctx.weight", constants.hidden_dim, constants.kv_dim, h_raw, d_raw, @ptrCast(sb + kb));
            dl.w_kv_ctx = self.pack_linear(d_stage, 2 * constants.kv_dim, constants.hidden_dim, mtp_kind, d_amax);
        }
        self.dspark.loaded = 1;
    }

    fn upload_bf16(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, name: []const u8, elems: u64) *anyopaque {
        const bytes: u64 = elems * 2;
        const dst: *anyopaque = self.arena.alloc_raw(bytes);
        upload_raw(sf, name, dst, bytes);
        return dst;
    }

    fn upload_layer_bf16(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, layer_idx: u32, suffix: []const u8, elems: u64) *anyopaque {
        var name_buf: [128]u8 = undefined;
        const tname = std.fmt.bufPrint(&name_buf, "model.llm.layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
        return self.upload_bf16(sf, tname, elems);
    }

    fn upload_f32(self: *ModelWeights, sf: *const safetensors.SafetensorsFile, name: []const u8, elems: u64) [*]f32 {
        const bytes: u64 = elems * 4;
        const dst: [*]f32 = self.arena.alloc_many(f32, elems);
        upload_raw(sf, name, @ptrCast(dst), bytes);
        return dst;
    }

    fn pack_linear(
        self: *ModelWeights,
        src_bf16_dev: *const anyopaque,
        rows: u32,
        cols: u32,
        kind: constants.WeightKind,
        d_amax: *f32,
    ) cuda.CLinearWeight {
        tiger.assert(rows > 0 and cols > 0 and cols % 16 == 0);
        var lw: cuda.CLinearWeight = .{
            .kind = @intFromEnum(kind),
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .global_scale = 1.0,
        };
        switch (kind) {
            .bf16 => {
                const bytes: u64 = @as(u64, rows) * cols * 2;
                lw.data = self.arena.alloc_raw(bytes);
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2d_async(lw.data.?, src_bf16_dev, @intCast(bytes), null));
            },
            .fp8 => {
                lw.data = self.arena.alloc_raw(@as(u64, rows) * cols);
                lw.row_scales = self.arena.alloc_many(f32, rows);
                cuda.squid_pack_bf16_to_fp8(src_bf16_dev, lw.data.?, lw.row_scales.?, @intCast(rows), @intCast(cols));
            },
            .nvfp4 => {
                lw.data = self.arena.alloc_raw(@as(u64, rows) * (cols / 2));
                lw.scales = self.arena.alloc_many(u8, @as(u64, rows) * (cols / 16));
                lw.global_scale = cuda.squid_pack_bf16_to_nvfp4(
                    src_bf16_dev,
                    lw.data.?,
                    lw.scales.?,
                    d_amax,
                    @intCast(rows),
                    @intCast(cols),
                );
            },
        }
        return lw;
    }

    fn load_expert_bank(
        self: *ModelWeights,
        sf: *const safetensors.SafetensorsFile,
        layer_idx: u32,
        wname: []const u8,
        rows: u32,
        cols: u32,
        quant_mode: constants.QuantMode,
        d_stage: *anyopaque,
    ) cuda.CExpertBank {
        var eb: cuda.CExpertBank = .{
            .n_experts = constants.expert_count,
            .rows = @intCast(rows),
            .cols = @intCast(cols),
        };
        var nbuf: [128]u8 = undefined;
        const ckpt_nvfp4: bool = constants.is_nvfp4_expert_layer(layer_idx);
        if (ckpt_nvfp4) {
            eb.is_nvfp4 = 1;
            const w_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 2);
            const s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 16);
            eb.data = self.arena.alloc_many(u8, w_bytes);
            eb.scales = self.arena.alloc_many(u8, s_bytes);

            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, wn, @ptrCast(eb.data.?), w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, sn, @ptrCast(eb.scales.?), s_bytes);

            const an = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_amax", .{ layer_idx, wname }) catch unreachable;
            const amax_slice = sf.data_slice(sf.lookup(an));
            const dn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_outer_scale_denominator", .{ layer_idx, wname }) catch unreachable;
            const denom_slice = sf.data_slice(sf.lookup(dn));
            tiger.assert(amax_slice.len == constants.expert_count * 4);
            tiger.assert(denom_slice.len >= 4);

            const denom: f32 = @bitCast(std.mem.readInt(u32, denom_slice[0..4], .little));
            const inv_denom: f32 = 1.0 / denom;
            var h_scales: [constants.expert_count]f32 = undefined;
            var e: u32 = 0;
            while (e < constants.expert_count) : (e += 1) {
                const amax_val: f32 = @bitCast(std.mem.readInt(u32, amax_slice[e * 4 ..][0..4], .little));
                h_scales[e] = amax_val * inv_denom;
            }
            eb.expert_scale = self.arena.alloc_many(f32, constants.expert_count);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(eb.expert_scale.?), @ptrCast(&h_scales), @sizeOf(@TypeOf(h_scales))));
        } else if (quant_mode != .nvfp4) {
            eb.is_nvfp4 = 0;
            const w_bytes: u64 = @as(u64, constants.expert_count) * rows * cols;
            const s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 32);
            eb.data = self.arena.alloc_many(u8, w_bytes);
            eb.scales = self.arena.alloc_many(u8, s_bytes);
            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, wn, @ptrCast(eb.data.?), w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, sn, @ptrCast(eb.scales.?), s_bytes);
        } else {
            eb.is_nvfp4 = 1;
            const mx_w_bytes: u64 = @as(u64, constants.expert_count) * rows * cols;
            const mx_s_bytes: u64 = @as(u64, constants.expert_count) * rows * (cols / 32);
            const tmp_w: [*]u8 = @ptrCast(d_stage);
            const tmp_s: [*]u8 = tmp_w + @as(usize, @intCast(mx_w_bytes));
            const wn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, wn, @ptrCast(tmp_w), mx_w_bytes);
            const sn = std.fmt.bufPrint(&nbuf, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ layer_idx, wname }) catch unreachable;
            upload_raw(sf, sn, @ptrCast(tmp_s), mx_s_bytes);

            eb.expert_scale = self.arena.alloc_many(f32, constants.expert_count);
            eb.data = self.arena.alloc_many(u8, @as(u64, constants.expert_count) * rows * (cols / 2));
            eb.scales = self.arena.alloc_many(u8, @as(u64, constants.expert_count) * rows * (cols / 16));
            cuda.squid_convert_mxfp8_to_nvfp4_experts(
                tmp_w,
                tmp_s,
                eb.data.?,
                eb.scales.?,
                eb.expert_scale.?,
                constants.expert_count,
                @intCast(rows),
                @intCast(cols),
            );
        }
        return eb;
    }

    fn load_dspark(
        self: *ModelWeights,
        sf_main: *const safetensors.SafetensorsFile,
        sf_mtp: *const safetensors.SafetensorsFile,
        mtp_kind: constants.WeightKind,
        d_stage: *anyopaque,
        d_amax: *f32,
        has_pruned_draft_vocab: bool,
        draft_vocab_host: []const i32,
        draft_vocab_count: u32,
    ) void {
        // fuse_weight [4, 4] BF16 -> row-wise softmax F32 [4, 4]
        const fw_slice = sf_mtp.data_slice(sf_mtp.lookup("fuse_weight"));
        tiger.assert(fw_slice.len == 16 * 2);
        var alpha: [16]f32 = undefined;
        var row: u32 = 0;
        while (row < 4) : (row += 1) {
            var vals: [4]f64 = undefined;
            var max_val: f64 = -1e30;
            var col: u32 = 0;
            while (col < 4) : (col += 1) {
                const bf16_bits: u16 = std.mem.readInt(u16, fw_slice[(row * 4 + col) * 2 ..][0..2], .little);
                const f32_bits: u32 = @as(u32, bf16_bits) << 16;
                const val_f32: f32 = @bitCast(f32_bits);
                vals[col] = @floatCast(val_f32);
                if (vals[col] > max_val) max_val = vals[col];
            }
            var sum_exp: f64 = 0.0;
            col = 0;
            while (col < 4) : (col += 1) {
                vals[col] = @exp(vals[col] - max_val);
                sum_exp += vals[col];
            }
            col = 0;
            while (col < 4) : (col += 1) {
                alpha[row * 4 + col] = @floatCast(vals[col] / sum_exp);
            }
        }
        self.dspark.fuse_alpha = self.arena.alloc_many(f32, 16);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.dspark.fuse_alpha.?), @ptrCast(&alpha), @sizeOf(@TypeOf(alpha))));

        self.dspark.mask_embed = self.upload_bf16(sf_mtp, "mask_embed", constants.hidden_dim);
        self.dspark.norm_w = self.upload_bf16(sf_mtp, "norm.weight", constants.hidden_dim);
        self.dspark.markov_dflare_norm = self.upload_bf16(sf_mtp, "markov_head.dflare_norm.weight", constants.hidden_dim);
        self.dspark.markov_w1 = self.upload_bf16(sf_mtp, "markov_head.markov_w1.weight", @as(u64, constants.vocab_padded_count) * constants.mtp_markov_rank);
        self.dspark.rnn_bias = self.upload_bf16(sf_mtp, "markov_head.rnn_proj.bias", 3 * constants.mtp_markov_rank);
        self.dspark.hidden_out_w = self.upload_bf16(sf_mtp, "markov_head.hidden_out.weight", @as(u64, constants.hidden_dim) * constants.mtp_markov_rank);
        self.dspark.hidden_out_b = self.upload_bf16(sf_mtp, "markov_head.hidden_out.bias", constants.hidden_dim);

        // Split rnn_proj.weight [768, 5120] -> w_sp [768, 512], w_h [768, 2304], w_f [768, 2304]
        {
            const rnn_bytes: u64 = (3 * constants.mtp_markov_rank) * (2 * constants.mtp_markov_rank + 2 * constants.hidden_dim) * 2;
            upload_raw(sf_mtp, "markov_head.rnn_proj.weight", d_stage, rnn_bytes);
            self.dspark.rnn_w_sp = self.arena.alloc_raw(768 * 512 * 2);
            self.dspark.rnn_w_h = self.arena.alloc_raw(768 * @as(u64, constants.hidden_dim) * 2);
            self.dspark.rnn_w_f = self.arena.alloc_raw(768 * @as(u64, constants.hidden_dim) * 2);
            cuda.squid_split_dspark_rnn(d_stage, self.dspark.rnn_w_sp.?, self.dspark.rnn_w_h.?, self.dspark.rnn_w_f.?);
        }

        // Folded Markov head (with optional pruned 32K/49K draft vocab)
        {
            const ui = sf_main.lookup("model.llm.unembed.weight");
            const h_unembed = sf_main.data_slice(ui);
            if (has_pruned_draft_vocab) {
                tiger.assert(draft_vocab_host.len == draft_vocab_count);
                self.dspark.draft_vocab_size = @intCast(draft_vocab_count);
                self.dspark.draft_vocab_ids = self.arena.alloc_many(i32, draft_vocab_count);
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(
                    @ptrCast(self.dspark.draft_vocab_ids.?),
                    @ptrCast(draft_vocab_host.ptr),
                    draft_vocab_host.len * @sizeOf(i32),
                ));

                // Inverse LUT [VOCAB_UNPADDED] -> draft idx or -1 for exact Leviathan/Chen rejection sampling
                self.dspark.inv_draft_vocab_ids = self.arena.alloc_many(i32, constants.vocab_unpadded_count);
                const h_inv = tiger.alloc_host_zeroed(i32, constants.vocab_unpadded_count);
                defer tiger.free_host_slice(i32, h_inv);
                @memset(h_inv, -1);
                for (draft_vocab_host, 0..) |tok_id, idx| {
                    tiger.assert(tok_id >= 0 and tok_id < constants.vocab_unpadded_count);
                    h_inv[@intCast(tok_id)] = @intCast(idx);
                }
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(
                    @ptrCast(self.dspark.inv_draft_vocab_ids.?),
                    @ptrCast(h_inv.ptr),
                    h_inv.len * @sizeOf(i32),
                ));

                const row_bytes: usize = constants.hidden_dim * 2;
                const h_sub = tiger.alloc_host_zeroed(u8, @as(usize, draft_vocab_count) * row_bytes);
                defer tiger.free_host_slice(u8, h_sub);
                for (draft_vocab_host, 0..) |tok_id, idx| {
                    const src_off: usize = @as(usize, @intCast(tok_id)) * row_bytes;
                    @memcpy(h_sub[idx * row_bytes ..][0..row_bytes], h_unembed[src_off..][0..row_bytes]);
                }
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_stage, @ptrCast(h_sub.ptr), h_sub.len));
                self.dspark.unembed_draft = self.pack_linear(d_stage, draft_vocab_count, constants.hidden_dim, mtp_kind, d_amax);
            } else {
                self.dspark.draft_vocab_size = constants.vocab_unpadded_count;
                self.dspark.draft_vocab_ids = null;
                self.dspark.inv_draft_vocab_ids = null;
                const usable_bytes: usize = @as(usize, constants.vocab_unpadded_count) * constants.hidden_dim * 2;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_stage, @ptrCast(h_unembed.ptr), usable_bytes));
                self.dspark.unembed_draft = self.unembed;
            }

            self.dspark.fold_b = self.arena.alloc_many(f32, draft_vocab_count);
            var d_fold_bf16_opt: ?*anyopaque = null;
            const fold_bytes: u64 = @as(u64, draft_vocab_count) * constants.mtp_markov_rank * 2;
            tiger.assert_cuda(cuda.squid_cuda_malloc(&d_fold_bf16_opt, @intCast(fold_bytes)));
            const d_fold_bf16: *anyopaque = d_fold_bf16_opt.?;
            defer tiger.assert_cuda(cuda.squid_cuda_free(d_fold_bf16));

            cuda.squid_compute_dspark_fold(
                d_stage,
                self.dspark.hidden_out_w.?,
                self.dspark.hidden_out_b.?,
                self.dspark.fold_b.?,
                d_fold_bf16,
                @intCast(draft_vocab_count),
                constants.hidden_dim,
                constants.mtp_markov_rank,
            );
            self.dspark.fold_w = self.pack_linear(d_fold_bf16, draft_vocab_count, constants.mtp_markov_rank, mtp_kind, d_amax);
            tiger.assert_cuda(cuda.squid_cuda_device_sync());
            tiger.print_stdout(
                "[dspark] Draft vocab size={d} ({s})\n",
                .{ draft_vocab_count, if (has_pruned_draft_vocab) "pruned" else "full" },
            );
        }

        // 3 DSpark layers
        var l: u32 = 0;
        while (l < constants.mtp_layer_count) : (l += 1) {
            const dl: *cuda.CDsparkLayerWeights = &self.dspark.layers[l];
            dl.is_local = if (constants.is_mtp_local_layer(l)) 1 else 0;
            var nbuf: [128]u8 = undefined;
            const stage_bytes_ptr: [*]u8 = @ptrCast(d_stage);

            dl.input_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, constants.hidden_dim);
            dl.post_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, constants.hidden_dim);
            dl.dflare_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.dflare_norm.weight", .{l}) catch unreachable, constants.hidden_dim);
            dl.q_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, constants.head_dim);
            dl.k_norm = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, constants.head_dim);
            dl.k_norm_ctx = self.upload_bf16(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm_ctx.weight", .{l}) catch unreachable, constants.head_dim);

            // w_qkv [5120, 2304]
            const qb: u64 = @as(u64, constants.query_dim) * constants.hidden_dim * 2;
            const kb: u64 = @as(u64, constants.kv_dim) * constants.hidden_dim * 2;
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr), qb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(qb))), kb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(qb + kb))), kb);
            dl.w_qkv = self.pack_linear(d_stage, constants.dspark_qkv_out_dim, constants.hidden_dim, mtp_kind, d_amax);

            // w_o [2304, 3072]
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, d_stage, @as(u64, constants.hidden_dim) * constants.query_dim * 2);
            dl.w_o = self.pack_linear(d_stage, constants.hidden_dim, constants.query_dim, mtp_kind, d_amax);

            // w_gu [18432, 2304]
            const gb: u64 = @as(u64, constants.mtp_intermediate_dim) * constants.hidden_dim * 2;
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr), gb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(gb))), gb);
            dl.w_gu = self.pack_linear(d_stage, 2 * constants.mtp_intermediate_dim, constants.hidden_dim, mtp_kind, d_amax);

            // w_down [2304, 9216]
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, d_stage, @as(u64, constants.hidden_dim) * constants.mtp_intermediate_dim * 2);
            dl.w_down = self.pack_linear(d_stage, constants.hidden_dim, constants.mtp_intermediate_dim, mtp_kind, d_amax);

            // w_kv_ctx [2048, 2304]
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj_ctx.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr), kb);
            upload_raw(sf_mtp, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj_ctx.weight", .{l}) catch unreachable, @ptrCast(stage_bytes_ptr + @as(usize, @intCast(kb))), kb);
            dl.w_kv_ctx = self.pack_linear(d_stage, 2 * constants.kv_dim, constants.hidden_dim, mtp_kind, d_amax);
        }
        self.dspark.loaded = 1;
    }

    fn load_ablation(self: *ModelWeights, sf_abl: *const safetensors.SafetensorsFile, has_detectors: bool) void {
        const visits: usize = constants.visit_count;
        const hidden: usize = constants.hidden_dim;
        const dir_elems: usize = visits * hidden;
        const plain_layout: bool = sf_abl.lookup_optional("attn_w") != null;

        const dir_host = tiger.alloc_host_zeroed(f32, dir_elems);
        defer tiger.free_host_slice(f32, dir_host);

        var w_host: [constants.visit_count]f32 = @splat(0.0);
        var c_host: [constants.visit_count]f32 = @splat(0.0);
        var tau_host: [constants.visit_count]f32 = @splat(0.0);
        var params_host: [constants.visit_count * 4]f32 = @splat(0.0);

        var gate_val: [1]f32 = .{0.0};
        if (!plain_layout) {
            read_f32_or_bf16(sf_abl, "gate", &gate_val);
        }
        const floor_val: f32 = if (!plain_layout and gate_val[0] != 0.0) 0.0 else -std.math.floatMax(f32);

        // Attention ablation
        read_f32_or_bf16(sf_abl, "attn_dirs", dir_host);
        normalize_rows_f32(dir_host, visits, hidden);
        self.ablation.attn_dirs = self.arena.alloc_many(f32, dir_elems);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.attn_dirs.?), @ptrCast(dir_host.ptr), dir_elems * 4));

        read_f32_or_bf16(sf_abl, if (plain_layout) "attn_w" else "attn_weight", &w_host);
        if (plain_layout) @memset(&c_host, 0.0) else read_f32_or_bf16(sf_abl, "attn_offset", &c_host);
        if (has_detectors) {
            read_f32_or_bf16(sf_abl, "attn_det", dir_host);
            self.ablation.attn_det = self.arena.alloc_many(f32, dir_elems);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.attn_det.?), @ptrCast(dir_host.ptr), dir_elems * 4));
            read_f32_or_bf16(sf_abl, "attn_threshold", &tau_host);
        } else {
            @memset(&tau_host, 0.0);
        }
        for (0..visits) |v| {
            self.ablation.attn_nonzero[v] = (w_host[v] != 0.0);
            params_host[4 * v + 0] = w_host[v];
            params_host[4 * v + 1] = c_host[v];
            params_host[4 * v + 2] = floor_val;
            params_host[4 * v + 3] = tau_host[v];
        }
        self.ablation.attn_params = self.arena.alloc_many(f32, visits * 4);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.attn_params.?), @ptrCast(&params_host), visits * 4 * 4));

        // MLP ablation
        read_f32_or_bf16(sf_abl, "mlp_dirs", dir_host);
        normalize_rows_f32(dir_host, visits, hidden);
        self.ablation.mlp_dirs = self.arena.alloc_many(f32, dir_elems);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.mlp_dirs.?), @ptrCast(dir_host.ptr), dir_elems * 4));

        read_f32_or_bf16(sf_abl, if (plain_layout) "mlp_w" else "mlp_weight", &w_host);
        if (plain_layout) @memset(&c_host, 0.0) else read_f32_or_bf16(sf_abl, "mlp_offset", &c_host);
        if (has_detectors) {
            read_f32_or_bf16(sf_abl, "mlp_det", dir_host);
            self.ablation.mlp_det = self.arena.alloc_many(f32, dir_elems);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.mlp_det.?), @ptrCast(dir_host.ptr), dir_elems * 4));
            read_f32_or_bf16(sf_abl, "mlp_threshold", &tau_host);
        } else {
            @memset(&tau_host, 0.0);
        }
        for (0..visits) |v| {
            self.ablation.mlp_nonzero[v] = (w_host[v] != 0.0);
            params_host[4 * v + 0] = w_host[v];
            params_host[4 * v + 1] = c_host[v];
            params_host[4 * v + 2] = floor_val;
            params_host[4 * v + 3] = tau_host[v];
        }
        self.ablation.mlp_params = self.arena.alloc_many(f32, visits * 4);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(@ptrCast(self.ablation.mlp_params.?), @ptrCast(&params_host), visits * 4 * 4));

        self.ablation.active = true;
        self.ablation.has_detectors = has_detectors;
    }
};

fn read_f32_or_bf16(sf: *const safetensors.SafetensorsFile, name: []const u8, dst: []f32) void {
    const info = sf.lookup(name);
    const raw = sf.data_slice(info);
    if (info.byte_length == dst.len * 4) {
        for (0..dst.len) |i| {
            const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
            dst[i] = @bitCast(bits);
        }
    } else if (info.byte_length == dst.len * 2) {
        for (0..dst.len) |i| {
            const b16: u32 = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
            dst[i] = @bitCast(b16 << 16);
        }
    } else {
        tiger.print_stderr("[ablation] Unexpected byte_length for {s}: got {d}, expected {d} or {d}\n", .{ name, info.byte_length, dst.len * 4, dst.len * 2 });
        @panic("Invalid ablation tensor size");
    }
}

fn normalize_rows_f32(buf: []f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        const row = buf[r * cols .. (r + 1) * cols];
        var ss: f64 = 0.0;
        for (row) |v| {
            const d: f64 = v;
            ss += d * d;
        }
        const norm: f32 = @max(@as(f32, @floatCast(@sqrt(ss))), 1e-12);
        for (row) |*v| v.* /= norm;
    }
}

fn upload_raw(sf: *const safetensors.SafetensorsFile, name: []const u8, dst_dev: *anyopaque, expected_bytes: u64) void {
    const info = sf.lookup(name);
    tiger.assert(info.byte_length == expected_bytes);
    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(dst_dev, sf.data_ptr(info), @intCast(expected_bytes)));
}

fn stage_layer_raw(sf: *const safetensors.SafetensorsFile, layer_idx: u32, suffix: []const u8, dst_dev: [*]u8, expected_bytes: u64) usize {
    var name_buf: [128]u8 = undefined;
    const tname = std.fmt.bufPrint(&name_buf, "model.llm.layers.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
    upload_raw(sf, tname, @ptrCast(dst_dev), expected_bytes);
    return @intCast(expected_bytes);
}

fn try_load_draft_vocab(model_directory: []const u8, quant_mode: constants.QuantMode) ?[]i32 {
    if (quant_mode == .exact) return null;
    const env_dv = tiger.getenv("SQUID_DRAFT_VOCAB");
    var prefer_49k: bool = false;
    if (env_dv) |ptr| {
        const s = std.mem.span(ptr);
        if (std.mem.eql(u8, s, "full")) return null;
        if (std.mem.eql(u8, s, "49k")) prefer_49k = true;
    }
    const candidates: [2][]const u8 = if (prefer_49k)
        .{ "/draft_vocab_49k.bin", "/draft_vocab_32k.bin" }
    else
        .{ "/draft_vocab_32k.bin", "/draft_vocab_49k.bin" };

    var path_buf: [1024]u8 = undefined;
    for (candidates) |suffix| {
        const path_z = tiger.format_path_z(&path_buf, model_directory, suffix);
        const fd: i32 = tiger.open(path_z, tiger.O_RDONLY);
        if (fd < 0) continue;
        defer _ = tiger.close(fd);
        const sz_i64: i64 = tiger.lseek(fd, 0, tiger.SEEK_END);
        if (sz_i64 <= 0 or @mod(sz_i64, 4) != 0) continue;
        _ = tiger.lseek(fd, 0, tiger.SEEK_SET);
        const count: usize = @intCast(@divExact(sz_i64, 4));
        if (count == 0 or count > constants.vocab_unpadded_count) continue;
        const ids = tiger.alloc_host_zeroed(i32, count);
        const raw_dst: [*]u8 = @ptrCast(ids.ptr);
        const want_bytes: usize = count * 4;
        var got: usize = 0;
        var iter: u32 = 0;
        while (got < want_bytes and iter < 4096) : (iter += 1) {
            const n: isize = tiger.read(fd, raw_dst + got, want_bytes - got);
            if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
            if (n <= 0) break;
            got += @intCast(n);
        }
        if (got == want_bytes) {
            var all_valid: bool = true;
            for (ids) |id| {
                if (id < 0 or id >= @as(i32, @intCast(constants.vocab_unpadded_count))) {
                    all_valid = false;
                    break;
                }
            }
            if (all_valid) return ids;
        }
        tiger.free_host_slice(i32, ids);
    }
    return null;
}

/// Pass 1: exact 256-byte-aligned VRAM byte accounting matching Pass 2 allocation-for-allocation.
fn compute_total_vram_bytes(
    quant_mode: constants.QuantMode,
    dense_kind: constants.WeightKind,
    mtp_kind: constants.WeightKind,
    embed_on_host: bool,
    has_mtp: bool,
    has_pruned_draft_vocab: bool,
    draft_vocab_count: u32,
    has_ablation: bool,
    has_ablation_detectors: bool,
) u64 {
    const a = constants.vram_alignment_bytes;
    var total: u64 = 0;

    if (!embed_on_host) {
        total += tiger.align_up_u64(@as(u64, constants.vocab_padded_count) * constants.hidden_dim * 2, a);
    }
    total += tiger.align_up_u64(constants.hidden_dim * 2, a); // embed_norm
    total += tiger.align_up_u64(constants.hidden_dim * 2, a); // final_norm
    total += constants.linear_weight_vram_bytes(constants.vocab_unpadded_count, constants.hidden_dim, dense_kind); // unembed

    var l: u32 = 1;
    while (l < constants.layer_count) : (l += 1) {
        const elems: u64 = if (constants.is_global_layer(l)) 3 * constants.hidden_dim else constants.hidden_dim;
        total += tiger.align_up_u64(elems * 4, a);
    }
    l = 0;
    while (l < constants.layer_count) : (l += 1) {
        total += tiger.align_up_u64(constants.hidden_dim * 4, a);
    }
    total += tiger.align_up_u64(constants.hidden_dim * 4, a); // output_query
    total += tiger.align_up_u64(24 * constants.hidden_dim * 4, a); // cross_pass_queries
    total += tiger.align_up_u64(24 * constants.hidden_dim * 4, a); // cross_pass_kv_queries

    l = 0;
    while (l < constants.layer_count) : (l += 1) {
        const is_glob = constants.is_global_layer(l);
        if (is_glob) {
            total += 3 * tiger.align_up_u64(constants.hidden_dim * 2, a);
        } else {
            total += tiger.align_up_u64(constants.hidden_dim * 2, a);
        }
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // mlp_norm

        total += constants.linear_weight_vram_bytes(constants.qrgs_out_dim, constants.hidden_dim, dense_kind);
        total += constants.linear_weight_vram_bytes(2 * constants.kv_dim, constants.hidden_dim, dense_kind);
        total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.query_dim, dense_kind);

        if (is_glob) {
            total += 2 * tiger.align_up_u64(constants.conv_tap_count * constants.kv_dim * 2, a);
            total += 2 * tiger.align_up_u64(constants.kv_head_count * constants.conv_tap_count * constants.hidden_dim * 2, a);
        } else {
            total += 2 * tiger.align_up_u64(constants.kv_dim * constants.conv_tap_count * 2, a);
        }
        total += 2 * tiger.align_up_u64(constants.hidden_dim * constants.conv_tap_count * 2, a); // attn_sconv, mlp_sconv
        total += tiger.align_up_u64(constants.head_count * constants.head_dim * 2, a); // q_clip_scale
        total += tiger.align_up_u64(constants.head_count * constants.rel_dim * 2, a); // rel_clip_scale
        total += tiger.align_up_u64(constants.rel_dim * constants.rel_extent * 2, a); // rel_logits_proj

        if (l >= constants.loop_start_layer and l < constants.loop_end_layer) {
            var s: u32 = 0;
            while (s < 4) : (s += 1) {
                if (quant_mode == .nvfp4) {
                    total += constants.linear_weight_vram_bytes(constants.gate_rank, 2 * constants.hidden_dim, .nvfp4);
                    total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.gate_rank, .nvfp4);
                    total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.gate_rank, .nvfp4);
                } else {
                    total += tiger.align_up_u64(@as(u64, constants.gate_rank) * 2 * constants.hidden_dim * 2, a);
                    total += 2 * tiger.align_up_u64(@as(u64, constants.hidden_dim) * constants.gate_rank * 2, a);
                }
            }
        }

        if (quant_mode == .nvfp4) {
            total += constants.linear_weight_vram_bytes(constants.expert_count, constants.hidden_dim, .nvfp4);
        } else {
            total += tiger.align_up_u64(@as(u64, constants.expert_count) * constants.hidden_dim * 2, a);
        }
        total += tiger.align_up_u64(constants.expert_count * 4, a); // router_bias

        // Expert banks w13 [2304, 2304] and w2 [2304, 1152]
        const dims: [2][2]u32 = .{
            .{ 2 * constants.intermediate_dim, constants.hidden_dim },
            .{ constants.hidden_dim, constants.intermediate_dim },
        };
        const is_nvfp4_bank: bool = constants.is_nvfp4_expert_layer(l) or quant_mode == .nvfp4;
        for (dims) |d| {
            const rows: u64 = d[0];
            const cols: u64 = d[1];
            if (is_nvfp4_bank) {
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 2), a);
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 16), a);
                total += tiger.align_up_u64(constants.expert_count * 4, a);
            } else {
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * cols, a);
                total += tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 32), a);
            }
        }
    }

    if (has_mtp) {
        total += tiger.align_up_u64(16 * 4, a); // fuse_alpha
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // mask_embed
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // norm_w
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // markov_dflare_norm
        total += tiger.align_up_u64(@as(u64, constants.vocab_padded_count) * constants.mtp_markov_rank * 2, a); // markov_w1
        total += tiger.align_up_u64(3 * constants.mtp_markov_rank * 2, a); // rnn_bias
        total += tiger.align_up_u64(@as(u64, constants.hidden_dim) * constants.mtp_markov_rank * 2, a); // hidden_out_w
        total += tiger.align_up_u64(constants.hidden_dim * 2, a); // hidden_out_b
        total += tiger.align_up_u64(768 * 512 * 2, a); // rnn_w_sp
        total += 2 * tiger.align_up_u64(768 * @as(u64, constants.hidden_dim) * 2, a); // rnn_w_h, rnn_w_f

        if (has_pruned_draft_vocab) {
            total += tiger.align_up_u64(@as(u64, draft_vocab_count) * 4, a); // draft_vocab_ids
            total += tiger.align_up_u64(@as(u64, constants.vocab_unpadded_count) * 4, a); // inv_draft_vocab_ids
            total += constants.linear_weight_vram_bytes(draft_vocab_count, constants.hidden_dim, mtp_kind); // unembed_draft
        }
        total += tiger.align_up_u64(@as(u64, draft_vocab_count) * 4, a); // fold_b
        total += constants.linear_weight_vram_bytes(draft_vocab_count, constants.mtp_markov_rank, mtp_kind); // fold_w

        var ml: u32 = 0;
        while (ml < constants.mtp_layer_count) : (ml += 1) {
            total += 3 * tiger.align_up_u64(constants.hidden_dim * 2, a); // input/post/dflare norms
            total += 3 * tiger.align_up_u64(constants.head_dim * 2, a); // q/k/k_ctx norms
            total += constants.linear_weight_vram_bytes(constants.dspark_qkv_out_dim, constants.hidden_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.query_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(2 * constants.mtp_intermediate_dim, constants.hidden_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(constants.hidden_dim, constants.mtp_intermediate_dim, mtp_kind);
            total += constants.linear_weight_vram_bytes(2 * constants.kv_dim, constants.hidden_dim, mtp_kind);
        }
    }

    if (has_ablation) {
        const dir_bytes: u64 = @as(u64, constants.visit_count) * constants.hidden_dim * 4;
        const param_bytes: u64 = @as(u64, constants.visit_count) * 4 * 4;
        // attn_dirs, attn_det?, attn_params, mlp_dirs, mlp_det?, mlp_params
        total += 2 * tiger.align_up_u64(dir_bytes, a);
        total += 2 * tiger.align_up_u64(param_bytes, a);
        if (has_ablation_detectors) {
            total += 2 * tiger.align_up_u64(dir_bytes, a);
        }
    }

    return total;
}

fn to_cstr(buf: []u8, s: []const u8) [*:0]const u8 {
    tiger.assert(s.len < buf.len);
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return @ptrCast(buf[0..s.len :0]);
}

fn stream_gguf_to_bf16_dev(
    gf: *const gguf.GgufFile,
    info: *const gguf.TensorInfo,
    n_rows: u64,
    cols: u64,
    h_raw: []u8,
    d_raw: [*]u8,
    dst_bf16_dev: *anyopaque,
) void {
    const rb = gguf.row_bytes(info.ggml_type, cols);
    const max_rows: u64 = @intCast(@min(h_raw.len, 32 * 1024 * 1024) / @as(usize, @intCast(rb)));
    tiger.assert(max_rows >= 1);
    const dst_bytes_ptr: [*]u8 = @ptrCast(dst_bf16_dev);
    var r0: u64 = 0;
    while (r0 < n_rows) {
        const cnt = tiger.min_u64(max_rows, n_rows - r0);
        const rbytes: usize = @intCast(cnt * rb);
        gf.read_range(info.abs_offset + r0 * rb, h_raw[0..rbytes]);
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d(d_raw, h_raw.ptr, rbytes));
        const out_off: usize = @intCast(r0 * cols * 2);
        cuda.squid_dequant_gguf_to_bf16(info.ggml_type, d_raw, @ptrCast(dst_bytes_ptr + out_off), @intCast(cnt), @intCast(cols));
        r0 += cnt;
    }
}

fn stage_gguf_layer_bf16(
    gf: *const gguf.GgufFile,
    layer_idx: u32,
    suffix: []const u8,
    cols: u64,
    rows: u64,
    h_raw: []u8,
    d_raw: [*]u8,
    dst_bf16_dev: *anyopaque,
) usize {
    var nb: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "blk.{d}.{s}", .{ layer_idx, suffix }) catch unreachable;
    const info = gf.require(name);
    tiger.assert(info.dims[0] == cols and info.dims[1] == rows);
    stream_gguf_to_bf16_dev(gf, info, rows, cols, h_raw, d_raw, dst_bf16_dev);
    return @intCast(rows * cols * 2);
}

fn compute_total_vram_bytes_gguf(
    gf_main: *const gguf.GgufFile,
    quant_mode: constants.QuantMode,
    dense_kind: constants.WeightKind,
    mtp_kind: constants.WeightKind,
    embed_on_host: bool,
    has_mtp: bool,
    has_pruned_draft_vocab: bool,
    draft_vocab_count: u32,
    has_ablation: bool,
    has_ablation_detectors: bool,
) u64 {
    const a = constants.vram_alignment_bytes;
    var total = compute_total_vram_bytes(
        quant_mode,
        dense_kind,
        mtp_kind,
        embed_on_host,
        has_mtp,
        has_pruned_draft_vocab,
        draft_vocab_count,
        has_ablation,
        has_ablation_detectors,
    );
    var nb: [128]u8 = undefined;
    var l: u32 = 0;
    while (l < constants.layer_count) : (l += 1) {
        const dims: [2][2]u32 = .{
            .{ 2 * constants.intermediate_dim, constants.hidden_dim },
            .{ constants.hidden_dim, constants.intermediate_dim },
        };
        const is_nvfp4_bank: bool = constants.is_nvfp4_expert_layer(l) or quant_mode == .nvfp4;
        for (dims) |d| {
            const rows: u64 = d[0];
            const cols: u64 = d[1];
            if (is_nvfp4_bank) {
                total -= tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 2), a);
                total -= tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 16), a);
                total -= tiger.align_up_u64(constants.expert_count * 4, a);
            } else {
                total -= tiger.align_up_u64(@as(u64, constants.expert_count) * rows * cols, a);
                total -= tiger.align_up_u64(@as(u64, constants.expert_count) * rows * (cols / 32), a);
            }
        }
        const w13_n = std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_up_exps.weight", .{l}) catch unreachable;
        total += tiger.align_up_u64(gf_main.require(w13_n).byte_size, a);
        const s13_n = std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_up_exps.scale", .{l}) catch unreachable;
        if (gf_main.find(s13_n) != null) total += tiger.align_up_u64(constants.expert_count * 4, a);

        const w2_n = std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.weight", .{l}) catch unreachable;
        total += tiger.align_up_u64(gf_main.require(w2_n).byte_size, a);
        const s2_n = std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.scale", .{l}) catch unreachable;
        if (gf_main.find(s2_n) != null) total += tiger.align_up_u64(constants.expert_count * 4, a);
    }
    return total;
}
