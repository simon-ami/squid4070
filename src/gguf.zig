// TigerStyle POSIX bounded-streaming GGUF v3 reader.
// Reads ONLY the file header/metadata/tensor directory into bounded host buffers, keeping `fd` open
// for bounded chunked reads (`read_range`) so WSL2 host RAM stays small even for 8.5 GB models.

const std = @import("std");
const tiger = @import("tiger.zig");

pub const tensor_count_max: u32 = 1024;
pub const hash_slot_count: u32 = 2048;
pub const name_arena_bytes_max: u32 = 256 * 1024;
pub const default_alignment: u64 = 32;

comptime {
    if ((hash_slot_count & (hash_slot_count - 1)) != 0) {
        @compileError("hash_slot_count must be a power of two");
    }
    if (hash_slot_count < tensor_count_max * 2) {
        @compileError("hash_slot_count must provide <= 50% load factor");
    }
}

pub const GgmlType = enum(u32) {
    f32 = 0,
    q8_0 = 8,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq2_s = 22,
    iq4_xs = 23,
    bf16 = 30,
    nvfp4 = 40,
    q2_0 = 42,
    _,
};

pub fn block_values(ggml_type: u32) u64 {
    return switch (ggml_type) {
        0 => 1,
        8, 20 => 32,
        12, 13, 14, 16, 17, 18, 19, 22, 23 => 256,
        30 => 1,
        40, 42 => 64,
        else => {
            tiger.print_stderr("[gguf] Unsupported ggml_type={d}\n", .{ggml_type});
            @panic("Unsupported GGUF ggml_type");
        },
    };
}

pub fn block_bytes(ggml_type: u32) u64 {
    return switch (ggml_type) {
        0 => 4,
        8 => 34,
        12 => 144,
        13 => 176,
        14 => 210,
        16 => 66,
        17 => 74,
        18 => 98,
        19 => 50,
        20 => 18,
        22 => 82,
        23 => 136,
        30 => 2,
        40 => 36,
        42 => 18,
        else => {
            tiger.print_stderr("[gguf] Unsupported ggml_type={d}\n", .{ggml_type});
            @panic("Unsupported GGUF ggml_type");
        },
    };
}

pub fn row_bytes(ggml_type: u32, cols: u64) u64 {
    const bv = block_values(ggml_type);
    const bb = block_bytes(ggml_type);
    tiger.assert(cols % bv == 0);
    return (cols / bv) * bb;
}

pub fn tensor_bytes(ggml_type: u32, dims: [4]u64, n_dims: u32) u64 {
    tiger.assert(n_dims >= 1 and n_dims <= 4);
    const rb = row_bytes(ggml_type, dims[0]);
    var n_rows: u64 = 1;
    var d: u32 = 1;
    while (d < n_dims) : (d += 1) {
        n_rows *= dims[d];
    }
    return n_rows * rb;
}

pub const TensorInfo = struct {
    name_offset: u32 = 0,
    name_length: u16 = 0,
    n_dims: u32 = 0,
    dims: [4]u64 = .{ 1, 1, 1, 1 },
    ggml_type: u32 = 0,
    rel_offset: u64 = 0,
    abs_offset: u64 = 0,
    byte_size: u64 = 0,
};

const BufferedReader = struct {
    fd: i32,
    file_pos: u64 = 0,
    buf_pos: usize = 0,
    buf_len: usize = 0,
    buf: []u8,

    fn init(fd: i32) BufferedReader {
        tiger.assert(fd >= 0);
        const rc = tiger.lseek(fd, 0, tiger.SEEK_SET);
        tiger.assert(rc == 0);
        return .{
            .fd = fd,
            .buf = tiger.alloc_host_zeroed(u8, 128 * 1024),
        };
    }

    fn deinit(self: *BufferedReader) void {
        tiger.free_host_slice(u8, self.buf);
    }

    fn offset(self: *const BufferedReader) u64 {
        return self.file_pos - @as(u64, self.buf_len - self.buf_pos);
    }

    fn refill(self: *BufferedReader) void {
        self.buf_pos = 0;
        self.buf_len = 0;
        var iter: u32 = 0;
        while (iter < 64) : (iter += 1) {
            const n = tiger.read(self.fd, self.buf.ptr, self.buf.len);
            if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
            if (n <= 0) break;
            self.buf_len = @intCast(n);
            self.file_pos += @as(u64, self.buf_len);
            return;
        }
        @panic("Unexpected EOF reading GGUF header");
    }

    fn read_bytes(self: *BufferedReader, dst: []u8) void {
        var done: usize = 0;
        while (done < dst.len) {
            if (self.buf_pos == self.buf_len) self.refill();
            const avail = self.buf_len - self.buf_pos;
            const take = @min(avail, dst.len - done);
            @memcpy(dst[done .. done + take], self.buf[self.buf_pos .. self.buf_pos + take]);
            self.buf_pos += take;
            done += take;
        }
    }

    fn skip_bytes(self: *BufferedReader, count: u64) void {
        var rem: u64 = count;
        while (rem > 0) {
            const avail: u64 = @intCast(self.buf_len - self.buf_pos);
            if (avail > 0) {
                const take = @min(avail, rem);
                self.buf_pos += @intCast(take);
                rem -= take;
            } else {
                const new_pos = self.offset() + rem;
                const s = tiger.lseek(self.fd, @intCast(new_pos), tiger.SEEK_SET);
                tiger.assert(s == @as(i64, @intCast(new_pos)));
                self.file_pos = new_pos;
                self.buf_pos = 0;
                self.buf_len = 0;
                rem = 0;
            }
        }
    }

    fn read_u8(self: *BufferedReader) u8 {
        var b: [1]u8 = undefined;
        self.read_bytes(&b);
        return b[0];
    }

    fn read_u32(self: *BufferedReader) u32 {
        var b: [4]u8 = undefined;
        self.read_bytes(&b);
        return std.mem.readInt(u32, &b, .little);
    }

    fn read_u64(self: *BufferedReader) u64 {
        var b: [8]u8 = undefined;
        self.read_bytes(&b);
        return std.mem.readInt(u64, &b, .little);
    }
};

fn skip_kv_value(r: *BufferedReader, val_type: u32, is_alignment: bool, alignment_out: *u64, depth: u32) void {
    tiger.assert(depth < 16);
    switch (val_type) {
        0, 1, 7 => r.skip_bytes(1),
        2, 3 => r.skip_bytes(2),
        4 => {
            const v = r.read_u32();
            if (is_alignment and v > 0) alignment_out.* = v;
        },
        5, 6 => r.skip_bytes(4),
        10 => {
            const v = r.read_u64();
            if (is_alignment and v > 0) alignment_out.* = v;
        },
        11, 12 => r.skip_bytes(8),
        8 => {
            const slen = r.read_u64();
            r.skip_bytes(slen);
        },
        9 => {
            const elem_type = r.read_u32();
            const count = r.read_u64();
            const fixed_size: u64 = switch (elem_type) {
                0, 1, 7 => 1,
                2, 3 => 2,
                4, 5, 6 => 4,
                10, 11, 12 => 8,
                else => 0,
            };
            if (fixed_size > 0) {
                r.skip_bytes(count * fixed_size);
            } else {
                var i: u64 = 0;
                while (i < count) : (i += 1) {
                    skip_kv_value(r, elem_type, false, alignment_out, depth + 1);
                }
            }
        },
        else => {
            tiger.print_stderr("[gguf] Unsupported KV value type {d}\n", .{val_type});
            @panic("Unsupported GGUF KV value type");
        },
    }
}

pub const GgufFile = struct {
    file_descriptor: i32 = -1,
    file_bytes: u64 = 0,
    alignment: u64 = default_alignment,
    data_start: u64 = 0,
    tensor_count: u32 = 0,
    tensors: []TensorInfo = &.{},
    hash_table: [hash_slot_count]u16 = @splat(empty_slot),
    name_arena: []u8 = &.{},
    name_arena_used: u32 = 0,

    const empty_slot: u16 = 0xffff;

    pub fn open_file(self: *GgufFile, path_z: [*:0]const u8) bool {
        tiger.assert(self.file_descriptor < 0);
        const fd: i32 = tiger.open(path_z, tiger.O_RDONLY);
        if (fd < 0) return false;

        const size_i64: i64 = tiger.lseek(fd, 0, tiger.SEEK_END);
        if (size_i64 < 24) {
            _ = tiger.close(fd);
            return false;
        }

        self.file_descriptor = fd;
        self.file_bytes = @intCast(size_i64);
        self.alignment = default_alignment;
        self.data_start = 0;
        self.tensor_count = 0;
        self.tensors = tiger.alloc_host_zeroed(TensorInfo, tensor_count_max);
        self.name_arena = tiger.alloc_host_zeroed(u8, name_arena_bytes_max);
        self.name_arena_used = 0;
        @memset(&self.hash_table, empty_slot);

        var r = BufferedReader.init(fd);
        defer r.deinit();

        var magic: [4]u8 = undefined;
        r.read_bytes(&magic);
        if (!std.mem.eql(u8, &magic, "GGUF")) {
            self.close_file();
            return false;
        }
        const version = r.read_u32();
        if (version != 3) {
            self.close_file();
            return false;
        }
        const n_tensors_u64 = r.read_u64();
        const n_kv = r.read_u64();
        tiger.assert(n_tensors_u64 > 0 and n_tensors_u64 <= tensor_count_max);

        var key_buf: [512]u8 = undefined;
        var kv_i: u64 = 0;
        while (kv_i < n_kv) : (kv_i += 1) {
            const klen = r.read_u64();
            var is_align: bool = false;
            if (klen < key_buf.len) {
                r.read_bytes(key_buf[0..@intCast(klen)]);
                is_align = std.mem.eql(u8, key_buf[0..@intCast(klen)], "general.alignment");
            } else {
                r.skip_bytes(klen);
            }
            const vtype = r.read_u32();
            skip_kv_value(&r, vtype, is_align, &self.alignment, 0);
        }

        const n_tensors: u32 = @intCast(n_tensors_u64);
        var t_i: u32 = 0;
        while (t_i < n_tensors) : (t_i += 1) {
            const tlen_u64 = r.read_u64();
            tiger.assert(tlen_u64 > 0 and tlen_u64 < 512);
            const tlen: u16 = @intCast(tlen_u64);
            tiger.assert(self.name_arena_used + tlen <= name_arena_bytes_max);
            const name_off = self.name_arena_used;
            r.read_bytes(self.name_arena[name_off .. name_off + tlen]);
            self.name_arena_used += tlen;

            const ndims = r.read_u32();
            tiger.assert(ndims >= 1 and ndims <= 4);
            var dims: [4]u64 = .{ 1, 1, 1, 1 };
            var d: u32 = 0;
            while (d < ndims) : (d += 1) {
                dims[d] = r.read_u64();
                tiger.assert(dims[d] > 0);
            }
            const gtype = r.read_u32();
            const rel_off = r.read_u64();
            const bsize = tensor_bytes(gtype, dims, ndims);

            self.tensors[t_i] = .{
                .name_offset = name_off,
                .name_length = tlen,
                .n_dims = ndims,
                .dims = dims,
                .ggml_type = gtype,
                .rel_offset = rel_off,
                .abs_offset = 0,
                .byte_size = bsize,
            };
        }

        self.tensor_count = n_tensors;
        self.data_start = tiger.align_up_u64(r.offset(), self.alignment);

        t_i = 0;
        while (t_i < n_tensors) : (t_i += 1) {
            const abs_off = self.data_start + self.tensors[t_i].rel_offset;
            tiger.assert(abs_off + self.tensors[t_i].byte_size <= self.file_bytes);
            self.tensors[t_i].abs_offset = abs_off;
            self.insert_hash(t_i);
        }

        return true;
    }

    pub fn close_file(self: *GgufFile) void {
        if (self.tensors.len > 0) {
            tiger.free_host_slice(TensorInfo, self.tensors);
            self.tensors = &.{};
        }
        if (self.name_arena.len > 0) {
            tiger.free_host_slice(u8, self.name_arena);
            self.name_arena = &.{};
        }
        if (self.file_descriptor >= 0) {
            _ = tiger.close(self.file_descriptor);
            self.file_descriptor = -1;
        }
        self.file_bytes = 0;
        self.data_start = 0;
        self.tensor_count = 0;
        self.name_arena_used = 0;
    }

    fn insert_hash(self: *GgufFile, index: u32) void {
        const info = &self.tensors[index];
        const name = self.name_arena[info.name_offset .. info.name_offset + info.name_length];
        const mask: u32 = hash_slot_count - 1;
        var slot: u32 = @intCast(tiger.hash_bytes_u64(name) & mask);
        var probe: u32 = 0;
        while (probe < hash_slot_count) : (probe += 1) {
            if (self.hash_table[slot] == empty_slot) {
                self.hash_table[slot] = @intCast(index);
                return;
            }
            slot = (slot + 1) & mask;
        }
        @panic("GGUF hash table full");
    }

    pub fn find(self: *const GgufFile, name: []const u8) ?*const TensorInfo {
        tiger.assert(self.file_descriptor >= 0);
        tiger.assert(name.len > 0);
        const mask: u32 = hash_slot_count - 1;
        var slot: u32 = @intCast(tiger.hash_bytes_u64(name) & mask);
        var probe: u32 = 0;
        while (probe < hash_slot_count) : (probe += 1) {
            const index = self.hash_table[slot];
            if (index == empty_slot) break;
            const info = &self.tensors[index];
            const entry_name = self.name_arena[info.name_offset .. info.name_offset + info.name_length];
            if (std.mem.eql(u8, entry_name, name)) return info;
            slot = (slot + 1) & mask;
        }
        return null;
    }

    pub fn require(self: *const GgufFile, name: []const u8) *const TensorInfo {
        if (self.find(name)) |info| return info;
        tiger.print_stderr("[gguf] Missing tensor: {s}\n", .{name});
        @panic("GGUF tensor not found");
    }

    pub fn read_range(self: *const GgufFile, abs_offset: u64, dst: []u8) void {
        tiger.assert(self.file_descriptor >= 0);
        tiger.assert(abs_offset + dst.len <= self.file_bytes);
        const s = tiger.lseek(self.file_descriptor, @intCast(abs_offset), tiger.SEEK_SET);
        tiger.assert(s == @as(i64, @intCast(abs_offset)));
        var got: usize = 0;
        var iter: u64 = 0;
        const max_iter: u64 = @as(u64, dst.len) + 65536;
        while (got < dst.len and iter < max_iter) : (iter += 1) {
            const n = tiger.read(self.file_descriptor, dst[got..].ptr, dst.len - got);
            if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
            if (n <= 0) break;
            got += @intCast(n);
        }
        tiger.assert(got == dst.len);
    }

    pub fn read_tensor_full(self: *const GgufFile, name: []const u8, dst_buf: []u8) *const TensorInfo {
        const info = self.require(name);
        tiger.assert(dst_buf.len >= info.byte_size);
        self.read_range(info.abs_offset, dst_buf[0..@intCast(info.byte_size)]);
        return info;
    }
};

test "parse real Inkling-2-Mini target and DSpark GGUF headers (CPU-only)" {
    var gf_main: GgufFile = .{};
    const ok_main = gf_main.open_file("/mnt/d/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ2_XXS.gguf");
    try std.testing.expect(ok_main);
    defer gf_main.close_file();
    try std.testing.expectEqual(@as(u32, 734), gf_main.tensor_count);
    try std.testing.expectEqual(@as(u64, 11978688), gf_main.data_start);
    const w13_0 = gf_main.require("blk.0.ffn_gate_up_exps.weight");
    try std.testing.expectEqual(@as(u32, 19), w13_0.ggml_type);
    try std.testing.expectEqual(@as(u64, 132710400), w13_0.byte_size);

    var gf_draft: GgufFile = .{};
    const ok_draft = gf_draft.open_file("/mnt/d/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf");
    try std.testing.expect(ok_draft);
    defer gf_draft.close_file();
    try std.testing.expectEqual(@as(u32, 54), gf_draft.tensor_count);
    try std.testing.expectEqual(@as(u64, 11933280), gf_draft.data_start);
    const jp = gf_draft.require("joint_proj.weight");
    try std.testing.expectEqual(@as(u32, 30), jp.ggml_type);
    try std.testing.expectEqual(@as(u64, 5120), jp.dims[0]);
    try std.testing.expectEqual(@as(u64, 768), jp.dims[1]);
}

test "parse Inkling-2-Mini-UD-IQ3_XXS GGUF header (CPU-only)" {
    var gf_main: GgufFile = .{};
    const ok_main = gf_main.open_file("/mnt/d/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf");
    try std.testing.expect(ok_main);
    defer gf_main.close_file();
    try std.testing.expectEqual(@as(u32, 744), gf_main.tensor_count);
    const w13_9 = gf_main.require("blk.9.ffn_gate_up_exps.weight");
    try std.testing.expectEqual(@as(u32, 18), w13_9.ggml_type);
    const w2_20 = gf_main.require("blk.20.ffn_down_exps.weight");
    try std.testing.expectEqual(@as(u32, 20), w2_20.ggml_type);
    const mq_6 = gf_main.require("blk.6.loop_msg_query.weight");
    try std.testing.expectEqual(@as(u32, 0), mq_6.ggml_type);
    try std.testing.expectEqual(@as(u64, 2304), mq_6.dims[0]);
    try std.testing.expectEqual(@as(u64, 4), mq_6.dims[1]);
}
