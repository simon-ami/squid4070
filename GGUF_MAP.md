# Inkling-2-Mini GGUF tensor map

| File | Tensors | Bytes | Data start (bytes) |
|---|---:|---:|---:|
| `Inkling-2-Mini-UD-IQ2_XXS.gguf` | 734 | 8,486,236,352 | 11,978,688 |
| `dspark-Inkling-2-Mini-Q8_0.gguf` | 54 | 402,278,560 | 11,933,280 |

Python read both GGUF v3 tensor directories on WSL2. Each file has unique tensor names, aligned non-overlapping tensor ranges, and a complete group match below. No extra tensors appear in either directory. SHA-256 of the bytes up to data start is `23cbd2c27136d2c91e8efbdc3cd419810a305931c2314f882f9b5df235f27d69` for the target and `88af7968bdff1362d66bfd4bbd0aa9e8e0d496ef1f6c138a50f94b67b533fe1f` for the draft.

## Layout and storage

- GGUF lists `[ne0, ne1, ...]` with `ne0` contiguous. A matrix `[K,N]` decodes into row-major `[N,K]` by writing `dst[r*K+c] = src(c,r)`. A bank `[K,N,E]` uses `dst[(e*N+r)*K+c] = src(c,r,e)`. The decoded element order is the same in both conventions. The tables list transposes, tap reversal, row reordering, slices, and fusions.
- Decode each quantized row to F32 blocks along `ne0`. Keep F32 destinations in F32. For BF16 fields, round F32 to BF16 using round-to-nearest-even, or copy BF16 input bits. For `LinearWeight` fields, assemble the listed rows, then pack the row-major matrix in the kind selected by the existing loader. FP8 uses one F32 scale per row. Native NVFP4 uses one E4M3 scale per 16 columns plus a tensor scale.
- `l` is the physical layer. The target advertises 48 visits but stores 24 physical blocks, `blk.0` to `blk.23`. Visits 0–5 and 42–47 map to physical 0–5 and 18–23. Visits 6–41 map to `6 + (visit-6)%12`. Reuse the physical weights for each pass. Global layers are 5, 11, 17, and 23. Controllers belong to physical layers 6–17.
- Row ranges are zero-based and half-open in destination expressions. Layer ranges in the **Layers** and **Type by layer** columns are inclusive.

### GGML types in the files

| ID | Type | Values per block | Bytes per block |
|---:|---|---:|---:|
| 0 | F32 | 1 | 4 |
| 8 | Q8_0 | 32 | 34 |
| 12 | Q4_K | 256 | 144 |
| 13 | Q5_K | 256 | 176 |
| 14 | Q6_K | 256 | 210 |
| 16 | IQ2_XXS | 256 | 66 |
| 17 | IQ2_XS | 256 | 74 |
| 18 | IQ3_XXS | 256 | 98 |
| 19 | IQ1_S | 256 | 50 |
| 22 | IQ2_S | 256 | 82 |
| 23 | IQ4_XS | 256 | 136 |
| 30 | BF16 | 1 | 2 |
| 40 | NVFP4 | 64 | 36 |
| 42 | Q2_0 | 64 | 18 |

Q2_0 stores little-endian FP16 `d` followed by 16 code bytes. Lane `j` decodes to `d * (((qs[j/4] >> (2*(j%4))) & 3) - 1)`.

GGUF NVFP4 stores four unsigned E4M3 scales, followed by 32 bytes of E2M1 codes. In each 16-lane group, the eight low nibbles encode lanes 0–7 and the eight high nibbles encode lanes 8–15. For a valid scale byte with exponent `e = byte >> 3` and mantissa `m = byte & 7`, the scale is `m*2^-9` when `e=0`, otherwise `(1+m/8)*2^(e-7)`. The reference returns zero for `0x7f`. The reference decoder halves the scale and uses the doubled E2M1 lookup values `0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12`. The external expert scale is a separate multiplier. Native squid NVFP4 instead puts adjacent lanes `2j` and `2j+1` in each byte and stores scales in a separate array.

## Target

The root fields below are in `ModelWeights`. All `blk.l.*` layer fields are in `ModelWeights.layers[l]` unless the table names another root field.

| Tensor | GGUF dims | Layers | Type by layer | Destination and transformation |
|---|---|---|---|---|
| `loop_kv_res_query.weight` | `[2304, 24]` | — | 0 | `cross_pass_kv_queries` F32 [24,2304]. For core offset i, rows 2i and 2i+1 are K and V. |
| `loop_res_query.weight` | `[2304, 24]` | — | 0 | `cross_pass_queries` F32 [24,2304]. For core offset i, rows 2i and 2i+1 are Q/attention and MLP. |
| `output.weight` | `[2304, 201024]` | — | 23 | `unembed`. Keep rows [0,200058), then pack [200058,2304]. Also the source of the draft output and fold. |
| `output_norm.weight` | `[2304]` | — | 0 | `final_norm`, BF16. |
| `output_res_query.weight` | `[2304]` | — | 0 | `output_query`, F32. |
| `token_embd.weight` | `[2304, 201024]` | — | 23 | `embed`, BF16 [201024,2304]. Keep all rows. |
| `token_embd_norm.weight` | `[2304]` | — | 0 | `embed_norm`, BF16. |
| `blk.l.attn_q.weight` | `[2304, 3072]` | 0–23 | 8 (6–13), 12 (18–19,23), 13 (3,16), 14 (0–2,4–5,14–15,17), 18 (20–22) | `w_qrgs` rows [0,3072), Q output in head-major order. |
| `blk.l.attn_rel.weight` | `[2304, 384]` | 0–23 | 8 (0–1,4,6,8–12,14–17), 12 (21–23), 13 (20), 14 (2–3,5,7,13,18–19) | `w_qrgs` rows [3072,3456), R output in head-major order. |
| `blk.l.attn_gate.weight` | `[2304, 3072]` | 0–23 | 8 (1–11,13), 12 (19–23), 13 (18), 14 (0,12,14–17) | `w_qrgs` rows [3456,6528), gate output in head-major order. |
| `blk.l.attn_sink_proj.weight` | `[2304, 24]` | 0–23 | 30 | `w_qrgs` rows [6528,6552), one output per head. |
| `blk.l.attn_k.weight` | `[2304, 1024]` | 0–23 | 8 (0–3,5–14,17), 12 (20–22), 13 (18–19,23), 14 (4,15–16) | `w_kv` rows [0,1024), also the `w_k` view. |
| `blk.l.attn_v.weight` | `[2304, 1024]` | 0–23 | 8 (0–18), 13 (20–23), 14 (19) | `w_kv` rows [1024,2048), also the `w_v` view. Advance the data and scale pointers by 1024 rows. |
| `blk.l.attn_output.weight` | `[3072, 2304]` | 0–23 | 8 (0–17), 13 (19–23), 14 (18) | `w_o`, [2304,3072]. |
| `blk.l.attn_norm.weight` | `[2304]` | 0–4,6–10,12–16,18–22 | 0 | `attn_norm`, BF16. Local layers. |
| `blk.l.attn_q_inp_norm.weight` | `[2304]` | 5,11,17,23 | 0 | `q_in_norm`, BF16. Global layers. |
| `blk.l.attn_k_inp_norm.weight` | `[2304]` | 5,11,17,23 | 0 | `k_in_norm`, BF16. Global layers. |
| `blk.l.attn_v_inp_norm.weight` | `[2304]` | 5,11,17,23 | 0 | `v_in_norm`, BF16. Global layers. |
| `blk.l.ffn_norm.weight` | `[2304]` | 0–23 | 0 | `mlp_norm`, BF16. |
| `blk.l.attn_k_conv.weight` | `[4, 1024]` | 0–23 | 0 | `k_sconv_w`, BF16. Local: dst[c,t] = src(t,c), oldest to newest. Global: dst[lag,c] = src(3-lag,c), newest to oldest. |
| `blk.l.attn_v_conv.weight` | `[4, 1024]` | 0–23 | 0 | `v_sconv_w`, BF16. Same indexing as K. |
| `blk.l.attn_k_conv_proj.weight` | `[2304, 32]` | 5,11,17,23 | 30 | `k_sconv_proj`, BF16 [32,2304]. Row 4*head+lag, lag 0 current. |
| `blk.l.attn_v_conv_proj.weight` | `[2304, 32]` | 5,11,17,23 | 30 | `v_sconv_proj`, BF16 [32,2304]. Row 4*head+lag, lag 0 current. |
| `blk.l.attn_post_conv.weight` | `[4, 2304]` | 0–23 | 0 | `attn_sconv_w`, BF16 [2304,4]. Oldest to newest per channel. |
| `blk.l.ffn_post_conv.weight` | `[4, 2304]` | 0–23 | 0 | `mlp_sconv_w`, BF16 [2304,4]. Oldest to newest per channel. |
| `blk.l.attn_q_clip.weight` | `[128, 24]` | 0–23 | 0 | `q_clip_scale`, BF16 [24,128]. |
| `blk.l.attn_rel_clip.weight` | `[16, 24]` | 0–23 | 0 | `rel_clip_scale`, BF16 [24,16]. |
| `blk.l.attn_rel_bias.weight` | `[16, 1024]` | 0–23 | 0 | `rel_logits_proj`, BF16 [16,1024]. Transpose the decoded [1024,16]: dst[k,d] = src(k,d), at GGUF linear offset 16*d+k. |
| `blk.l.attn_res_query.weight` | `[2304]` | 1–4,6–10,12–16,18–22 | 0 | `ModelWeights.attn_queries[l-1]`, F32. One row on local layers, three contiguous Q/K/V rows on global layers. Layer 0 has no tensor. |
| `blk.l.attn_res_query.weight` | `[2304, 3]` | 5,11,17,23 | 0 | `ModelWeights.attn_queries[l-1]`, F32. One row on local layers, three contiguous Q/K/V rows on global layers. Layer 0 has no tensor. |
| `blk.l.ffn_res_query.weight` | `[2304]` | 0–23 | 0 | `ModelWeights.mlp_queries[l]`, F32. |
| `blk.l.ffn_gate_inp.weight` | `[2304, 128]` | 0–23 | 0 | `router_w` BF16 or `router_nvfp4` in NVFP4 mode, [128,2304]. |
| `blk.l.exp_probs_b.bias` | `[128]` | 0–23 | 0 | `router_bias`, F32 selection bias [128]. |
| `blk.l.ffn_gate_up_exps.weight` | `[2304, 2304, 128]` | 0–23 | 16 (5,7–8,12–14,18), 17 (9–11,15,17,22), 19 (0–4,6), 22 (16,19–21,23) | `w13.data` and `w13.scales` after expert packing. Interleave gate and up output rows as described below. |
| `blk.l.ffn_gate_up_exps.scale` | `[128]` | 0–19 | 0 | `w13.expert_scale` outer factor, combined with the packer's factor if needed. |
| `blk.l.ffn_down_exps.weight` | `[1152, 2304, 128]` | 0–23 | 40 (5,8–9,11,17–19), 42 (0–4,6–7,10,12–16,20–23) | `w2.data` and `w2.scales` after expert packing. |
| `blk.l.ffn_down_exps.scale` | `[128]` | 0–19 | 0 | `w2.expert_scale` outer factor, combined with the packer's factor if needed. |
| `blk.l.loop_attn_down.weight` | `[4608, 64]` | 6–17 | 0 | `ctrl[0].down_w` BF16 or `ctrl[0].down_nvfp4` in NVFP4 mode, [64,4608]. Columns are [current, message]. |
| `blk.l.loop_attn_forget.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[0].forget_w` BF16 or `ctrl[0].forget_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_attn_gain.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[0].gain_w` BF16 or `ctrl[0].gain_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_k_down.weight` | `[4608, 64]` | 6–17 | 0 | `ctrl[1].down_w` BF16 or `ctrl[1].down_nvfp4` in NVFP4 mode, [64,4608]. Columns are [current, message]. |
| `blk.l.loop_k_forget.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[1].forget_w` BF16 or `ctrl[1].forget_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_k_gain.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[1].gain_w` BF16 or `ctrl[1].gain_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_v_down.weight` | `[4608, 64]` | 6–17 | 0 | `ctrl[2].down_w` BF16 or `ctrl[2].down_nvfp4` in NVFP4 mode, [64,4608]. Columns are [current, message]. |
| `blk.l.loop_v_forget.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[2].forget_w` BF16 or `ctrl[2].forget_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_v_gain.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[2].gain_w` BF16 or `ctrl[2].gain_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_mlp_down.weight` | `[4608, 64]` | 6–17 | 0 | `ctrl[3].down_w` BF16 or `ctrl[3].down_nvfp4` in NVFP4 mode, [64,4608]. Columns are [current, message]. |
| `blk.l.loop_mlp_forget.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[3].forget_w` BF16 or `ctrl[3].forget_nvfp4` in NVFP4 mode, [2304,64]. |
| `blk.l.loop_mlp_gain.weight` | `[64, 2304]` | 6–17 | 0 | `ctrl[3].gain_w` BF16 or `ctrl[3].gain_nvfp4` in NVFP4 mode, [2304,64]. |

### Expert banks

- `w13` has logical shape `[128,2304,2304]`. Within each GGUF expert, rows 0–1151 are gate and rows 1152–2303 are up. Convert them to squid rows `2*i` and `2*i+1` respectively. Keep the input columns and expert order.
- `w2` has logical shape `[128,2304,1152]`. Keep rows, columns, and expert order.
- Each scale tensor is F32 `[128]`. For a selected expert `e`, llama.cpp multiplies the entire fused gate/up dot product by `S13[e]` before SiTU. It multiplies the down dot product by `S2[e]` before weighted reduction. With no scale tensor, the multiplier is 1. The GGUF vectors contain the outer factors directly. The target has both vectors on every layer 0–19. Across these layers `S13` spans 3.77341e-5 to 2.73769e-4 and `S2` spans 5.34791e-5 to 3.16986e-4. All 5,120 values are finite, positive, and non-unit.
- For each native NVFP4 squid bank, write `data[E,rows,cols/2]`, `scales[E,rows,cols/16]`, and `expert_scale[E]`. GGUF NVFP4 can preserve its codes and valid E4M3 scales by reordering nibbles into adjacent pairs, gathering scales into the separate array, mapping GGML `0x7f` scales to zero if present, and using the GGUF outer factor directly. For IQ or Q2_0, decode the blocks and requantize. The existing MXFP8-to-NVFP4 converter uses the per-expert factor `max(amax/(448*6), 1e-12)` and the block factor `E4M3(min(block_amax/(6*expert_factor),448))`. It rounds scaled values to the nearest E2M1 code, with midpoint ties away from zero. If repacking unscaled GGUF values, multiply the new packer's outer factor by the GGUF factor once. If folding the GGUF factor into decoded weights before packing, use the new factor alone.
- To match the existing expert bank selection, use NVFP4 for layers 0–19 and for every layer in NVFP4 mode. In Exact or FP8 mode, use native MXFP8 for layers 20–23. That packs `data[E,rows,cols]` as E4M3, `scales[E,rows,cols/32]` as UE8M0, and uses no runtime expert-scale array. The GGUF has no scale tensors on those four layers. Requantizing IQ/Q2_0 changes the decoded weights. The current `ExpertBank` has no native IQ or Q2_0 path. The GGUF loader must supply the repacking step.

## DSpark

Root fields are in `ModelWeights.dspark`. Layer fields are in `ModelWeights.dspark.layers[l]`. Layers 0 and 2 are local. Layer 1 is global. Keep target feature order `[5,23,41,48]`. Pack the fused draft matrices as BF16 in Exact mode or with `SQUID_MTP_BF16=1`, and as native NVFP4 otherwise.

| Tensor | GGUF dims | Layers | Type by layer | Destination and transformation |
|---|---|---|---|---|
| `dspark_fuse.weight` | `[4, 4]` | — | 0 | `fuse_alpha`, F32 row-wise softmax, each row over the four target features. Rows 0–2 feed the draft layers, row 3 feeds Markov context. |
| `dspark_hidden_out.bias` | `[2304]` | — | 0 | `hidden_out_b`, BF16 [2304]. |
| `dspark_hidden_out.weight` | `[256, 2304]` | — | 30 | `hidden_out_w`, BF16 [2304,256]. |
| `dspark_norm.weight` | `[2304]` | — | 0 | `markov_dflare_norm`, BF16. |
| `joint_proj.bias` | `[768]` | — | 0 | `rnn_bias`, BF16 [768]. Keep gate, candidate, Markov order. |
| `joint_proj.weight` | `[5120, 768]` | — | 30 | `rnn_w_sp`, `rnn_w_h`, `rnn_w_f`. Copy BF16 column slices below. |
| `markov_w1.weight` | `[256, 201024]` | — | 30 | `markov_w1`, BF16 [201024,256]. Keep all rows. |
| `mask_embd.weight` | `[2304]` | — | 0 | `mask_embed`, BF16. |
| `output_norm.weight` | `[2304]` | — | 0 | `norm_w`, BF16. |
| `blk.l.attn_norm.weight` | `[2304]` | 0–2 | 0 | `input_norm`, BF16. |
| `blk.l.ffn_norm.weight` | `[2304]` | 0–2 | 0 | `post_norm`, BF16. |
| `blk.l.dflare_norm.weight` | `[2304]` | 0–2 | 0 | `dflare_norm`, BF16. |
| `blk.l.attn_q_norm.weight` | `[128]` | 0–2 | 0 | `q_norm`, BF16. |
| `blk.l.attn_k_norm.weight` | `[128]` | 0–2 | 0 | `k_norm`, BF16. |
| `blk.l.attn_k_ctx_norm.weight` | `[128]` | 0–2 | 0 | `k_norm_ctx`, BF16. |
| `blk.l.attn_q.weight` | `[2304, 3072]` | 0–2 | 8 | `w_qkv` rows [0,3072). |
| `blk.l.attn_k.weight` | `[2304, 1024]` | 0–2 | 8 | `w_qkv` rows [3072,4096). |
| `blk.l.attn_v.weight` | `[2304, 1024]` | 0–2 | 8 | `w_qkv` rows [4096,5120). |
| `blk.l.attn_output.weight` | `[3072, 2304]` | 0–2 | 8 | `w_o`, [2304,3072]. |
| `blk.l.attn_k_ctx.weight` | `[2304, 1024]` | 0–2 | 8 | `w_kv_ctx` rows [0,1024). |
| `blk.l.attn_v_ctx.weight` | `[2304, 1024]` | 0–2 | 8 | `w_kv_ctx` rows [1024,2048). |
| `blk.l.ffn_gate.weight` | `[2304, 9216]` | 0–2 | 8 | `w_gu` rows [0,9216). |
| `blk.l.ffn_up.weight` | `[2304, 9216]` | 0–2 | 8 | `w_gu` rows [9216,18432). |
| `blk.l.ffn_down.weight` | `[9216, 2304]` | 0–2 | 8 | `w_down`, [2304,9216]. |

### Fused recurrent head

`joint_proj.weight` is BF16 with GGUF dimensions `[5120,768]`. For every output row, split its columns as follows.

| Input | Source columns | Destination |
|---|---|---|
| State and previous token's `markov_w1` row | `[0,256)` and `[256,512)` | `rnn_w_sp[768,512]` |
| Current draft hidden row after `norm_w` | `[512,2816)` | `rnn_w_h[768,2304]` |
| Fused target context after `markov_dflare_norm` | `[2816,5120)` | `rnn_w_f[768,2304]` |

Keep rows 0–255 for sigmoid gate, 256–511 for tanh candidate, and 512–767 for tanh Markov latent. Keep the same order in the bias. Start with zero state per draft block and update `state = gate * state + (1-gate) * candidate`. Read the padded `markov_w1` table by previous token ID. The latent produces a hidden correction via `hidden_out_w @ latent + hidden_out_b`. This draft stores the hidden-space projection used to build the folded head. Its logits use the target output matrix `U`, truncated to 200,058 vocabulary rows.

For the current squid folded implementation, compute `fold_w = U_stage @ hidden_out_w` and `fold_b = U_stage @ hidden_out_b` using the selected target rows staged in BF16. Keep `fold_b` in F32 and pack `fold_w` from the BF16-rounded product. At full vocabulary, alias `unembed_draft` to target `unembed`. If an external pruned token list is configured, gather its target rows in list order for both `unembed_draft` and the fold, and build both token-ID maps from that list. Neither GGUF supplies a pruned token list. The draft has no separate `markov_w2`, confidence head or vocabulary projection tensor.

## Source and verification

- [Legacy loop layout, dd330cc0](https://github.com/unslothai/llama.cpp/blob/dd330cc0024fb381461a43e58db9af6f37c1467f/src/models/inkling2-loop.cpp) covers the target physical blocks, reused visits, and controller mapping. [Inkling2 tensor conversion](https://github.com/unslothai/llama.cpp/blob/dd330cc0024fb381461a43e58db9af6f37c1467f/conversion/inkling2.py) covers expert row order, static taps and relative bias.
- [Expert-scale loader, 688251e6](https://github.com/unslothai/llama.cpp/blob/688251e6291871d2e5ff8532196e5aa570091b05/src/llama-model.cpp#L1903) aliases the fused companion scale to the up-scale slot. [Expert graph](https://github.com/unslothai/llama.cpp/blob/688251e6291871d2e5ff8532196e5aa570091b05/src/llama-graph.cpp#L2583) applies it to both outputs before the split and activation.
- [Fused DSpark head, b5ed9f28](https://github.com/unslothai/llama.cpp/blob/b5ed9f2845358124d1cbce91f475011047515e74/src/models/dspark.cpp#L197) matches this draft's 5120-column projection, padded lookup and hidden correction. The generic DSpark head in `inkling2-expert-scales` takes only one hidden-width feature vector. The fused head explicitly supplies both hidden and target context.
- Compared `pin/inkling2` (`5f3eb91c`), `inkling2` (`26a2a424`), `inkling2-expert-scales` (`688251e6`), `base/inkling2-attn-chunk-378` (`be9c519b`) and `master` (`768c7300`). Their GGML common and CUDA dequant/vecdot headers are identical, as are the 12 C dequantizer bodies after removing comments and whitespace. The pinned Inkling2 graph and converter use a separate-expert format. Use the legacy loop and fused DSpark sources above for the files in this map.
- The destination shapes, short-conv indexing and fusion order were checked against squid checkout `c3b35127`, specifically `cuda/weights.hpp`, `cuda/config.hpp`, `src/weights.zig`, `src/engine.zig`, and the CUDA kernels.
- `tests/ref_slices.f32` stores 48 decoded F32 rows in the manifest order in `tests/test_gguf_quants.cu`. The quantized references came from the actual [GGML C dequantizers at 688251e6](https://github.com/unslothai/llama.cpp/blob/688251e6291871d2e5ff8532196e5aa570091b05/ggml/src/ggml-quants.c), compiled with `gcc -O2 -ffp-contract=off`. The fixture SHA-256 is `596080ccca33e73b4c3eae746801ddc5643993ec0cbe4419c6585c08adaa4006`. The test reads the matching raw rows by absolute GGUF offset and checks their packed-byte FNV-1a checksums. The CPU test passed at zero maximum absolute error and zero NaN/Inf for every type. CUDA compiled for `sm_89`. GPU execution is pending.

## Expanded tensor directories

The tables above give the destination and transformation for each record below. Entries retain the on-disk directory order.

### `Inkling-2-Mini-UD-IQ2_XXS.gguf`

```text
tensor	[ne0,ne1,...]	type
loop_kv_res_query.weight	[2304,24]	0
loop_res_query.weight	[2304,24]	0
output.weight	[2304,201024]	23
output_norm.weight	[2304]	0
output_res_query.weight	[2304]	0
token_embd.weight	[2304,201024]	23
token_embd_norm.weight	[2304]	0
blk.0.attn_gate.weight	[2304,3072]	14
blk.0.attn_k.weight	[2304,1024]	8
blk.0.attn_k_conv.weight	[4,1024]	0
blk.0.attn_norm.weight	[2304]	0
blk.0.attn_output.weight	[3072,2304]	8
blk.0.attn_post_conv.weight	[4,2304]	0
blk.0.attn_q.weight	[2304,3072]	14
blk.0.attn_q_clip.weight	[128,24]	0
blk.0.attn_rel.weight	[2304,384]	8
blk.0.attn_rel_bias.weight	[16,1024]	0
blk.0.attn_rel_clip.weight	[16,24]	0
blk.0.attn_sink_proj.weight	[2304,24]	30
blk.0.attn_v.weight	[2304,1024]	8
blk.0.attn_v_conv.weight	[4,1024]	0
blk.0.exp_probs_b.bias	[128]	0
blk.0.ffn_down_exps.scale	[128]	0
blk.0.ffn_down_exps.weight	[1152,2304,128]	42
blk.0.ffn_gate_inp.weight	[2304,128]	0
blk.0.ffn_gate_up_exps.scale	[128]	0
blk.0.ffn_gate_up_exps.weight	[2304,2304,128]	19
blk.0.ffn_norm.weight	[2304]	0
blk.0.ffn_post_conv.weight	[4,2304]	0
blk.0.ffn_res_query.weight	[2304]	0
blk.1.attn_gate.weight	[2304,3072]	8
blk.1.attn_k.weight	[2304,1024]	8
blk.1.attn_k_conv.weight	[4,1024]	0
blk.1.attn_norm.weight	[2304]	0
blk.1.attn_output.weight	[3072,2304]	8
blk.1.attn_post_conv.weight	[4,2304]	0
blk.1.attn_q.weight	[2304,3072]	14
blk.1.attn_q_clip.weight	[128,24]	0
blk.1.attn_rel.weight	[2304,384]	8
blk.1.attn_rel_bias.weight	[16,1024]	0
blk.1.attn_rel_clip.weight	[16,24]	0
blk.1.attn_res_query.weight	[2304]	0
blk.1.attn_sink_proj.weight	[2304,24]	30
blk.1.attn_v.weight	[2304,1024]	8
blk.1.attn_v_conv.weight	[4,1024]	0
blk.1.exp_probs_b.bias	[128]	0
blk.1.ffn_down_exps.scale	[128]	0
blk.1.ffn_down_exps.weight	[1152,2304,128]	42
blk.1.ffn_gate_inp.weight	[2304,128]	0
blk.1.ffn_gate_up_exps.scale	[128]	0
blk.1.ffn_gate_up_exps.weight	[2304,2304,128]	19
blk.1.ffn_norm.weight	[2304]	0
blk.1.ffn_post_conv.weight	[4,2304]	0
blk.1.ffn_res_query.weight	[2304]	0
blk.2.attn_gate.weight	[2304,3072]	8
blk.2.attn_k.weight	[2304,1024]	8
blk.2.attn_k_conv.weight	[4,1024]	0
blk.2.attn_norm.weight	[2304]	0
blk.2.attn_output.weight	[3072,2304]	8
blk.2.attn_post_conv.weight	[4,2304]	0
blk.2.attn_q.weight	[2304,3072]	14
blk.2.attn_q_clip.weight	[128,24]	0
blk.2.attn_rel.weight	[2304,384]	14
blk.2.attn_rel_bias.weight	[16,1024]	0
blk.2.attn_rel_clip.weight	[16,24]	0
blk.2.attn_res_query.weight	[2304]	0
blk.2.attn_sink_proj.weight	[2304,24]	30
blk.2.attn_v.weight	[2304,1024]	8
blk.2.attn_v_conv.weight	[4,1024]	0
blk.2.exp_probs_b.bias	[128]	0
blk.2.ffn_down_exps.scale	[128]	0
blk.2.ffn_down_exps.weight	[1152,2304,128]	42
blk.2.ffn_gate_inp.weight	[2304,128]	0
blk.2.ffn_gate_up_exps.scale	[128]	0
blk.2.ffn_gate_up_exps.weight	[2304,2304,128]	19
blk.2.ffn_norm.weight	[2304]	0
blk.2.ffn_post_conv.weight	[4,2304]	0
blk.2.ffn_res_query.weight	[2304]	0
blk.3.attn_gate.weight	[2304,3072]	8
blk.3.attn_k.weight	[2304,1024]	8
blk.3.attn_k_conv.weight	[4,1024]	0
blk.3.attn_norm.weight	[2304]	0
blk.3.attn_output.weight	[3072,2304]	8
blk.3.attn_post_conv.weight	[4,2304]	0
blk.3.attn_q.weight	[2304,3072]	13
blk.3.attn_q_clip.weight	[128,24]	0
blk.3.attn_rel.weight	[2304,384]	14
blk.3.attn_rel_bias.weight	[16,1024]	0
blk.3.attn_rel_clip.weight	[16,24]	0
blk.3.attn_res_query.weight	[2304]	0
blk.3.attn_sink_proj.weight	[2304,24]	30
blk.3.attn_v.weight	[2304,1024]	8
blk.3.attn_v_conv.weight	[4,1024]	0
blk.3.exp_probs_b.bias	[128]	0
blk.3.ffn_down_exps.scale	[128]	0
blk.3.ffn_down_exps.weight	[1152,2304,128]	42
blk.3.ffn_gate_inp.weight	[2304,128]	0
blk.3.ffn_gate_up_exps.scale	[128]	0
blk.3.ffn_gate_up_exps.weight	[2304,2304,128]	19
blk.3.ffn_norm.weight	[2304]	0
blk.3.ffn_post_conv.weight	[4,2304]	0
blk.3.ffn_res_query.weight	[2304]	0
blk.4.attn_gate.weight	[2304,3072]	8
blk.4.attn_k.weight	[2304,1024]	14
blk.4.attn_k_conv.weight	[4,1024]	0
blk.4.attn_norm.weight	[2304]	0
blk.4.attn_output.weight	[3072,2304]	8
blk.4.attn_post_conv.weight	[4,2304]	0
blk.4.attn_q.weight	[2304,3072]	14
blk.4.attn_q_clip.weight	[128,24]	0
blk.4.attn_rel.weight	[2304,384]	8
blk.4.attn_rel_bias.weight	[16,1024]	0
blk.4.attn_rel_clip.weight	[16,24]	0
blk.4.attn_res_query.weight	[2304]	0
blk.4.attn_sink_proj.weight	[2304,24]	30
blk.4.attn_v.weight	[2304,1024]	8
blk.4.attn_v_conv.weight	[4,1024]	0
blk.4.exp_probs_b.bias	[128]	0
blk.4.ffn_down_exps.scale	[128]	0
blk.4.ffn_down_exps.weight	[1152,2304,128]	42
blk.4.ffn_gate_inp.weight	[2304,128]	0
blk.4.ffn_gate_up_exps.scale	[128]	0
blk.4.ffn_gate_up_exps.weight	[2304,2304,128]	19
blk.4.ffn_norm.weight	[2304]	0
blk.4.ffn_post_conv.weight	[4,2304]	0
blk.4.ffn_res_query.weight	[2304]	0
blk.5.attn_gate.weight	[2304,3072]	8
blk.5.attn_k.weight	[2304,1024]	8
blk.5.attn_k_conv.weight	[4,1024]	0
blk.5.attn_k_conv_proj.weight	[2304,32]	30
blk.5.attn_k_inp_norm.weight	[2304]	0
blk.5.attn_output.weight	[3072,2304]	8
blk.5.attn_post_conv.weight	[4,2304]	0
blk.5.attn_q.weight	[2304,3072]	14
blk.5.attn_q_clip.weight	[128,24]	0
blk.5.attn_q_inp_norm.weight	[2304]	0
blk.5.attn_rel.weight	[2304,384]	14
blk.5.attn_rel_bias.weight	[16,1024]	0
blk.5.attn_rel_clip.weight	[16,24]	0
blk.5.attn_res_query.weight	[2304,3]	0
blk.5.attn_sink_proj.weight	[2304,24]	30
blk.5.attn_v.weight	[2304,1024]	8
blk.5.attn_v_conv.weight	[4,1024]	0
blk.5.attn_v_conv_proj.weight	[2304,32]	30
blk.5.attn_v_inp_norm.weight	[2304]	0
blk.5.exp_probs_b.bias	[128]	0
blk.5.ffn_down_exps.scale	[128]	0
blk.5.ffn_down_exps.weight	[1152,2304,128]	40
blk.5.ffn_gate_inp.weight	[2304,128]	0
blk.5.ffn_gate_up_exps.scale	[128]	0
blk.5.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.5.ffn_norm.weight	[2304]	0
blk.5.ffn_post_conv.weight	[4,2304]	0
blk.5.ffn_res_query.weight	[2304]	0
blk.6.attn_gate.weight	[2304,3072]	8
blk.6.attn_k.weight	[2304,1024]	8
blk.6.attn_k_conv.weight	[4,1024]	0
blk.6.attn_norm.weight	[2304]	0
blk.6.attn_output.weight	[3072,2304]	8
blk.6.attn_post_conv.weight	[4,2304]	0
blk.6.attn_q.weight	[2304,3072]	8
blk.6.attn_q_clip.weight	[128,24]	0
blk.6.attn_rel.weight	[2304,384]	8
blk.6.attn_rel_bias.weight	[16,1024]	0
blk.6.attn_rel_clip.weight	[16,24]	0
blk.6.attn_res_query.weight	[2304]	0
blk.6.attn_sink_proj.weight	[2304,24]	30
blk.6.attn_v.weight	[2304,1024]	8
blk.6.attn_v_conv.weight	[4,1024]	0
blk.6.exp_probs_b.bias	[128]	0
blk.6.ffn_down_exps.scale	[128]	0
blk.6.ffn_down_exps.weight	[1152,2304,128]	42
blk.6.ffn_gate_inp.weight	[2304,128]	0
blk.6.ffn_gate_up_exps.scale	[128]	0
blk.6.ffn_gate_up_exps.weight	[2304,2304,128]	19
blk.6.ffn_norm.weight	[2304]	0
blk.6.ffn_post_conv.weight	[4,2304]	0
blk.6.ffn_res_query.weight	[2304]	0
blk.6.loop_attn_down.weight	[4608,64]	0
blk.6.loop_attn_forget.weight	[64,2304]	0
blk.6.loop_attn_gain.weight	[64,2304]	0
blk.6.loop_k_down.weight	[4608,64]	0
blk.6.loop_k_forget.weight	[64,2304]	0
blk.6.loop_k_gain.weight	[64,2304]	0
blk.6.loop_mlp_down.weight	[4608,64]	0
blk.6.loop_mlp_forget.weight	[64,2304]	0
blk.6.loop_mlp_gain.weight	[64,2304]	0
blk.6.loop_v_down.weight	[4608,64]	0
blk.6.loop_v_forget.weight	[64,2304]	0
blk.6.loop_v_gain.weight	[64,2304]	0
blk.7.attn_gate.weight	[2304,3072]	8
blk.7.attn_k.weight	[2304,1024]	8
blk.7.attn_k_conv.weight	[4,1024]	0
blk.7.attn_norm.weight	[2304]	0
blk.7.attn_output.weight	[3072,2304]	8
blk.7.attn_post_conv.weight	[4,2304]	0
blk.7.attn_q.weight	[2304,3072]	8
blk.7.attn_q_clip.weight	[128,24]	0
blk.7.attn_rel.weight	[2304,384]	14
blk.7.attn_rel_bias.weight	[16,1024]	0
blk.7.attn_rel_clip.weight	[16,24]	0
blk.7.attn_res_query.weight	[2304]	0
blk.7.attn_sink_proj.weight	[2304,24]	30
blk.7.attn_v.weight	[2304,1024]	8
blk.7.attn_v_conv.weight	[4,1024]	0
blk.7.exp_probs_b.bias	[128]	0
blk.7.ffn_down_exps.scale	[128]	0
blk.7.ffn_down_exps.weight	[1152,2304,128]	42
blk.7.ffn_gate_inp.weight	[2304,128]	0
blk.7.ffn_gate_up_exps.scale	[128]	0
blk.7.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.7.ffn_norm.weight	[2304]	0
blk.7.ffn_post_conv.weight	[4,2304]	0
blk.7.ffn_res_query.weight	[2304]	0
blk.7.loop_attn_down.weight	[4608,64]	0
blk.7.loop_attn_forget.weight	[64,2304]	0
blk.7.loop_attn_gain.weight	[64,2304]	0
blk.7.loop_k_down.weight	[4608,64]	0
blk.7.loop_k_forget.weight	[64,2304]	0
blk.7.loop_k_gain.weight	[64,2304]	0
blk.7.loop_mlp_down.weight	[4608,64]	0
blk.7.loop_mlp_forget.weight	[64,2304]	0
blk.7.loop_mlp_gain.weight	[64,2304]	0
blk.7.loop_v_down.weight	[4608,64]	0
blk.7.loop_v_forget.weight	[64,2304]	0
blk.7.loop_v_gain.weight	[64,2304]	0
blk.8.attn_gate.weight	[2304,3072]	8
blk.8.attn_k.weight	[2304,1024]	8
blk.8.attn_k_conv.weight	[4,1024]	0
blk.8.attn_norm.weight	[2304]	0
blk.8.attn_output.weight	[3072,2304]	8
blk.8.attn_post_conv.weight	[4,2304]	0
blk.8.attn_q.weight	[2304,3072]	8
blk.8.attn_q_clip.weight	[128,24]	0
blk.8.attn_rel.weight	[2304,384]	8
blk.8.attn_rel_bias.weight	[16,1024]	0
blk.8.attn_rel_clip.weight	[16,24]	0
blk.8.attn_res_query.weight	[2304]	0
blk.8.attn_sink_proj.weight	[2304,24]	30
blk.8.attn_v.weight	[2304,1024]	8
blk.8.attn_v_conv.weight	[4,1024]	0
blk.8.exp_probs_b.bias	[128]	0
blk.8.ffn_down_exps.scale	[128]	0
blk.8.ffn_down_exps.weight	[1152,2304,128]	40
blk.8.ffn_gate_inp.weight	[2304,128]	0
blk.8.ffn_gate_up_exps.scale	[128]	0
blk.8.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.8.ffn_norm.weight	[2304]	0
blk.8.ffn_post_conv.weight	[4,2304]	0
blk.8.ffn_res_query.weight	[2304]	0
blk.8.loop_attn_down.weight	[4608,64]	0
blk.8.loop_attn_forget.weight	[64,2304]	0
blk.8.loop_attn_gain.weight	[64,2304]	0
blk.8.loop_k_down.weight	[4608,64]	0
blk.8.loop_k_forget.weight	[64,2304]	0
blk.8.loop_k_gain.weight	[64,2304]	0
blk.8.loop_mlp_down.weight	[4608,64]	0
blk.8.loop_mlp_forget.weight	[64,2304]	0
blk.8.loop_mlp_gain.weight	[64,2304]	0
blk.8.loop_v_down.weight	[4608,64]	0
blk.8.loop_v_forget.weight	[64,2304]	0
blk.8.loop_v_gain.weight	[64,2304]	0
blk.9.attn_gate.weight	[2304,3072]	8
blk.9.attn_k.weight	[2304,1024]	8
blk.9.attn_k_conv.weight	[4,1024]	0
blk.9.attn_norm.weight	[2304]	0
blk.9.attn_output.weight	[3072,2304]	8
blk.9.attn_post_conv.weight	[4,2304]	0
blk.9.attn_q.weight	[2304,3072]	8
blk.9.attn_q_clip.weight	[128,24]	0
blk.9.attn_rel.weight	[2304,384]	8
blk.9.attn_rel_bias.weight	[16,1024]	0
blk.9.attn_rel_clip.weight	[16,24]	0
blk.9.attn_res_query.weight	[2304]	0
blk.9.attn_sink_proj.weight	[2304,24]	30
blk.9.attn_v.weight	[2304,1024]	8
blk.9.attn_v_conv.weight	[4,1024]	0
blk.9.exp_probs_b.bias	[128]	0
blk.9.ffn_down_exps.scale	[128]	0
blk.9.ffn_down_exps.weight	[1152,2304,128]	40
blk.9.ffn_gate_inp.weight	[2304,128]	0
blk.9.ffn_gate_up_exps.scale	[128]	0
blk.9.ffn_gate_up_exps.weight	[2304,2304,128]	17
blk.9.ffn_norm.weight	[2304]	0
blk.9.ffn_post_conv.weight	[4,2304]	0
blk.9.ffn_res_query.weight	[2304]	0
blk.9.loop_attn_down.weight	[4608,64]	0
blk.9.loop_attn_forget.weight	[64,2304]	0
blk.9.loop_attn_gain.weight	[64,2304]	0
blk.9.loop_k_down.weight	[4608,64]	0
blk.9.loop_k_forget.weight	[64,2304]	0
blk.9.loop_k_gain.weight	[64,2304]	0
blk.9.loop_mlp_down.weight	[4608,64]	0
blk.9.loop_mlp_forget.weight	[64,2304]	0
blk.9.loop_mlp_gain.weight	[64,2304]	0
blk.9.loop_v_down.weight	[4608,64]	0
blk.9.loop_v_forget.weight	[64,2304]	0
blk.9.loop_v_gain.weight	[64,2304]	0
blk.10.attn_gate.weight	[2304,3072]	8
blk.10.attn_k.weight	[2304,1024]	8
blk.10.attn_k_conv.weight	[4,1024]	0
blk.10.attn_norm.weight	[2304]	0
blk.10.attn_output.weight	[3072,2304]	8
blk.10.attn_post_conv.weight	[4,2304]	0
blk.10.attn_q.weight	[2304,3072]	8
blk.10.attn_q_clip.weight	[128,24]	0
blk.10.attn_rel.weight	[2304,384]	8
blk.10.attn_rel_bias.weight	[16,1024]	0
blk.10.attn_rel_clip.weight	[16,24]	0
blk.10.attn_res_query.weight	[2304]	0
blk.10.attn_sink_proj.weight	[2304,24]	30
blk.10.attn_v.weight	[2304,1024]	8
blk.10.attn_v_conv.weight	[4,1024]	0
blk.10.exp_probs_b.bias	[128]	0
blk.10.ffn_down_exps.scale	[128]	0
blk.10.ffn_down_exps.weight	[1152,2304,128]	42
blk.10.ffn_gate_inp.weight	[2304,128]	0
blk.10.ffn_gate_up_exps.scale	[128]	0
blk.10.ffn_gate_up_exps.weight	[2304,2304,128]	17
blk.10.ffn_norm.weight	[2304]	0
blk.10.ffn_post_conv.weight	[4,2304]	0
blk.10.ffn_res_query.weight	[2304]	0
blk.10.loop_attn_down.weight	[4608,64]	0
blk.10.loop_attn_forget.weight	[64,2304]	0
blk.10.loop_attn_gain.weight	[64,2304]	0
blk.10.loop_k_down.weight	[4608,64]	0
blk.10.loop_k_forget.weight	[64,2304]	0
blk.10.loop_k_gain.weight	[64,2304]	0
blk.10.loop_mlp_down.weight	[4608,64]	0
blk.10.loop_mlp_forget.weight	[64,2304]	0
blk.10.loop_mlp_gain.weight	[64,2304]	0
blk.10.loop_v_down.weight	[4608,64]	0
blk.10.loop_v_forget.weight	[64,2304]	0
blk.10.loop_v_gain.weight	[64,2304]	0
blk.11.attn_gate.weight	[2304,3072]	8
blk.11.attn_k.weight	[2304,1024]	8
blk.11.attn_k_conv.weight	[4,1024]	0
blk.11.attn_k_conv_proj.weight	[2304,32]	30
blk.11.attn_k_inp_norm.weight	[2304]	0
blk.11.attn_output.weight	[3072,2304]	8
blk.11.attn_post_conv.weight	[4,2304]	0
blk.11.attn_q.weight	[2304,3072]	8
blk.11.attn_q_clip.weight	[128,24]	0
blk.11.attn_q_inp_norm.weight	[2304]	0
blk.11.attn_rel.weight	[2304,384]	8
blk.11.attn_rel_bias.weight	[16,1024]	0
blk.11.attn_rel_clip.weight	[16,24]	0
blk.11.attn_res_query.weight	[2304,3]	0
blk.11.attn_sink_proj.weight	[2304,24]	30
blk.11.attn_v.weight	[2304,1024]	8
blk.11.attn_v_conv.weight	[4,1024]	0
blk.11.attn_v_conv_proj.weight	[2304,32]	30
blk.11.attn_v_inp_norm.weight	[2304]	0
blk.11.exp_probs_b.bias	[128]	0
blk.11.ffn_down_exps.scale	[128]	0
blk.11.ffn_down_exps.weight	[1152,2304,128]	40
blk.11.ffn_gate_inp.weight	[2304,128]	0
blk.11.ffn_gate_up_exps.scale	[128]	0
blk.11.ffn_gate_up_exps.weight	[2304,2304,128]	17
blk.11.ffn_norm.weight	[2304]	0
blk.11.ffn_post_conv.weight	[4,2304]	0
blk.11.ffn_res_query.weight	[2304]	0
blk.11.loop_attn_down.weight	[4608,64]	0
blk.11.loop_attn_forget.weight	[64,2304]	0
blk.11.loop_attn_gain.weight	[64,2304]	0
blk.11.loop_k_down.weight	[4608,64]	0
blk.11.loop_k_forget.weight	[64,2304]	0
blk.11.loop_k_gain.weight	[64,2304]	0
blk.11.loop_mlp_down.weight	[4608,64]	0
blk.11.loop_mlp_forget.weight	[64,2304]	0
blk.11.loop_mlp_gain.weight	[64,2304]	0
blk.11.loop_v_down.weight	[4608,64]	0
blk.11.loop_v_forget.weight	[64,2304]	0
blk.11.loop_v_gain.weight	[64,2304]	0
blk.12.attn_gate.weight	[2304,3072]	14
blk.12.attn_k.weight	[2304,1024]	8
blk.12.attn_k_conv.weight	[4,1024]	0
blk.12.attn_norm.weight	[2304]	0
blk.12.attn_output.weight	[3072,2304]	8
blk.12.attn_post_conv.weight	[4,2304]	0
blk.12.attn_q.weight	[2304,3072]	8
blk.12.attn_q_clip.weight	[128,24]	0
blk.12.attn_rel.weight	[2304,384]	8
blk.12.attn_rel_bias.weight	[16,1024]	0
blk.12.attn_rel_clip.weight	[16,24]	0
blk.12.attn_res_query.weight	[2304]	0
blk.12.attn_sink_proj.weight	[2304,24]	30
blk.12.attn_v.weight	[2304,1024]	8
blk.12.attn_v_conv.weight	[4,1024]	0
blk.12.exp_probs_b.bias	[128]	0
blk.12.ffn_down_exps.scale	[128]	0
blk.12.ffn_down_exps.weight	[1152,2304,128]	42
blk.12.ffn_gate_inp.weight	[2304,128]	0
blk.12.ffn_gate_up_exps.scale	[128]	0
blk.12.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.12.ffn_norm.weight	[2304]	0
blk.12.ffn_post_conv.weight	[4,2304]	0
blk.12.ffn_res_query.weight	[2304]	0
blk.12.loop_attn_down.weight	[4608,64]	0
blk.12.loop_attn_forget.weight	[64,2304]	0
blk.12.loop_attn_gain.weight	[64,2304]	0
blk.12.loop_k_down.weight	[4608,64]	0
blk.12.loop_k_forget.weight	[64,2304]	0
blk.12.loop_k_gain.weight	[64,2304]	0
blk.12.loop_mlp_down.weight	[4608,64]	0
blk.12.loop_mlp_forget.weight	[64,2304]	0
blk.12.loop_mlp_gain.weight	[64,2304]	0
blk.12.loop_v_down.weight	[4608,64]	0
blk.12.loop_v_forget.weight	[64,2304]	0
blk.12.loop_v_gain.weight	[64,2304]	0
blk.13.attn_gate.weight	[2304,3072]	8
blk.13.attn_k.weight	[2304,1024]	8
blk.13.attn_k_conv.weight	[4,1024]	0
blk.13.attn_norm.weight	[2304]	0
blk.13.attn_output.weight	[3072,2304]	8
blk.13.attn_post_conv.weight	[4,2304]	0
blk.13.attn_q.weight	[2304,3072]	8
blk.13.attn_q_clip.weight	[128,24]	0
blk.13.attn_rel.weight	[2304,384]	14
blk.13.attn_rel_bias.weight	[16,1024]	0
blk.13.attn_rel_clip.weight	[16,24]	0
blk.13.attn_res_query.weight	[2304]	0
blk.13.attn_sink_proj.weight	[2304,24]	30
blk.13.attn_v.weight	[2304,1024]	8
blk.13.attn_v_conv.weight	[4,1024]	0
blk.13.exp_probs_b.bias	[128]	0
blk.13.ffn_down_exps.scale	[128]	0
blk.13.ffn_down_exps.weight	[1152,2304,128]	42
blk.13.ffn_gate_inp.weight	[2304,128]	0
blk.13.ffn_gate_up_exps.scale	[128]	0
blk.13.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.13.ffn_norm.weight	[2304]	0
blk.13.ffn_post_conv.weight	[4,2304]	0
blk.13.ffn_res_query.weight	[2304]	0
blk.13.loop_attn_down.weight	[4608,64]	0
blk.13.loop_attn_forget.weight	[64,2304]	0
blk.13.loop_attn_gain.weight	[64,2304]	0
blk.13.loop_k_down.weight	[4608,64]	0
blk.13.loop_k_forget.weight	[64,2304]	0
blk.13.loop_k_gain.weight	[64,2304]	0
blk.13.loop_mlp_down.weight	[4608,64]	0
blk.13.loop_mlp_forget.weight	[64,2304]	0
blk.13.loop_mlp_gain.weight	[64,2304]	0
blk.13.loop_v_down.weight	[4608,64]	0
blk.13.loop_v_forget.weight	[64,2304]	0
blk.13.loop_v_gain.weight	[64,2304]	0
blk.14.attn_gate.weight	[2304,3072]	14
blk.14.attn_k.weight	[2304,1024]	8
blk.14.attn_k_conv.weight	[4,1024]	0
blk.14.attn_norm.weight	[2304]	0
blk.14.attn_output.weight	[3072,2304]	8
blk.14.attn_post_conv.weight	[4,2304]	0
blk.14.attn_q.weight	[2304,3072]	14
blk.14.attn_q_clip.weight	[128,24]	0
blk.14.attn_rel.weight	[2304,384]	8
blk.14.attn_rel_bias.weight	[16,1024]	0
blk.14.attn_rel_clip.weight	[16,24]	0
blk.14.attn_res_query.weight	[2304]	0
blk.14.attn_sink_proj.weight	[2304,24]	30
blk.14.attn_v.weight	[2304,1024]	8
blk.14.attn_v_conv.weight	[4,1024]	0
blk.14.exp_probs_b.bias	[128]	0
blk.14.ffn_down_exps.scale	[128]	0
blk.14.ffn_down_exps.weight	[1152,2304,128]	42
blk.14.ffn_gate_inp.weight	[2304,128]	0
blk.14.ffn_gate_up_exps.scale	[128]	0
blk.14.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.14.ffn_norm.weight	[2304]	0
blk.14.ffn_post_conv.weight	[4,2304]	0
blk.14.ffn_res_query.weight	[2304]	0
blk.14.loop_attn_down.weight	[4608,64]	0
blk.14.loop_attn_forget.weight	[64,2304]	0
blk.14.loop_attn_gain.weight	[64,2304]	0
blk.14.loop_k_down.weight	[4608,64]	0
blk.14.loop_k_forget.weight	[64,2304]	0
blk.14.loop_k_gain.weight	[64,2304]	0
blk.14.loop_mlp_down.weight	[4608,64]	0
blk.14.loop_mlp_forget.weight	[64,2304]	0
blk.14.loop_mlp_gain.weight	[64,2304]	0
blk.14.loop_v_down.weight	[4608,64]	0
blk.14.loop_v_forget.weight	[64,2304]	0
blk.14.loop_v_gain.weight	[64,2304]	0
blk.15.attn_gate.weight	[2304,3072]	14
blk.15.attn_k.weight	[2304,1024]	14
blk.15.attn_k_conv.weight	[4,1024]	0
blk.15.attn_norm.weight	[2304]	0
blk.15.attn_output.weight	[3072,2304]	8
blk.15.attn_post_conv.weight	[4,2304]	0
blk.15.attn_q.weight	[2304,3072]	14
blk.15.attn_q_clip.weight	[128,24]	0
blk.15.attn_rel.weight	[2304,384]	8
blk.15.attn_rel_bias.weight	[16,1024]	0
blk.15.attn_rel_clip.weight	[16,24]	0
blk.15.attn_res_query.weight	[2304]	0
blk.15.attn_sink_proj.weight	[2304,24]	30
blk.15.attn_v.weight	[2304,1024]	8
blk.15.attn_v_conv.weight	[4,1024]	0
blk.15.exp_probs_b.bias	[128]	0
blk.15.ffn_down_exps.scale	[128]	0
blk.15.ffn_down_exps.weight	[1152,2304,128]	42
blk.15.ffn_gate_inp.weight	[2304,128]	0
blk.15.ffn_gate_up_exps.scale	[128]	0
blk.15.ffn_gate_up_exps.weight	[2304,2304,128]	17
blk.15.ffn_norm.weight	[2304]	0
blk.15.ffn_post_conv.weight	[4,2304]	0
blk.15.ffn_res_query.weight	[2304]	0
blk.15.loop_attn_down.weight	[4608,64]	0
blk.15.loop_attn_forget.weight	[64,2304]	0
blk.15.loop_attn_gain.weight	[64,2304]	0
blk.15.loop_k_down.weight	[4608,64]	0
blk.15.loop_k_forget.weight	[64,2304]	0
blk.15.loop_k_gain.weight	[64,2304]	0
blk.15.loop_mlp_down.weight	[4608,64]	0
blk.15.loop_mlp_forget.weight	[64,2304]	0
blk.15.loop_mlp_gain.weight	[64,2304]	0
blk.15.loop_v_down.weight	[4608,64]	0
blk.15.loop_v_forget.weight	[64,2304]	0
blk.15.loop_v_gain.weight	[64,2304]	0
blk.16.attn_gate.weight	[2304,3072]	14
blk.16.attn_k.weight	[2304,1024]	14
blk.16.attn_k_conv.weight	[4,1024]	0
blk.16.attn_norm.weight	[2304]	0
blk.16.attn_output.weight	[3072,2304]	8
blk.16.attn_post_conv.weight	[4,2304]	0
blk.16.attn_q.weight	[2304,3072]	13
blk.16.attn_q_clip.weight	[128,24]	0
blk.16.attn_rel.weight	[2304,384]	8
blk.16.attn_rel_bias.weight	[16,1024]	0
blk.16.attn_rel_clip.weight	[16,24]	0
blk.16.attn_res_query.weight	[2304]	0
blk.16.attn_sink_proj.weight	[2304,24]	30
blk.16.attn_v.weight	[2304,1024]	8
blk.16.attn_v_conv.weight	[4,1024]	0
blk.16.exp_probs_b.bias	[128]	0
blk.16.ffn_down_exps.scale	[128]	0
blk.16.ffn_down_exps.weight	[1152,2304,128]	42
blk.16.ffn_gate_inp.weight	[2304,128]	0
blk.16.ffn_gate_up_exps.scale	[128]	0
blk.16.ffn_gate_up_exps.weight	[2304,2304,128]	22
blk.16.ffn_norm.weight	[2304]	0
blk.16.ffn_post_conv.weight	[4,2304]	0
blk.16.ffn_res_query.weight	[2304]	0
blk.16.loop_attn_down.weight	[4608,64]	0
blk.16.loop_attn_forget.weight	[64,2304]	0
blk.16.loop_attn_gain.weight	[64,2304]	0
blk.16.loop_k_down.weight	[4608,64]	0
blk.16.loop_k_forget.weight	[64,2304]	0
blk.16.loop_k_gain.weight	[64,2304]	0
blk.16.loop_mlp_down.weight	[4608,64]	0
blk.16.loop_mlp_forget.weight	[64,2304]	0
blk.16.loop_mlp_gain.weight	[64,2304]	0
blk.16.loop_v_down.weight	[4608,64]	0
blk.16.loop_v_forget.weight	[64,2304]	0
blk.16.loop_v_gain.weight	[64,2304]	0
blk.17.attn_gate.weight	[2304,3072]	14
blk.17.attn_k.weight	[2304,1024]	8
blk.17.attn_k_conv.weight	[4,1024]	0
blk.17.attn_k_conv_proj.weight	[2304,32]	30
blk.17.attn_k_inp_norm.weight	[2304]	0
blk.17.attn_output.weight	[3072,2304]	8
blk.17.attn_post_conv.weight	[4,2304]	0
blk.17.attn_q.weight	[2304,3072]	14
blk.17.attn_q_clip.weight	[128,24]	0
blk.17.attn_q_inp_norm.weight	[2304]	0
blk.17.attn_rel.weight	[2304,384]	8
blk.17.attn_rel_bias.weight	[16,1024]	0
blk.17.attn_rel_clip.weight	[16,24]	0
blk.17.attn_res_query.weight	[2304,3]	0
blk.17.attn_sink_proj.weight	[2304,24]	30
blk.17.attn_v.weight	[2304,1024]	8
blk.17.attn_v_conv.weight	[4,1024]	0
blk.17.attn_v_conv_proj.weight	[2304,32]	30
blk.17.attn_v_inp_norm.weight	[2304]	0
blk.17.exp_probs_b.bias	[128]	0
blk.17.ffn_down_exps.scale	[128]	0
blk.17.ffn_down_exps.weight	[1152,2304,128]	40
blk.17.ffn_gate_inp.weight	[2304,128]	0
blk.17.ffn_gate_up_exps.scale	[128]	0
blk.17.ffn_gate_up_exps.weight	[2304,2304,128]	17
blk.17.ffn_norm.weight	[2304]	0
blk.17.ffn_post_conv.weight	[4,2304]	0
blk.17.ffn_res_query.weight	[2304]	0
blk.17.loop_attn_down.weight	[4608,64]	0
blk.17.loop_attn_forget.weight	[64,2304]	0
blk.17.loop_attn_gain.weight	[64,2304]	0
blk.17.loop_k_down.weight	[4608,64]	0
blk.17.loop_k_forget.weight	[64,2304]	0
blk.17.loop_k_gain.weight	[64,2304]	0
blk.17.loop_mlp_down.weight	[4608,64]	0
blk.17.loop_mlp_forget.weight	[64,2304]	0
blk.17.loop_mlp_gain.weight	[64,2304]	0
blk.17.loop_v_down.weight	[4608,64]	0
blk.17.loop_v_forget.weight	[64,2304]	0
blk.17.loop_v_gain.weight	[64,2304]	0
blk.18.attn_gate.weight	[2304,3072]	13
blk.18.attn_k.weight	[2304,1024]	13
blk.18.attn_k_conv.weight	[4,1024]	0
blk.18.attn_norm.weight	[2304]	0
blk.18.attn_output.weight	[3072,2304]	14
blk.18.attn_post_conv.weight	[4,2304]	0
blk.18.attn_q.weight	[2304,3072]	12
blk.18.attn_q_clip.weight	[128,24]	0
blk.18.attn_rel.weight	[2304,384]	14
blk.18.attn_rel_bias.weight	[16,1024]	0
blk.18.attn_rel_clip.weight	[16,24]	0
blk.18.attn_res_query.weight	[2304]	0
blk.18.attn_sink_proj.weight	[2304,24]	30
blk.18.attn_v.weight	[2304,1024]	8
blk.18.attn_v_conv.weight	[4,1024]	0
blk.18.exp_probs_b.bias	[128]	0
blk.18.ffn_down_exps.scale	[128]	0
blk.18.ffn_down_exps.weight	[1152,2304,128]	40
blk.18.ffn_gate_inp.weight	[2304,128]	0
blk.18.ffn_gate_up_exps.scale	[128]	0
blk.18.ffn_gate_up_exps.weight	[2304,2304,128]	16
blk.18.ffn_norm.weight	[2304]	0
blk.18.ffn_post_conv.weight	[4,2304]	0
blk.18.ffn_res_query.weight	[2304]	0
blk.19.attn_gate.weight	[2304,3072]	12
blk.19.attn_k.weight	[2304,1024]	13
blk.19.attn_k_conv.weight	[4,1024]	0
blk.19.attn_norm.weight	[2304]	0
blk.19.attn_output.weight	[3072,2304]	13
blk.19.attn_post_conv.weight	[4,2304]	0
blk.19.attn_q.weight	[2304,3072]	12
blk.19.attn_q_clip.weight	[128,24]	0
blk.19.attn_rel.weight	[2304,384]	14
blk.19.attn_rel_bias.weight	[16,1024]	0
blk.19.attn_rel_clip.weight	[16,24]	0
blk.19.attn_res_query.weight	[2304]	0
blk.19.attn_sink_proj.weight	[2304,24]	30
blk.19.attn_v.weight	[2304,1024]	14
blk.19.attn_v_conv.weight	[4,1024]	0
blk.19.exp_probs_b.bias	[128]	0
blk.19.ffn_down_exps.scale	[128]	0
blk.19.ffn_down_exps.weight	[1152,2304,128]	40
blk.19.ffn_gate_inp.weight	[2304,128]	0
blk.19.ffn_gate_up_exps.scale	[128]	0
blk.19.ffn_gate_up_exps.weight	[2304,2304,128]	22
blk.19.ffn_norm.weight	[2304]	0
blk.19.ffn_post_conv.weight	[4,2304]	0
blk.19.ffn_res_query.weight	[2304]	0
blk.20.attn_gate.weight	[2304,3072]	12
blk.20.attn_k.weight	[2304,1024]	12
blk.20.attn_k_conv.weight	[4,1024]	0
blk.20.attn_norm.weight	[2304]	0
blk.20.attn_output.weight	[3072,2304]	13
blk.20.attn_post_conv.weight	[4,2304]	0
blk.20.attn_q.weight	[2304,3072]	18
blk.20.attn_q_clip.weight	[128,24]	0
blk.20.attn_rel.weight	[2304,384]	13
blk.20.attn_rel_bias.weight	[16,1024]	0
blk.20.attn_rel_clip.weight	[16,24]	0
blk.20.attn_res_query.weight	[2304]	0
blk.20.attn_sink_proj.weight	[2304,24]	30
blk.20.attn_v.weight	[2304,1024]	13
blk.20.attn_v_conv.weight	[4,1024]	0
blk.20.exp_probs_b.bias	[128]	0
blk.20.ffn_down_exps.weight	[1152,2304,128]	42
blk.20.ffn_gate_inp.weight	[2304,128]	0
blk.20.ffn_gate_up_exps.weight	[2304,2304,128]	22
blk.20.ffn_norm.weight	[2304]	0
blk.20.ffn_post_conv.weight	[4,2304]	0
blk.20.ffn_res_query.weight	[2304]	0
blk.21.attn_gate.weight	[2304,3072]	12
blk.21.attn_k.weight	[2304,1024]	12
blk.21.attn_k_conv.weight	[4,1024]	0
blk.21.attn_norm.weight	[2304]	0
blk.21.attn_output.weight	[3072,2304]	13
blk.21.attn_post_conv.weight	[4,2304]	0
blk.21.attn_q.weight	[2304,3072]	18
blk.21.attn_q_clip.weight	[128,24]	0
blk.21.attn_rel.weight	[2304,384]	12
blk.21.attn_rel_bias.weight	[16,1024]	0
blk.21.attn_rel_clip.weight	[16,24]	0
blk.21.attn_res_query.weight	[2304]	0
blk.21.attn_sink_proj.weight	[2304,24]	30
blk.21.attn_v.weight	[2304,1024]	13
blk.21.attn_v_conv.weight	[4,1024]	0
blk.21.exp_probs_b.bias	[128]	0
blk.21.ffn_down_exps.weight	[1152,2304,128]	42
blk.21.ffn_gate_inp.weight	[2304,128]	0
blk.21.ffn_gate_up_exps.weight	[2304,2304,128]	22
blk.21.ffn_norm.weight	[2304]	0
blk.21.ffn_post_conv.weight	[4,2304]	0
blk.21.ffn_res_query.weight	[2304]	0
blk.22.attn_gate.weight	[2304,3072]	12
blk.22.attn_k.weight	[2304,1024]	12
blk.22.attn_k_conv.weight	[4,1024]	0
blk.22.attn_norm.weight	[2304]	0
blk.22.attn_output.weight	[3072,2304]	13
blk.22.attn_post_conv.weight	[4,2304]	0
blk.22.attn_q.weight	[2304,3072]	18
blk.22.attn_q_clip.weight	[128,24]	0
blk.22.attn_rel.weight	[2304,384]	12
blk.22.attn_rel_bias.weight	[16,1024]	0
blk.22.attn_rel_clip.weight	[16,24]	0
blk.22.attn_res_query.weight	[2304]	0
blk.22.attn_sink_proj.weight	[2304,24]	30
blk.22.attn_v.weight	[2304,1024]	13
blk.22.attn_v_conv.weight	[4,1024]	0
blk.22.exp_probs_b.bias	[128]	0
blk.22.ffn_down_exps.weight	[1152,2304,128]	42
blk.22.ffn_gate_inp.weight	[2304,128]	0
blk.22.ffn_gate_up_exps.weight	[2304,2304,128]	17
blk.22.ffn_norm.weight	[2304]	0
blk.22.ffn_post_conv.weight	[4,2304]	0
blk.22.ffn_res_query.weight	[2304]	0
blk.23.attn_gate.weight	[2304,3072]	12
blk.23.attn_k.weight	[2304,1024]	13
blk.23.attn_k_conv.weight	[4,1024]	0
blk.23.attn_k_conv_proj.weight	[2304,32]	30
blk.23.attn_k_inp_norm.weight	[2304]	0
blk.23.attn_output.weight	[3072,2304]	13
blk.23.attn_post_conv.weight	[4,2304]	0
blk.23.attn_q.weight	[2304,3072]	12
blk.23.attn_q_clip.weight	[128,24]	0
blk.23.attn_q_inp_norm.weight	[2304]	0
blk.23.attn_rel.weight	[2304,384]	12
blk.23.attn_rel_bias.weight	[16,1024]	0
blk.23.attn_rel_clip.weight	[16,24]	0
blk.23.attn_res_query.weight	[2304,3]	0
blk.23.attn_sink_proj.weight	[2304,24]	30
blk.23.attn_v.weight	[2304,1024]	13
blk.23.attn_v_conv.weight	[4,1024]	0
blk.23.attn_v_conv_proj.weight	[2304,32]	30
blk.23.attn_v_inp_norm.weight	[2304]	0
blk.23.exp_probs_b.bias	[128]	0
blk.23.ffn_down_exps.weight	[1152,2304,128]	42
blk.23.ffn_gate_inp.weight	[2304,128]	0
blk.23.ffn_gate_up_exps.weight	[2304,2304,128]	22
blk.23.ffn_norm.weight	[2304]	0
blk.23.ffn_post_conv.weight	[4,2304]	0
blk.23.ffn_res_query.weight	[2304]	0
```

### `dspark-Inkling-2-Mini-Q8_0.gguf`

```text
tensor	[ne0,ne1,...]	type
dspark_fuse.weight	[4,4]	0
dspark_hidden_out.bias	[2304]	0
dspark_hidden_out.weight	[256,2304]	30
dspark_norm.weight	[2304]	0
joint_proj.bias	[768]	0
joint_proj.weight	[5120,768]	30
markov_w1.weight	[256,201024]	30
mask_embd.weight	[2304]	0
output_norm.weight	[2304]	0
blk.0.attn_k.weight	[2304,1024]	8
blk.0.attn_k_ctx.weight	[2304,1024]	8
blk.0.attn_k_ctx_norm.weight	[128]	0
blk.0.attn_k_norm.weight	[128]	0
blk.0.attn_norm.weight	[2304]	0
blk.0.attn_output.weight	[3072,2304]	8
blk.0.attn_q.weight	[2304,3072]	8
blk.0.attn_q_norm.weight	[128]	0
blk.0.attn_v.weight	[2304,1024]	8
blk.0.attn_v_ctx.weight	[2304,1024]	8
blk.0.dflare_norm.weight	[2304]	0
blk.0.ffn_down.weight	[9216,2304]	8
blk.0.ffn_gate.weight	[2304,9216]	8
blk.0.ffn_norm.weight	[2304]	0
blk.0.ffn_up.weight	[2304,9216]	8
blk.1.attn_k.weight	[2304,1024]	8
blk.1.attn_k_ctx.weight	[2304,1024]	8
blk.1.attn_k_ctx_norm.weight	[128]	0
blk.1.attn_k_norm.weight	[128]	0
blk.1.attn_norm.weight	[2304]	0
blk.1.attn_output.weight	[3072,2304]	8
blk.1.attn_q.weight	[2304,3072]	8
blk.1.attn_q_norm.weight	[128]	0
blk.1.attn_v.weight	[2304,1024]	8
blk.1.attn_v_ctx.weight	[2304,1024]	8
blk.1.dflare_norm.weight	[2304]	0
blk.1.ffn_down.weight	[9216,2304]	8
blk.1.ffn_gate.weight	[2304,9216]	8
blk.1.ffn_norm.weight	[2304]	0
blk.1.ffn_up.weight	[2304,9216]	8
blk.2.attn_k.weight	[2304,1024]	8
blk.2.attn_k_ctx.weight	[2304,1024]	8
blk.2.attn_k_ctx_norm.weight	[128]	0
blk.2.attn_k_norm.weight	[128]	0
blk.2.attn_norm.weight	[2304]	0
blk.2.attn_output.weight	[3072,2304]	8
blk.2.attn_q.weight	[2304,3072]	8
blk.2.attn_q_norm.weight	[128]	0
blk.2.attn_v.weight	[2304,1024]	8
blk.2.attn_v_ctx.weight	[2304,1024]	8
blk.2.dflare_norm.weight	[2304]	0
blk.2.ffn_down.weight	[9216,2304]	8
blk.2.ffn_gate.weight	[2304,9216]	8
blk.2.ffn_norm.weight	[2304]	0
blk.2.ffn_up.weight	[2304,9216]	8
```
