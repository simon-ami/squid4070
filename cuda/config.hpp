#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace squid {

// Pinned Inkling-2-Mini text + DSpark MTP geometry constants
constexpr int HIDDEN = 2304;
constexpr int NUM_LAYERS = 24;
constexpr int NUM_VISITS = 48;
constexpr int LOOP_START = 6;
constexpr int LOOP_END = 18;
constexpr int LOOP_PASSES = 3;
constexpr int BLOCK_LAYERS = 6;
constexpr int GATE_RANK = 64;

constexpr int NUM_HEADS = 24;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_DIM = NUM_HEADS * HEAD_DIM;     // 3072
constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 1024
constexpr int GQA_RATIO = NUM_HEADS / NUM_KV_HEADS; // 3

constexpr int D_REL = 16;
constexpr int R_DIM = NUM_HEADS * D_REL;        // 384
constexpr int REL_EXTENT = 1024;
constexpr int SWA_WINDOW = 1024;
constexpr int CONV_TAPS = 4;

constexpr int NUM_EXPERTS = 128;
constexpr int TOP_K = 8;
constexpr int INTERMEDIATE = 1152;
constexpr int VOCAB_PADDED = 201024;
constexpr int VOCAB_UNPADDED = 200058;
constexpr int EOS_TOKEN = 200006;

constexpr float RMS_EPS = 1e-6f;
constexpr float QK_CLIP = 22.627416997969522f;
constexpr float REL_CLIP = 8.0f;
constexpr int LOG_SCALE_FLOOR = 8192;
constexpr float LOG_SCALE_ALPHA = 0.1f;
constexpr float ROUTE_SCALE = 8.0f;
constexpr float GATE_SOFTCAP = 4.0f;
constexpr float UP_SOFTCAP = 25.0f;
constexpr float FINAL_SOFTCAP = 30.0f;
constexpr float INV_SQRT_HEAD_DIM = 0.08838834764831845f; // 1/sqrt(128)
constexpr float INV_SQRT_HIDDEN = 0.020833333333333332f;  // 1/sqrt(2304) = 1/48
constexpr float INV_SQRT_5 = 0.4472135954999579f;         // 1/sqrt(5)
constexpr float INV_SQRT_9 = 0.3333333333333333f;         // 1/sqrt(9)

// DSpark MTP constants
constexpr int MTP_LAYERS = 3;
constexpr int MTP_INTERMEDIATE = 9216;
constexpr int MTP_BLOCK_SIZE = 8; // anchor + 7 drafts
constexpr int MTP_MAX_DRAFTS = 7;
constexpr int MTP_MARKOV_RANK = 256;
constexpr float MTP_ROPE_THETA = 1000000.0f;
constexpr int MTP_TARGET_VISITS[4] = {5, 23, 41, 48};

constexpr bool is_global_layer(int physical_layer) {
    return (physical_layer + 1) % BLOCK_LAYERS == 0; // 5, 11, 17, 23
}

constexpr bool is_nvfp4_expert_layer(int physical_layer) {
    return physical_layer < 20; // 0..19 are NVFP4, 20..23 are MXFP8
}

constexpr bool is_mtp_local_layer(int l) {
    return l == 0 || l == 2;
}

enum class QuantMode {
    Exact, // Official checkpoint: NVFP4 L0-19, MXFP8 L20-23, BF16 dense & LM head
    FP8,   // Official experts + FP8 E4M3 dense attention & LM head
    NVFP4  // Aggressive all-NVFP4: L0-23 experts, dense attention, LM head, DSpark
};

enum class KVFormat {
    BF16 = 0, // 2.000 B/elem (exact parity)
    FP8  = 1, // 1.016 B/elem (E4M3 + per-head FP16 scale -> up to 448K context)
    Q4   = 2  // 0.531 B/elem (signed INT4 [-7..7] + group-64 FP16 scale -> up to 1M context!)
};

// Device-resident sampling config (uploaded via H2D before generate; zero graph rebuild!)
// inv_temp == 0.0f => pure greedy argmax with zero extra memory traffic.
struct SampleConfig {
    float inv_temp;
    float top_p;
    int32_t top_k;
    uint32_t pad;
    uint64_t seed;
};

// LoopCD-Hidden (arXiv:2610.02185 Eq. 2): h' = h_R + omega * (h_R - h_ref) before Coda layers.
// Lives in device memory so CUDA Graphs never rebuild when omega or ref_pass changes per request.
struct LoopCDConfig {
    float omega;      // 0.0 = disabled (bit-exact identity); > 0 = contrastive extrapolation
    int32_t ref_pass; // 1 = h_1 (after pass 0, vi=17), 2 = h_2 (after pass 1, vi=29)
};

constexpr int KV_SPLIT_K = 32;    // 8 KV heads x 64 bulk splits = 512 CTAs (3.0 blocks/SM!)
constexpr int KV_LOCAL_SPLITS = 17; // 17 x 64 = 1088 >= 1024+8 (1 single tile per CTA!)
constexpr int KV_TOTAL_SPLITS = KV_SPLIT_K + KV_LOCAL_SPLITS; // 81 slots in split_buf
constexpr int KV_Q4_GROUP = 64;  // 2 groups per 128-dim head
constexpr int KV_Q4_GROUPS_PER_HEAD = HEAD_DIM / KV_Q4_GROUP; // 2
constexpr int KV_Q4_SCALES_PER_TOK = NUM_KV_HEADS * KV_Q4_GROUPS_PER_HEAD; // 16

struct VisitInfo {
    int visit_idx;
    int physical_layer;
    int pass;   // 0 for prelude/coda, 0..2 for loop
    bool is_loop;
    bool is_global;
};

inline void build_visit_schedule(VisitInfo out[NUM_VISITS]) {
    int idx = 0;
    for (int l = 0; l < LOOP_START; ++l) {
        out[idx] = {idx, l, 0, false, is_global_layer(l)};
        idx++;
    }
    for (int p = 0; p < LOOP_PASSES; ++p) {
        for (int l = LOOP_START; l < LOOP_END; ++l) {
            out[idx] = {idx, l, p, true, is_global_layer(l)};
            idx++;
        }
    }
    for (int l = LOOP_END; l < NUM_LAYERS; ++l) {
        out[idx] = {idx, l, 0, false, is_global_layer(l)};
        idx++;
    }
}

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                           \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

} // namespace squid
