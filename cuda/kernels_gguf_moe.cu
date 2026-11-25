#include "config.hpp"
#include "kernels_common.cuh"
// GGUF grids have lane-dependent indices. Let these TU-local tables use the
// read-only global cache instead of serializing a warp on constant memory.
#pragma push_macro("__constant__")
#undef __constant__
#define __constant__
#include "gguf_quants.cuh"
#pragma pop_macro("__constant__")

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace squid {
namespace {

using namespace gguf;

constexpr int kThreads = 256;
constexpr int kDecodeTokens = 8;
constexpr int kUpThreads = 256;
constexpr int kPrefillThreads = 256;
constexpr int kPrefillRows = kPrefillThreads / 32 * 8;
constexpr int kDownRows = 4;
constexpr int kMaxTiles = 256;
constexpr int kTileExpert = NUM_EXPERTS + 2;
constexpr int kTileBegin = kTileExpert + kMaxTiles;
constexpr int kTileEnd = kTileBegin + kMaxTiles;
constexpr unsigned kWarpMask = 0xffffffffu;

static_assert(TOP_K == 8 && TOP_K * kDecodeTokens == 64);
static_assert(HIDDEN % kDownRows == 0);

template<class Block, int K>
constexpr int row_bytes = K / quant_traits<Block>::values * sizeof(Block);

__device__ __forceinline__ unsigned load_word(const uint16_t* block, int byte) {
    return (block[byte / 2] >> (8 * (byte & 1))) & 255;
}

__device__ __forceinline__ float half_value(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// Each chunk carries eight codes. IQ and Q2_0 leave the FP16 block scale
// outside the integer codes so prefill can feed exact values to the MMA.
template<class Block>
struct QuantChunk {
    uint32_t lo;
    uint32_t hi;
    int scale;
};

// Look up four IQ4_NL codes using two 8-entry byte tables. Bias values by
// 128 so the half2 conversion can recover signed integers by subtraction.
__device__ __forceinline__ uint32_t iq4_nl_codes(uint32_t codes) {
    const uint32_t indices = codes & 0x07070707;
    const uint32_t paired = (indices | (indices >> 4)) & 0x00770077;
    const uint32_t selector = paired | (paired >> 8);
    // ggml's kvalues_iq4nl, biased and packed in index order.
    const uint32_t low = __byte_perm(0x3f2d1801, 0x766a5d4f, selector);
    const uint32_t high = __byte_perm(0xa6998d81, 0xf1d9c5b5, selector);
    const uint32_t mask = ((codes & 0x08080808) >> 3) * 255;
    return (low & ~mask) | (high & mask);
}

template<class Block>
__device__ __forceinline__ QuantChunk<Block> load_chunk(const uint16_t* p, int chunk) {
    uint64_t grid;
    int scale;
    if constexpr (quant_traits<Block>::type == Type::iq2_xxs) {
        const auto* group = p + 1 + 4 * (chunk / 4);
        const int sub = chunk % 4;
        const unsigned flags = unsigned(group[2]) | (unsigned(group[3]) << 16);
        grid = __ldg(detail::iq2xxs_grid + load_word(group, sub));
        const unsigned code = (flags >> (7 * sub)) & 127;
        const unsigned signs = code | ((__popc(code) & 1) << 7);
        scale = (2 * (flags >> 28) + 1) | (signs << 8);
    } else if constexpr (quant_traits<Block>::type == Type::iq3_xxs) {
        const unsigned indices = p[1 + chunk];
        const auto* packed = p + 33 + 2 * (chunk / 4);
        const unsigned flags = unsigned(packed[0]) | (unsigned(packed[1]) << 16);
        const unsigned code = (flags >> (7 * (chunk % 4))) & 127;
        const unsigned signs = code | ((__popc(code) & 1) << 7);
        const uint32_t lo = __ldg(detail::iq3xxs_grid + (indices & 255));
        const uint32_t hi = __ldg(detail::iq3xxs_grid + (indices >> 8));
        grid = uint64_t(lo) | (uint64_t(hi) << 32);
        scale = (2 * (flags >> 28) + 1) | (signs << 8);
    } else if constexpr (quant_traits<Block>::type == Type::iq2_xs) {
        const unsigned packed = p[1 + chunk];
        grid = __ldg(detail::iq2xs_grid + (packed & 511));
        const int nibble = (load_word(p, 66 + chunk / 4) >> (4 * ((chunk / 2) & 1))) & 15;
        const unsigned code = packed >> 9;
        const unsigned signs = code | ((__popc(code) & 1) << 7);
        scale = (2 * nibble + 1) | (signs << 8);
    } else if constexpr (quant_traits<Block>::type == Type::iq2_s) {
        const int high = load_word(p, 66 + chunk / 4) >> (2 * (chunk % 4));
        grid = __ldg(detail::iq2s_grid + (load_word(p, 2 + chunk) | ((high & 3) << 8)));
        const int nibble = (load_word(p, 74 + chunk / 4) >> (4 * ((chunk / 2) & 1))) & 15;
        scale = (2 * nibble + 1) | (load_word(p, 34 + chunk) << 8);
    } else if constexpr (quant_traits<Block>::type == Type::iq1_s) {
        const unsigned flags = p[17 + chunk / 4];
        const int high = (flags >> (3 * (chunk % 4))) & 7;
        grid = __ldg(detail::iq1s_grid + (load_word(p, 2 + chunk) | (high << 8)));
        scale = (2 * ((flags >> 12) & 7) + 1) | ((flags >> 15) << 8);
    } else if constexpr (quant_traits<Block>::type == Type::q2_0) {
        grid = p[1 + chunk];
        scale = 1;
    } else if constexpr (quant_traits<Block>::type == Type::iq4_nl) {
        const auto* packed = p + 1 + 4 * (chunk % 2);
        const int shift = 4 * (chunk / 2);
        const uint32_t lo = (uint32_t(packed[0]) | (uint32_t(packed[1]) << 16)) >> shift;
        const uint32_t hi = (uint32_t(packed[2]) | (uint32_t(packed[3]) << 16)) >> shift;
        grid = uint64_t(iq4_nl_codes(lo)) | (uint64_t(iq4_nl_codes(hi)) << 32);
        scale = 1;
    } else {
        static_assert(quant_traits<Block>::type == Type::nvfp4);
        const auto* codes = p + 2 + 4 * (chunk / 2);
        grid = uint64_t(codes[0]) | (uint64_t(codes[1]) << 16) |
               (uint64_t(codes[2]) << 32) | (uint64_t(codes[3]) << 48);
        grid = (grid >> (4 * (chunk & 1))) & 0x0f0f0f0f0f0f0f0full;
        scale = load_word(p, chunk / 2);
    }
    return {uint32_t(grid), uint32_t(grid >> 32), scale};
}

template<class Block>
__device__ __forceinline__ float block_scale(const uint16_t* p) {
    if constexpr (quant_traits<Block>::type == Type::nvfp4) return 1.0f;
    if constexpr (quant_traits<Block>::type == Type::q2_0 ||
                  quant_traits<Block>::type == Type::iq4_nl) return half_value(p[0]);
    // IQ3_XXS uses d * (2 * scale + 1) / 4. The odd factor is in the chunk.
    if constexpr (quant_traits<Block>::type == Type::iq3_xxs) return half_value(p[0]) * 0.25f;
    return half_value(p[0]) * 0.125f;
}

__device__ __forceinline__ __half2 pair_from_bits(uint32_t bits) {
    return __half2(__half2_raw{uint16_t(bits), uint16_t(bits >> 16)});
}

template<class Block, bool ApplyScale = true>
__device__ __forceinline__ __half2 weight_pair(const QuantChunk<Block>& q, int j) {
    if constexpr (quant_traits<Block>::type == Type::q2_0) {
        const uint32_t code = q.lo >> (4 * j);
        const uint32_t hi = __byte_perm(0x403c00bc, 0, (code & 3) | ((code & 12) << 6));
        return pair_from_bits((hi << 8) & 0xff00ff00);
    } else {
        uint32_t codes = j < 2 ? q.lo : q.hi;
        if constexpr (quant_traits<Block>::type == Type::nvfp4) {
            // FP16 high bytes for the doubled E2M1 magnitudes.
            codes >>= 16 * (j & 1);
            const uint32_t hi = __byte_perm(0x42403c00, 0x4a484644, codes & 0x707);
            const uint32_t signs = ((codes & 8) << 12) | ((codes & 0x800) << 20);
            return pair_from_bits(((hi << 8) & 0xff00ff00) | signs);
        } else if constexpr (quant_traits<Block>::type == Type::iq4_nl) {
            const __half2 v = pair_from_bits(__byte_perm(codes, 0x64006400, 0x5150 + 0x0202 * (j & 1)));
            return __hsub2(v, __float2half2_rn(1152.0f));
        } else {
            const __half2 scale = __float2half2_rn(float(q.scale & 255));
            if constexpr (quant_traits<Block>::type == Type::iq1_s) {
                // Add one per byte to map {-1,0,1} to {0,1,2}.
                codes = __vadd4(codes, 0x01010101);
                const __half2 v = pair_from_bits(__byte_perm(codes, 0x64006400, 0x5150 + 0x0202 * (j & 1)));
                const __half2 base = __hsub2(v, __float2half2_rn(1025.0f));
                const __half2 offset = __float2half2_rn(q.scale & 256 ? -1.0f : 1.0f);
                const __half2 value = __hfma2(base, __float2half2_rn(8.0f), offset);
                if constexpr (ApplyScale) return __hmul2(value, scale);
                return value;
            } else {
                const __half2 v = pair_from_bits(__byte_perm(codes, 0x64006400, 0x5150 + 0x0202 * (j & 1)));
                __half2 magnitude = __hsub2(v, __float2half2_rn(1024.0f));
                if constexpr (ApplyScale) magnitude = __hmul2(magnitude, scale);
                const uint32_t sign = q.scale >> (8 + 2 * j);
                __half2_raw bits = magnitude;
                bits.x ^= (sign & 1) << 15;
                bits.y ^= (sign & 2) << 14;
                return __half2(bits);
            }
        }
    }
}

template<class Block>
__device__ __forceinline__ float chunk_scale(const QuantChunk<Block>& q) {
    if constexpr (quant_traits<Block>::type == Type::nvfp4)
        return detail::ue4m3_half(q.scale & 255);
    return 1.0f;
}

struct DotWeights {
    float2 pairs[4];
};

template<class Block>
__device__ __forceinline__ DotWeights decode_weights(const QuantChunk<Block>& q) {
    DotWeights w;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        w.pairs[j] = __half22float2(weight_pair<Block, false>(q, j));
    return w;
}

template<bool Cached = false, bool Vector = true>
__device__ __forceinline__ float dot_chunk(const DotWeights& w, const __half2* x) {
    float a = 0.0f;
    float b = 0.0f;
    uint4 vec{};
    if constexpr (Vector) {
        const auto* p = reinterpret_cast<const uint4*>(x);
        vec = Cached ? __ldg(p) : *p;
    }
    const uint32_t bits[4] = {vec.x, vec.y, vec.z, vec.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2 v = Vector ? pair_from_bits(bits[j]) : Cached ? __ldg(x + j) : x[j];
        const float2 pair = __half22float2(v);
        a = fmaf(w.pairs[j].x, pair.x, a);
        b = fmaf(w.pairs[j].y, pair.y, b);
    }
    return a + b;
}

__device__ __forceinline__ __half activate(float gate, float up) {
    const float g_cap = fast_tanh(gate * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
    const float u_cap = fast_tanh(up * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
    return __float2half((g_cap * fast_sigmoid(gate)) * u_cap);
}

__device__ __forceinline__ float outer_scale(const float* scales, int expert) {
    return scales ? __ldg(scales + expert) : 1.0f;
}

template<int MaxTokens>
__device__ __forceinline__ uint64_t slot_mask(int expert, int lo, int hi) {
    const uint32_t low = __ballot_sync(kWarpMask, lo == expert);
    if constexpr (MaxTokens <= 4) return low;
    const uint32_t high = __ballot_sync(kWarpMask, hi == expert);
    return low | (uint64_t(high) << 32);
}

template<class Block, int MaxTokens>
constexpr int up_lanes = MaxTokens == 8 && quant_traits<Block>::type != Type::iq2_s ? 8 : 16;

template<int Sub>
__device__ __forceinline__ QuantChunk<block_iq2_s> load_iq2s_pair(
    unsigned low, unsigned signs, unsigned flags, int lane) {
    const unsigned high = (flags >> (4 * (lane & 3) + 2 * Sub)) & 3;
    const unsigned nibble = (flags >> (16 + 4 * (lane & 3))) & 15;
    const uint64_t grid = __ldg(detail::iq2s_grid + (((low >> (8 * Sub)) & 255) | (high << 8)));
    const int scale = (2 * nibble + 1) | (((signs >> (8 * Sub)) & 255) << 8);
    return {uint32_t(grid), uint32_t(grid >> 32), scale};
}

template<class Block, int MaxTokens>
__global__ __launch_bounds__(kUpThreads, 4) void decode_up(
    const uint8_t* data, const float* scales, const __half* x,
    const int32_t* ids, __half* out, int slots) {
    constexpr int kLanes = up_lanes<Block, MaxTokens>;
    constexpr int kRows = kUpThreads / kLanes;
    constexpr int kQuant = quant_traits<Block>::values;
    constexpr int kChunks = kQuant / (8 * kLanes);
    static_assert(2 * INTERMEDIATE % kRows == 0);
    constexpr int kRowBytes = row_bytes<Block, HIDDEN>;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int slot = blockIdx.y;
    const int expert = ids[slot];
    const int lo = lane < slots ? ids[lane] : -1;
    const int hi = MaxTokens > 4 && lane + 32 < slots ? ids[lane + 32] : -1;
    const uint64_t routes = slot_mask<MaxTokens>(expert, lo, hi);
    // The first slot for each expert owns all matching output slots.
    if (__ffsll(routes) - 1 != slot) return;

    const int r = tid / kLanes;
    const int c = tid % kLanes;
    const int row = blockIdx.x * kRows + r;
    const auto* p = reinterpret_cast<const uint16_t*>(
        data + (size_t(expert) * (2 * INTERMEDIATE) + row) * kRowBytes);
    int tokens[MaxTokens];
    if constexpr (MaxTokens == 1) {
        tokens[0] = 0;
    } else {
        uint64_t pending = routes;
        #pragma unroll
        for (int m = 0; m < MaxTokens; ++m) {
            const int token = (__ffsll(pending) - 1) / TOP_K;
            tokens[m] = pending ? token : -1;
            pending &= ~(uint64_t(255) << (token * TOP_K));
        }
    }
    float acc[MaxTokens] = {};
    constexpr int kUnroll = MaxTokens == 1 ? 3 : 1;
    #pragma unroll kUnroll
    for (int b = 0; b < HIDDEN / kQuant; ++b) {
        const float d = block_scale<Block>(p);
        unsigned low = 0, signs = 0, flags = 0;
        if constexpr (quant_traits<Block>::type == Type::iq2_s) {
            // Each lane reads both low codes and sign bytes of a chunk pair.
            // The first eight lanes also read the block's high codes and scales.
            low = __ldcg(p + 1 + c);
            signs = __ldcg(p + 17 + c);
            const unsigned extra = c < 8 ? __ldcg(p + 33 + c) : 0;
            flags = __shfl_sync(kWarpMask, extra, c / 4, kLanes) |
                    (__shfl_sync(kWarpMask, extra, 4 + c / 4, kLanes) << 16);
        }
        auto accumulate = [&]<int Sub>() {
            const int chunk = kChunks * c + Sub;
            QuantChunk<Block> q;
            if constexpr (quant_traits<Block>::type == Type::iq2_s)
                q = load_iq2s_pair<Sub>(low, signs, flags, c);
            else
                q = load_chunk<Block>(p, chunk);
            const auto w = decode_weights(q);
            const float d_chunk = d * float(q.scale & 255);
            #pragma unroll
            for (int m = 0; m < MaxTokens; ++m) {
                if (MaxTokens == 1 || tokens[m] >= 0) {
                    const auto* v = reinterpret_cast<const __half2*>(x + tokens[m] * HIDDEN + b * kQuant + chunk * 8);
                    acc[m] = fmaf(dot_chunk<true>(w, v), d_chunk, acc[m]);
                }
            }
        };
        accumulate.template operator()<0>();
        accumulate.template operator()<1>();
        if constexpr (kChunks == 4) {
            accumulate.template operator()<2>();
            accumulate.template operator()<3>();
        }
        p += sizeof(Block) / 2;
    }
    const float es = outer_scale(scales, expert);
    #pragma unroll
    for (int m = 0; m < MaxTokens; ++m) {
        if (MaxTokens == 1 || tokens[m] >= 0) {
            float sum = acc[m];
            #pragma unroll
            for (int step = kLanes / 2; step > 0; step /= 2)
                sum += __shfl_down_sync(kWarpMask, sum, step, kLanes);
            const float up = __shfl_xor_sync(kWarpMask, sum, kLanes);
            if (c == 0 && r % 2 == 0) {
                const __half value = activate(sum * es, up * es);
                unsigned matches = (routes >> (tokens[m] * TOP_K)) & 255;
                for (; matches; matches &= matches - 1) {
                    const int dst = tokens[m] * TOP_K + __ffs(matches) - 1;
                    out[size_t(dst) * INTERMEDIATE + row / 2] = value;
                }
            }
        }
    }
}

__device__ __forceinline__ void stage_input(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src), "r"(bytes) : "memory");
}

template<class Block, int MaxTokens>
__global__ __launch_bounds__(kThreads, MaxTokens == 8 && quant_traits<Block>::type == Type::nvfp4 ? 4 : 5) void decode_down(
    const uint8_t* data, const float* scales, const __half* act,
    const int32_t* ids, const float* route_w, float* out, int slots) {
    constexpr int kK = INTERMEDIATE;
    constexpr int kQuant = quant_traits<Block>::values;
    constexpr int kStep = 64;
    constexpr int kChunks = kQuant / 8;
    constexpr int kBlocksPerStep = kStep / kQuant;
    static_assert(kStep % kQuant == 0 && kK % kStep == 0);
    constexpr int kRowBytes = row_bytes<Block, kK>;
    constexpr bool kVectorInput = MaxTokens == 1;
    constexpr int kWarps = kThreads / 32;
    __shared__ float partial[MaxTokens * TOP_K][kDownRows];
    __shared__ uint64_t routes[MaxTokens * TOP_K];
    __shared__ uint4 input_tile[MaxTokens == 1 ? TOP_K * kK / 8 : 1];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int r = lane / 8;
    const int c = lane % 8;
    const int row = blockIdx.x * kDownRows + r;
    const int lo = lane < slots ? ids[lane] : -1;
    const int hi = MaxTokens > 4 && lane + 32 < slots ? ids[lane + 32] : -1;
    const auto* x = reinterpret_cast<const __half2*>(act);
    if constexpr (MaxTokens == 1) {
        for (int i = tid; i < TOP_K * kK / 8; i += kThreads)
            input_tile[i] = __ldg(reinterpret_cast<const uint4*>(act) + i);
        __syncthreads();
        x = reinterpret_cast<const __half2*>(input_tile);
    }
    for (int slot = warp; slot < slots; slot += kWarps) {
        const int expert = __shfl_sync(kWarpMask, slot < 32 ? lo : hi, slot % 32);
        const uint64_t mask = slot_mask<MaxTokens>(expert, lo, hi);
        if (lane == 0) routes[slot] = mask;
        if (__ffsll(mask) - 1 != slot) continue;
        const auto* p = reinterpret_cast<const uint16_t*>(
            data + (size_t(expert) * HIDDEN + row) * kRowBytes);
        const float es = outer_scale(scales, expert);

        if (mask == (uint64_t(1) << slot)) {
            float sum = 0;
            const auto* v = x + slot * (kK / 2) + c * 4;
            #pragma unroll 2
            for (int b = 0; b < kK / kStep; ++b) {
                const auto* packed = p + b * (sizeof(Block) / 2);
                int chunk = c;
                // Eight lanes cover 64 values, split across two IQ4_NL blocks.
                if constexpr (kBlocksPerStep > 1) {
                    packed = p + (b * kBlocksPerStep + c / kChunks) * (sizeof(Block) / 2);
                    chunk = c % kChunks;
                }
                const auto q = load_chunk<Block>(packed, chunk);
                const float d = block_scale<Block>(packed) * chunk_scale(q);
                sum = fmaf(dot_chunk<MaxTokens != 1, kVectorInput>(decode_weights(q), v + b * (kStep / 2)), d, sum);
            }
            #pragma unroll
            for (int step = 4; step > 0; step /= 2)
                sum += __shfl_down_sync(kWarpMask, sum, step, 8);
            if (c == 0) partial[slot][r] = (sum * es) * route_w[slot];
        } else {
            // Combine repeated routes within each token in FP32.
            float sum[MaxTokens] = {};
            #pragma unroll 1
            for (int b = 0; b < kK / kStep; ++b) {
                const auto* packed = p + b * (sizeof(Block) / 2);
                int chunk = c;
                if constexpr (kBlocksPerStep > 1) {
                    packed = p + (b * kBlocksPerStep + c / kChunks) * (sizeof(Block) / 2);
                    chunk = c % kChunks;
                }
                const auto q = load_chunk<Block>(packed, chunk);
                const float d = block_scale<Block>(packed) * chunk_scale(q);
                const auto w = decode_weights(q);
                #pragma unroll
                for (int t = 0; t < MaxTokens; ++t) {
                    unsigned pending = (mask >> (t * TOP_K)) & 255;
                    if (pending) {
                        float dot = 0;
                        for (; pending; pending &= pending - 1) {
                            const int match = t * TOP_K + __ffs(pending) - 1;
                            const auto* v = x + match * (kK / 2) + b * (kStep / 2) + c * 4;
                            dot = fmaf(dot_chunk<MaxTokens != 1, kVectorInput>(w, v), route_w[match], dot);
                        }
                        sum[t] = fmaf(dot, d, sum[t]);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < MaxTokens; ++t) {
                const unsigned matches = (mask >> (t * TOP_K)) & 255;
                if (matches) {
                    float dot = sum[t];
                    #pragma unroll
                    for (int step = 4; step > 0; step /= 2)
                        dot += __shfl_down_sync(kWarpMask, dot, step, 8);
                    if (c == 0) partial[t * TOP_K + __ffs(matches) - 1][r] = dot * es;
                }
            }
        }
    }
    __syncthreads();
    const int token = tid / kDownRows;
    if (token < slots / TOP_K) {
        const int r_out = tid % kDownRows;
        float sum = 0;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            const int slot = token * TOP_K + k;
            const unsigned matches = (routes[slot] >> (token * TOP_K)) & 255;
            if (__ffs(matches) - 1 == k) sum += partial[slot][r_out];
        }
        out[size_t(token) * HIDDEN + blockIdx.x * kDownRows + r_out] = sum;
    }
}

// Tell the compiler that the opaque shared addresses read staged data.
__device__ __forceinline__ void load_weight_fragment(uint32_t& b0, uint32_t& b1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(b0), "=r"(b1) : "r"(uint32_t(__cvta_generic_to_shared(p))) : "memory");
}

__device__ __forceinline__ void load_input_fragment(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(uint32_t(__cvta_generic_to_shared(p))) : "memory");
}

// The four lanes of a quad each decode eight consecutive weights. Transpose
// their packed pairs into the column-major B fragments expected by mma.
struct WeightFragments {
    uint32_t a, b, c, d;
};

template<class Block>
__device__ __forceinline__ WeightFragments weight_fragments(const QuantChunk<Block>& q, int lane) {
    uint32_t bits[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        __half2 pair = weight_pair(q, j);
        if constexpr (quant_traits<Block>::type == Type::nvfp4)
            pair = __hmul2(pair, __float2half2_rn(chunk_scale(q)));
        const __half2_raw raw = pair;
        bits[j] = unsigned(raw.x) | (unsigned(raw.y) << 16);
    }

    const bool odd = lane & 1;
    const uint32_t low = __shfl_xor_sync(kWarpMask, odd ? bits[0] : bits[1], 1);
    const uint32_t high = __shfl_xor_sync(kWarpMask, odd ? bits[2] : bits[3], 1);
    const uint32_t a = odd ? low : bits[0];
    const uint32_t b = odd ? bits[1] : low;
    const uint32_t c = odd ? high : bits[2];
    const uint32_t d = odd ? bits[3] : high;
    const bool upper = lane & 2;
    const uint32_t even = __shfl_xor_sync(kWarpMask, upper ? a : c, 2);
    const uint32_t other = __shfl_xor_sync(kWarpMask, upper ? b : d, 2);
    return {upper ? even : a, upper ? other : b, upper ? c : even, upper ? d : other};
}

template<class Block>
constexpr int prefill_min_blocks = quant_traits<Block>::type == Type::nvfp4 ? 6 :
                                   quant_traits<Block>::type == Type::iq2_xxs ||
                                   quant_traits<Block>::type == Type::iq3_xxs ||
                                   quant_traits<Block>::type == Type::q2_0 ? 3 : 4;

template<class Block, bool Up>
__global__ __launch_bounds__(kPrefillThreads, prefill_min_blocks<Block>) void prefill(const uint8_t* data, const float* scales, const __half* input,
                        const int32_t* offsets, const int32_t* sorted_tok,
                        const int32_t* sorted_slot, const float* sorted_w,
                        __half* out_act, float* out_slots) {
    constexpr int kK = Up ? HIDDEN : INTERMEDIATE;
    constexpr int kR = Up ? 2 * INTERMEDIATE : HIDDEN;
    constexpr int kRows = kPrefillRows;
    constexpr int kTokens = 32;
    constexpr bool kRegisterWeights = quant_traits<Block>::type == Type::q2_0 ||
                                      quant_traits<Block>::type == Type::iq2_xxs ||
                                      quant_traits<Block>::type == Type::iq3_xxs;
    constexpr int kStep = quant_traits<Block>::type == Type::nvfp4 ||
                          quant_traits<Block>::type == Type::iq4_nl ? 64 :
                          Up && kRegisterWeights ? 256 : 128;
    constexpr int kQuant = quant_traits<Block>::values;
    // Apply each down block's scale after its own MMA group.
    constexpr int kGroup = Up ? kStep : kQuant;
    constexpr int kGroups = kStep / kGroup;
    static_assert(kK % kStep == 0 && kStep % kGroup == 0);
    static_assert(Up ? kQuant % kStep == 0 : kGroup % 16 == 0);
    constexpr int kRowBytes = row_bytes<Block, kK>;

    const int tile = blockIdx.y;
    if (tile >= offsets[NUM_EXPERTS + 1]) return;
    const int start = offsets[kTileBegin + tile] + blockIdx.z * kTokens;
    const int n = min(kTokens, offsets[kTileEnd + tile] - start);
    if (n <= 0) return;
    const int expert = offsets[kTileExpert + tile];
    const int row = blockIdx.x * kRows;

    __shared__ uint4 x[kTokens][kStep / 8 + 1];
    __shared__ __align__(16) __half2 w[kRegisterWeights ? 1 : kRows][kRegisterWeights ? 1 : kStep / 2 + 4];
    __shared__ const __half* source[kTokens];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int c = lane % 4 * 2;
    if (tid < n) {
        const int idx = start + tid;
        source[tid] = input + size_t(Up ? sorted_tok[idx] : idx) * kK;
    }
    __syncthreads();

    float sum[2][4] = {};
    const auto* packed = data + (size_t(expert) * kR + row) * kRowBytes;
    #pragma unroll 1
    for (int k = 0; k < kK; k += kStep) {
        for (int i = tid; i < kTokens * (kStep / 8); i += kPrefillThreads) {
            const int m = i / (kStep / 8);
            const int kk = i % (kStep / 8);
            const int src = m < n ? m : 0;
            stage_input(&x[m][kk], source[src] + k + kk * 8, m < n ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        if constexpr (!kRegisterWeights) {
            for (int i = tid; i < kRows * (kStep / 8); i += kPrefillThreads) {
                const int r = i / (kStep / 8);
                const int kk = i % (kStep / 8) * 8;
                const auto* p = reinterpret_cast<const uint16_t*>(
                    packed + r * kRowBytes + ((k + kk) / kQuant) * sizeof(Block));
                const auto q = load_chunk<Block>(p, (k + kk) % kQuant / 8);
                uint32_t bits[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    __half2 pair = weight_pair(q, j);
                    if constexpr (quant_traits<Block>::type == Type::nvfp4)
                        pair = __hmul2(pair, __float2half2_rn(chunk_scale(q)));
                    const __half2_raw raw = pair;
                    bits[j] = unsigned(raw.x) | (unsigned(raw.y) << 16);
                }
                *reinterpret_cast<uint4*>(&w[r][kk / 2]) = make_uint4(bits[0], bits[1], bits[2], bits[3]);
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        #pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            const int sub = group * kGroup;
            float part[2][4] = {};
            if constexpr (kRegisterWeights) {
                const auto* weight = reinterpret_cast<const uint16_t*>(
                    packed + (warp * 8 + lane / 4) * kRowBytes + ((k + sub) / kQuant) * sizeof(Block));
                #pragma unroll 2
                for (int kk = 0; kk < kGroup; kk += 32) {
                    const auto q = load_chunk<Block>(weight, kk / 8 + lane % 4);
                    const auto w = weight_fragments(q, lane);
                    #pragma unroll
                    for (int m = 0; m < kTokens / 16; ++m) {
                        if (m * 16 < n) {
                            uint32_t a0, a1, a2, a3;
                            const auto* input = &x[m * 16 + lane % 16][(sub + kk) / 8 + lane / 16];
                            load_input_fragment(a0, a1, a2, a3, input);
                            mma_m16n8k16_f32_f16(part[m][0], part[m][1], part[m][2], part[m][3],
                                                  a0, a1, a2, a3, w.a, w.b,
                                                  part[m][0], part[m][1], part[m][2], part[m][3]);
                            load_input_fragment(a0, a1, a2, a3, input + 2);
                            mma_m16n8k16_f32_f16(part[m][0], part[m][1], part[m][2], part[m][3],
                                                  a0, a1, a2, a3, w.c, w.d,
                                                  part[m][0], part[m][1], part[m][2], part[m][3]);
                        }
                    }
                }
            } else {
                #pragma unroll 4
                for (int kk = 0; kk < kGroup; kk += 16) {
                    const int offset = sub + kk;
                    uint32_t b0, b1;
                    load_weight_fragment(b0, b1, &w[warp * 8 + lane % 8][offset / 2 + 4 * ((lane / 8) & 1)]);
                    #pragma unroll
                    for (int m = 0; m < kTokens / 16; ++m) {
                        if (m * 16 < n) {
                            uint32_t a0, a1, a2, a3;
                            load_input_fragment(a0, a1, a2, a3, &x[m * 16 + lane % 16][offset / 8 + lane / 16]);
                            mma_m16n8k16_f32_f16(part[m][0], part[m][1], part[m][2], part[m][3],
                                                  a0, a1, a2, a3, b0, b1,
                                                  part[m][0], part[m][1], part[m][2], part[m][3]);
                        }
                    }
                }
            }
            const auto* p = packed + (warp * 8 + c) * kRowBytes + ((k + sub) / kQuant) * sizeof(Block);
            const float d0 = block_scale<Block>(reinterpret_cast<const uint16_t*>(p));
            const float d1 = block_scale<Block>(reinterpret_cast<const uint16_t*>(p + kRowBytes));
            #pragma unroll
            for (int m = 0; m < kTokens / 16; ++m) {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    sum[m][j] = fmaf(part[m][j], j % 2 ? d1 : d0, sum[m][j]);
            }
        }
        __syncthreads();
    }

    const float es = outer_scale(scales, expert);
    const int col = row + warp * 8 + c;
    #pragma unroll
    for (int m = 0; m < kTokens / 16; ++m) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int tok = m * 16 + lane / 4 + half * 8;
            if (tok < n) {
                const int idx = start + tok;
                const float a = sum[m][half * 2] * es;
                const float b = sum[m][half * 2 + 1] * es;
                if constexpr (Up) {
                    out_act[size_t(idx) * INTERMEDIATE + col / 2] = activate(a, b);
                } else {
                    const int dst = sorted_tok[idx] * TOP_K + sorted_slot[idx];
                    const float route = sorted_w[idx];
                    out_slots[size_t(dst) * HIDDEN + col] = a * route;
                    out_slots[size_t(dst) * HIDDEN + col + 1] = b * route;
                }
            }
        }
    }
}

template<class Block> struct BlockTag { using BlockType = Block; };

template<class F>
void dispatch_up(uint8_t type, F launch) {
    switch (static_cast<Type>(type)) {
        case Type::iq1_s: launch(BlockTag<block_iq1_s>{}); return;
        case Type::iq3_xxs: launch(BlockTag<block_iq3_xxs>{}); return;
        case Type::iq2_xxs: launch(BlockTag<block_iq2_xxs>{}); return;
        case Type::iq2_xs: launch(BlockTag<block_iq2_xs>{}); return;
        case Type::iq2_s: launch(BlockTag<block_iq2_s>{}); return;
        default: std::fprintf(stderr, "Unsupported GGUF MoE gate/up type %u\n", unsigned(type));
    }
    std::abort();
}

template<class F>
void dispatch_down(uint8_t type, F launch) {
    switch (static_cast<Type>(type)) {
        case Type::nvfp4: launch(BlockTag<block_nvfp4>{}); return;
        case Type::q2_0: launch(BlockTag<block_q2_0>{}); return;
        case Type::iq4_nl: launch(BlockTag<block_iq4_nl>{}); return;
        default: std::fprintf(stderr, "Unsupported GGUF MoE down type %u\n", unsigned(type));
    }
    std::abort();
}

} // namespace

void launch_moe_gate_up_gguf_h(uint8_t gguf_type, const uint8_t* w13_data, const float* expert_scale,
                               const __half* x_h, const int32_t* route_ids, __half* out_act_h,
                               int B, cudaStream_t stream) {
    dispatch_up(gguf_type, [&](auto tag) {
        using Block = typename decltype(tag)::BlockType;
        for (int start = 0; start < B; start += kDecodeTokens) {
            const int n = B - start < kDecodeTokens ? B - start : kDecodeTokens;
            const int slots = n * TOP_K;
            const __half* x = x_h + size_t(start) * HIDDEN;
            const int32_t* ids = route_ids + size_t(start) * TOP_K;
            __half* out = out_act_h + size_t(start) * TOP_K * INTERMEDIATE;
            auto launch = [&]<int MaxTokens>() {
                constexpr int kRows = kUpThreads / up_lanes<Block, MaxTokens>;
                const dim3 grid(2 * INTERMEDIATE / kRows, slots);
                decode_up<Block, MaxTokens><<<grid, kUpThreads, 0, stream>>>(
                    w13_data, expert_scale, x, ids, out, slots);
            };
            if (n == 1) launch.template operator()<1>();
            else if (n == 2) launch.template operator()<2>();
            else if (n <= 4) launch.template operator()<4>();
            else launch.template operator()<8>();
        }
    });
}

void launch_moe_down_gguf_h(uint8_t gguf_type, const uint8_t* w2_data, const float* expert_scale,
                            const __half* act_h, const int32_t* route_ids, const float* route_w,
                            float* out_down, int B, cudaStream_t stream) {
    dispatch_down(gguf_type, [&](auto tag) {
        using Block = typename decltype(tag)::BlockType;
        for (int start = 0; start < B; start += kDecodeTokens) {
            const int n = B - start < kDecodeTokens ? B - start : kDecodeTokens;
            const int slots = n * TOP_K;
            const size_t offset = size_t(start) * TOP_K;
            auto launch = [&]<int MaxTokens>() {
                decode_down<Block, MaxTokens><<<HIDDEN / kDownRows, kThreads, 0, stream>>>(
                    w2_data, expert_scale, act_h + offset * INTERMEDIATE, route_ids + offset,
                    route_w + offset, out_down + size_t(start) * HIDDEN, slots);
            };
            if (n == 1) launch.template operator()<1>();
            else if (n == 2) launch.template operator()<2>();
            else if (n <= 4) launch.template operator()<4>();
            else launch.template operator()<8>();
        }
    });
}

void launch_prefill_moe_gate_up_gguf_h(
    uint8_t gguf_type, const uint8_t* w13_data, const float* expert_scale,
    const __half* x_h, const int32_t* expert_offsets, const int32_t* sorted_tok,
    const int32_t*, __half* out_act_h, int B, cudaStream_t stream) {
    if (B <= 0) return;
    dispatch_up(gguf_type, [&](auto tag) {
        using Block = typename decltype(tag)::BlockType;
        prefill<Block, true><<<dim3(2 * INTERMEDIATE / kPrefillRows, kMaxTiles, B <= 512 ? 1 : 4),
                                kPrefillThreads, 0, stream>>>(
            w13_data, expert_scale, x_h, expert_offsets, sorted_tok, nullptr, nullptr, out_act_h, nullptr);
    });
}

void launch_prefill_moe_down_gguf_h(
    uint8_t gguf_type, const uint8_t* w2_data, const float* expert_scale,
    const __half* act_h, const int32_t* expert_offsets, const int32_t* sorted_tok,
    const int32_t* sorted_slot, const float* sorted_w, float* out_slots, int B, cudaStream_t stream) {
    if (B <= 0) return;
    dispatch_down(gguf_type, [&](auto tag) {
        using Block = typename decltype(tag)::BlockType;
        prefill<Block, false><<<dim3(HIDDEN / kPrefillRows, kMaxTiles, B <= 512 ? 1 : 4),
                                 kPrefillThreads, 0, stream>>>(
            w2_data, expert_scale, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, nullptr, out_slots);
    });
}

} // namespace squid
