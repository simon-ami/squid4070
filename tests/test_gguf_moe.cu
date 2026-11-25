#include "config.hpp"
#include "gguf_quants.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace squid {
void launch_moe_gate_up_gguf_h(uint8_t, const uint8_t*, const float*, const __half*,
                               const int32_t*, __half*, int, cudaStream_t);
void launch_moe_down_gguf_h(uint8_t, const uint8_t*, const float*, const __half*,
                            const int32_t*, const float*, float*, int, cudaStream_t);
void launch_prefill_moe_gate_up_gguf_h(uint8_t, const uint8_t*, const float*, const __half*,
                                       const int32_t*, const int32_t*, const int32_t*, __half*, int, cudaStream_t);
void launch_prefill_moe_down_gguf_h(uint8_t, const uint8_t*, const float*, const __half*,
                                    const int32_t*, const int32_t*, const int32_t*, const float*,
                                    float*, int, cudaStream_t);
} // namespace squid

namespace {

using namespace squid;
using namespace squid::gguf;

constexpr const char* kModel =
    "/mnt/d/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ2_XXS.gguf";
constexpr const char* kModelIq3 =
    "/mnt/d/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ3_XXS.gguf";
constexpr int kMaxTiles = 256;
constexpr int kTileExpert = NUM_EXPERTS + 2;
constexpr int kTileBegin = kTileExpert + kMaxTiles;
constexpr int kTileEnd = kTileBegin + kMaxTiles;
constexpr float kTolerance = 0.01f;

void check_cuda(cudaError_t status, const char* expr, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "line %d: %s: %s\n", line, expr, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA(expr) check_cuda((expr), #expr, __LINE__)

template<class T>
struct DeviceBuffer {
    explicit DeviceBuffer(size_t count, size_t pad = 0) {
        CUDA(cudaMalloc(&base, (count + pad) * sizeof(T)));
        ptr = base + pad;
    }
    ~DeviceBuffer() { CUDA(cudaFree(base)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const T* data, size_t count) {
        CUDA(cudaMemcpy(ptr, data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* base = nullptr;
    T* ptr = nullptr;
};

struct Tensor {
    uint64_t offset;
    std::vector<uint64_t> dims;
    Type type;
};

class GgufFile {
public:
    explicit GgufFile(const char* path) : file_(path, std::ios::binary) {
        if (!file_ || read<uint32_t>() != 0x46554747 || read<uint32_t>() != 3)
            throw std::runtime_error("Expected a GGUF v3 file");
        const uint64_t tensors = read<uint64_t>();
        const uint64_t metadata = read<uint64_t>();
        uint32_t alignment = 32;
        for (uint64_t i = 0; i < metadata; ++i) {
            const std::string key = read_string();
            const uint32_t type = read<uint32_t>();
            if (key == "general.alignment" && type == 4) alignment = read<uint32_t>();
            else skip_metadata(type, 1);
        }
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw std::runtime_error("Invalid GGUF alignment");
        for (uint64_t i = 0; i < tensors; ++i) {
            std::string name = read_string();
            const uint32_t rank = read<uint32_t>();
            if (rank < 1 || rank > 4) throw std::runtime_error("Invalid GGUF tensor rank");
            std::vector<uint64_t> dims(rank);
            for (uint64_t& d : dims) d = read<uint64_t>();
            const auto type = static_cast<Type>(read<uint32_t>());
            const uint64_t offset = read<uint64_t>();
            if (!directory_.emplace(name, Tensor{offset, std::move(dims), type}).second)
                throw std::runtime_error("Duplicate tensor " + name);
        }
        const auto end = static_cast<uint64_t>(file_.tellg());
        data_start_ = (end + alignment - 1) & ~(uint64_t(alignment) - 1);
    }

    bool contains(const std::string& name) const { return directory_.contains(name); }
    Type type_of(const std::string& name) const { return directory_.at(name).type; }

    void read_tensor(const std::string& name, Type type, std::vector<uint64_t> dims,
                     void* data, size_t bytes) {
        const Tensor& t = directory_.at(name);
        if (t.type != type || t.dims != dims)
            throw std::runtime_error("Unexpected type or shape for " + name);
        file_.seekg(data_start_ + t.offset);
        read_bytes(data, bytes);
    }

private:
    void read_bytes(void* data, size_t count) {
        if (!file_.read(static_cast<char*>(data), count))
            throw std::runtime_error("Incomplete GGUF read");
    }

    template<class T>
    T read() {
        T value;
        read_bytes(&value, sizeof(value));
        return value;
    }

    std::string read_string() {
        std::string result(read<uint64_t>(), '\0');
        read_bytes(result.data(), result.size());
        return result;
    }

    void skip(uint64_t bytes) {
        file_.ignore(bytes);
        if (uint64_t(file_.gcount()) != bytes)
            throw std::runtime_error("Invalid GGUF metadata offset");
    }

    void skip_metadata(uint32_t type, uint64_t count) {
        static constexpr int sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type >= std::size(sizes)) throw std::runtime_error("Invalid GGUF metadata type");
        if (sizes[type] != 0) {
            skip(count * sizes[type]);
        } else if (type == 8) {
            for (uint64_t i = 0; i < count; ++i) skip(read<uint64_t>());
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                const uint32_t item = read<uint32_t>();
                const uint64_t length = read<uint64_t>();
                skip_metadata(item, length);
            }
        }
    }

    std::ifstream file_;
    std::unordered_map<std::string, Tensor> directory_;
    uint64_t data_start_ = 0;
};

struct TestCase {
    TestCase(int batch, bool prefill, bool up, unsigned seed, bool repeated_slot = false)
        : b(batch), prefill(prefill), repeated_slot(repeated_slot), ids(b * TOP_K), weights(b * TOP_K),
          offsets(kTileEnd + kMaxTiles), sorted_tok(b * TOP_K), sorted_slot(b * TOP_K),
          sorted_w(b * TOP_K), input(size_t(up ? b : b * TOP_K) * (up ? HIDDEN : INTERMEDIATE)),
          reference(size_t(up || prefill ? b * TOP_K : b) * (up ? INTERMEDIATE : HIDDEN)) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> activation(0.0f, up ? 0.65f : 0.75f);
        std::uniform_real_distribution<float> weight(0.1f, 1.0f);
        std::uniform_int_distribution<int> expert(0, NUM_EXPERTS - 1);
        const int shared = expert(rng);
        for (int token = 0; token < b; ++token) {
            std::array<bool, NUM_EXPERTS> taken{};
            float total = 0.0f;
            const int shared_slot = token % TOP_K;
            for (int k = 0; k < TOP_K; ++k) {
                int e = shared;
                if (k != shared_slot || (prefill && token % 3 == 0)) {
                    do e = expert(rng); while (taken[e] || e == shared);
                }
                taken[e] = true;
                const int pos = token * TOP_K + k;
                ids[pos] = e;
                weights[pos] = weight(rng);
                total += weights[pos];
            }
            for (int k = 0; k < TOP_K; ++k) weights[token * TOP_K + k] /= total;
        }
        if (repeated_slot) ids[TOP_K - 1] = ids[0];
        // Give the oracle distinct down inputs even when two tokens share an expert.
        for (auto& value : input) value = __float2half(activation(rng));
        for (int pos = 0; pos < b * TOP_K; ++pos) by_expert[ids[pos]].push_back(pos);

        int pos = 0;
        int nt = 0;
        const int tr = b <= 512 ? 32 : 128;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            offsets[e] = pos;
            auto& routes = by_expert[e];
            // The GPU sort uses atomic cursors, so the within-expert order can vary.
            if (prefill) std::shuffle(routes.begin(), routes.end(), rng);
            for (size_t i = 0; i < routes.size(); ++i) {
                if (i % tr == 0) {
                    if (nt == kMaxTiles) throw std::runtime_error("Too many expert tiles");
                    offsets[kTileExpert + nt] = e;
                    offsets[kTileBegin + nt] = pos;
                    offsets[kTileEnd + nt] = pos + std::min<size_t>(tr, routes.size() - i);
                    ++nt;
                }
                const int src = routes[i];
                sorted_tok[pos] = src / TOP_K;
                sorted_slot[pos] = src % TOP_K;
                sorted_w[pos] = weights[src];
                ++pos;
            }
        }
        offsets[NUM_EXPERTS] = pos;
        offsets[NUM_EXPERTS + 1] = nt;
    }

    int b;
    bool prefill;
    bool repeated_slot;
    std::vector<int32_t> ids;
    std::vector<float> weights;
    std::vector<int32_t> offsets;
    std::vector<int32_t> sorted_tok;
    std::vector<int32_t> sorted_slot;
    std::vector<float> sorted_w;
    std::vector<__half> input;
    std::vector<float> reference;
    std::array<std::vector<int>, NUM_EXPERTS> by_expert;
};

float reference_activation(float gate, float up) {
    const float g_cap = std::tanh(gate / GATE_SOFTCAP) * GATE_SOFTCAP;
    const float u_cap = std::tanh(up / UP_SOFTCAP) * UP_SOFTCAP;
    return (g_cap * (1.0f / (1.0f + std::exp(-gate)))) * u_cap;
}

// Four rows by two input vectors. Use F32 SIMD only for the dot product.
__attribute__((target("avx2,fma")))
void cpu_dots(const float* w, const float* a, const float* b, int width, float out[2][4]) {
    __m256 acc[2][4] = {};
    for (int c = 0; c < width; c += 8) {
        const __m256 x0 = _mm256_loadu_ps(a + c);
        const __m256 x1 = _mm256_loadu_ps(b + c);
        for (int r = 0; r < 4; ++r) {
            const __m256 v = _mm256_loadu_ps(w + r * width + c);
            acc[0][r] = _mm256_fmadd_ps(v, x0, acc[0][r]);
            acc[1][r] = _mm256_fmadd_ps(v, x1, acc[1][r]);
        }
    }
    for (int n = 0; n < 2; ++n) {
        for (int r = 0; r < 4; ++r) {
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, acc[n][r]);
            out[n][r] = (lanes[0] + lanes[4]) + (lanes[1] + lanes[5]) +
                        (lanes[2] + lanes[6]) + (lanes[3] + lanes[7]);
        }
    }
}

template<class Block, bool Up>
void make_reference(const std::vector<uint8_t>& raw, const std::vector<float>& scales,
                    std::vector<TestCase>& cases) {
    constexpr int kK = Up ? HIDDEN : INTERMEDIATE;
    constexpr int kR = Up ? 2 * INTERMEDIATE : HIDDEN;
    constexpr int kWidth = quant_traits<Block>::values;
    constexpr int kRowBytes = kK / kWidth * sizeof(Block);

    struct Destination {
        float* out;
        float route;
    };
    std::vector<std::vector<float>> decode_slots(cases.size());
    if constexpr (!Up) {
        for (size_t i = 0; i < cases.size(); ++i) {
            if (!cases[i].prefill) decode_slots[i].resize(size_t(cases[i].b) * TOP_K * HIDDEN);
        }
    }

    std::atomic<int> next_expert{0};
    auto worker = [&] {
        for (int e; (e = next_expert.fetch_add(1, std::memory_order_relaxed)) < NUM_EXPERTS;) {
            std::vector<Destination> destinations;
            std::vector<float> inputs;
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                auto& tc = cases[ci];
                for (size_t j = 0; j < tc.by_expert[e].size(); ++j) {
                    const int pos = tc.by_expert[e][j];
                    const int idx = tc.offsets[e] + j;
                    if constexpr (Up) {
                        const int dst = tc.prefill ? idx : pos;
                        destinations.push_back({tc.reference.data() + size_t(dst) * INTERMEDIATE, 1.0f});
                    } else {
                        float* dst = tc.prefill ? tc.reference.data() : decode_slots[ci].data();
                        destinations.push_back({dst + size_t(pos) * HIDDEN, tc.weights[pos]});
                    }
                    const int src = Up ? pos / TOP_K : (tc.prefill ? idx : pos);
                    const __half* x = tc.input.data() + size_t(src) * kK;
                    const auto offset = inputs.size();
                    inputs.resize(offset + kK);
                    for (int c = 0; c < kK; ++c) inputs[offset + c] = __half2float(x[c]);
                }
            }
            if (destinations.empty()) continue;
            const float es = scales.empty() ? 1.0f : scales[e];
            for (int row = 0; row < kR; row += 4) {
                float w[4][kK];
                for (int r = 0; r < 4; ++r) {
                    // Read the original GGUF order independently of GPU staging.
                    const int disk_row = Up ? (row + r) / 2 + ((row + r) & 1) * INTERMEDIATE : row + r;
                    const uint8_t* p = raw.data() + (size_t(e) * kR + disk_row) * kRowBytes;
                    for (int c = 0; c < kK; c += kWidth) {
                        Block block;
                        std::memcpy(&block, p + (c / kWidth) * sizeof(Block), sizeof(block));
                        for (int j = 0; j < kWidth; ++j)
                            w[r][c + j] = dequant_value(block, j);
                    }
                }
                for (size_t n = 0; n < destinations.size(); n += 2) {
                    const size_t second = std::min(n + 1, destinations.size() - 1);
                    float dot[2][4];
                    cpu_dots(&w[0][0], inputs.data() + n * kK, inputs.data() + second * kK, kK, dot);
                    for (int m = 0; m < 2 && n + m < destinations.size(); ++m) {
                        const auto [result, route] = destinations[n + m];
                        if constexpr (Up) {
                            result[row / 2] = reference_activation(dot[m][0] * es, dot[m][1] * es);
                            result[row / 2 + 1] = reference_activation(dot[m][2] * es, dot[m][3] * es);
                        } else {
                            for (int r = 0; r < 4; ++r) result[row + r] = (dot[m][r] * es) * route;
                        }
                    }
                }
            }
        }
    };

    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    for (unsigned i = 1; i < threads; ++i) workers.emplace_back(worker);
    worker();
    for (auto& thread : workers) thread.join();

    if constexpr (!Up) {
        for (size_t i = 0; i < cases.size(); ++i) {
            auto& tc = cases[i];
            if (tc.prefill) continue;
            for (int token = 0; token < tc.b; ++token) {
                for (int k = 0; k < TOP_K; ++k) {
                    const float* src = decode_slots[i].data() + size_t(token * TOP_K + k) * HIDDEN;
                    for (int r = 0; r < HIDDEN; ++r) tc.reference[size_t(token) * HIDDEN + r] += src[r];
                }
            }
        }
    }
}

float as_float(float v) { return v; }
float as_float(__half v) { return __half2float(v); }

template<class T>
bool check_result(const TestCase& tc, const std::vector<T>& a, const std::vector<T>& b,
                  const std::vector<T>& graph) {
    float max_err = 0.0f;
    size_t worst = 0;
    size_t nonfinite = 0, different = 0, graph_different = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float observed = as_float(a[i]);
        const float expected = tc.reference[i];
        if (!std::isfinite(expected) || !std::isfinite(observed) ||
            !std::isfinite(as_float(b[i])) || !std::isfinite(as_float(graph[i]))) ++nonfinite;
        const float err = std::abs(observed - expected);
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
        different += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
        graph_different += std::memcmp(&a[i], &graph[i], sizeof(T)) != 0;
    }
    std::printf("  %-7s B=%4d  n=%8zu  max_abs_err=%.7g  NaN/Inf=%zu  bit_mismatches=%zu  graph_mismatches=%zu%s\n",
                tc.prefill ? "prefill" : "decode", tc.b, a.size(), max_err, nonfinite,
                different, graph_different, tc.repeated_slot ? "  repeated slot expert" : "");
    if (max_err > kTolerance)
        std::printf("    index=%zu GPU=%.9g CPU=%.9g\n", worst, as_float(a[worst]), tc.reference[worst]);
    return max_err <= kTolerance && nonfinite == 0 && different == 0 && graph_different == 0;
}

template<bool Up>
bool run_gpu(const TestCase& tc, uint8_t type, const uint8_t* bank, const float* scale,
             size_t expert_bytes, cudaStream_t stream) {
    using Output = std::conditional_t<Up, __half, float>;
    DeviceBuffer<__half> input(tc.input.size());
    DeviceBuffer<int32_t> ids(tc.ids.size());
    DeviceBuffer<float> weights(tc.weights.size());
    DeviceBuffer<int32_t> offsets(tc.offsets.size());
    DeviceBuffer<int32_t> sorted_tok(tc.sorted_tok.size());
    DeviceBuffer<int32_t> sorted_slot(tc.sorted_slot.size());
    DeviceBuffer<float> sorted_w(tc.sorted_w.size());
    DeviceBuffer<Output> output(tc.reference.size());
    input.upload(tc.input.data(), tc.input.size());
    ids.upload(tc.ids.data(), tc.ids.size());
    weights.upload(tc.weights.data(), tc.weights.size());
    offsets.upload(tc.offsets.data(), tc.offsets.size());
    sorted_tok.upload(tc.sorted_tok.data(), tc.sorted_tok.size());
    sorted_slot.upload(tc.sorted_slot.data(), tc.sorted_slot.size());
    sorted_w.upload(tc.sorted_w.data(), tc.sorted_w.size());

    auto launch = [&] {
        if constexpr (Up) {
            if (tc.prefill)
                launch_prefill_moe_gate_up_gguf_h(type, bank, scale, input.ptr, offsets.ptr,
                                                 sorted_tok.ptr, sorted_slot.ptr, output.ptr, tc.b, stream);
            else
                launch_moe_gate_up_gguf_h(type, bank, scale, input.ptr, ids.ptr, output.ptr, tc.b, stream);
        } else {
            if (tc.prefill)
                launch_prefill_moe_down_gguf_h(type, bank, scale, input.ptr, offsets.ptr, sorted_tok.ptr,
                                              sorted_slot.ptr, sorted_w.ptr, output.ptr, tc.b, stream);
            else
                launch_moe_down_gguf_h(type, bank, scale, input.ptr, ids.ptr, weights.ptr,
                                      output.ptr, tc.b, stream);
        }
        CUDA(cudaPeekAtLastError());
    };

    const size_t bytes = tc.reference.size() * sizeof(Output);
    std::vector<Output> first(tc.reference.size()), second(first.size()), replay(first.size());
    CUDA(cudaMemsetAsync(output.ptr, 0xff, bytes, stream));
    launch();
    CUDA(cudaMemcpyAsync(first.data(), output.ptr, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA(cudaMemsetAsync(output.ptr, 0x7f, bytes, stream));
    launch();
    CUDA(cudaMemcpyAsync(second.data(), output.ptr, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&exec, graph, 0));
    CUDA(cudaMemsetAsync(output.ptr, 0xff, bytes, stream));
    CUDA(cudaGraphLaunch(exec, stream));
    CUDA(cudaMemcpyAsync(replay.data(), output.ptr, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));

    const bool ok = check_result(tc, first, second, replay);
    const bool bench_b2 = type == uint8_t(Type::iq3_xxs) || type == uint8_t(Type::iq4_nl);
    if (tc.prefill ? tc.b == 512 :
        !tc.repeated_slot && (tc.b == 1 || (tc.b == 2 && bench_b2) || tc.b == 4 || tc.b == 8)) {
        const int repeats = tc.prefill ? 24 : 300;
        const int warmups = tc.prefill ? 12 : 150;
        for (int n = 0; n < warmups; ++n) CUDA(cudaGraphLaunch(exec, stream));
        cudaEvent_t start, end;
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&end));
        CUDA(cudaEventRecord(start, stream));
        for (int n = 0; n < repeats; ++n) CUDA(cudaGraphLaunch(exec, stream));
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, start, end));
        CUDA(cudaEventDestroy(start));
        CUDA(cudaEventDestroy(end));
        const float us = ms * 1000.0f / repeats;
        const size_t reads = tc.prefill ? tc.offsets[NUM_EXPERTS + 1] :
            std::count_if(tc.by_expert.begin(), tc.by_expert.end(), [](const auto& v) { return !v.empty(); });
        std::printf("    benchmark: %9.2f us  %7.2f effective GB/s  (%zu expert %s)\n",
                    us, double(reads * expert_bytes) / (1000.0 * us), reads,
                    tc.prefill ? "tiles" : "banks");
    }
    CUDA(cudaGraphExecDestroy(exec));
    CUDA(cudaGraphDestroy(graph));
    return ok;
}

template<class Block, bool Up>
bool run_tensor(GgufFile& file, cudaStream_t stream, int layer, bool scaled, const char* format) {
    constexpr int kK = Up ? HIDDEN : INTERMEDIATE;
    constexpr int kR = Up ? 2 * INTERMEDIATE : HIDDEN;
    constexpr size_t kRowBytes = kK / quant_traits<Block>::values * sizeof(Block);
    constexpr size_t kExpertBytes = kR * kRowBytes;
    constexpr Type type = quant_traits<Block>::type;
    const std::string stem = "blk." + std::to_string(layer) + (Up ? ".ffn_gate_up_exps" : ".ffn_down_exps");
    std::printf("%s.weight (%s, %s)\n", stem.c_str(), format, scaled ? "scaled" : "nullptr scale");

    std::vector<uint8_t> raw(NUM_EXPERTS * kExpertBytes);
    file.read_tensor(stem + ".weight", type, {kK, kR, NUM_EXPERTS}, raw.data(), raw.size());
    if (scaled != file.contains(stem + ".scale"))
        throw std::runtime_error("Unexpected scale tensor for " + stem);
    std::vector<float> scale(scaled ? NUM_EXPERTS : 0);
    if (scaled)
        file.read_tensor(stem + ".scale", Type::f32, {NUM_EXPERTS}, scale.data(), scale.size() * sizeof(float));

    // Force a two-byte-aligned bank base even for the formats with four-byte strides.
    DeviceBuffer<uint8_t> bank(raw.size(), 2);
    DeviceBuffer<float> device_scale(NUM_EXPERTS);
    if (scaled) device_scale.upload(scale.data(), scale.size());
    if constexpr (Up) {
        std::vector<uint8_t> interleaved(raw.size());
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            const uint8_t* src = raw.data() + e * kExpertBytes;
            uint8_t* dst = interleaved.data() + e * kExpertBytes;
            for (int r = 0; r < INTERMEDIATE; ++r) {
                std::memcpy(dst + 2 * r * kRowBytes, src + r * kRowBytes, kRowBytes);
                std::memcpy(dst + (2 * r + 1) * kRowBytes, src + (INTERMEDIATE + r) * kRowBytes, kRowBytes);
            }
        }
        bank.upload(interleaved.data(), interleaved.size());
    } else {
        bank.upload(raw.data(), raw.size());
    }

    std::vector<TestCase> cases;
    for (int b : {1, 2, 4, 8, 9}) cases.emplace_back(b, false, Up, 1000 + b);
    cases.emplace_back(1, false, Up, 3001, true);
    for (int b : {1, 37, 512, 700}) cases.emplace_back(b, true, Up, 2000 + b);
    make_reference<Block, Up>(raw, scale, cases);

    bool ok = true;
    for (const auto& tc : cases) {
        ok &= run_gpu<Up>(tc, uint8_t(type), bank.ptr, scaled ? device_scale.ptr : nullptr,
                          kExpertBytes, stream);
    }
    return ok;
}

bool run_model(const char* path, cudaStream_t stream, const std::string& filter) {
    GgufFile file(path);
    std::printf("\nModel: %s\n", path);
    bool ok = true;
    int selected = 0;
    auto run = [&]<class Block, bool Up>(int layer, bool scaled, const char* format) {
        const std::string name = "blk." + std::to_string(layer) +
                                 (Up ? ".ffn_gate_up_exps" : ".ffn_down_exps");
        if (!filter.empty() && filter != name) return;
        ++selected;
        ok &= run_tensor<Block, Up>(file, stream, layer, scaled, format);
    };
    const Type gate_type = file.type_of("blk.9.ffn_gate_up_exps.weight");
    if (gate_type == Type::iq3_xxs) {
        run.operator()<block_iq3_xxs, true>(9, true, "IQ3_XXS");
        run.operator()<block_iq3_xxs, true>(20, false, "IQ3_XXS");
        run.operator()<block_iq4_nl, false>(20, false, "IQ4_NL");
        run.operator()<block_nvfp4, false>(0, true, "NVFP4");
    } else if (gate_type == Type::iq2_xs) {
        run.operator()<block_iq1_s, true>(0, true, "IQ1_S");
        run.operator()<block_iq2_xxs, true>(5, true, "IQ2_XXS");
        run.operator()<block_iq2_xs, true>(9, true, "IQ2_XS");
        run.operator()<block_iq2_s, true>(16, true, "IQ2_S");
        run.operator()<block_iq2_s, true>(23, false, "IQ2_S");
        run.operator()<block_q2_0, false>(0, true, "Q2_0");
        run.operator()<block_q2_0, false>(20, false, "Q2_0");
        run.operator()<block_nvfp4, false>(5, true, "NVFP4");
    } else {
        throw std::runtime_error("Unsupported model: " + std::string(path));
    }
    if (selected == 0) throw std::runtime_error("Unknown tensor stem " + filter);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (argc > 3) throw std::runtime_error("Usage: test_gguf_moe [GGUF] [tensor stem]");
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
            throw std::runtime_error("CPU reference requires AVX2 and FMA");
        const std::string filter = argc == 3 ? argv[2] : "";
        cudaDeviceProp device;
        CUDA(cudaGetDeviceProperties(&device, 0));
        std::printf("GPU: %s (sm_%d%d)\n", device.name, device.major, device.minor);
        std::puts("Benchmarks use warm CUDA Graph replay and logical packed-weight bytes.");

        cudaStream_t stream;
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bool ok = run_model(argc >= 2 ? argv[1] : kModel, stream, filter);
        if (argc == 1) ok &= run_model(kModelIq3, stream, filter);
        CUDA(cudaStreamDestroy(stream));
        if (!ok) return EXIT_FAILURE;
        std::puts(filter.empty() ? "ALL GGUF MOE TESTS PASS" : "SELECTED GGUF MOE TESTS PASS");
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }
}
