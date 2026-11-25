#include "gguf_quants.cuh"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace gguf = squid::gguf;
using gguf::Type;

constexpr auto model_dir = "/mnt/d/models/unsloth/Inkling-2-Mini-GGUF";
constexpr auto target_name = "Inkling-2-Mini-UD-IQ2_XXS.gguf";
constexpr auto iq3_name = "Inkling-2-Mini-UD-IQ3_XXS.gguf";
constexpr auto draft_name = "dspark-Inkling-2-Mini-Q8_0.gguf";
constexpr float error_limit = 1e-5f;

struct Sample {
    Type type;
    bool draft;
    const char* tensor;
    int row;
    uint64_t offset;
    int count;
    uint64_t checksum;
};

// Absolute GGUF offsets from the v3 tensor directories, with FNV-1a of each packed row.
// ref_slices.f32 contains each corresponding float row in this order.
constexpr Sample samples[] = {
    {Type::f32, false, "loop_kv_res_query.weight", 0, 11978688ull, 2304, 0x51b3db1476b0b327ull},
    {Type::f32, false, "blk.11.loop_mlp_down.weight", 32, 4343535552ull, 4608, 0x08f13c19d72f59f5ull},
    {Type::f32, false, "blk.23.ffn_res_query.weight", 0, 8486227136ull, 2304, 0x99c41b9fc159d4f7ull},
    {Type::f32, true, "dspark_fuse.weight", 0, 11933280ull, 4, 0xb92d6b59d2dd0641ull},
    {Type::f32, true, "blk.2.ffn_norm.weight", 0, 379708576ull, 2304, 0x774e543c5982ad7dull},
    {Type::q8_0, false, "blk.0.attn_k.weight", 0, 510361536ull, 2304, 0x780010cb86b2d451ull},
    {Type::q8_0, false, "blk.9.attn_gate.weight", 1536, 3165703104ull, 2304, 0x49340c0c6ecbe193ull},
    {Type::q8_0, false, "blk.18.attn_v.weight", 1023, 6391695920ull, 2304, 0x21215735bcf70839ull},
    {Type::q8_0, true, "blk.0.attn_k.weight", 0, 123941536ull, 2304, 0x8239c945f1581cafull},
    {Type::q8_0, true, "blk.2.ffn_up.weight", 4608, 390998176ull, 2304, 0xba0b2e90e922996cull},
    {Type::q4_k, false, "blk.18.attn_q.weight", 0, 6384285376ull, 2304, 0x2289b8eed0f8c1bbull},
    {Type::q4_k, false, "blk.21.attn_rel.weight", 192, 7529638080ull, 2304, 0xc8d23684a174f8b3ull},
    {Type::q4_k, false, "blk.23.attn_rel.weight", 383, 8169802672ull, 2304, 0x4a5f163c604d38adull},
    {Type::q5_k, false, "blk.3.attn_q.weight", 0, 1289957056ull, 2304, 0x0b2e28aae0931d41ull},
    {Type::q5_k, false, "blk.20.attn_v.weight", 512, 7201171136ull, 2304, 0x1e0889e57e2a0f01ull},
    {Type::q5_k, false, "blk.23.attn_v.weight", 1023, 8171629712ull, 2304, 0xa434f8ff4d7f2432ull},
    {Type::q6_k, false, "blk.0.attn_gate.weight", 0, 504555456ull, 2304, 0xbcc65df31fe106b9ull},
    {Type::q6_k, false, "blk.14.attn_gate.weight", 1536, 4969029312ull, 2304, 0x8cd33c8ff55939f4ull},
    {Type::q6_k, false, "blk.19.attn_v.weight", 1023, 6776603486ull, 2304, 0xfbb0c3b17691ba15ull},
    {Type::iq2_xxs, false, "blk.5.ffn_gate_up_exps.weight", 0, 2003254464ull, 2304, 0x61e0b835c5438ff8ull},
    {Type::iq2_xxs, false, "blk.12.ffn_gate_up_exps.weight", 147456, 4559064000ull, 2304, 0x70c2d292c1822c7eull},
    {Type::iq2_xxs, false, "blk.18.ffn_gate_up_exps.weight", 294911, 6759176046ull, 2304, 0xd6518693e5eb8c59ull},
    {Type::iq2_xs, false, "blk.9.ffn_gate_up_exps.weight", 0, 3383019456ull, 2304, 0x41877b0da263904bull},
    {Type::iq2_xs, false, "blk.15.ffn_gate_up_exps.weight", 147456, 5492624064ull, 2304, 0xdb0d5638127c9611ull},
    {Type::iq2_xs, false, "blk.22.ffn_gate_up_exps.weight", 294911, 8154568230ull, 2304, 0x38419b4fa48d2ce1ull},
    {Type::iq3_xxs, false, "blk.20.attn_q.weight", 0, 7196843200ull, 2304, 0x4ab43a4fe68011f2ull},
    {Type::iq3_xxs, false, "blk.21.attn_q.weight", 1536, 7528022208ull, 2304, 0xb77629cb00122e97ull},
    {Type::iq3_xxs, false, "blk.22.attn_q.weight", 3071, 7859089742ull, 2304, 0x6008e470ccc226ffull},
    {Type::iq1_s, false, "blk.0.ffn_gate_up_exps.weight", 0, 626642880ull, 2304, 0x2a440e9c38f9c56dull},
    {Type::iq1_s, false, "blk.3.ffn_gate_up_exps.weight", 147456, 1461359040ull, 2304, 0x7fe5a1f3691e27b3ull},
    {Type::iq1_s, false, "blk.6.ffn_gate_up_exps.weight", 294911, 2436722430ull, 2304, 0x87148afbcd60a46dull},
    {Type::iq2_s, false, "blk.16.ffn_gate_up_exps.weight", 0, 5720907456ull, 2304, 0x56bbb72248241f82ull},
    {Type::iq2_s, false, "blk.20.ffn_gate_up_exps.weight", 147456, 7407552704ull, 2304, 0x50868ab7ec3d68adull},
    {Type::iq2_s, false, "blk.23.ffn_gate_up_exps.weight", 294911, 8486180318ull, 2304, 0x3d3f04f9f1808dd3ull},
    {Type::iq4_xs, false, "output.weight", 0, 12421056ull, 2304, 0xa8e8c0c3e48cd9c6ull},
    {Type::iq4_xs, false, "token_embd.weight", 100512, 381519552ull, 2304, 0xbb47c100ccab0112ull},
    {Type::iq4_xs, false, "token_embd.weight", 201023, 504545016ull, 2304, 0x806d52734a1677e2ull},
    {Type::bf16, false, "blk.0.attn_sink_proj.weight", 0, 527276480ull, 2304, 0x2f201227caf9c190ull},
    {Type::bf16, false, "blk.12.attn_sink_proj.weight", 12, 4372164032ull, 2304, 0xc343809b13045679ull},
    {Type::bf16, false, "blk.23.attn_v_conv_proj.weight", 31, 8171790528ull, 2304, 0xb7c8edeab104e75full},
    {Type::bf16, true, "dspark_hidden_out.weight", 0, 11942560ull, 256, 0xf08a502f8a3a02b2ull},
    {Type::bf16, true, "markov_w1.weight", 100512, 72460960ull, 256, 0x109dc936f75b3cc6ull},
    {Type::nvfp4, false, "blk.5.ffn_down_exps.weight", 0, 1810971328ull, 1152, 0x56f2b16729d0f6c0ull},
    {Type::nvfp4, false, "blk.11.ffn_down_exps.weight", 147456, 4045028800ull, 1152, 0x2c298f8619ceab32ull},
    {Type::nvfp4, false, "blk.19.ffn_down_exps.weight", 294911, 6967725112ull, 1152, 0x82d143e2938496f1ull},
    {Type::q2_0, false, "blk.0.ffn_down_exps.weight", 0, 529911232ull, 1152, 0xfa266160e6fb8831ull},
    {Type::q2_0, false, "blk.12.ffn_down_exps.weight", 147456, 4422519232ull, 1152, 0xcb468de33c400384ull},
    {Type::q2_0, false, "blk.23.ffn_down_exps.weight", 294911, 8267356028ull, 1152, 0xd4848e8722935824ull},
};

std::ifstream open(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    return file;
}

void read_exact(std::ifstream& file, void* data, std::size_t bytes) {
    if (!file.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Truncated GGUF row or reference file");
    }
}

uint64_t fnv1a(std::span<const std::byte> bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (const auto byte : bytes) {
        hash = (hash ^ std::to_integer<uint8_t>(byte)) * 1099511628211ull;
    }
    return hash;
}

struct Results {
    unsigned rows = 0;
    float cpu_ref = 0;
    float gpu_ref = 0;
    float gpu_cpu = 0;
};

void compare(std::span<const float> actual, std::span<const float> expected, float& maximum) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("NaN or Inf at column " + std::to_string(i));
        }
        const float error = std::abs(actual[i] - expected[i]);
        if (!(error < error_limit)) {
            throw std::runtime_error("Absolute error " + std::to_string(error) +
                                     " at column " + std::to_string(i));
        }
        maximum = std::max(maximum, error);
    }
}

#if defined(__CUDACC__)
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <class T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(&data, count * sizeof(T)));
    }

    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <class Block>
constexpr int values_per_thread = std::is_same_v<Block, gguf::block_iq4_nl>
    ? gguf::quant_traits<Block>::values : 1;

template <class Block>
__global__ void decode_kernel(const Block* blocks, float* out, int count) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_thread<Block>;
    constexpr int width = gguf::quant_traits<Block>::values;
    if (index < count) {
        if constexpr (values_per_thread<Block> > 1)
            gguf::dequant_row(blocks + index / width, out + index, width);
        else
            out[index] = gguf::dequant_value(blocks[index / width], index % width);
    }
}

template <class Block>
void gpu_compare(std::span<const Block> packed, std::span<const float> cpu,
                 std::span<const float> reference, Results& results) {
    const int count = static_cast<int>(cpu.size());
    DeviceBuffer<Block> input(packed.size());
    DeviceBuffer<float> output(count);
    std::vector<float> decoded(count);
    cuda_check(cudaMemcpy(input.data, packed.data(), packed.size_bytes(), cudaMemcpyHostToDevice));
    constexpr int threads = 256;
    const int work = count / values_per_thread<Block>;
    decode_kernel<<<(work + threads - 1) / threads, threads>>>(input.data, output.data, count);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpy(decoded.data(), output.data, count * sizeof(float), cudaMemcpyDeviceToHost));
    compare(decoded, reference, results.gpu_ref);
    compare(decoded, cpu, results.gpu_cpu);
}
#endif

template <class Block>
void check_sample(const Sample& sample, std::ifstream& model, std::ifstream& ref, Results& results) {
    constexpr int width = gguf::quant_traits<Block>::values;
    if (sample.count <= 0 || sample.count % width) {
        throw std::runtime_error("Invalid row width");
    }
    std::vector<Block> packed(sample.count / width);
    std::vector<float> cpu(sample.count);
    std::vector<float> reference(sample.count);
    auto bytes = std::as_writable_bytes(std::span(packed));
    model.seekg(sample.offset);
    read_exact(model, bytes.data(), bytes.size());
    if (fnv1a(bytes) != sample.checksum) {
        throw std::runtime_error("GGUF row checksum mismatch");
    }
    read_exact(ref, reference.data(), reference.size() * sizeof(float));
    gguf::dequant_row(packed.data(), cpu.data(), cpu.size());
    compare(cpu, reference, results.cpu_ref);
#if defined(__CUDACC__)
    gpu_compare<Block>(packed, cpu, reference, results);
#endif
    ++results.rows;
}

Results check_iq4_nl(const std::filesystem::path& dir) {
    using Block = gguf::block_iq4_nl;
    constexpr int width = gguf::quant_traits<Block>::values;
    constexpr int count = 1152;
    // First, middle and last expert rows of blk.20.ffn_down_exps.weight.
    constexpr std::array offsets = {9793813696ull, 9889365184ull, 9984916024ull};
    constexpr std::array checksums = {
        0xe1cc6a28593a4fb5ull, 0xcfc44ae5cc8b2853ull, 0x5f027500ec09df6dull,
    };
    // ggml dequantize_row_iq4_nl codebook and split-nibble order.
    constexpr int codebook[] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    auto model = open(dir / iq3_name);
    Results result;
    for (std::size_t row = 0; row < offsets.size(); ++row) {
        std::array<Block, count / width> packed;
        std::array<float, count> cpu, reference;
        const auto bytes = std::as_writable_bytes(std::span(packed));
        model.seekg(offsets[row]);
        read_exact(model, bytes.data(), bytes.size());
        if (fnv1a(bytes) != checksums[row])
            throw std::runtime_error("IQ4_NL packed row checksum mismatch");

        for (std::size_t b = 0; b < packed.size(); ++b) {
            const auto& block = packed[b];
            const int exp = (block.d >> 10) & 31;
            if (exp == 31) throw std::runtime_error("Nonfinite IQ4_NL block scale");
            const int mantissa = block.d & 1023;
            const float magnitude = exp == 0 ? std::ldexp(float(mantissa), -24)
                                             : std::ldexp(float(1024 + mantissa), exp - 25);
            const float d = (block.d & 0x8000) ? -magnitude : magnitude;
            for (int j = 0; j < 16; ++j) {
                reference[b * width + j] = d * codebook[block.qs[j] & 15];
                reference[b * width + j + 16] = d * codebook[block.qs[j] >> 4];
            }
        }
        gguf::dequant_row(packed.data(), cpu.data(), count);
        compare(cpu, reference, result.cpu_ref);
#if defined(__CUDACC__)
        gpu_compare<Block>(packed, cpu, reference, result);
#endif
        ++result.rows;
    }
    if (result.cpu_ref != 0 || result.gpu_ref != 0 || result.gpu_cpu != 0)
        throw std::runtime_error("IQ4_NL dequantization must match exactly");
    return result;
}

struct TestType {
    Type type;
    const char* name;
    void (*check)(const Sample&, std::ifstream&, std::ifstream&, Results&);
};

template <class Block>
constexpr TestType test_type(const char* name) {
    return {gguf::quant_traits<Block>::type, name, check_sample<Block>};
}

constexpr std::array types = {
    test_type<gguf::block_q2_0>("Q2_0"),
    test_type<gguf::block_nvfp4>("NVFP4"),
    test_type<gguf::block_iq2_xxs>("IQ2_XXS"),
    test_type<gguf::block_iq2_xs>("IQ2_XS"),
    test_type<gguf::block_iq2_s>("IQ2_S"),
    test_type<gguf::block_iq1_s>("IQ1_S"),
    test_type<gguf::block_iq4_xs>("IQ4_XS"),
    test_type<gguf::block_iq3_xxs>("IQ3_XXS"),
    test_type<gguf::block_q8_0>("Q8_0"),
    test_type<gguf::block_q6_k>("Q6_K"),
    test_type<gguf::block_q5_k>("Q5_K"),
    test_type<gguf::block_q4_k>("Q4_K"),
    test_type<gguf::block_bf16>("BF16"),
    test_type<gguf::block_f32>("F32"),
};

void print_result(const char* name, const Results& result) {
    std::printf("%-8s rows=%u CPU/reference max_abs_err=%.9g", name, result.rows, result.cpu_ref);
#if defined(__CUDACC__)
    std::printf(" GPU/reference max_abs_err=%.9g GPU/CPU max_abs_err=%.9g",
                result.gpu_ref, result.gpu_cpu);
#endif
    std::puts(" NaN/Inf=0");
}

void run(const std::filesystem::path& dir, const std::filesystem::path& reference_path) {
    auto target = open(dir / target_name);
    auto draft = open(dir / draft_name);
    auto reference = open(reference_path);
    std::array<Results, types.size()> results{};

    for (const auto& sample : samples) {
        const auto type = std::find_if(types.begin(), types.end(),
                                       [&](const TestType& t) { return t.type == sample.type; });
        if (type == types.end()) {
            throw std::runtime_error("Unsupported fixture type");
        }
        try {
            type->check(sample, sample.draft ? draft : target, reference,
                        results[type - types.begin()]);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(sample.draft ? draft_name : target_name) +
                " " + sample.tensor + " row " + std::to_string(sample.row) + ": " + error.what());
        }
    }
    if (reference.peek() != std::char_traits<char>::eof() || reference.bad()) {
        throw std::runtime_error("Unexpected trailing reference data");
    }

    for (std::size_t i = 0; i < types.size(); ++i) {
        const auto& result = results[i];
        if (!result.rows) {
            throw std::runtime_error(std::string("Missing samples for ") + types[i].name);
        }
        print_result(types[i].name, result);
    }
    print_result("IQ4_NL", check_iq4_nl(dir));
    std::printf("ALL %zu QUANT TYPES PASS\n", types.size() + 1);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "Usage: %s [GGUF_DIRECTORY] [REF_SLICES_F32]\n", argv[0]);
        return 1;
    }
    try {
        const auto fixture = std::filesystem::path(__FILE__).parent_path() / "ref_slices.f32";
        run(argc >= 2 ? argv[1] : model_dir, argc >= 3 ? std::filesystem::path(argv[2]) : fixture);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
