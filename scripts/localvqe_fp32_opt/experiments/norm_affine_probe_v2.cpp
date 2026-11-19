// Isolated A/B using the actual GGML CPU reference and frozen Layer inputs.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace {
constexpr float eps = 1e-5f;
constexpr size_t frames = 750;
constexpr int rounds = 9, iterations = 1024;

void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Preserve GGML AArch64 vec_sum / vec_cvar reduction order, FP32 rounding
// boundaries, and normalize before applying the per-channel affine.
void norm_affine(ggml_tensor* dst, const ggml_tensor* x,
                 const ggml_tensor* gamma, const ggml_tensor* beta,
                 int ith, int nth, void*) {
    GGML_ASSERT(ith == 0 && nth == 1 && x->ne[1] == 1 && x->ne[3] == 1);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32 &&
                beta->type == GGML_TYPE_F32 && ggml_is_contiguous(x) &&
                ggml_is_contiguous(dst) && ggml_is_contiguous(gamma) &&
                ggml_is_contiguous(beta));
    const int f = int(x->ne[0]), c = int(x->ne[2]), n = f * c;
    GGML_ASSERT(f > 0 && c > 0 && ggml_nelements(gamma) == c &&
                ggml_nelements(beta) == c);
    const auto* in = static_cast<const float*>(x->data);
    const auto* g = static_cast<const float*>(gamma->data);
    const auto* b = static_cast<const float*>(beta->data);
    auto* out = static_cast<float*>(dst->data);

    double sum = 0.0;
    for (int k = 0; k < n; ++k) sum += double(in[k]);
    const float mean = float(sum) / float(n);
    const auto vm = vdupq_n_f32(mean);
    double ss = 0.0;
    int k = 0;
    for (; k + 3 < n; k += 4) {
        const auto z = vsubq_f32(vld1q_f32(in + k), vm);
        ss += double(vaddvq_f32(vmulq_f32(z, z)));
    }
    for (; k < n; ++k) {
        const float z = in[k] - mean;
        ss += double(z * z);
    }
    const float scale = 1.0f / std::sqrt(float(ss / double(n)) + eps);
    const auto vs = vdupq_n_f32(scale);
    for (int ch = 0; ch < c; ++ch, in += f, out += f) {
        const auto vg = vdupq_n_f32(g[ch]), vb = vdupq_n_f32(b[ch]);
        int i = 0;
        for (; i + 3 < f; i += 4) {
            const auto z = vsubq_f32(vld1q_f32(in + i), vm);
            const auto normalized = vmulq_f32(z, vs);
            vst1q_f32(out + i, vaddq_f32(vmulq_f32(normalized, vg), vb));
        }
        for (; i < f; ++i) {
            const float normalized = (in[i] - mean) * scale;
            const float scaled = normalized * g[ch];
            out[i] = scaled + b[ch];
        }
    }
}

struct Branch {
    explicit Branch(int f, int c, const float* g, const float* beta,
                    ggml_threadpool* pool, bool fused) {
        ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
        need(ctx != nullptr, "graph arena");
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, f, 1, c);
        gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c);
        std::memcpy(gamma->data, g, 4 * c);
        std::memcpy(bias->data, beta, 4 * c);
        if (fused) {
            output = ggml_map_custom3(ctx, input, gamma, bias, norm_affine, 1, nullptr);
        } else {
            auto* perm = ggml_cont(ctx, ggml_permute(ctx, input, 0, 2, 1, 3));
            auto* flat = ggml_reshape_2d(ctx, perm, f * c, 1);
            auto* norm = ggml_reshape_3d(ctx, ggml_norm(ctx, flat, eps), f, c, 1);
            auto* g3 = ggml_reshape_3d(ctx, gamma, 1, c, 1);
            auto* b3 = ggml_reshape_3d(ctx, bias, 1, c, 1);
            auto* affine = ggml_add(ctx, ggml_mul(ctx, norm, g3), b3);
            output = ggml_cont(ctx, ggml_permute(ctx, affine, 0, 2, 1, 3));
        }
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            switch (ggml_graph_node(graph, i)->op) {
                case GGML_OP_NONE: case GGML_OP_PERMUTE: case GGML_OP_RESHAPE: break;
                default: ++kernels; break;
            }
        }
        plan = ggml_graph_plan(graph, 1, pool);
        work.resize(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
    }
    ~Branch() { ggml_free(ctx); }
    Branch(const Branch&) = delete;
    Branch& operator=(const Branch&) = delete;
    void run() { need(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
                      "ggml compute failed"); }
    void set(const std::vector<float>& source) {
        need(source.size() == size_t(ggml_nelements(input)), "input extent");
        std::memcpy(input->data, source.data(), source.size() * sizeof(float));
    }
    const float* data() const { return static_cast<const float*>(output->data); }
    ggml_context* ctx = nullptr;
    ggml_tensor *input = nullptr, *gamma = nullptr, *bias = nullptr, *output = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    int kernels = 0;
};

struct Site { const char* tap; const char* norm; int f, c; };
struct Slice { size_t count, offset; bool fp32; };

uint64_t cpu_ns() {
    timespec t{};
    need(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0, "clock_gettime");
    return uint64_t(t.tv_sec) * 1000000000 + uint64_t(t.tv_nsec);
}

double median(std::array<double, rounds> x) {
    std::sort(x.begin(), x.end());
    return x[rounds / 2];
}

const float* weights(ggml_context* ctx, const std::string& key, int n) {
    auto* t = ggml_get_tensor(ctx, key.c_str());
    need(t && t->type == GGML_TYPE_F32 && t->data && ggml_nelements(t) == n,
         "missing or incompatible weights");
    return static_cast<const float*>(t->data);
}
} // namespace

int main(int argc, char** argv) {
    try {
        need(argc == 4, "usage: norm-probe MODEL.gguf META.csv STATES.bin");
        std::ifstream meta(argv[2]), tape(argv[3], std::ios::binary);
        need(meta.good() && tape.good(), "cannot open frozen taps");
        const std::array<Site, 4> sites{{
            {"tap_mic_e1", "mic_enc2.norm", 128, 16},
            {"tap_mic_e2", "dec2.skip_norm", 64, 20},
            {"tap_mic_e3", "mic_enc4.norm", 32, 20},
            {"tap_mic_e5", "dec5.skip_norm", 8, 20},
        }};
        std::array<Slice, sites.size()> slices{};
        std::string line, name, type, count, offset;
        need(bool(std::getline(meta, line)) &&
             line == "name,dtype,count,byte_offset", "invalid tap header");
        size_t rowbytes = 0;
        while (std::getline(meta, line)) {
            std::istringstream row(line);
            need(bool(std::getline(row, name, ',')) && bool(std::getline(row, type, ',')) &&
                 bool(std::getline(row, count, ',')) && bool(std::getline(row, offset)),
                 "invalid tap row");
            const Slice slice{std::stoull(count), std::stoull(offset), type == "f32"};
            need(slice.fp32 || type == "i64", "invalid tap dtype");
            const size_t bytes = slice.count * (slice.fp32 ? sizeof(float) : sizeof(int64_t));
            need(slice.offset == rowbytes, "invalid tap layout");
            rowbytes += bytes;
            for (size_t i = 0; i < sites.size(); ++i) {
                if (name == sites[i].tap) slices[i] = slice;
            }
        }
        tape.seekg(0, std::ios::end);
        need(rowbytes > 0 && tape.tellg() == std::streamoff(rowbytes * frames),
             "tap tape extent mismatch");
        ggml_context* weight_ctx = nullptr;
        auto* model = gguf_init_from_file(argv[1], {false, &weight_ctx});
        need(model && weight_ctx, "could not load frozen gguf");
        auto defaults = ggml_threadpool_params_default(1);
        auto* pool = ggml_threadpool_new(&defaults);
        need(pool != nullptr, "could not create a shared one-thread pool");
        size_t elements = 0;
        for (size_t j = 0; j < sites.size(); ++j) {
            const auto& [tap, name, f, c] = sites[j];
            const auto slice = slices[j];
            need(slice.fp32 && slice.count == size_t(f * c), "tap shape does not match");
            const float* g = weights(weight_ctx, std::string(name) + ".weight", c);
            const float* b = weights(weight_ctx, std::string(name) + ".bias", c);
            Branch ref(f, c, g, b, pool, false), candidate(f, c, g, b, pool, true);
            need(ref.kernels == 5 && candidate.kernels == 1, "unexpected executed nodes");
            std::vector<float> input(slice.count);
            const auto size = slice.count * sizeof(float);
            for (size_t i = 0; i < frames; ++i) {
                tape.seekg(std::streamoff(rowbytes * i + slice.offset));
                need(bool(tape.read(reinterpret_cast<char*>(input.data()), size)),
                     "tap read failure");
                ref.set(input);
                candidate.set(input);
                ref.run();
                candidate.run();
                if (std::memcmp(ref.data(), candidate.data(), size) != 0) {
                    std::fprintf(stderr, "bit mismatch %s frame=%zu\n", name, i);
                    return 2;
                }
                for (size_t k = 0; k < slice.count; ++k)
                    need(std::isfinite(ref.data()[k]), "non-finite reference output");
            }
            elements += frames * slice.count;
            auto& first_bias = *static_cast<float*>(candidate.bias->data);
            const float saved_bias = first_bias;
            first_bias += 0.125f;
            candidate.run();
            need(std::memcmp(ref.data(), candidate.data(), size) != 0,
                 "bias negative control not rejected");
            first_bias = saved_bias;
            candidate.run();
            need(std::memcmp(ref.data(), candidate.data(), size) == 0,
                 "bias restoration failed");

            std::array<double, rounds> x{}, y{};
            auto time = [](Branch& branch) {
                const auto start = cpu_ns();
                for (int i = 0; i < iterations; ++i) branch.run();
                return double(cpu_ns() - start) / iterations;
            };
            for (int i = 0; i < rounds; ++i) {
                if (i % 2 == 0) { x[i] = time(ref); y[i] = time(candidate); }
                else { y[i] = time(candidate); x[i] = time(ref); }
            }
            std::printf("%s,%s,%d,%d,%zu,%d,%d,%.1f,%.1f,wrong_bias_rejected\n",
                        tap, name, f, c, frames, ref.kernels, candidate.kernels,
                        median(x), median(y));
        }
        ggml_threadpool_free(pool);
        gguf_free(model);
        ggml_free(weight_ctx);
        std::printf("PASS real_tap_tensors=%zu frames_per_tensor=%zu bit_equal_scalars=%zu "
                    "alternating_rounds=%d iterations=%d\n",
                    sites.size(), frames, elements, rounds, iterations);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
