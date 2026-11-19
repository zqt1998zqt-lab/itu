#include "norm_affine_v2.hpp"

#include "ggml-cpu.h"
#include "gguf.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kEps = 1e-5f;
constexpr int kFrames = 64;

void need(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Site {
    const char * tap;
    const char * norm;
    int f;
    int c;
};

struct Branch {
    Branch(int f, int c, const float * gamma_data, const float * beta_data,
           ggml_threadpool * pool, bool fused)
        : ctx(ggml_init({8 * 1024 * 1024, nullptr, false})) {
        need(ctx != nullptr, "graph arena");
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, f, 1, c);
        gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c);
        beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c);
        need(input != nullptr && gamma != nullptr && beta != nullptr,
             "tensor allocation");
        std::memcpy(gamma->data, gamma_data, sizeof(float) * size_t(c));
        std::memcpy(beta->data, beta_data, sizeof(float) * size_t(c));

        if (fused) {
            output = itu_fusion_r1::norm_affine_or_null(
                ctx, input, gamma, beta, kEps);
            need(output != nullptr, "compatible v2 candidate was rejected");
        } else {
            auto * permuted =
                ggml_cont(ctx, ggml_permute(ctx, input, 0, 2, 1, 3));
            auto * flat = ggml_reshape_2d(ctx, permuted, f * c, 1);
            auto * norm = ggml_reshape_3d(
                ctx, ggml_norm(ctx, flat, kEps), f, c, 1);
            auto * gamma3 = ggml_reshape_3d(ctx, gamma, 1, c, 1);
            auto * beta3 = ggml_reshape_3d(ctx, beta, 1, c, 1);
            auto * affine =
                ggml_add(ctx, ggml_mul(ctx, norm, gamma3), beta3);
            output =
                ggml_cont(ctx, ggml_permute(ctx, affine, 0, 2, 1, 3));
        }

        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            switch (ggml_graph_node(graph, i)->op) {
                case GGML_OP_NONE:
                case GGML_OP_PERMUTE:
                case GGML_OP_RESHAPE:
                    break;
                default:
                    ++kernels;
                    break;
            }
        }
        plan = ggml_graph_plan(graph, 1, pool);
        work.resize(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
    }

    ~Branch() {
        ggml_free(ctx);
    }

    Branch(const Branch &) = delete;
    Branch & operator=(const Branch &) = delete;

    void set(const std::vector<float> & values) {
        need(values.size() == size_t(ggml_nelements(input)),
             "input extent");
        std::memcpy(input->data, values.data(),
                    values.size() * sizeof(float));
    }

    void run() {
        need(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
             "ggml compute failed");
    }

    const float * data() const {
        return static_cast<const float *>(output->data);
    }

    ggml_context * ctx = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * beta = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    int kernels = 0;
};

const float * real_weights(ggml_context * ctx, const std::string & key,
                           int count) {
    auto * tensor = ggml_get_tensor(ctx, key.c_str());
    need(tensor != nullptr && tensor->type == GGML_TYPE_F32 &&
             tensor->data != nullptr &&
             ggml_nelements(tensor) == count,
         "missing or incompatible real weight");
    return static_cast<const float *>(tensor->data);
}

void rejection_gates() {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    need(ctx != nullptr, "rejection graph arena");
    constexpr int f = 7;
    constexpr int c = 3;
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, f, 1, c);
    auto * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c);
    auto * beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c);
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, x, gamma, beta, kEps) != nullptr,
         "positive gate rejected");

    need(itu_fusion_r1::norm_affine_or_null(
             ctx, x, gamma, beta, 0.0f) == nullptr,
         "wrong epsilon accepted");
    const float next_eps = std::nextafter(
        kEps, std::numeric_limits<float>::infinity());
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, x, gamma, beta, next_eps) == nullptr,
         "next epsilon accepted");

    auto * bad_rank = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, f, 2, c);
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, bad_rank, gamma, beta, kEps) == nullptr,
         "non-single second dimension accepted");

    auto * bad_channels = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, f, 1, c + 1);
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, bad_channels, gamma, beta, kEps) == nullptr,
         "channel mismatch accepted");

    auto * bad_gamma = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, c + 1);
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, x, bad_gamma, beta, kEps) == nullptr,
         "gamma shape mismatch accepted");

    auto * f16_x = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F16, f, 1, c);
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, f16_x, gamma, beta, kEps) == nullptr,
         "non-F32 input accepted");

    auto * f16_beta = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F16, c);
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, x, gamma, f16_beta, kEps) == nullptr,
         "non-F32 beta accepted");

    auto * stride_base = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, f * (c + 1));
    auto * strided = ggml_view_3d(
        ctx, stride_base, f, 1, c,
        sizeof(float) * size_t(f),
        sizeof(float) * size_t(f + 1), 0);
    need(!ggml_is_contiguous(strided), "stride fixture is contiguous");
    need(itu_fusion_r1::norm_affine_or_null(
             ctx, strided, gamma, beta, kEps) == nullptr,
         "strided input accepted");

    ggml_tensor oversized_dim{};
    oversized_dim.type = GGML_TYPE_F32;
    oversized_dim.ne[0] =
        static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
    oversized_dim.ne[1] = 1;
    oversized_dim.ne[2] = 1;
    oversized_dim.ne[3] = 1;
    need(!itu_fusion_r1::detail::is_norm_affine_compatible_v2(
             &oversized_dim, gamma, beta, kEps),
         "single dimension above INT_MAX accepted");

    ggml_tensor oversized_product{};
    oversized_product.type = GGML_TYPE_F32;
    oversized_product.ne[0] = std::numeric_limits<int>::max();
    oversized_product.ne[1] = 1;
    oversized_product.ne[2] = 2;
    oversized_product.ne[3] = 1;
    need(!itu_fusion_r1::detail::is_norm_affine_compatible_v2(
             &oversized_product, gamma, beta, kEps),
         "F*C above INT_MAX accepted");

    ggml_free(ctx);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        need(argc == 2, "usage: test-norm-v2 FROZEN_MODEL.gguf");
#if !defined(__aarch64__) || !defined(__ARM_NEON)
        std::fprintf(stderr, "SKIP: AArch64 NEON is required\n");
        return 77;
#else
        rejection_gates();

        ggml_context * weight_ctx = nullptr;
        gguf_context * model =
            gguf_init_from_file(argv[1], {false, &weight_ctx});
        need(model != nullptr && weight_ctx != nullptr,
             "could not load frozen GGUF");

        const std::array<Site, 4> sites{{
            {"tap_mic_e1", "mic_enc2.norm", 128, 16},
            {"tap_mic_e2", "dec2.skip_norm", 64, 20},
            {"tap_mic_e3", "mic_enc4.norm", 32, 20},
            {"tap_mic_e5", "dec5.skip_norm", 8, 20},
        }};

        auto pool_params = ggml_threadpool_params_default(1);
        ggml_threadpool * pool = ggml_threadpool_new(&pool_params);
        need(pool != nullptr, "single-thread pool");

        size_t compared_scalars = 0;
        size_t negative_controls = 0;
        for (const Site & site : sites) {
            const float * gamma_data = real_weights(
                weight_ctx, std::string(site.norm) + ".weight", site.c);
            const float * beta_data = real_weights(
                weight_ctx, std::string(site.norm) + ".bias", site.c);

            Branch reference(
                site.f, site.c, gamma_data, beta_data, pool, false);
            Branch candidate(
                site.f, site.c, gamma_data, beta_data, pool, true);
            need(reference.kernels == 5,
                 "reference graph no longer has five kernels");
            need(candidate.kernels == 1,
                 "candidate graph no longer has one kernel");
            need(candidate.output->op == GGML_OP_MAP_CUSTOM3,
                 "candidate is not GGML custom3");
            need(std::strcmp(candidate.output->name, "itu_fusion_norm") == 0,
                 "candidate name mismatch");
            need(candidate.output->type == GGML_TYPE_F32 &&
                     ggml_is_contiguous(candidate.output),
                 "candidate output is not contiguous F32");

            std::vector<float> input(
                size_t(site.f) * size_t(site.c));
            uint32_t state = 0x6d2b79f5u ^
                             uint32_t(site.f * 131 + site.c);
            for (int frame = 0; frame < kFrames; ++frame) {
                for (size_t i = 0; i < input.size(); ++i) {
                    state = state * 1664525u + 1013904223u;
                    const float random =
                        float((state >> 8) & 0x00ffffffu) *
                            (4.0f / 16777216.0f) -
                        2.0f;
                    const float shaped =
                        0.5f * std::sin(0.013f * float(i) +
                                        0.017f * float(frame)) +
                        0.25f * std::cos(0.007f * float(i + site.c));
                    input[i] = random + shaped;
                }
                reference.set(input);
                candidate.set(input);
                reference.run();
                candidate.run();
                const size_t bytes = input.size() * sizeof(float);
                if (std::memcmp(reference.data(), candidate.data(),
                                bytes) != 0) {
                    std::fprintf(stderr,
                                 "bit mismatch %s frame=%d\n",
                                 site.tap, frame);
                    return 2;
                }
                for (size_t i = 0; i < input.size(); ++i) {
                    need(std::isfinite(reference.data()[i]),
                         "non-finite reference output");
                }
                compared_scalars += input.size();
            }

            auto & first_bias =
                *static_cast<float *>(candidate.beta->data);
            const float saved_bias = first_bias;
            first_bias += 0.125f;
            candidate.run();
            need(std::memcmp(reference.data(), candidate.data(),
                             input.size() * sizeof(float)) != 0,
                 "wrong-bias negative control was not rejected");
            first_bias = saved_bias;
            candidate.run();
            need(std::memcmp(reference.data(), candidate.data(),
                             input.size() * sizeof(float)) == 0,
                 "bias restoration did not recover parity");
            ++negative_controls;

            std::printf(
                "%s,%s,%d,%d,%d,5,1,bit_equal,wrong_bias_rejected\n",
                site.tap, site.norm, site.f, site.c, kFrames);
        }

        ggml_threadpool_free(pool);
        gguf_free(model);
        ggml_free(weight_ctx);
        std::printf(
            "PASS v2 real_weights=4 shapes=4 frames_per_shape=%d "
            "bit_equal_scalars=%zu negative_controls=%zu\n",
            kFrames, compared_scalars, negative_controls);
        return 0;
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
