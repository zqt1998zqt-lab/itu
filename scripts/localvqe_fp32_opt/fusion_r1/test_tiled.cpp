#include "tiled_conv.hpp"

#include "ggml-cpu.h"
#include "gguf.h"
#include "llamafile/sgemm.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int F = 128, W = 4, H = 4, C = 16, O = 54, FP = 131;
constexpr int K = W * H * C, T = 16;

void need(bool x, const char * m) {
    if (!x) throw std::runtime_error(m);
}

struct Branch {
    Branch(ggml_tensor * weight, bool candidate, ggml_threadpool * pool)
        : ctx(ggml_init({8 * 1024 * 1024, nullptr, false})) {
        need(ctx != nullptr, "graph arena");
        window = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, FP, H, C, 1);
        if (candidate) {
            output = itu_fusion_r1::tiled_dec1_or_null(ctx, weight, window);
        } else {
            im2col = ggml_im2col(ctx, weight, window, 1, 1, 0, 0, 1, 1,
                                 true, GGML_TYPE_F32);
            auto * a = ggml_reshape_2d(ctx, im2col, K, F);
            auto * b = ggml_reshape_2d(ctx, weight, K, O);
            output = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, a, b), F, 1, O, 1);
        }
        need(output != nullptr, "candidate/reference output");
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            switch (ggml_graph_node(graph, i)->op) {
                case GGML_OP_IM2COL: ++im2col_nodes; break;
                case GGML_OP_MUL_MAT: ++mul_mat_nodes; break;
                case GGML_OP_CUSTOM: ++custom_nodes; break;
                default: break;
            }
        }
        plan = ggml_graph_plan(graph, 1, pool);
        work.resize(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
    }
    ~Branch() { ggml_free(ctx); }
    void run() {
        need(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS,
             "ggml compute failed");
    }
    void fill(uint32_t & state) {
        auto * x = static_cast<float *>(window->data);
        for (size_t i = 0; i < size_t(FP) * H * C; ++i) {
            state = state * 1664525u + 1013904223u;
            x[i] = float((state >> 8) & 0x00ffffffu) / 8388608.0f - 1.0f;
        }
    }
    ggml_context * ctx = nullptr;
    ggml_tensor * window = nullptr;
    ggml_tensor * im2col = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan{};
    std::vector<unsigned char> work;
    int im2col_nodes = 0, mul_mat_nodes = 0, custom_nodes = 0;
};

void rejection_gates() {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    need(ctx != nullptr, "rejection arena");
    auto * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, O);
    auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, FP, H, C, 1);
    need(itu_fusion_r1::tiled_dec1_or_null(ctx, w, x) != nullptr,
         "positive gate");
    auto * xbad = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, FP - 1, H, C, 1);
    auto * wbad = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, O - 1);
    need(itu_fusion_r1::tiled_dec1_or_null(ctx, w, xbad) == nullptr,
         "bad window accepted");
    need(itu_fusion_r1::tiled_dec1_or_null(ctx, wbad, x) == nullptr,
         "bad weight accepted");
    auto * base = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, (FP + 1) * H * C);
    auto * strided = ggml_view_4d(
        ctx, base, FP, H, C, 1,
        sizeof(float) * (FP + 1),
        sizeof(float) * (FP + 1) * H,
        sizeof(float) * (FP + 1) * H * C, 0);
    need(!ggml_is_contiguous(strided), "stride fixture");
    need(itu_fusion_r1::tiled_dec1_or_null(ctx, w, strided) == nullptr,
         "strided window accepted");
    ggml_free(ctx);
}

void direct_controls(const Branch & ref, const ggml_tensor * weight,
                     ggml_threadpool * pool) {
    ggml_compute_params p{};
    p.ith = 0;
    p.nth = 1;
    p.threadpool = pool;
    p.use_ref = false;
    const auto * a = static_cast<const float *>(ref.im2col->data);
    const auto * b = static_cast<const float *>(weight->data);
    const auto * r = static_cast<const float *>(ref.output->data);
    std::vector<float> direct(O * F);
    need(llamafile_sgemm(&p, T, O, K, a, K, b, K, direct.data(), F,
                         GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32),
         "direct SGEMM rejected");
    for (int ch = 0; ch < O; ++ch) {
        for (int q = 0; q < T; ++q) {
            need(std::memcmp(&direct[ch * F + q], &r[ch * F + q],
                             sizeof(float)) == 0,
                 "direct SGEMM differs from reference");
        }
    }

    const auto * x = static_cast<const float *>(ref.window->data);
    const int64_t sh = ref.window->nb[1] / sizeof(float);
    const int64_t sc = ref.window->nb[2] / sizeof(float);
    std::vector<float> wrong_k(T * K);
    for (int q = 0; q < T; ++q) {
        for (int ic = 0; ic < C; ++ic) {
            for (int kh = 0; kh < H; ++kh) {
                for (int kw = 0; kw < W; ++kw) {
                    const int wi = kw * (H * C) + kh * C + ic;
                    wrong_k[q * K + wi] =
                        x[q + kh * sh + ic * sc + kw];
                }
            }
        }
    }
    std::vector<float> wrong_k_out(O * F);
    need(llamafile_sgemm(&p, T, O, K, wrong_k.data(), K, b, K,
                         wrong_k_out.data(), F, GGML_TYPE_F32,
                         GGML_TYPE_F32, GGML_TYPE_F32),
         "wrong-K rejected");
    need(std::memcmp(wrong_k_out.data(), direct.data(),
                     direct.size() * sizeof(float)) != 0,
         "wrong-K did not change output");

    std::vector<float> wrong_s(O * F,
                               std::numeric_limits<float>::quiet_NaN());
    need(llamafile_sgemm(&p, T, O, K, a, K, b, K, wrong_s.data(), T,
                         GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32),
         "wrong-stride rejected");
    bool mismatch = false;
    for (int ch = 0; ch < O; ++ch) {
        for (int q = 0; q < T; ++q) {
            if (std::memcmp(&wrong_s[ch * F + q], &direct[ch * F + q],
                            sizeof(float)) != 0) {
                mismatch = true;
            }
        }
    }
    need(mismatch, "wrong-stride did not change layout");
}
}

int main(int argc, char ** argv) {
    try {
        need(argc == 2, "usage: test-tiled FROZEN_MODEL.gguf");
#if !defined(__aarch64__) || !defined(__ARM_NEON)
        std::fprintf(stderr, "SKIP: AArch64 NEON required\n");
        return 77;
#else
        rejection_gates();
        ggml_context * wc = nullptr;
        gguf_context * model = gguf_init_from_file(argv[1], {false, &wc});
        need(model != nullptr && wc != nullptr, "GGUF");
        auto * weight = ggml_get_tensor(wc, "dec1.deconv.conv.weight");
        need(weight && weight->data && weight->type == GGML_TYPE_F32,
             "real dec1 weight");
        auto pp = ggml_threadpool_params_default(1);
        auto * pool = ggml_threadpool_new(&pp);
        need(pool != nullptr, "pool");
        Branch ref(weight, false, pool), cand(weight, true, pool);
        need(ref.im2col_nodes == 1 && ref.mul_mat_nodes == 1 &&
                 ref.custom_nodes == 0,
             "reference graph shape");
        need(cand.im2col_nodes == 0 && cand.mul_mat_nodes == 0 &&
                 cand.custom_nodes == 1,
             "candidate graph shape");
        need(std::strcmp(cand.output->name, "itu_fusion_tiled_dec1") == 0,
             "candidate name");

        constexpr int frames = 16;
        for (int frame = 0; frame < frames; ++frame) {
            uint32_t seed = 0x91e10da5u + uint32_t(frame) * 7919u;
            uint32_t ref_seed = seed, cand_seed = seed;
            ref.fill(ref_seed);
            cand.fill(cand_seed);
            ref.run();
            cand.run();
            need(std::memcmp(ref.output->data, cand.output->data,
                             size_t(F) * O * sizeof(float)) == 0,
                 "tiled candidate differs from full IM2COL");
            if (frame == 0) direct_controls(ref, weight, pool);
        }
        ggml_threadpool_free(pool);
        gguf_free(model);
        ggml_free(wc);
        std::printf("PASS shape=%dx%dx%dx%d k=%d tile=%d frames=%d "
                    "bit_exact=1 custom=%d im2col=%d mul_mat=%d "
                    "wrong_k_rejected=1 wrong_stride_rejected=1\n",
                    F, 1, O, 1, K, T, frames, cand.custom_nodes,
                    cand.im2col_nodes, cand.mul_mat_nodes);
        return 0;
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 2;
    }
}
