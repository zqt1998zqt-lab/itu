#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "llamafile/sgemm.h"

#include <cstdint>
#include <cstring>

namespace itu_fusion_r1 {
namespace detail {

constexpr int kDec1Fout = 128;
constexpr int kDec1KW = 4;
constexpr int kDec1KH = 4;
constexpr int kDec1Cin = 16;
constexpr int kDec1Cout = 54;
constexpr int kDec1Fp = 131;
constexpr int kDec1K = kDec1KW * kDec1KH * kDec1Cin;
constexpr int kDec1Tile = 16;
constexpr int kDec1Tiles = kDec1Fout / kDec1Tile;

inline bool is_tiled_dec1_layout_compatible(const ggml_tensor * weight,
                                            const ggml_tensor * window) {
    if (weight == nullptr || window == nullptr) {
        return false;
    }
    if (weight->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32) {
        return false;
    }
    if (weight->ne[0] != kDec1KW || weight->ne[1] != kDec1KH ||
        weight->ne[2] != kDec1Cin || weight->ne[3] != kDec1Cout) {
        return false;
    }
    if (window->ne[0] != kDec1Fp || window->ne[1] != kDec1KH ||
        window->ne[2] != kDec1Cin || window->ne[3] != 1) {
        return false;
    }
    return ggml_is_contiguous(weight) && ggml_is_contiguous(window);
}

inline bool is_tiled_dec1_compatible(const ggml_context * ctx,
                                     const ggml_tensor * weight,
                                     const ggml_tensor * window) {
    return ctx != nullptr && is_tiled_dec1_layout_compatible(weight, window);
}

#if defined(__aarch64__) && defined(__ARM_NEON)

// llamafile_sgemm() uses params->threadpool for its one-thread barrier and
// chunk counter even when nth == 1. A custom-op callback does not receive the
// outer ggml_compute_params, so keep a dedicated one-worker pool. Priming an
// empty graph sets n_graph=1, which makes the barrier a no-op and allocates no
// secondary OS thread. The callback is intentionally single-task only.
class tiled_dec1_runtime {
public:
    tiled_dec1_runtime() {
        ggml_threadpool_params params = ggml_threadpool_params_default(1);
        pool_ = ggml_threadpool_new(&params);
        if (pool_ == nullptr) {
            return;
        }

        ctx_ = ggml_init({256 * 1024, nullptr, true});
        if (ctx_ == nullptr) {
            return;
        }
        graph_ = ggml_new_graph_custom(ctx_, 1, false);
        if (graph_ == nullptr) {
            return;
        }
        plan_ = ggml_graph_plan(graph_, 1, pool_);
        if (plan_.work_size != 0 ||
            ggml_graph_compute(graph_, &plan_) != GGML_STATUS_SUCCESS) {
            return;
        }
        ready_ = true;
    }

    ~tiled_dec1_runtime() {
        if (pool_ != nullptr) {
            ggml_threadpool_free(pool_);
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
        }
    }

    tiled_dec1_runtime(const tiled_dec1_runtime &) = delete;
    tiled_dec1_runtime & operator=(const tiled_dec1_runtime &) = delete;

    bool ready() const {
        return ready_;
    }

    ggml_threadpool * pool() const {
        return pool_;
    }

private:
    ggml_threadpool * pool_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_cplan plan_{};
    bool ready_ = false;
};

inline tiled_dec1_runtime & dec1_runtime() {
    static tiled_dec1_runtime runtime;
    return runtime;
}

inline void fill_dec1_tile(float * tile, const float * window,
                           int64_t stride_kh, int64_t stride_cin,
                           int f0) {
    // Exact ggml_im2col K order for a [kW, kH, Cin, Cout] weight:
    // k = ic*(kH*kW) + kh*kW + kw.
    for (int p = 0; p < kDec1Tile; ++p) {
        float * dst = tile + p * kDec1K;
        for (int ic = 0; ic < kDec1Cin; ++ic) {
            for (int kh = 0; kh < kDec1KH; ++kh) {
                const float * src = window + (f0 + p) +
                                    kh * stride_kh + ic * stride_cin;
                std::memcpy(dst + ic * (kDec1KH * kDec1KW) +
                                kh * kDec1KW,
                            src, kDec1KW * sizeof(float));
            }
        }
    }
}

inline void tiled_dec1_callback(ggml_tensor * dst, int ith, int nth, void *) {
    if (ith != 0 || nth != 1) {
        GGML_ABORT("itu_fusion_tiled_dec1 requires one GGML task");
    }
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr ||
        dst->data == nullptr || dst->src[0]->data == nullptr ||
        dst->src[1]->data == nullptr) {
        GGML_ABORT("itu_fusion_tiled_dec1 has null tensor data");
    }
    const ggml_tensor * window = dst->src[0];
    const ggml_tensor * weight = dst->src[1];
    if (!is_tiled_dec1_layout_compatible(weight, window)) {
        GGML_ABORT("itu_fusion_tiled_dec1 shape changed after build");
    }
    if (dst->ne[0] != kDec1Fout || dst->ne[1] != 1 ||
        dst->ne[2] != kDec1Cout || dst->ne[3] != 1 ||
        dst->nb[0] != sizeof(float) ||
        dst->nb[1] != sizeof(float) * kDec1Fout ||
        dst->nb[2] != sizeof(float) * kDec1Fout ||
        dst->nb[3] != sizeof(float) * kDec1Fout * kDec1Cout) {
        GGML_ABORT("itu_fusion_tiled_dec1 output layout mismatch");
    }

    tiled_dec1_runtime & runtime = dec1_runtime();
    if (!runtime.ready()) {
        GGML_ABORT("itu_fusion_tiled_dec1 runtime is not ready");
    }

    alignas(64) float tile[kDec1Tile * kDec1K];
    static_assert(sizeof(tile) == 16 * 1024,
                  "itu_fusion_tiled_dec1 workspace must stay 16 KiB");

    const auto * window_data = static_cast<const float *>(window->data);
    const auto * weight_data = static_cast<const float *>(weight->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t stride_kh = window->nb[1] / sizeof(float);
    const int64_t stride_cin = window->nb[2] / sizeof(float);

    for (int tile_index = 0; tile_index < kDec1Tiles; ++tile_index) {
        const int f0 = tile_index * kDec1Tile;
        fill_dec1_tile(tile, window_data, stride_kh, stride_cin, f0);

        ggml_compute_params params{};
        params.ith = 0;
        params.nth = 1;
        params.threadpool = runtime.pool();
        params.use_ref = false;

        // Same D path: C = A^T * B. A is the 16x256 tile, B is the
        // channel-major [256,54] weight, and C keeps the final 128-position
        // stride so no post-GEMM copy is needed.
        if (!llamafile_sgemm(&params, kDec1Tile, kDec1Cout, kDec1K,
                             tile, kDec1K,
                             weight_data, kDec1K,
                             output + f0, kDec1Fout,
                             GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32)) {
            GGML_ABORT("itu_fusion_tiled_dec1 SGEMM rejected the shape");
        }
    }
}

#endif

} // namespace detail

inline ggml_tensor * tiled_dec1_or_null(ggml_context * ctx,
                                        ggml_tensor * weight,
                                        ggml_tensor * window) {
    if (!detail::is_tiled_dec1_compatible(ctx, weight, window)) {
        return nullptr;
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    detail::tiled_dec1_runtime & runtime = detail::dec1_runtime();
    if (!runtime.ready()) {
        return nullptr;
    }

    ggml_tensor * args[2] = {window, weight};
    ggml_tensor * out = ggml_custom_4d(
        ctx, GGML_TYPE_F32, detail::kDec1Fout, 1, detail::kDec1Cout, 1,
        args, 2, detail::tiled_dec1_callback, 1, nullptr);
    if (out == nullptr || out->type != GGML_TYPE_F32 ||
        out->ne[0] != detail::kDec1Fout || out->ne[1] != 1 ||
        out->ne[2] != detail::kDec1Cout || out->ne[3] != 1 ||
        !ggml_is_contiguous(out)) {
        return nullptr;
    }
    ggml_set_name(out, "itu_fusion_tiled_dec1");
    return out;
#else
    (void) ctx;
    return nullptr;
#endif
}

} // namespace itu_fusion_r1
