#pragma once

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace itu_fusion_r1 {

namespace detail {

inline bool is_norm_affine_compatible_v2(const ggml_tensor * x,
                                         const ggml_tensor * gamma,
                                         const ggml_tensor * beta,
                                         float eps) {
    constexpr float required_eps = 1e-5f;

    if (x == nullptr || gamma == nullptr || beta == nullptr) {
        return false;
    }
    if (std::memcmp(&eps, &required_eps, sizeof(eps)) != 0) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || gamma->type != GGML_TYPE_F32 ||
        beta->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->ne[0] <= 0 || x->ne[1] != 1 || x->ne[2] <= 0 || x->ne[3] != 1) {
        return false;
    }
    if (gamma->ne[0] != x->ne[2] || gamma->ne[1] != 1 ||
        gamma->ne[2] != 1 || gamma->ne[3] != 1) {
        return false;
    }
    if (beta->ne[0] != x->ne[2] || beta->ne[1] != 1 ||
        beta->ne[2] != 1 || beta->ne[3] != 1) {
        return false;
    }

    constexpr int64_t max_int = std::numeric_limits<int>::max();
    if (x->ne[0] > max_int || x->ne[2] > max_int ||
        x->ne[0] > max_int / x->ne[2]) {
        return false;
    }

    const size_t f = static_cast<size_t>(x->ne[0]);
    const size_t c = static_cast<size_t>(x->ne[2]);
    const size_t element = sizeof(float);
    if (x->nb[0] != element || x->nb[1] != element * f ||
        x->nb[2] != element * f || x->nb[3] != element * f * c) {
        return false;
    }
    if (gamma->nb[0] != element || gamma->nb[1] != element * c ||
        gamma->nb[2] != element * c || gamma->nb[3] != element * c) {
        return false;
    }
    if (beta->nb[0] != element || beta->nb[1] != element * c ||
        beta->nb[2] != element * c || beta->nb[3] != element * c) {
        return false;
    }
    return ggml_is_contiguous(x) && ggml_is_contiguous(gamma) &&
           ggml_is_contiguous(beta);
}

#if defined(__aarch64__) && defined(__ARM_NEON)

#if defined(__GNUC__) && !defined(__clang__)
#define ITU_FUSION_R1_V2_GCC_FP_CONTRACT_OFF \
    __attribute__((optimize("fp-contract=off")))
#else
#define ITU_FUSION_R1_V2_GCC_FP_CONTRACT_OFF
#endif

inline ITU_FUSION_R1_V2_GCC_FP_CONTRACT_OFF
void norm_affine_callback_v2(ggml_tensor * dst,
                             const ggml_tensor * x,
                             const ggml_tensor * gamma,
                             const ggml_tensor * beta,
                             int ith,
                             int nth,
                             void *) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (ith != 0 || nth != 1) {
        return;
    }

    const int f = static_cast<int>(x->ne[0]);
    const int c = static_cast<int>(x->ne[2]);
    const int n = f * c;
    const auto * in = static_cast<const float *>(x->data);
    const auto * g = static_cast<const float *>(gamma->data);
    const auto * b = static_cast<const float *>(beta->data);
    auto * out = static_cast<float *>(dst->data);

    // Match ggml_compute_forward_norm_f32 on AArch64: double accumulation
    // for the mean, then one AArch64 vaddvq_f32 result per four inputs.
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += static_cast<double>(in[k]);
    }
    const float mean = static_cast<float>(sum) / static_cast<float>(n);
    const float32x4_t mean4 = vdupq_n_f32(mean);

    double squared_sum = 0.0;
    int k = 0;
    for (; k + 3 < n; k += 4) {
        const float32x4_t centered =
            vsubq_f32(vld1q_f32(in + k), mean4);
        squared_sum += static_cast<double>(
            vaddvq_f32(vmulq_f32(centered, centered)));
    }
    for (; k < n; ++k) {
        const float centered = in[k] - mean;
        squared_sum += static_cast<double>(centered * centered);
    }

    const float variance =
        static_cast<float>(squared_sum / static_cast<double>(n));
    const float scale = 1.0f / std::sqrt(variance + 1e-5f);
    const float32x4_t scale4 = vdupq_n_f32(scale);

    for (int ch = 0; ch < c; ++ch, in += f, out += f) {
        const float32x4_t gamma4 = vdupq_n_f32(g[ch]);
        const float32x4_t beta4 = vdupq_n_f32(b[ch]);
        int i = 0;
        for (; i + 3 < f; i += 4) {
            const float32x4_t centered =
                vsubq_f32(vld1q_f32(in + i), mean4);
            const float32x4_t normalized = vmulq_f32(centered, scale4);
            vst1q_f32(out + i,
                      vaddq_f32(vmulq_f32(normalized, gamma4), beta4));
        }
        for (; i < f; ++i) {
            const float normalized = (in[i] - mean) * scale;
            const float scaled = normalized * g[ch];
            out[i] = scaled + b[ch];
        }
    }
}

#undef ITU_FUSION_R1_V2_GCC_FP_CONTRACT_OFF

#endif

} // namespace detail

inline ggml_tensor * norm_affine_or_null(ggml_context * ctx,
                                         ggml_tensor * x,
                                         ggml_tensor * gamma,
                                         ggml_tensor * beta,
                                         float eps) {
    if (ctx == nullptr ||
        !detail::is_norm_affine_compatible_v2(x, gamma, beta, eps)) {
        return nullptr;
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    ggml_tensor * out = ggml_map_custom3(
        ctx, x, gamma, beta, detail::norm_affine_callback_v2, 1, nullptr);
    if (out == nullptr || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(out)) {
        return nullptr;
    }

    const size_t f = static_cast<size_t>(out->ne[0]);
    const size_t c = static_cast<size_t>(out->ne[2]);
    const size_t element = sizeof(float);
    if (out->ne[0] != x->ne[0] || out->ne[1] != 1 ||
        out->ne[2] != x->ne[2] || out->ne[3] != 1 ||
        out->nb[0] != element || out->nb[1] != element * f ||
        out->nb[2] != element * f || out->nb[3] != element * f * c) {
        return nullptr;
    }
    ggml_set_name(out, "itu_fusion_norm");
    return out;
#else
    (void) ctx;
    return nullptr;
#endif
}

} // namespace itu_fusion_r1
