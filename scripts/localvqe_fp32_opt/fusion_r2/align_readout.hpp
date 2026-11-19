#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace itu_fusion_r2 {

namespace detail {

constexpr int64_t kAlignF = 64;
constexpr int64_t kAlignD = 16;
constexpr int64_t kAlignC = 20;

inline bool is_align_readout_compatible(const ggml_tensor * ref_full,
                                        const ggml_tensor * attn) {
    if (ref_full == nullptr || attn == nullptr) {
        return false;
    }
    if (ref_full->type != GGML_TYPE_F32 ||
        attn->type != GGML_TYPE_F32) {
        return false;
    }
    if (ref_full->ne[0] != kAlignF ||
        ref_full->ne[1] != kAlignD ||
        ref_full->ne[2] != kAlignC ||
        ref_full->ne[3] != 1) {
        return false;
    }
    if (attn->ne[0] != kAlignD ||
        attn->ne[1] != 1 ||
        attn->ne[2] != 1 ||
        attn->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(ref_full) || !ggml_is_contiguous(attn)) {
        return false;
    }

    constexpr size_t element = sizeof(float);
    constexpr size_t stride_f = element * kAlignF;
    constexpr size_t stride_d = stride_f * kAlignD;
    constexpr size_t stride_c = stride_d * kAlignC;
    return ref_full->nb[0] == element &&
           ref_full->nb[1] == stride_f &&
           ref_full->nb[2] == stride_d &&
           ref_full->nb[3] == stride_c &&
           attn->nb[0] == element &&
           attn->nb[1] == element * kAlignD &&
           attn->nb[2] == element * kAlignD &&
           attn->nb[3] == element * kAlignD;
}

#if defined(__aarch64__) && defined(__ARM_NEON)

#if defined(__GNUC__) && !defined(__clang__)
#define ITU_FUSION_R2_ALIGN_GCC_FP_CONTRACT_OFF \
    __attribute__((optimize("fp-contract=off")))
#else
#define ITU_FUSION_R2_ALIGN_GCC_FP_CONTRACT_OFF
#endif

inline ITU_FUSION_R2_ALIGN_GCC_FP_CONTRACT_OFF
void align_readout_callback(ggml_tensor * dst,
                            int ith,
                            int nth,
                            void *) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (ith != 0 || nth != 1) {
        GGML_ABORT("itu_fusion_align_readout requires one GGML task");
    }
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr ||
        dst->data == nullptr || dst->src[0]->data == nullptr ||
        dst->src[1]->data == nullptr) {
        GGML_ABORT("itu_fusion_align_readout has null tensor data");
    }

    const ggml_tensor * ref_full = dst->src[0];
    const ggml_tensor * attn = dst->src[1];
    if (!is_align_readout_compatible(ref_full, attn)) {
        GGML_ABORT("itu_fusion_align_readout input layout changed");
    }
    if (dst->type != GGML_TYPE_F32 ||
        dst->ne[0] != kAlignF ||
        dst->ne[1] != 1 ||
        dst->ne[2] != kAlignC ||
        dst->ne[3] != 1 ||
        dst->nb[0] != sizeof(float) ||
        dst->nb[1] != sizeof(float) * kAlignF ||
        dst->nb[2] != sizeof(float) * kAlignF ||
        dst->nb[3] != sizeof(float) * kAlignF * kAlignC ||
        !ggml_is_contiguous(dst)) {
        GGML_ABORT("itu_fusion_align_readout output layout mismatch");
    }

    const auto * ref_data = static_cast<const float *>(ref_full->data);
    const auto * attn_data = static_cast<const float *>(attn->data);
    auto * output = static_cast<float *>(dst->data);

    // Four adjacent frequencies are independent rows. Each lane still visits
    // d=0..15 in order and adds a rounded FP32 product to its own double
    // accumulator. No product is contracted into the double addition.
    for (int64_t c = 0; c < kAlignC; ++c) {
        const float * channel =
            ref_data + static_cast<size_t>(c * kAlignD * kAlignF);
        float * channel_out = output + static_cast<size_t>(c * kAlignF);

        for (int64_t f = 0; f < kAlignF; f += 4) {
            float64x2_t sum_lo = vdupq_n_f64(0.0);
            float64x2_t sum_hi = vdupq_n_f64(0.0);

            for (int64_t d = 0; d < kAlignD; ++d) {
                const float * row =
                    channel + static_cast<size_t>(d * kAlignF + f);
                const float32x4_t products =
                    vmulq_f32(vld1q_f32(row),
                              vdupq_n_f32(attn_data[d]));
                sum_lo = vaddq_f64(
                    sum_lo, vcvt_f64_f32(vget_low_f32(products)));
                sum_hi = vaddq_f64(
                    sum_hi, vcvt_f64_f32(vget_high_f32(products)));
            }

            vst1_f32(channel_out + f, vcvt_f32_f64(sum_lo));
            vst1_f32(channel_out + f + 2, vcvt_f32_f64(sum_hi));
        }
    }
}

#undef ITU_FUSION_R2_ALIGN_GCC_FP_CONTRACT_OFF

#endif

} // namespace detail

inline ggml_tensor * align_readout_or_null(ggml_context * ctx,
                                           ggml_tensor * ref_full,
                                           ggml_tensor * attn) {
    if (ctx == nullptr ||
        !detail::is_align_readout_compatible(ref_full, attn)) {
        return nullptr;
    }

#if defined(__aarch64__) && defined(__ARM_NEON)
    ggml_tensor * args[2] = {ref_full, attn};
    ggml_tensor * out = ggml_custom_4d(
        ctx, GGML_TYPE_F32,
        detail::kAlignF, 1, detail::kAlignC, 1,
        args, 2, detail::align_readout_callback, 1, nullptr);
    if (out == nullptr ||
        out->type != GGML_TYPE_F32 ||
        out->ne[0] != detail::kAlignF ||
        out->ne[1] != 1 ||
        out->ne[2] != detail::kAlignC ||
        out->ne[3] != 1 ||
        out->nb[0] != sizeof(float) ||
        out->nb[1] != sizeof(float) * detail::kAlignF ||
        out->nb[2] != sizeof(float) * detail::kAlignF ||
        out->nb[3] != sizeof(float) * detail::kAlignF * detail::kAlignC ||
        !ggml_is_contiguous(out)) {
        return nullptr;
    }
    ggml_set_name(out, "itu_fusion_align_readout");
    return out;
#else
    (void) ref_full;
    (void) attn;
    return nullptr;
#endif
}

} // namespace itu_fusion_r2
