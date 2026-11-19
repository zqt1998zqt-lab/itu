#pragma once

#include "ggml.h"

#include <limits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include "ggml-cpu/vec.h"
#endif

namespace itu_fusion_r1 {

namespace detail {

inline bool is_bias_silu_compatible(const ggml_tensor * prebias_conv,
                                    const ggml_tensor * broadcast_bias) {
    if (prebias_conv == nullptr || broadcast_bias == nullptr) {
        return false;
    }
    if (prebias_conv->type != GGML_TYPE_F32 ||
        broadcast_bias->type != GGML_TYPE_F32) {
        return false;
    }
    if (prebias_conv->ne[0] <= 0 || prebias_conv->ne[0] >
            std::numeric_limits<int>::max() ||
        prebias_conv->ne[1] != 1 || prebias_conv->ne[2] <= 0 ||
        prebias_conv->ne[3] != 1) {
        return false;
    }
    if (broadcast_bias->ne[0] != 1 || broadcast_bias->ne[1] != 1 ||
        broadcast_bias->ne[2] != prebias_conv->ne[2] ||
        broadcast_bias->ne[3] != 1) {
        return false;
    }
    return ggml_is_contiguous(prebias_conv) &&
           ggml_is_contiguous(broadcast_bias);
}

#if defined(__aarch64__) && defined(__ARM_NEON)

inline void bias_silu_callback(ggml_tensor * dst,
                               const ggml_tensor * prebias_conv,
                               const ggml_tensor * broadcast_bias,
                               int ith,
                               int nth,
                               void *) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    GGML_ASSERT(ith == 0 && nth == 1);
    GGML_ASSERT(dst != nullptr && prebias_conv != nullptr &&
                broadcast_bias != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 &&
                prebias_conv->type == GGML_TYPE_F32 &&
                broadcast_bias->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) &&
                ggml_is_contiguous(prebias_conv) &&
                ggml_is_contiguous(broadcast_bias));
    GGML_ASSERT(ggml_are_same_shape(dst, prebias_conv));

    const int f = static_cast<int>(prebias_conv->ne[0]);
    const int64_t channels = prebias_conv->ne[2];
    const auto * input = static_cast<const float *>(prebias_conv->data);
    const auto * bias = static_cast<const float *>(broadcast_bias->data);
    auto * output = static_cast<float *>(dst->data);

    for (int64_t channel = 0; channel < channels; ++channel) {
        const float channel_bias = bias[channel];
        float * row = output + channel * f;
        const float * src = input + channel * f;

        // Keep the standalone GGML ADD rounding point before invoking the
        // exact SILU implementation used by GGML's unary CPU op.
        for (int i = 0; i < f; ++i) {
            row[i] = src[i] + channel_bias;
        }
        ggml_vec_silu_f32(f, row, row);
    }
}

#endif

} // namespace detail

inline ggml_tensor * bias_silu_or_null(
        ggml_context * ctx,
        ggml_tensor * prebias_conv,
        ggml_tensor * broadcast_bias) {
    if (ctx == nullptr ||
        !detail::is_bias_silu_compatible(prebias_conv, broadcast_bias)) {
        return nullptr;
    }

#if defined(__aarch64__) && defined(__ARM_NEON)
    ggml_tensor * out = ggml_map_custom2(
        ctx, prebias_conv, broadcast_bias,
        detail::bias_silu_callback, 1, nullptr);
    if (out == nullptr || out->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(out, prebias_conv) ||
        !ggml_is_contiguous(out)) {
        return nullptr;
    }
    ggml_set_name(out, "itu_fusion_bias_silu");
    return out;
#else
    (void) prebias_conv;
    (void) broadcast_bias;
    return nullptr;
#endif
}

} // namespace itu_fusion_r1
