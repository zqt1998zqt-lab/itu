#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "gru_batch.hpp requires an AArch64 NEON target"
#endif

#if defined(__FAST_MATH__)
#error "gru_batch.hpp forbids fast-math"
#endif

#include <arm_neon.h>

namespace daf_gru_batch {

constexpr int kLanes = 4;

struct stats {
    std::uint64_t calls = 0;
    std::uint64_t lane_steps = 0;
};

inline stats& counters() {
    static thread_local stats value;
    return value;
}

inline void reset_counters() {
    counters() = stats{};
}

// Batch four independent GRUCell states.  The input is feature-major with the
// four lane values adjacent: input[feature * 4 + lane].  The state stays in
// its original state-major layout: h[lane * nh + hidden].
//
// For each gate and input index this performs exactly:
//   mul = weight * input
//   sum = sum + mul
// for all four lanes.  It deliberately uses separate NEON fmul/fadd
// instructions under -ffp-contract=off; it never uses FMA.  Activations are
// evaluated lane-by-lane with the scalar expf/tanhf calls used by the original
// gru_cell, and the final state expression keeps the original operation order.
template <int NIN, int NH>
inline void gru_cell4(const float* input_soa, float* h, int active,
                      const float* wih, const float* whh,
                      const float* bih, const float* bhh) {
    alignas(16) float h_soa[NH * kLanes];
    alignas(16) float gi[(3 * NH) * kLanes];
    alignas(16) float gh[(3 * NH) * kLanes];

    for (int i = 0; i < NH; ++i) {
        const float lanes[kLanes] = {
            active > 0 ? h[0 * NH + i] : 0.0f,
            active > 1 ? h[1 * NH + i] : 0.0f,
            active > 2 ? h[2 * NH + i] : 0.0f,
            active > 3 ? h[3 * NH + i] : 0.0f,
        };
        vst1q_f32(h_soa + i * kLanes, vld1q_f32(lanes));
    }

    for (int g = 0; g < 3 * NH; ++g) {
        float32x4_t s = vdupq_n_f32(bih[g]);
        float32x4_t t = vdupq_n_f32(bhh[g]);
        const float* w = wih + static_cast<std::size_t>(g) * NIN;
        const float* v = whh + static_cast<std::size_t>(g) * NH;
        for (int i = 0; i < NIN; ++i) {
            const float32x4_t xv = vld1q_f32(input_soa + i * kLanes);
            const float32x4_t product = vmulq_f32(vdupq_n_f32(w[i]), xv);
            s = vaddq_f32(s, product);
        }
        for (int i = 0; i < NH; ++i) {
            const float32x4_t hv = vld1q_f32(h_soa + i * kLanes);
            const float32x4_t product = vmulq_f32(vdupq_n_f32(v[i]), hv);
            t = vaddq_f32(t, product);
        }
        vst1q_f32(gi + g * kLanes, s);
        vst1q_f32(gh + g * kLanes, t);
    }

#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (int lane = 0; lane < active; ++lane) {
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int i = 0; i < NH; ++i) {
            const float r = 1.0f /
                (1.0f + expf(-(gi[(0 * NH + i) * kLanes + lane] +
                              gh[(0 * NH + i) * kLanes + lane])));
            const float z = 1.0f /
                (1.0f + expf(-(gi[(1 * NH + i) * kLanes + lane] +
                              gh[(1 * NH + i) * kLanes + lane])));
            const float n = tanhf(
                gi[(2 * NH + i) * kLanes + lane] +
                r * gh[(2 * NH + i) * kLanes + lane]);
            const float old_h = h[lane * NH + i];
            h[lane * NH + i] = (1.0f - z) * n + z * old_h;
        }
    }

    counters().calls += 1;
    counters().lane_steps += static_cast<std::uint64_t>(active);
}

}  // namespace daf_gru_batch
