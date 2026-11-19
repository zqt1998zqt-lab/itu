#pragma once

// Exact FP32 GCC-PHAT 32768-point FFT plan.
//
// The plan is built once during DSO initialization and is read-only during
// audio callbacks. try_fft() performs no allocation, file I/O, printing, or
// locking. The caller must compile this translation unit with
// -ffp-contract=off (the frozen DAF recipe already does).
//
// Mode selection is sampled once at DSO initialization:
//   ITU_LVQ_GCC_FFT=table     -> enabled
//   ITU_LVQ_GCC_FFT=original  -> disabled
//   unset                     -> ITU_GCC_FFT_TABLE_DEFAULT (0 by default)
//   any other value           -> disabled

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace itu_gcc_fft_table {

static constexpr std::size_t kSize = 32768;
static constexpr std::size_t kStageCount = 15;
static constexpr std::size_t kTwiddleCount = kSize - 1;
// Number of i<j pairs produced by the original in-place bit-reversal loop
// for n=32768. It is 16256, not n/2.
static constexpr std::size_t kSwapCount = 16256;

struct Swap {
    std::size_t i;
    std::size_t j;
};

struct Twiddle {
    double re;
    double im;
};

struct Stage {
    std::size_t len;
    std::size_t offset;
};

struct Plan {
    std::vector<Swap> forward_swaps;
    std::vector<Swap> inverse_swaps;
    std::vector<Twiddle> forward_twiddles;
    std::vector<Twiddle> inverse_twiddles;
    std::vector<Stage> stages;
    bool valid;
};

struct Config {
    bool enabled;
};

struct Runtime {
    Config config;
    Plan plan;
};

namespace detail {

static inline std::vector<Swap> make_swaps() {
    std::vector<Swap> swaps;
    swaps.reserve(kSwapCount);
    for (std::size_t i = 1, j = 0; i < kSize; ++i) {
        std::size_t bit = kSize >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swaps.push_back(Swap{i, j});
    }
    return swaps;
}

static inline void make_twiddles(std::vector<Twiddle>& out, bool inv) {
    out.clear();
    out.reserve(kTwiddleCount);
    for (std::size_t len = 2; len <= kSize; len <<= 1) {
        const double ang = (inv ? 2.0 : -2.0) * M_PI / (double)len;
        const double wr = cos(ang);
        const double wi = sin(ang);
        double cr = 1.0;
        double ci = 0.0;
        for (std::size_t k = 0; k < len / 2; ++k) {
            out.push_back(Twiddle{cr, ci});
            const double ncr = cr * wr - ci * wi;
            ci = cr * wi + ci * wr;
            cr = ncr;
        }
    }
}

static inline Plan make_plan() {
    Plan plan{};
    plan.forward_swaps = make_swaps();
    plan.inverse_swaps = make_swaps();
    make_twiddles(plan.forward_twiddles, false);
    make_twiddles(plan.inverse_twiddles, true);

    plan.stages.reserve(kStageCount);
    std::size_t offset = 0;
    for (std::size_t len = 2; len <= kSize; len <<= 1) {
        plan.stages.push_back(Stage{len, offset});
        offset += len / 2;
    }

    plan.valid =
        plan.forward_swaps.size() == kSwapCount &&
        plan.inverse_swaps.size() == kSwapCount &&
        plan.forward_twiddles.size() == kTwiddleCount &&
        plan.inverse_twiddles.size() == kTwiddleCount &&
        plan.stages.size() == kStageCount &&
        offset == kTwiddleCount;
    return plan;
}

static inline Config read_config() {
    const char* value = std::getenv("ITU_LVQ_GCC_FFT");
    if (value != nullptr) {
        if (std::strcmp(value, "table") == 0) return Config{true};
        if (std::strcmp(value, "original") == 0) return Config{false};
        return Config{false};
    }
#if defined(ITU_GCC_FFT_TABLE_DEFAULT) && ITU_GCC_FFT_TABLE_DEFAULT
    return Config{true};
#else
    return Config{false};
#endif
}

static inline Runtime make_runtime() {
    const Config config = read_config();
    return Runtime{config, config.enabled ? make_plan() : Plan{}};
}

static inline void fft_inplace_table(std::vector<float>& re,
                                     std::vector<float>& im, bool inv,
                                     const Plan& plan) {
    const std::size_t n = re.size();
    const std::vector<Swap>& swaps =
        inv ? plan.inverse_swaps : plan.forward_swaps;
    const std::vector<Twiddle>& twiddles =
        inv ? plan.inverse_twiddles : plan.forward_twiddles;

    for (const Swap& swap : swaps) {
        std::swap(re[swap.i], re[swap.j]);
        std::swap(im[swap.i], im[swap.j]);
    }

    for (const Stage& stage : plan.stages) {
        const std::size_t len = stage.len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t k = 0; k < len / 2; ++k) {
                const Twiddle tw = twiddles[stage.offset + k];
                const double cr = tw.re;
                const double ci = tw.im;
                const float ur = re[i + k];
                const float ui = im[i + k];
                const float vr = (float)(re[i + k + len / 2] * cr -
                                         im[i + k + len / 2] * ci);
                const float vi = (float)(re[i + k + len / 2] * ci +
                                         im[i + k + len / 2] * cr);
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
            }
        }
    }

    if (inv) {
        const float s = 1.0f / (float)n;
        for (std::size_t i = 0; i < n; ++i) {
            re[i] *= s;
            im[i] *= s;
        }
    }
}

}  // namespace detail

// One immutable runtime plan per linked image. Dynamic initialization occurs
// during DSO load; callbacks only read the vectors.
inline const Runtime kRuntime = detail::make_runtime();

inline bool enabled() {
    return kRuntime.config.enabled;
}

inline bool try_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    if (!enabled() || re.size() != kSize || im.size() != kSize) return false;
    if (!kRuntime.plan.valid) return false;
    detail::fft_inplace_table(re, im, inv, kRuntime.plan);
    return true;
}

}  // namespace itu_gcc_fft_table
