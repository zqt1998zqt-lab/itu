#ifndef ITU_GCC_FFT_TABLE_DEFAULT
#define ITU_GCC_FFT_TABLE_DEFAULT 1
#endif

#include "gcc_fft_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

volatile std::uint64_t bench_sink = 0;

// BEGIN_EXTRACTED_FFT_INPLACE
static void fft_inplace(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; i++) {            // bit reversal
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = (inv ? 2.0 : -2.0) * M_PI / (double)len;
        const double wr = cos(ang), wi = sin(ang);
        for (size_t i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (size_t k = 0; k < len / 2; k++) {
                const float ur = re[i+k], ui = im[i+k];
                const float vr = (float)(re[i+k+len/2]*cr - im[i+k+len/2]*ci);
                const float vi = (float)(re[i+k+len/2]*ci + im[i+k+len/2]*cr);
                re[i+k] = ur + vr;       im[i+k] = ui + vi;
                re[i+k+len/2] = ur - vr; im[i+k+len/2] = ui - vi;
                const double ncr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr; cr = ncr;
            }
        }
    }
    if (inv) {
        const float s = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
    }
}
// END_EXTRACTED_FFT_INPLACE

enum class CaseKind {
    zeros,
    impulse,
    ramp,
    lcg,
    tail,
};

constexpr std::size_t kTail = 32;
constexpr int kBenchmarkIterations = 5;

std::uint32_t next_u32(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

void fill_case(CaseKind kind, std::vector<float>& re,
               std::vector<float>& im) {
    re.assign(itu_gcc_fft_table::kSize, 0.0f);
    im.assign(itu_gcc_fft_table::kSize, 0.0f);
    std::uint32_t state = 0x12345678u;
    switch (kind) {
        case CaseKind::zeros:
            break;
        case CaseKind::impulse:
            re[0] = 1.0f;
            break;
        case CaseKind::ramp:
            for (std::size_t i = 0; i < re.size(); ++i) {
                re[i] = static_cast<float>(static_cast<int>(i % 257) - 128) /
                        257.0f;
                im[i] = static_cast<float>(static_cast<int>(i % 131) - 65) /
                        131.0f;
            }
            break;
        case CaseKind::lcg:
            for (std::size_t i = 0; i < re.size(); ++i) {
                const std::uint32_t a = next_u32(state);
                const std::uint32_t b = next_u32(state);
                re[i] = static_cast<float>(static_cast<std::int32_t>(a)) /
                        2147483648.0f;
                im[i] = static_cast<float>(static_cast<std::int32_t>(b)) /
                        2147483648.0f;
            }
            break;
        case CaseKind::tail:
            for (std::size_t i = re.size() - kTail; i < re.size(); ++i) {
                re[i] = static_cast<float>(i - (re.size() - kTail));
                im[i] = -re[i] * 0.5f;
            }
            break;
    }
}

bool bitwise_equal(const std::vector<float>& a,
                   const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool tail_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.size() < kTail) return false;
    const std::size_t begin = a.size() - kTail;
    return std::memcmp(a.data() + begin, b.data() + begin,
                       kTail * sizeof(float)) == 0;
}

std::uint64_t now_ns() {
    struct timespec ts{};
    if (::clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

bool check_case(CaseKind kind, bool inv) {
    std::vector<float> input_re;
    std::vector<float> input_im;
    fill_case(kind, input_re, input_im);

    std::vector<float> original_re = input_re;
    std::vector<float> original_im = input_im;
    fft_inplace(original_re, original_im, inv);

    std::vector<float> table_re = input_re;
    std::vector<float> table_im = input_im;
    const bool hit = itu_gcc_fft_table::try_fft(table_re, table_im, inv);
    if (itu_gcc_fft_table::enabled()) {
        if (!hit) return false;
        if (!bitwise_equal(original_re, table_re)) return false;
        if (!bitwise_equal(original_im, table_im)) return false;
        if (!tail_equal(original_re, table_re)) return false;
        if (!tail_equal(original_im, table_im)) return false;
    } else if (hit) {
        return false;
    }
    return true;
}

bool check_non_size_rejected() {
    std::vector<float> re(16384, 0.0f);
    std::vector<float> im(16384, 0.0f);
    return !itu_gcc_fft_table::try_fft(re, im, false);
}

bool check_negative_controls() {
    std::vector<float> input_re;
    std::vector<float> input_im;
    fill_case(CaseKind::zeros, input_re, input_im);
    std::vector<float> original_re = input_re;
    std::vector<float> original_im = input_im;
    fft_inplace(original_re, original_im, false);

    std::vector<float> perturbed_re = input_re;
    std::vector<float> perturbed_im = input_im;
    perturbed_re[0] = 1.0f;
    fft_inplace(perturbed_re, perturbed_im, false);
    if (bitwise_equal(original_re, perturbed_re)) return false;

    if (itu_gcc_fft_table::enabled()) {
        std::vector<float> table_re = input_re;
        std::vector<float> table_im = input_im;
        if (!itu_gcc_fft_table::try_fft(table_re, table_im, false)) return false;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &table_re.back(), sizeof(bits));
        bits ^= 1u;
        std::memcpy(&table_re.back(), &bits, sizeof(bits));
        if (bitwise_equal(original_re, table_re)) return false;
    }
    return true;
}

template <typename Fn>
std::uint64_t benchmark(const std::vector<float>& input_re,
                        const std::vector<float>& input_im, Fn fn) {
    std::uint64_t total = 0;
    for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
        std::vector<float> re = input_re;
        std::vector<float> im = input_im;
        const std::uint64_t start = now_ns();
        const bool ok = fn(re, im);
        const std::uint64_t end = now_ns();
        if (!ok || start == 0 || end < start) return 0;
        total += end - start;
        bench_sink += static_cast<std::uint64_t>(re[0] != 0.0f) +
                      static_cast<std::uint64_t>(im.back() != 0.0f);
    }
    return total / kBenchmarkIterations;
}

void run_benchmark() {
    std::vector<float> input_re;
    std::vector<float> input_im;
    fill_case(CaseKind::lcg, input_re, input_im);
    const std::uint64_t original_ns = benchmark(
        input_re, input_im,
        [](std::vector<float>& re, std::vector<float>& im) {
            fft_inplace(re, im, false);
            return true;
        });
    std::uint64_t table_ns = 0;
    if (itu_gcc_fft_table::enabled()) {
        table_ns = benchmark(
            input_re, input_im,
            [](std::vector<float>& re, std::vector<float>& im) {
                return itu_gcc_fft_table::try_fft(re, im, false);
            });
    }
    std::printf("BENCH original_ns=%llu table_ns=%llu iterations=%d\n",
                static_cast<unsigned long long>(original_ns),
                static_cast<unsigned long long>(table_ns),
                kBenchmarkIterations);
}

}  // namespace

int main() {
    const CaseKind cases[] = {
        CaseKind::zeros,
        CaseKind::impulse,
        CaseKind::ramp,
        CaseKind::lcg,
        CaseKind::tail,
    };
    for (CaseKind kind : cases) {
        if (!check_case(kind, false)) return 10;
        if (!check_case(kind, true)) return 11;
    }
    if (!check_non_size_rejected()) return 12;
    if (!check_negative_controls()) return 13;
    run_benchmark();
    std::printf("MODE enabled=%d macro_default=%d\n",
                itu_gcc_fft_table::enabled() ? 1 : 0,
                ITU_GCC_FFT_TABLE_DEFAULT ? 1 : 0);
    std::printf("TABLE_PASS cases=5 directions=2 tail=%zu\n", kTail);
    return 0;
}
