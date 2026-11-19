#include "gru_batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct rng {
    std::uint64_t state;

    std::uint32_t next_u32() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<std::uint32_t>(state >> 16);
    }

    float signed_unit() {
        return static_cast<float>(static_cast<std::int32_t>(next_u32())) /
               2147483648.0f;
    }
};

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Byte-for-byte copy of the frozen scalar GRU accumulation/activation order.
template <int NIN, int NH>
void scalar_gru(const float* x, float* h,
                const float* wih, const float* whh,
                const float* bih, const float* bhh,
                float* gi, float* gh) {
    for (int g = 0; g < 3 * NH; g++) {
        float s = bih[g];
        const float* w = wih + static_cast<std::size_t>(g) * NIN;
        for (int i = 0; i < NIN; i++) s += w[i] * x[i];
        gi[g] = s;
        float t = bhh[g];
        const float* v = whh + static_cast<std::size_t>(g) * NH;
        for (int i = 0; i < NH; i++) t += v[i] * h[i];
        gh[g] = t;
    }
    for (int i = 0; i < NH; i++) {
        const float r = 1.0f / (1.0f + expf(-(gi[i] + gh[i])));
        const float z = 1.0f / (1.0f + expf(-(gi[NH+i] + gh[NH+i])));
        const float n = tanhf(gi[2*NH+i] + r * gh[2*NH+i]);
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

template <int NIN, int NH>
void batch_gru_steps(const float* input_aos, int count,
                     float* h, const float* wih, const float* whh,
                     const float* bih, const float* bhh) {
    for (int base = 0; base < count; base += daf_gru_batch::kLanes) {
        const int active =
            std::min(daf_gru_batch::kLanes, count - base);
        alignas(16) float input_soa[NIN * daf_gru_batch::kLanes];
        for (int lane = 0; lane < daf_gru_batch::kLanes; ++lane) {
            for (int i = 0; i < NIN; ++i) {
                input_soa[i * daf_gru_batch::kLanes + lane] =
                    lane < active
                        ? input_aos[(base + lane) * NIN + i]
                        : 0.0f;
            }
        }
        daf_gru_batch::gru_cell4<NIN, NH>(
            input_soa, h + static_cast<std::size_t>(base) * NH, active,
            wih, whh, bih, bhh);
    }
}

template <int NIN, int NH>
bool first_mismatch(const std::vector<float>& reference,
                    const std::vector<float>& actual,
                    int step, int* index) {
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (float_bits(reference[i]) != float_bits(actual[i])) {
            *index = static_cast<int>(i);
            std::printf(
                "  first mismatch step=%d index=%d lane=%d hidden=%d "
                "ref=%a(0x%08x) batch=%a(0x%08x)\n",
                step, *index, *index / NH, *index % NH,
                static_cast<double>(reference[i]), float_bits(reference[i]),
                static_cast<double>(actual[i]), float_bits(actual[i]));
            return true;
        }
    }
    return false;
}

template <int NIN, int NH>
bool run_positive_case(const char* label, int count, int steps,
                       bool extremes, std::uint64_t seed) {
    rng random{seed};
    std::vector<float> wih(static_cast<std::size_t>(3 * NH) * NIN);
    std::vector<float> whh(static_cast<std::size_t>(3 * NH) * NH);
    std::vector<float> bih(static_cast<std::size_t>(3 * NH));
    std::vector<float> bhh(static_cast<std::size_t>(3 * NH));
    std::vector<float> h_ref(static_cast<std::size_t>(count) * NH);
    std::vector<float> h_batch(static_cast<std::size_t>(count) * NH);
    std::vector<float> input(static_cast<std::size_t>(count) * NIN);
    std::vector<float> gi(3 * NH), gh(3 * NH);

    const float extreme_weights[] = {
        0.0f, 0.03125f, -0.03125f, 0.25f, -0.25f, 0.75f, -0.75f
    };
    const float extreme_values[] = {
        0.0f, 1e-30f, -1e-30f, 0.125f, -0.125f, 0.5f, -0.5f,
        2.0f, -2.0f, 17.0f, -17.0f, 64.0f, -64.0f
    };
    auto value = [&](const float* choices, int n) {
        return choices[random.next_u32() % static_cast<std::uint32_t>(n)];
    };

    for (float& item : wih)
        item = extremes ? value(extreme_weights, 7) : 0.75f * random.signed_unit();
    for (float& item : whh)
        item = extremes ? value(extreme_weights, 7) : 0.5f * random.signed_unit();
    for (float& item : bih)
        item = extremes ? 4.0f * random.signed_unit() : 0.25f * random.signed_unit();
    for (float& item : bhh)
        item = extremes ? 4.0f * random.signed_unit() : 0.25f * random.signed_unit();
    for (std::size_t i = 0; i < h_ref.size(); ++i) {
        h_ref[i] = extremes
            ? value(extreme_values, 13)
            : 2.0f * random.signed_unit();
    }
    h_batch = h_ref;

    daf_gru_batch::reset_counters();
    for (int step = 0; step < steps; ++step) {
        for (float& item : input) {
            item = extremes
                ? value(extreme_values, 13)
                : 2.0f * random.signed_unit();
        }
        for (int lane = 0; lane < count; ++lane) {
            scalar_gru<NIN, NH>(
                input.data() + static_cast<std::size_t>(lane) * NIN,
                h_ref.data() + static_cast<std::size_t>(lane) * NH,
                wih.data(), whh.data(), bih.data(), bhh.data(),
                gi.data(), gh.data());
        }
        batch_gru_steps<NIN, NH>(
            input.data(), count, h_batch.data(),
            wih.data(), whh.data(), bih.data(), bhh.data());

        int mismatch_index = -1;
        if (first_mismatch<NIN, NH>(h_ref, h_batch, step, &mismatch_index))
            return false;
    }

    const std::uint64_t expected_calls =
        static_cast<std::uint64_t>((count + daf_gru_batch::kLanes - 1) /
                                   daf_gru_batch::kLanes) * steps;
    const std::uint64_t expected_lane_steps =
        static_cast<std::uint64_t>(count) * steps;
    const auto& stats = daf_gru_batch::counters();
    if (stats.calls != expected_calls ||
        stats.lane_steps != expected_lane_steps) {
        std::printf(
            "FAIL %s counters calls=%llu/%llu lane_steps=%llu/%llu\n",
            label,
            static_cast<unsigned long long>(stats.calls),
            static_cast<unsigned long long>(expected_calls),
            static_cast<unsigned long long>(stats.lane_steps),
            static_cast<unsigned long long>(expected_lane_steps));
        return false;
    }

    std::printf(
        "PASS %s states=%d steps=%d calls=%llu lane_steps=%llu bitexact=1\n",
        label, count, steps,
        static_cast<unsigned long long>(stats.calls),
        static_cast<unsigned long long>(stats.lane_steps));
    return true;
}

template <int NIN, int NH>
bool wrongstate_negative_control() {
    constexpr int count = NIN == 18 ? 129 : 128;
    rng random{0x9e3779b97f4a7c15ULL};
    std::vector<float> wih(static_cast<std::size_t>(3 * NH) * NIN);
    std::vector<float> whh(static_cast<std::size_t>(3 * NH) * NH);
    std::vector<float> bih(static_cast<std::size_t>(3 * NH));
    std::vector<float> bhh(static_cast<std::size_t>(3 * NH));
    std::vector<float> h_ref(static_cast<std::size_t>(count) * NH);
    std::vector<float> h_batch(static_cast<std::size_t>(count) * NH);
    std::vector<float> input(static_cast<std::size_t>(count) * NIN);
    std::vector<float> gi(3 * NH), gh(3 * NH);
    for (float& item : wih) item = random.signed_unit();
    for (float& item : whh) item = random.signed_unit();
    for (float& item : bih) item = 0.25f * random.signed_unit();
    for (float& item : bhh) item = 0.25f * random.signed_unit();
    for (float& item : h_ref) item = random.signed_unit();
    for (float& item : input) item = random.signed_unit();
    h_batch = h_ref;
    h_batch[37 * NH + 3] += 0.125f;

    for (int lane = 0; lane < count; ++lane) {
        scalar_gru<NIN, NH>(
            input.data() + static_cast<std::size_t>(lane) * NIN,
            h_ref.data() + static_cast<std::size_t>(lane) * NH,
            wih.data(), whh.data(), bih.data(), bhh.data(),
            gi.data(), gh.data());
    }
    batch_gru_steps<NIN, NH>(
        input.data(), count, h_batch.data(),
        wih.data(), whh.data(), bih.data(), bhh.data());
    int index = -1;
    const bool detected = first_mismatch<NIN, NH>(h_ref, h_batch, 0, &index);
    std::printf(
        "NEGATIVE_CONTROL wrongstate expected_failure detected=%d index=%d\n",
        detected ? 1 : 0, index);
    return detected;
}

template <int NIN, int NH>
bool shift_negative_control() {
    constexpr int count = 4;
    rng random{0xd1b54a32d192ed03ULL};
    std::vector<float> wih(static_cast<std::size_t>(3 * NH) * NIN);
    std::vector<float> whh(static_cast<std::size_t>(3 * NH) * NH);
    std::vector<float> bih(static_cast<std::size_t>(3 * NH));
    std::vector<float> bhh(static_cast<std::size_t>(3 * NH));
    std::vector<float> h_ref(static_cast<std::size_t>(count) * NH);
    std::vector<float> h_batch(static_cast<std::size_t>(count) * NH);
    std::vector<float> input(static_cast<std::size_t>(count) * NIN);
    std::vector<float> gi(3 * NH), gh(3 * NH);
    for (float& item : wih) item = random.signed_unit();
    for (float& item : whh) item = random.signed_unit();
    for (float& item : bih) item = 0.25f * random.signed_unit();
    for (float& item : bhh) item = 0.25f * random.signed_unit();
    for (float& item : h_ref) item = random.signed_unit();
    for (float& item : input) item = random.signed_unit();
    h_batch = h_ref;

    for (int lane = 0; lane < count; ++lane) {
        scalar_gru<NIN, NH>(
            input.data() + static_cast<std::size_t>(lane) * NIN,
            h_ref.data() + static_cast<std::size_t>(lane) * NH,
            wih.data(), whh.data(), bih.data(), bhh.data(),
            gi.data(), gh.data());
    }

    alignas(16) float shifted[NIN * daf_gru_batch::kLanes];
    for (int lane = 0; lane < daf_gru_batch::kLanes; ++lane) {
        for (int i = 0; i < NIN; ++i) {
            shifted[i * daf_gru_batch::kLanes + lane] =
                input[lane * NIN + (i + 1) % NIN];
        }
    }
    daf_gru_batch::gru_cell4<NIN, NH>(
        shifted, h_batch.data(), count,
        wih.data(), whh.data(), bih.data(), bhh.data());
    int index = -1;
    const bool detected = first_mismatch<NIN, NH>(h_ref, h_batch, 0, &index);
    std::printf(
        "NEGATIVE_CONTROL shift expected_failure detected=%d index=%d\n",
        detected ? 1 : 0, index);
    return detected;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= run_positive_case<18, 16>(
        "bin129_random", 129, 128, false, 0x243f6a8885a308d3ULL);
    ok &= run_positive_case<18, 16>(
        "bin129_extremes", 129, 128, true, 0x13198a2e03707344ULL);
    ok &= run_positive_case<10, 8>(
        "partition128_random", 128, 128, false, 0xa4093822299f31d0ULL);
    ok &= run_positive_case<10, 8>(
        "partition128_extremes", 128, 128, true, 0x082efa98ec4e6c89ULL);
    ok &= wrongstate_negative_control<18, 16>();
    ok &= shift_negative_control<18, 16>();
    std::printf("RESULT %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
