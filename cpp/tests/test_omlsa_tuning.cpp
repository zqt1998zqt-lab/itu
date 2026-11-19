#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OMLSA_ORIGINAL_GOLDEN
#include "itu/omlsa_config.hpp"
#endif
#include "omlsa_imcra.hpp"

namespace {

std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};

void* counted_allocate(std::size_t size) {
  if (count_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
  throw std::bad_alloc();
}

}  // namespace

void* operator new(std::size_t size) { return counted_allocate(size); }
void* operator new[](std::size_t size) { return counted_allocate(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

class TestFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw TestFailure(message);
}

class XorShift32 {
 public:
  explicit XorShift32(std::uint32_t seed) : state_(seed ? seed : 1u) {}

  std::uint32_t next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return state_;
  }

  float bipolar() {
    return (static_cast<float>(next() >> 8) * (1.f / 16777216.f)) * 2.f - 1.f;
  }

 private:
  std::uint32_t state_;
};

struct EngineRun {
  std::vector<float> output;
  std::vector<float> mean_gain;
};

EngineRun run_engine(itu::detail::OmlsaImcra& engine,
                     const std::vector<float>& input, bool drain) {
  const unsigned hop = engine.delay();
  require(input.size() % hop == 0, "test input is not frame aligned");
  const std::size_t frames = input.size() / hop;
  EngineRun result;
  result.output.reserve(input.size() + (drain ? hop : 0));
  result.mean_gain.reserve(frames + (drain ? 1 : 0));
  std::vector<float> frame(hop);
  for (std::size_t frame_index = 0; frame_index < frames; ++frame_index) {
    std::copy_n(input.data() + frame_index * hop, hop, frame.data());
    engine.process(frame.data());
    result.output.insert(result.output.end(), frame.begin(), frame.end());
    result.mean_gain.push_back(engine.mean_gain());
  }
  if (drain) {
    std::fill(frame.begin(), frame.end(), 0.f);
    engine.process(frame.data());
    result.output.insert(result.output.end(), frame.begin(), frame.end());
    result.mean_gain.push_back(engine.mean_gain());
  }
  return result;
}

void require_finite(const std::vector<float>& values, const std::string& label) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]))
      throw TestFailure(label + " has a non-finite sample at index " + std::to_string(i));
  }
}

void require_bitwise_equal(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           const std::string& label) {
  require(actual.size() == expected.size(), label + " length differs");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    std::uint32_t actual_bits = 0, expected_bits = 0;
    std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
    std::memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
    if (actual_bits != expected_bits)
      throw TestFailure(label + " differs at sample " + std::to_string(i) +
                        " (actual " + std::to_string(actual[i]) +
                        ", expected " + std::to_string(expected[i]) + ")");
  }
}

double rms(const float* samples, std::size_t count) {
  if (count == 0) return 0;
  double sum = 0;
  for (std::size_t i = 0; i < count; ++i) sum += double(samples[i]) * samples[i];
  return std::sqrt(sum / count);
}

std::vector<float> golden_input(unsigned rate, std::size_t frames) {
  const unsigned hop = rate / 100;
  std::vector<float> input(frames * hop);
  XorShift32 random(0x9e3779b9u ^ rate);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (unsigned i = 0; i < hop; ++i) {
      const std::size_t n = frame * hop + i;
      const double time = double(n) / rate;
      float signal = 0.035f * random.bipolar();
      if (frame >= 10 && frame < 30) {
        signal += 0.16f * float(std::sin(2.0 * 3.14159265358979323846 * 440.0 * time));
        signal += 0.07f * float(std::sin(2.0 * 3.14159265358979323846 * 1320.0 * time));
      } else if (frame >= 30 && frame < 38) {
        signal += 0.004f * float(std::sin(2.0 * 3.14159265358979323846 * 660.0 * time));
      }
      if (n == 173) signal += 0.45f;
      input[n] = signal;
    }
  }
  return input;
}

void hash_float(std::uint64_t& hash, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (unsigned shift = 0; shift < 32; shift += 8) {
    hash ^= (bits >> shift) & 0xffu;
    hash *= 1099511628211ULL;
  }
}

std::uint64_t run_digest(const EngineRun& run) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (float value : run.output) hash_float(hash, value);
  for (float value : run.mean_gain) hash_float(hash, value);
  return hash;
}

#ifdef OMLSA_ORIGINAL_GOLDEN
EngineRun run_default_baseline(unsigned rate) {
  itu::detail::OmlsaImcra engine(rate);
  return run_engine(engine, golden_input(rate, 48), true);
}
#endif

#ifndef OMLSA_ORIGINAL_GOLDEN

using itu::detail::OmlsaImcra;
using itu::detail::SpectralGainRule;

template <typename Callable>
void require_invalid_argument(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  } catch (const std::exception& error) {
    throw TestFailure(label + " threw the wrong exception: " + error.what());
  }
  throw TestFailure(label + " did not throw");
}

void test_validation_contract() {
  const itu::OmlsaTuningConfig defaults;
  require(defaults.noise_update_scale == 1.f, "default noise_update_scale changed");
  require(defaults.speech_recovery == 0.f, "default speech_recovery changed");
  require(defaults.release_ms == 60.f, "default release_ms changed");

  struct InvalidTuning {
    const char* label;
    itu::OmlsaTuningConfig tuning;
  };
  const std::array<InvalidTuning, 15> invalid = {{
      {"noise nan", {std::numeric_limits<float>::quiet_NaN(), 0.f, 60.f}},
      {"noise +inf", {std::numeric_limits<float>::infinity(), 0.f, 60.f}},
      {"noise -inf", {-std::numeric_limits<float>::infinity(), 0.f, 60.f}},
      {"noise low", {0.099f, 0.f, 60.f}},
      {"noise high", {1.001f, 0.f, 60.f}},
      {"speech nan", {1.f, std::numeric_limits<float>::quiet_NaN(), 60.f}},
      {"speech +inf", {1.f, std::numeric_limits<float>::infinity(), 60.f}},
      {"speech -inf", {1.f, -std::numeric_limits<float>::infinity(), 60.f}},
      {"speech low", {1.f, -0.001f, 60.f}},
      {"speech high", {1.f, 1.001f, 60.f}},
      {"release nan", {1.f, 0.f, std::numeric_limits<float>::quiet_NaN()}},
      {"release +inf", {1.f, 0.f, std::numeric_limits<float>::infinity()}},
      {"release -inf", {1.f, 0.f, -std::numeric_limits<float>::infinity()}},
      {"release low", {1.f, 0.f, -0.001f}},
      {"release high", {1.f, 0.f, 200.001f}},
  }};
  for (const InvalidTuning& item : invalid) {
    require_invalid_argument(
        [&] { OmlsaImcra engine(16000, -18.f, true, SpectralGainRule::omlsa, item.tuning); },
        item.label);
  }

  const std::array<float, 5> invalid_floors = {
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(), -40.001f, 0.001f};
  for (float floor_db : invalid_floors)
    require_invalid_argument([&] { OmlsaImcra engine(16000, floor_db); }, "invalid floor_db");

  for (unsigned rate : {0u, 8000u, 44100u})
    require_invalid_argument([&] { OmlsaImcra engine(rate); }, "invalid sample rate");

  OmlsaImcra floor_low(16000, -40.f, true, SpectralGainRule::omlsa, {0.1f, 0.f, 0.f});
  OmlsaImcra floor_high(16000, 0.f, true, SpectralGainRule::omlsa, {1.f, 1.f, 200.f});
  require(floor_low.delay() == 160, "floor lower-bound engine has wrong delay");
  require(floor_high.delay() == 160, "floor upper-bound engine has wrong delay");

  require_invalid_argument(
      [&] {
        OmlsaImcra engine(16000, -18.f, true, SpectralGainRule::super_gaussian_map,
                          {0.5f, 0.f, 60.f});
      },
      "noise tuning with super_gaussian_map");
  require_invalid_argument(
      [&] {
        OmlsaImcra engine(16000, -18.f, true, SpectralGainRule::super_gaussian_map,
                          {1.f, 0.2f, 60.f});
      },
      "speech recovery with super_gaussian_map");

  OmlsaImcra disabled_super_gaussian(16000, -18.f, true,
                                     SpectralGainRule::super_gaussian_map, {1.f, 0.f, 200.f});
  require(disabled_super_gaussian.delay() == 160,
          "disabled tuning should preserve the super-Gaussian baseline path");
  std::cout << "PASS validation: NaN/Inf/out-of-range rejected; legal boundaries accepted\n";
}

void test_floor_zero_is_unity_wola() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const unsigned hop = rate / 100;
    constexpr std::size_t frames = 37;
    std::vector<float> input(frames * hop);
    XorShift32 random(0x12345678u ^ rate);
    for (std::size_t n = 0; n < input.size(); ++n) {
      const double time = double(n) / rate;
      input[n] = 0.31f * random.bipolar() +
                 0.27f * float(std::sin(2.0 * 3.14159265358979323846 * 317.0 * time));
    }
    input[hop + 7] += 0.5f;
    OmlsaImcra engine(rate, 0.f, true, SpectralGainRule::omlsa, {0.5f, 0.75f, 120.f});
    EngineRun run = run_engine(engine, input, true);
    require(engine.delay() == hop, "unity WOLA delay is not exactly one hop");
    require(run.output.size() == input.size() + hop, "unity WOLA drain length is wrong");
    double max_error = 0;
    for (std::size_t i = 0; i < input.size(); ++i)
      max_error = std::max(max_error, std::abs(double(run.output[hop + i] - input[i])));
    for (unsigned i = 0; i < hop; ++i)
      max_error = std::max(max_error, std::abs(double(run.output[hop + input.size() - hop + i] -
                                                        input[input.size() - hop + i])));
    require_finite(run.output, "floor-zero WOLA output");
    require(max_error < 2e-6, "floor-zero WOLA error exceeds 2e-6 at rate " +
                                  std::to_string(rate) + ": " + std::to_string(max_error));
    std::cout << "PASS unity WOLA rate=" << rate << " delay=" << engine.delay()
              << " maxerr=" << std::setprecision(10) << max_error << '\n';
  }
}

void test_zero_and_denormal_inputs_stay_finite() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const unsigned hop = rate / 100;
    OmlsaImcra zero_engine(rate, -18.f, true, SpectralGainRule::omlsa, {0.35f, 0.4f, 75.f});
    EngineRun zero = run_engine(zero_engine, std::vector<float>(32 * hop, 0.f), true);
    require_finite(zero.output, "all-zero input output");
    require(std::all_of(zero.output.begin(), zero.output.end(),
                        [](float x) { return x == 0.f; }),
            "all-zero input should remain exactly zero");

    std::vector<float> mixed(45 * hop);
    XorShift32 random(0xdeadbeefu ^ rate);
    const float denormal = std::numeric_limits<float>::denorm_min();
    for (std::size_t n = 0; n < mixed.size(); ++n) {
      if (n % 11 == 0)
        mixed[n] = 0.f;
      else if (n % 17 == 0)
        mixed[n] = (n & 1) ? denormal : -denormal;
      else if (n % 19 == 0)
        mixed[n] = (n & 1) ? std::numeric_limits<float>::min()
                           : -std::numeric_limits<float>::min();
      else
        mixed[n] = 0.8f * random.bipolar();
    }
    OmlsaImcra mixed_engine(rate, -18.f, true, SpectralGainRule::omlsa, {0.1f, 1.f, 0.f});
    EngineRun mixed_run = run_engine(mixed_engine, mixed, true);
    require_finite(mixed_run.output, "denormal/random mixed output");
    require(std::all_of(mixed_run.mean_gain.begin(), mixed_run.mean_gain.end(),
                        [](float x) { return std::isfinite(x) && x >= 0.f && x <= 1.f + 1e-6f; }),
            "denormal/random mixed input produced an invalid mean gain");
    std::cout << "PASS finite zero/denormal/random rate=" << rate << '\n';
  }
}

void test_disabled_path() {
  const std::array<unsigned, 3> rates = {16000u, 24000u, 48000u};
  for (std::size_t rate_index = 0; rate_index < rates.size(); ++rate_index) {
    const unsigned rate = rates[rate_index];
    const std::vector<float> input = golden_input(rate, 48);
    OmlsaImcra default_engine(rate);
    EngineRun default_run = run_engine(default_engine, input, true);
    // Floating-point FFT digests depend on the platform and compiler. Compare
    // default and explicitly disabled paths in this build; the paired probe
    // separately verifies against the original header on the same platform.

    for (float release_ms : {0.f, 60.f, 200.f}) {
      itu::OmlsaTuningConfig explicit_disabled{1.f, 0.f, release_ms};
      OmlsaImcra explicit_engine(rate, -18.f, true, SpectralGainRule::omlsa,
                                 explicit_disabled);
      EngineRun explicit_run = run_engine(explicit_engine, input, true);
      require_bitwise_equal(explicit_run.output, default_run.output,
                            "explicit-disabled output rate " + std::to_string(rate));
      require_bitwise_equal(explicit_run.mean_gain, default_run.mean_gain,
                            "explicit-disabled gain rate " + std::to_string(rate));
    }
    std::cout << "PASS default/explicit-disabled rate=" << rate
              << " digest=0x" << std::hex << std::setw(16) << std::setfill('0')
              << run_digest(default_run) << std::dec << std::setfill(' ') << '\n';
  }
}

void test_causality_prefix_invariance() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  constexpr std::size_t prefix_frames = 43;
  constexpr std::size_t future_frames = 19;
  std::vector<float> prefix = golden_input(rate, prefix_frames);
  std::vector<float> future_a(future_frames * hop), future_b(future_frames * hop);
  XorShift32 random_a(0xabcdu), random_b(0x1234u);
  for (std::size_t n = 0; n < future_a.size(); ++n) {
    const double time = double(n) / rate;
    future_a[n] = 0.2f * random_a.bipolar() +
                  0.3f * float(std::sin(2.0 * 3.14159265358979323846 * 880.0 * time));
    future_b[n] = 0.05f * random_b.bipolar() -
                  0.25f * float(std::sin(2.0 * 3.14159265358979323846 * 233.0 * time));
  }
  OmlsaImcra engine_a(rate, -18.f, true, SpectralGainRule::omlsa, {0.45f, 0.65f, 90.f});
  OmlsaImcra engine_b(rate, -18.f, true, SpectralGainRule::omlsa, {0.45f, 0.65f, 90.f});
  EngineRun prefix_a = run_engine(engine_a, prefix, false);
  EngineRun prefix_b = run_engine(engine_b, prefix, false);
  require_bitwise_equal(prefix_a.output, prefix_b.output, "equal-prefix output");
  run_engine(engine_a, future_a, false);
  run_engine(engine_b, future_b, false);
  require_bitwise_equal(prefix_a.output, prefix_b.output,
                        "prefix output changed after processing different futures");
  std::cout << "PASS causality: " << prefix_frames << " common frames unchanged by differing futures\n";
}

void test_noise_only_suppression() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  constexpr std::size_t frames = 600;
  std::vector<float> input(frames * hop);
  XorShift32 random(0x51ed270bu);
  for (float& sample : input) sample = 0.04f * random.bipolar();

  OmlsaImcra engine(rate, -18.f, true, SpectralGainRule::omlsa, {0.25f, 0.5f, 150.f});
  EngineRun run = run_engine(engine, input, true);
  require_finite(run.output, "600-frame noise-only output");
  require(run.output.size() == input.size() + hop, "600-frame noise output has wrong drain length");
  const double input_rms = rms(input.data(), input.size());
  const double output_rms = rms(run.output.data() + hop, input.size());
  const double ratio = output_rms / input_rms;
  require(ratio <= 1.0001, "noise-only average was amplified: ratio=" + std::to_string(ratio));
  const auto min_gain = std::min_element(run.mean_gain.begin(), run.mean_gain.end());
  require(min_gain != run.mean_gain.end() && std::isfinite(*min_gain),
          "noise-only gain trace is empty or non-finite");
  require(*min_gain < 0.999f, "noise-only test produced a vacuous all-unity gain");
  std::cout << "PASS 600-frame noise seed=0x51ed270b input_rms=" << std::setprecision(9)
            << input_rms << " output_rms=" << output_rms << " ratio=" << ratio
            << " min_mean_gain=" << *min_gain << '\n';
}

void test_speech_recovery_changes_near_end_tail() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  constexpr std::size_t frames = 76;
  constexpr std::size_t burst_begin = 14;
  constexpr std::size_t tail_begin = 34;
  constexpr std::size_t tail_end = 56;
  std::vector<float> input(frames * hop);
  XorShift32 random(0x600df00du);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (unsigned i = 0; i < hop; ++i) {
      const std::size_t n = frame * hop + i;
      const double time = double(n) / rate;
      float signal = 0.004f * random.bipolar();
      if (frame >= burst_begin && frame < tail_begin) {
        signal += 0.16f * float(std::sin(2.0 * 3.14159265358979323846 * 440.0 * time));
        signal += 0.09f * float(std::sin(2.0 * 3.14159265358979323846 * 880.0 * time));
        signal += 0.05f * float(std::sin(2.0 * 3.14159265358979323846 * 1320.0 * time));
      } else if (frame >= tail_begin && frame < tail_end) {
        signal += 0.0025f * float(std::sin(2.0 * 3.14159265358979323846 * 660.0 * time));
      }
      input[n] = signal;
    }
  }
  OmlsaImcra baseline(rate, -18.f, true, SpectralGainRule::omlsa, {1.f, 0.f, 150.f});
  OmlsaImcra recovered(rate, -18.f, true, SpectralGainRule::omlsa, {1.f, 0.8f, 150.f});
  EngineRun baseline_run = run_engine(baseline, input, true);
  EngineRun recovered_run = run_engine(recovered, input, true);
  require_finite(recovered_run.output, "speech-recovery output");
  double max_difference = 0;
  std::size_t differing_samples = 0;
  for (std::size_t i = 0; i < baseline_run.output.size(); ++i) {
    const double difference = std::abs(double(recovered_run.output[i] - baseline_run.output[i]));
    max_difference = std::max(max_difference, difference);
    differing_samples += difference != 0;
  }
  require(differing_samples > 0, "speech recovery never changed output");
  const std::size_t tail_offset = hop + tail_begin * hop;
  const std::size_t tail_count = (tail_end - tail_begin) * hop;
  const std::size_t noise_offset = hop + tail_end * hop;
  const std::size_t noise_count = (frames - tail_end) * hop;
  const double tail_input_rms = rms(input.data() + tail_begin * hop, tail_count);
  const double tail_baseline_rms = rms(baseline_run.output.data() + tail_offset, tail_count);
  const double tail_recovered_rms = rms(recovered_run.output.data() + tail_offset, tail_count);
  const double noise_input_rms = rms(input.data() + tail_end * hop, noise_count);
  const double noise_baseline_rms = rms(baseline_run.output.data() + noise_offset, noise_count);
  const double noise_recovered_rms = rms(recovered_run.output.data() + noise_offset, noise_count);
  std::cout << "PASS speech recovery triggered differing_samples=" << differing_samples
            << " maxdiff=" << std::setprecision(9) << max_difference
            << " tail_input_rms=" << tail_input_rms
            << " tail_baseline_rms=" << tail_baseline_rms
            << " tail_recovered_rms=" << tail_recovered_rms
            << " noise_input_rms=" << noise_input_rms
            << " noise_baseline_rms=" << noise_baseline_rms
            << " noise_recovered_rms=" << noise_recovered_rms << '\n';
}

void test_process_is_allocation_free() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    OmlsaImcra engine(rate, -18.f, true, SpectralGainRule::omlsa, {0.35f, 0.7f, 120.f});
    std::vector<float> frame(engine.delay());
    XorShift32 random(rate);
    for (float& sample : frame) sample = 0.1f * random.bipolar();
    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    for (unsigned iteration = 0; iteration < 8; ++iteration) engine.process(frame.data());
    count_allocations.store(false, std::memory_order_relaxed);
    const std::size_t allocations = allocation_count.load(std::memory_order_relaxed);
    require(allocations == 0, "process allocated " + std::to_string(allocations) +
                                  " times after construction at rate " + std::to_string(rate));
    std::cout << "PASS allocation-free process rate=" << rate << '\n';
  }
}

#endif  // OMLSA_ORIGINAL_GOLDEN

}  // namespace

int main() {
#ifdef OMLSA_ORIGINAL_GOLDEN
  const std::array<unsigned, 3> rates = {16000u, 24000u, 48000u};
  for (unsigned rate : rates) {
    std::cout << rate << " 0x" << std::hex << std::setw(16) << std::setfill('0')
              << run_digest(run_default_baseline(rate)) << std::dec << '\n';
  }
  return 0;
#else
  try {
    test_validation_contract();
    test_floor_zero_is_unity_wola();
    test_zero_and_denormal_inputs_stay_finite();
    test_disabled_path();
    test_causality_prefix_invariance();
    test_noise_only_suppression();
    test_speech_recovery_changes_near_end_tail();
    test_process_is_allocation_free();
    std::cout << "ALL OMLSA TUNING TESTS PASSED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
#endif
}
