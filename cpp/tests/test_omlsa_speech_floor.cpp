#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "itu/omlsa_config.hpp"
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

itu::OmlsaTuningConfig make_tuning(float noise_update_scale = 1.f,
                                   float speech_recovery = 0.f,
                                   float release_ms = 60.f,
                                   float speech_floor_db = -40.f) {
  itu::OmlsaTuningConfig tuning;
  tuning.noise_update_scale = noise_update_scale;
  tuning.speech_recovery = speech_recovery;
  tuning.release_ms = release_ms;
  tuning.speech_floor_db = speech_floor_db;
  return tuning;
}

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

bool same_bits(float left, float right) {
  std::uint32_t left_bits = 0, right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

void require_bitwise_equal(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           const std::string& label) {
  require(actual.size() == expected.size(), label + " length differs");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!same_bits(actual[i], expected[i]))
      throw TestFailure(label + " differs at sample " + std::to_string(i));
  }
}

void require_pointer_bits(const float* actual, const float* expected,
                          unsigned count, const std::string& label) {
  for (unsigned i = 0; i < count; ++i) {
    if (!same_bits(actual[i], expected[i]))
      throw TestFailure(label + " differs at bin " + std::to_string(i));
  }
}

void require_finite(const std::vector<float>& values, const std::string& label) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]))
      throw TestFailure(label + " has a non-finite sample at index " +
                        std::to_string(i));
  }
}

void require_finite_frame(const itu::detail::NsSpectralFrame& frame,
                          const std::string& label) {
  for (unsigned i = 0; i < frame.bins; ++i) {
    if (!std::isfinite(frame.power[i]) || !std::isfinite(frame.noise[i]) ||
        !std::isfinite(frame.xi[i]) || !std::isfinite(frame.gamma[i]) ||
        !std::isfinite(frame.probability[i]) || !std::isfinite(frame.gain[i]))
      throw TestFailure(label + " has a non-finite spectral value at bin " +
                        std::to_string(i));
    if (frame.gain[i] < 0.f || frame.gain[i] > 1.f)
      throw TestFailure(label + " gain is outside [0,1] at bin " +
                        std::to_string(i));
  }
}

struct EngineRun {
  std::vector<float> output;
  std::vector<float> mean_gain;
};

EngineRun run_engine(itu::detail::OmlsaImcra& engine,
                     const std::vector<float>& input) {
  const unsigned hop = engine.delay();
  require(input.size() % hop == 0, "input is not frame aligned");
  const std::size_t frames = input.size() / hop;
  EngineRun result;
  result.output.reserve(input.size());
  result.mean_gain.reserve(frames);
  std::vector<float> frame(hop);
  for (std::size_t frame_index = 0; frame_index < frames; ++frame_index) {
    std::copy_n(input.data() + frame_index * hop, hop, frame.data());
    engine.process(frame.data());
    result.output.insert(result.output.end(), frame.begin(), frame.end());
    result.mean_gain.push_back(engine.mean_gain());
  }
  return result;
}

std::vector<float> probe_input(unsigned rate, std::size_t frames) {
  const unsigned hop = rate / 100;
  std::vector<float> input(frames * hop);
  XorShift32 random(0x7f4a7c15u ^ rate);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (unsigned i = 0; i < hop; ++i) {
      const std::size_t n = frame * hop + i;
      const double time = double(n) / rate;
      float sample = 0.03f * random.bipolar();
      if (frame >= 9 && frame < 25) {
        sample += 0.15f * float(std::sin(
            2.0 * 3.14159265358979323846 * 440.0 * time));
        sample += 0.07f * float(std::sin(
            2.0 * 3.14159265358979323846 * 1320.0 * time));
      }
      if (n == 173) sample += 0.4f;
      input[n] = sample;
    }
  }
  return input;
}

void test_validation_and_disabled_path() {
  const itu::OmlsaTuningConfig defaults;
  require(defaults.speech_floor_db == -40.f,
          "default speech_floor_db is not -40 dB");

  const std::array<float, 5> invalid_floors = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(), -40.001f, 0.001f};
  for (float floor_db : invalid_floors) {
    require_invalid_argument(
        [&] {
          itu::detail::OmlsaImcra engine(
              16000, -18.f, true, itu::detail::SpectralGainRule::omlsa,
              make_tuning(1.f, 0.f, 60.f, floor_db));
        },
        "invalid speech_floor_db");
  }

  itu::detail::OmlsaImcra disabled_boundary(
      16000, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      make_tuning(1.f, 0.f, 60.f, -40.f));
  itu::detail::OmlsaImcra enabled_boundary(
      16000, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      make_tuning(1.f, 0.f, 60.f, 0.f));
  require(disabled_boundary.delay() == 160 && enabled_boundary.delay() == 160,
          "legal speech floor boundary was not accepted");

  std::array<itu::OmlsaTuningConfig, 3> enabled_tunings = {
      make_tuning(0.5f, 0.f, 60.f, -40.f),
      make_tuning(1.f, 0.2f, 60.f, -40.f),
      make_tuning(1.f, 0.f, 60.f, -6.f)};
  for (const itu::OmlsaTuningConfig& tuning : enabled_tunings) {
    require_invalid_argument(
        [&] {
          itu::detail::OmlsaImcra engine(
              16000, -18.f, true,
              itu::detail::SpectralGainRule::super_gaussian_map, tuning);
        },
        "non-OM-LSA enabled tuning");
  }

  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const std::vector<float> input = probe_input(rate, 48);
    itu::detail::OmlsaImcra default_engine(rate);
    itu::detail::OmlsaImcra explicit_disabled(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        make_tuning(1.f, 0.f, 60.f, -40.f));
    const EngineRun default_run = run_engine(default_engine, input);
    const EngineRun disabled_run = run_engine(explicit_disabled, input);
    require_bitwise_equal(disabled_run.output, default_run.output,
                          "explicit -40 output rate " + std::to_string(rate));
    require_bitwise_equal(disabled_run.mean_gain, default_run.mean_gain,
                          "explicit -40 gain rate " + std::to_string(rate));
  }
  std::cout << "PASS validation and bit-exact explicit -40 path\n";
}

void test_noise_only_is_not_uniformly_lifted() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  constexpr std::size_t frames = 600;
  itu::detail::OmlsaImcra baseline(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      make_tuning(1.f, 0.f, 60.f, -40.f));
  itu::detail::OmlsaImcra floored(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      make_tuning(1.f, 0.f, 60.f, -6.f));
  XorShift32 random(0x51ed270bu);
  std::vector<float> original(hop), baseline_frame(hop), floored_frame(hop);
  double input_energy = 0, output_energy = 0;
  float minimum_mean_gain = 1.f;
  std::size_t observed_bins = 0, lifted_bins = 0;

  for (std::size_t frame_index = 0; frame_index < frames; ++frame_index) {
    for (float& sample : original) {
      sample = 0.04f * random.bipolar();
      input_energy += double(sample) * sample;
    }
    baseline_frame = original;
    floored_frame = original;
    baseline.process(baseline_frame.data());
    floored.process(floored_frame.data());
    const auto baseline_spectrum = baseline.spectral_frame();
    const auto floored_spectrum = floored.spectral_frame();
    require_finite_frame(floored_spectrum, "noise-only floor frame");
    require_pointer_bits(floored_spectrum.noise, baseline_spectrum.noise,
                         baseline_spectrum.bins,
                         "noise-only estimator");
    require_pointer_bits(floored_spectrum.gamma, baseline_spectrum.gamma,
                         baseline_spectrum.bins,
                         "noise-only gamma");
    require_pointer_bits(floored_spectrum.xi, baseline_spectrum.xi,
                         baseline_spectrum.bins,
                         "noise-only xi");
    for (unsigned k = 0; k < baseline_spectrum.bins; ++k) {
      require(floored_spectrum.gain[k] >= baseline_spectrum.gain[k],
              "speech floor reduced a noise-only gain");
      ++observed_bins;
      lifted_bins += !same_bits(floored_spectrum.gain[k],
                                baseline_spectrum.gain[k]);
    }
    for (float sample : floored_frame) output_energy += double(sample) * sample;
    minimum_mean_gain = std::min(minimum_mean_gain, floored.mean_gain());
  }

  require(lifted_bins < observed_bins,
          "speech floor uniformly lifted noise-only gains");
  require(minimum_mean_gain < 0.5f,
          "noise-only suppression collapsed under the speech floor");
  const double output_ratio = std::sqrt(output_energy / input_energy);
  require(std::isfinite(output_ratio) && output_ratio <= 1.0001,
          "noise-only output was amplified");
  std::cout << "PASS 600-frame noise-only floor lifted=" << lifted_bins
            << "/" << observed_bins << " min_mean_gain=" << minimum_mean_gain
            << " rms_ratio=" << output_ratio << '\n';
}

void test_speech_continuation_never_lowers_gain() {
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
      float sample = 0.004f * random.bipolar();
      if (frame >= burst_begin && frame < tail_begin) {
        sample += 0.16f * float(std::sin(
            2.0 * 3.14159265358979323846 * 440.0 * time));
        sample += 0.09f * float(std::sin(
            2.0 * 3.14159265358979323846 * 880.0 * time));
        sample += 0.05f * float(std::sin(
            2.0 * 3.14159265358979323846 * 1320.0 * time));
      } else if (frame >= tail_begin && frame < tail_end) {
        sample += 0.0025f * float(std::sin(
            2.0 * 3.14159265358979323846 * 660.0 * time));
      }
      input[n] = sample;
    }
  }

  itu::detail::OmlsaImcra baseline(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      make_tuning(1.f, 0.f, 60.f, -40.f));
  itu::detail::OmlsaImcra floored(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      make_tuning(1.f, 0.f, 60.f, -6.f));
  std::vector<float> baseline_output, floored_output;
  baseline_output.reserve(input.size());
  floored_output.reserve(input.size());
  std::vector<float> original(hop), baseline_frame(hop), floored_frame(hop);
  std::size_t differing_bins = 0, differing_samples = 0;
  for (std::size_t frame_index = 0; frame_index < frames; ++frame_index) {
    std::copy_n(input.data() + frame_index * hop, hop, original.begin());
    baseline_frame = original;
    floored_frame = original;
    baseline.process(baseline_frame.data());
    floored.process(floored_frame.data());
    const auto baseline_spectrum = baseline.spectral_frame();
    const auto floored_spectrum = floored.spectral_frame();
    require_finite_frame(floored_spectrum, "speech continuation frame");
    require_pointer_bits(floored_spectrum.noise, baseline_spectrum.noise,
                         baseline_spectrum.bins,
                         "speech continuation estimator");
    require_pointer_bits(floored_spectrum.gamma, baseline_spectrum.gamma,
                         baseline_spectrum.bins,
                         "speech continuation gamma");
    require_pointer_bits(floored_spectrum.xi, baseline_spectrum.xi,
                         baseline_spectrum.bins,
                         "speech continuation xi");
    for (unsigned k = 0; k < baseline_spectrum.bins; ++k) {
      require(floored_spectrum.gain[k] >= baseline_spectrum.gain[k],
              "speech floor lowered a gain below baseline");
      differing_bins += !same_bits(floored_spectrum.gain[k],
                                   baseline_spectrum.gain[k]);
    }
    for (unsigned i = 0; i < hop; ++i)
      differing_samples += !same_bits(floored_frame[i], baseline_frame[i]);
    baseline_output.insert(baseline_output.end(), baseline_frame.begin(),
                           baseline_frame.end());
    floored_output.insert(floored_output.end(), floored_frame.begin(),
                          floored_frame.end());
  }
  require_finite(baseline_output, "speech continuation baseline output");
  require_finite(floored_output, "speech continuation floored output");
  require(differing_bins > 0, "speech floor never found continuation evidence");
  require(differing_samples > 0, "speech floor never changed synthesis output");
  std::cout << "PASS speech continuation synthesis-only changed_bins="
            << differing_bins << " changed_samples=" << differing_samples << '\n';
}

void test_zero_silent_and_startup_are_finite() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const unsigned hop = rate / 100;
    itu::detail::OmlsaImcra zero_engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        make_tuning(1.f, 0.f, 60.f, 0.f));
    std::vector<float> frame(hop, 0.f);
    for (unsigned frame_index = 0; frame_index < 32; ++frame_index) {
      zero_engine.process(frame.data());
      require_finite(frame, "silent input output");
      require_finite_frame(zero_engine.spectral_frame(), "silent spectral frame");
      require(std::all_of(frame.begin(), frame.end(),
                          [](float sample) { return sample == 0.f; }),
              "silent input did not remain exactly zero");
    }

    itu::detail::OmlsaImcra startup_engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        make_tuning(1.f, 0.f, 60.f, 0.f));
    XorShift32 random(0x9e3779b9u ^ rate);
    for (unsigned frame_index = 0; frame_index < 24; ++frame_index) {
      for (unsigned i = 0; i < hop; ++i) {
        const double time = double(frame_index * hop + i) / rate;
        frame[i] = 0.3f * random.bipolar() + 0.25f * float(std::sin(
            2.0 * 3.14159265358979323846 * 173.0 * time));
      }
      if (frame_index == 0) frame[7] += 0.8f;
      startup_engine.process(frame.data());
      require_finite(frame, "startup input output");
      require_finite_frame(startup_engine.spectral_frame(),
                           "startup spectral frame");
      require(std::isfinite(startup_engine.mean_gain()) &&
                  startup_engine.mean_gain() >= 0.f &&
                  startup_engine.mean_gain() <= 1.f,
              "startup mean gain is invalid");
    }
  }
  std::cout << "PASS zero/silent and startup frames stay finite\n";
}

void test_process_is_allocation_free() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    itu::detail::OmlsaImcra engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        make_tuning(1.f, 0.f, 60.f, -6.f));
    std::vector<float> frame(engine.delay());
    XorShift32 random(rate);
    for (float& sample : frame) sample = 0.1f * random.bipolar();
    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
    for (unsigned iteration = 0; iteration < 12; ++iteration)
      engine.process(frame.data());
    count_allocations.store(false, std::memory_order_relaxed);
    const std::size_t allocations =
        allocation_count.load(std::memory_order_relaxed);
    require(allocations == 0,
            "process allocated " + std::to_string(allocations) +
                " times at rate " + std::to_string(rate));
  }
  std::cout << "PASS allocation-free process with speech floor\n";
}

}  // namespace

int main() {
  try {
    test_validation_and_disabled_path();
    test_noise_only_is_not_uniformly_lifted();
    test_speech_continuation_never_lowers_gain();
    test_zero_silent_and_startup_are_finite();
    test_process_is_allocation_free();
    std::cout << "ALL OMLSA SPEECH FLOOR TESTS PASSED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
