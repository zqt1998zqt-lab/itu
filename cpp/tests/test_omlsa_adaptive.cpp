#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OMLSA_ORIGINAL_GOLDEN
#include "itu/omlsa_config.hpp"
#endif
#include "omlsa_adaptive_control.hpp"
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

template <typename Callable>
void require_invalid_argument(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  } catch (const std::exception& error) {
    throw TestFailure(label + " threw the wrong exception: " + error.what());
  }
  throw TestFailure(label + " was accepted");
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

  float unit() { return (static_cast<float>(next() >> 8) + .5f) / 16777216.f; }

 private:
  std::uint32_t state_;
};

bool same_bits(float left, float right) {
  std::uint32_t left_bits = 0, right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

double rms(const float* samples, std::size_t count) {
  double energy = 0;
  for (std::size_t i = 0; i < count; ++i)
    energy += double(samples[i]) * samples[i];
  return count ? std::sqrt(energy / count) : 0.;
}

void require_finite(const std::vector<float>& values, const std::string& label) {
  for (std::size_t i = 0; i < values.size(); ++i)
    require(std::isfinite(values[i]), label + " non-finite sample=" + std::to_string(i));
}

void require_bitwise_equal(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           const std::string& label) {
  require(actual.size() == expected.size(), label + " size differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(same_bits(actual[i], expected[i]),
            label + " differs at index " + std::to_string(i));
}

itu::OmlsaTuningConfig mild_tuning() {
  itu::OmlsaTuningConfig tuning;
  tuning.noise_update_scale = .5f;
  tuning.speech_recovery = .75f;
  tuning.release_ms = 120.f;
  return tuning;
}

#ifndef OMLSA_ORIGINAL_GOLDEN

itu::OmlsaTuningConfig adaptive_tuning(bool all_features = false) {
  itu::OmlsaTuningConfig tuning;
  tuning.adaptive_noise_tracking = true;
  tuning.high_snr_transparency = all_features;
  tuning.low_band_protection = all_features;
  return tuning;
}

struct EngineRun {
  std::vector<float> output;
  std::vector<float> mean_gain;
};

EngineRun run_engine(itu::detail::OmlsaImcra& engine,
                     const std::vector<float>& input,
                     bool count_process_allocations = false) {
  const unsigned hop = engine.delay();
  require(input.size() % hop == 0, "scenario is not 10 ms frame aligned");
  EngineRun result;
  result.output.reserve(input.size());
  result.mean_gain.reserve(input.size() / hop);
  std::vector<float> frame(hop);
  if (count_process_allocations) {
    allocation_count.store(0, std::memory_order_relaxed);
    count_allocations.store(true, std::memory_order_relaxed);
  }
  for (std::size_t offset = 0; offset < input.size(); offset += hop) {
    std::copy_n(input.data() + offset, hop, frame.begin());
    engine.process(frame.data());
    result.output.insert(result.output.end(), frame.begin(), frame.end());
    result.mean_gain.push_back(engine.mean_gain());
  }
  if (count_process_allocations) {
    count_allocations.store(false, std::memory_order_relaxed);
    const std::size_t allocations =
        allocation_count.load(std::memory_order_relaxed);
    require(allocations == 0,
            "process allocated " + std::to_string(allocations) + " times");
  }
  return result;
}

std::vector<float> mixed_input(unsigned rate, std::size_t frames) {
  const unsigned hop = rate / 100;
  std::vector<float> input(frames * hop);
  XorShift32 random(0x6a09e667u ^ rate);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const float noise_gain = frame < 110 ? .025f : .08f;
    for (unsigned i = 0; i < hop; ++i) {
      const std::size_t n = frame * hop + i;
      const double time = double(n) / rate;
      float sample = noise_gain * random.bipolar();
      if (frame >= 40 && frame < 80) {
        sample += .14f * float(std::sin(2.0 * 3.14159265358979323846 * 440.0 * time));
        sample += .08f * float(std::sin(2.0 * 3.14159265358979323846 * 1320.0 * time));
      } else if (frame >= 80 && frame < 95) {
        sample += .003f * float(std::sin(2.0 * 3.14159265358979323846 * 660.0 * time));
      }
      input[n] = sample;
    }
  }
  return input;
}

std::vector<float> tone_input(unsigned rate, std::size_t frames) {
  const unsigned hop = rate / 100;
  std::vector<float> input(frames * hop);
  for (std::size_t n = 0; n < input.size(); ++n)
    input[n] = .12f * float(std::sin(
        2.0 * 3.14159265358979323846 * 440.0 * double(n) / rate));
  return input;
}

void test_controller_pause_and_speech() {
  constexpr unsigned rate = 24000;
  constexpr unsigned fft = 512;
  constexpr unsigned bins = fft / 2 + 1;
  itu::detail::OmlsaAdaptiveControl control(rate, fft, bins);
  XorShift32 random(0xbb67ae85u);
  std::vector<float> power(bins);
  std::array<float, 160> speech{};
  std::array<float, 160> pause{};
  for (unsigned frame = 0; frame < speech.size(); ++frame) {
    for (unsigned k = 0; k < bins; ++k) {
      const float texture = .75f + .5f * random.unit();
      power[k] = .006f * texture;
    }
    // Ground truth is local to this test. Only the power vector is observed.
    if (frame >= 60 && frame < 90) {
      for (unsigned k = 18; k <= 72; k += 6) power[k] += .10f;
    } else if (frame >= 90 && frame < 100) {
      for (unsigned k = 24; k <= 54; k += 6) power[k] += .0015f;
    }
    control.observe(power.data());
    speech[frame] = control.speech();
    pause[frame] = control.pause();
  }
  float noise_pause = 0;
  for (unsigned frame = 48; frame < 60; ++frame)
    noise_pause = std::max(noise_pause, pause[frame]);
  float onset_speech = 0, onset_pause = 1;
  for (unsigned frame = 60; frame < 64; ++frame) {
    onset_speech = std::max(onset_speech, speech[frame]);
    onset_pause = std::min(onset_pause, pause[frame]);
  }
  float recovered_pause = 0;
  for (unsigned frame = 140; frame < 160; ++frame)
    recovered_pause = std::max(recovered_pause, pause[frame]);
  require(noise_pause > .7f, "stable noise never raised pause: " +
                                  std::to_string(noise_pause));
  require(onset_speech > .3f, "harmonic energy did not raise speech: " +
                                  std::to_string(onset_speech));
  require(onset_pause < .2f, "speech did not rapidly clear pause: " +
                                 std::to_string(onset_pause));
  require(recovered_pause > .7f, "pause did not recover after speech: " +
                                     std::to_string(recovered_pause));
  std::cout << "PASS controller stable_pause=" << noise_pause
            << " onset_speech=" << onset_speech
            << " onset_pause=" << onset_pause
            << " recovered_pause=" << recovered_pause << '\n';
}

void test_controller_narrow_tone_continuity() {
  constexpr unsigned rate = 24000;
  constexpr unsigned fft = 512;
  constexpr unsigned bins = fft / 2 + 1;
  constexpr unsigned cold_bin = 20;
  constexpr unsigned other_bin = 80;
  constexpr float floor_power = 1e-6f;
  constexpr float tone_power = 1.f;

  auto tone_frame = [&](std::vector<float>& power, unsigned peak_bin) {
    power.assign(bins, floor_power);
    power[peak_bin] = tone_power;
  };
  auto flat_frame = [&](std::vector<float>& power) {
    power.assign(bins, 1e-3f);
  };
  auto noise_frame = [&](std::vector<float>& power, XorShift32& random) {
    power.resize(bins);
    for (float& value : power) value = 1e-4f * (.25f + random.unit());
  };
  auto silent_frame = [&](std::vector<float>& power) {
    power.assign(bins, 0.f);
  };

  itu::detail::OmlsaAdaptiveControl continuous(rate, fft, bins);
  std::vector<float> power(bins);
  tone_frame(power, cold_bin);
  for (unsigned frame = 0; frame < 30; ++frame) continuous.observe(power.data());
  require(continuous.narrow_tone_protection(cold_bin) == .98f,
          "continuous cold tone center was not protected");
  require(continuous.narrow_tone_protection(cold_bin + 2) == .98f &&
              continuous.narrow_tone_protection(cold_bin - 2) == .98f,
          "continuous cold tone +/-2 bins were not protected");
  require(continuous.narrow_tone_protection(cold_bin + 3) == .70f &&
              continuous.narrow_tone_protection(cold_bin - 3) == .70f,
          "continuous cold tone +/-3 bins lacked shoulder protection");
  require(continuous.narrow_tone_protection(cold_bin + 4) == 0.f,
          "continuous cold tone protection leaked beyond +/-3 bins");

  for (unsigned interruption = 0; interruption < 3; ++interruption) {
    itu::detail::OmlsaAdaptiveControl control(rate, fft, bins);
    XorShift32 random(0x9e3779b9u + interruption);
    tone_frame(power, cold_bin);
    for (unsigned frame = 0; frame < 4; ++frame) control.observe(power.data());
    require(control.narrow_tone_protection(cold_bin) == .98f,
            "tone was not locked before interruption");
    if (interruption == 0)
      flat_frame(power);
    else if (interruption == 1)
      noise_frame(power, random);
    else
      silent_frame(power);
    control.observe(power.data());
    require(control.narrow_tone_protection(cold_bin) == 0.f,
            "interruption did not clear cold-tone protection");
    tone_frame(power, cold_bin);
    for (unsigned frame = 0; frame < 20; ++frame) control.observe(power.data());
    require(control.narrow_tone_protection(cold_bin) == 0.f,
            "same-bin tone was shielded again after a continuity break");
  }

  itu::detail::OmlsaAdaptiveControl different_bin(rate, fft, bins);
  tone_frame(power, cold_bin);
  for (unsigned frame = 0; frame < 4; ++frame) different_bin.observe(power.data());
  require(different_bin.narrow_tone_protection(cold_bin) == .98f,
          "different-bin control lacked an initial cold lock");
  tone_frame(power, other_bin);
  different_bin.observe(power.data());
  require(different_bin.narrow_tone_protection(other_bin) == 0.f,
          "different-bin later tone was incorrectly protected");
  for (unsigned frame = 0; frame < 20; ++frame)
    different_bin.observe(power.data());
  require(different_bin.narrow_tone_protection(other_bin) == 0.f,
          "different-bin tone reacquired protection after cold frames");

  itu::detail::OmlsaAdaptiveControl late_only(rate, fft, bins);
  flat_frame(power);
  for (unsigned frame = 0; frame < 6; ++frame) late_only.observe(power.data());
  tone_frame(power, cold_bin);
  for (unsigned frame = 0; frame < 30; ++frame)
    late_only.observe(power.data());
  require(late_only.narrow_tone_protection(cold_bin) == 0.f,
          "late-only tone received cold-start protection");

  std::cout << "PASS controller narrow tone sustained center/edge/far="
            << ".98/.70/0; interrupted flat,noise,silence same-bin=0; "
               "different-bin=0; late-only=0\n";
}

void test_off_config_bitwise() {
  const std::array<unsigned, 3> rates = {16000u, 24000u, 48000u};
  for (unsigned rate : rates) {
    const std::vector<float> input = mixed_input(rate, 130);
    itu::detail::OmlsaImcra defaults(rate);
    itu::OmlsaTuningConfig off;
    itu::detail::OmlsaImcra explicit_off(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa, off);
    const EngineRun baseline = run_engine(defaults, input);
    const EngineRun candidate = run_engine(explicit_off, input);
    require(baseline.output.size() == candidate.output.size(), "off output size");
    for (std::size_t i = 0; i < baseline.output.size(); ++i)
      require(same_bits(baseline.output[i], candidate.output[i]),
              "off output differs at " + std::to_string(i) +
                  " rate=" + std::to_string(rate));
    for (std::size_t i = 0; i < baseline.mean_gain.size(); ++i)
      require(same_bits(baseline.mean_gain[i], candidate.mean_gain[i]),
              "off gain differs at " + std::to_string(i) +
                  " rate=" + std::to_string(rate));

    itu::detail::OmlsaImcra mild_default(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        mild_tuning());
    itu::OmlsaTuningConfig mild_off = mild_tuning();
    mild_off.adaptive_noise_tracking = false;
    mild_off.high_snr_transparency = false;
    mild_off.low_band_protection = false;
    itu::detail::OmlsaImcra mild_explicit_off(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa, mild_off);
    const EngineRun mild_baseline = run_engine(mild_default, input);
    const EngineRun mild_candidate = run_engine(mild_explicit_off, input);
    for (std::size_t i = 0; i < mild_baseline.output.size(); ++i)
      require(same_bits(mild_baseline.output[i], mild_candidate.output[i]),
              "mild off output differs at " + std::to_string(i) +
                  " rate=" + std::to_string(rate));
    for (std::size_t i = 0; i < mild_baseline.mean_gain.size(); ++i)
      require(same_bits(mild_baseline.mean_gain[i],
                        mild_candidate.mean_gain[i]),
              "mild off gain differs at " + std::to_string(i) +
                  " rate=" + std::to_string(rate));
  }
  std::cout << "PASS off config bitwise 16/24/48k\n";
}

void test_mixed_finite_passive_delay_and_allocation() {
  const std::array<unsigned, 3> rates = {16000u, 24000u, 48000u};
  for (unsigned rate : rates) {
    itu::detail::OmlsaImcra engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        adaptive_tuning(true));
    require(engine.delay() == rate / 100,
            "adaptive delay is not one 10 ms hop at rate " + std::to_string(rate));
    const std::vector<float> input = mixed_input(rate, 150);
    const EngineRun run = run_engine(engine, input, true);
    require_finite(run.output, "adaptive mixed output");
    for (float gain : run.mean_gain)
      require(std::isfinite(gain) && gain >= 0.f && gain <= 1.000001f,
              "adaptive mean gain outside [0,1]");
    const double input_rms = rms(input.data(), input.size());
    const double output_rms = rms(run.output.data(), run.output.size());
    require(output_rms <= 1.05 * input_rms,
            "adaptive output amplified: ratio=" +
                std::to_string(output_rms / input_rms));
    std::cout << "PASS adaptive finite/passive/allocation rate=" << rate
              << " delay=" << engine.delay() << " ratio="
              << std::setprecision(7) << output_rms / input_rms << '\n';
  }
}

void test_direct_supergaussian_rejects_adaptive_flags() {
  for (unsigned flag = 0; flag < 3; ++flag) {
    itu::OmlsaTuningConfig tuning;
    if (flag == 0) tuning.adaptive_noise_tracking = true;
    if (flag == 1) tuning.high_snr_transparency = true;
    if (flag == 2) tuning.low_band_protection = true;
    require_invalid_argument(
        [&] {
          itu::detail::OmlsaImcra engine(
              16000, -18.f, true,
              itu::detail::SpectralGainRule::super_gaussian_map, tuning);
        },
        "direct super_gaussian_map adaptive flag " + std::to_string(flag));
  }
  std::cout << "PASS direct super-Gaussian rejects all three adaptive flags\n";
}

void test_pause_floor_is_relative_to_constructor_floor() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  std::array<double, 3> minimum_gain{};
  const std::array<float, 3> floors = {-9.f, -12.f, -18.f};
  for (std::size_t index = 0; index < floors.size(); ++index) {
    itu::detail::OmlsaImcra engine(
        rate, floors[index], true, itu::detail::SpectralGainRule::omlsa,
        adaptive_tuning(false));
    XorShift32 random(0xa54ff53au ^ unsigned(index));
    std::vector<float> frame(hop);
    double observed_minimum = 1.;
    for (unsigned frame_index = 0; frame_index < 100; ++frame_index) {
      for (float& sample : frame) sample = .02f * random.bipolar();
      engine.set_capture_evidence(0.f);
      engine.process(frame.data());
      require_finite(frame, "relative pause floor output");
      if (frame_index < 60) continue;
      const auto spectrum = engine.spectral_frame();
      for (unsigned k = 1; k + 1 < spectrum.bins; ++k)
        observed_minimum = std::min(observed_minimum, double(spectrum.gain[k]));
    }
    const double constructor_floor = std::pow(10., floors[index] / 20.);
    const double relative_pause_floor = constructor_floor * .35481339;
    require(observed_minimum >= relative_pause_floor - 2e-6,
            "pause depth crossed the relative constructor floor at floor_db=" +
                std::to_string(floors[index]) + ": minimum=" +
                std::to_string(observed_minimum) + " expected>=" +
                std::to_string(relative_pause_floor));
    minimum_gain[index] = observed_minimum;
  }
  require(minimum_gain[0] > minimum_gain[1] &&
              minimum_gain[1] > minimum_gain[2],
          "different constructor floors collapsed to one hard-coded pause floor");
  std::cout << "PASS relative pause floors -9/-12/-18="
            << minimum_gain[0] << '/' << minimum_gain[1] << '/'
            << minimum_gain[2] << '\n';
}

double mean_estimator_noise(itu::detail::OmlsaImcra& engine,
                            unsigned first_bin, unsigned last_bin) {
  double sum = 0;
  unsigned count = 0;
  for (unsigned k = first_bin; k < last_bin; ++k) {
    sum += engine.estimator().noise(k);
    ++count;
  }
  return count ? sum / count : 0.;
}

void test_legacy_step_recovery_after_pause() {
  // This exercises the legacy process(..., step_recovery) estimator path. It
  // is deliberately not evidence that the new setter/pause PSD reopen works.
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  itu::detail::OmlsaImcra engine(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      adaptive_tuning(false));
  std::vector<float> frame(hop);
  XorShift32 random(0x3c6ef372u);
  for (unsigned frame_index = 0; frame_index < 100; ++frame_index) {
    for (float& sample : frame) sample = .025f * random.bipolar();
    engine.process(frame.data());
  }
  const unsigned first_bin = 8;
  const unsigned last_bin = engine.fft_size() / 2 - 7;
  const double before = mean_estimator_noise(engine, first_bin, last_bin);
  const std::size_t updates_before = engine.estimator().step_updates();
  for (unsigned frame_index = 0; frame_index < 50; ++frame_index) {
    for (float& sample : frame) sample = .08f * random.bipolar();
    engine.process(frame.data(), nullptr, 1.f);
  }
  const double after = mean_estimator_noise(engine, first_bin, last_bin);
  const std::size_t updates = engine.estimator().step_updates() - updates_before;
  require(updates > 0, "broadband step did not trigger estimator recovery");
  require(after > before * 1.05,
          "triggered noise step did not raise estimator: before=" +
              std::to_string(before) + " after=" + std::to_string(after));
  std::cout << "PASS legacy step_recovery before=" << before
            << " after=" << after << " bin_updates=" << updates << '\n';
}

void test_pause_psd_reopen_without_legacy_step() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  itu::detail::OmlsaImcra pause_engine(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      adaptive_tuning(false));
  itu::detail::OmlsaImcra control_engine(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      adaptive_tuning(false));
  XorShift32 random(0x1f83d9abu);
  std::vector<float> frame(hop);
  for (unsigned frame_index = 0; frame_index < 100; ++frame_index) {
    for (float& sample : frame) sample = .025f * random.bipolar();
    std::vector<float> pause_frame = frame;
    std::vector<float> control_frame = frame;
    // Synthetic capture evidence, not an external VAD or detector-accuracy
    // claim. Both engines see identical evidence while pause is established.
    pause_engine.set_capture_evidence(0.f);
    control_engine.set_capture_evidence(0.f);
    pause_engine.process(pause_frame.data());
    control_engine.process(control_frame.data());
  }
  const unsigned first_bin = 8;
  const unsigned last_bin = pause_engine.fft_size() / 2 - 7;
  const double pause_before =
      mean_estimator_noise(pause_engine, first_bin, last_bin);
  const double control_before =
      mean_estimator_noise(control_engine, first_bin, last_bin);
  require(std::abs(pause_before - control_before) < 1e-7,
          "pause/control estimator states diverged before the step");
  const std::size_t pause_steps_before = pause_engine.estimator().step_updates();
  const std::size_t control_steps_before = control_engine.estimator().step_updates();
  for (unsigned frame_index = 0; frame_index < 50; ++frame_index) {
    for (float& sample : frame) sample = .08f * random.bipolar();
    std::vector<float> pause_frame = frame;
    std::vector<float> control_frame = frame;
    // Same synthetic evidence family after the step. The control value is
    // above the quiet threshold, while zero keeps the genuine pause path open.
    pause_engine.set_capture_evidence(0.f);
    control_engine.set_capture_evidence(.4f);
    // Default step_recovery=0: this test exercises only pause_confidence and
    // independent_noise reopening in ImcraNoiseEstimator::update().
    pause_engine.process(pause_frame.data());
    control_engine.process(control_frame.data());
  }
  const double pause_after =
      mean_estimator_noise(pause_engine, first_bin, last_bin);
  const double control_after =
      mean_estimator_noise(control_engine, first_bin, last_bin);
  const std::size_t pause_steps =
      pause_engine.estimator().step_updates() - pause_steps_before;
  const std::size_t control_steps =
      control_engine.estimator().step_updates() - control_steps_before;
  require(pause_steps == 0 && control_steps == 0,
          "pause PSD comparison accidentally used legacy step_recovery");
  require(pause_after > pause_before * 1.05,
          "pause PSD did not reopen: before=" + std::to_string(pause_before) +
              " after=" + std::to_string(pause_after));
  require(pause_after > control_after * 1.05,
          "pause PSD did not exceed the no-pause control: pause=" +
              std::to_string(pause_after) + " control=" +
              std::to_string(control_after));
  std::cout << "PASS synthetic pause PSD reopen before=" << pause_before
            << " pause_after=" << pause_after
            << " control_after=" << control_after
            << " legacy_steps=" << pause_steps + control_steps << '\n';
}

void test_setter_coldstart_tone_and_continuous_weak_speech() {
  const std::array<unsigned, 3> rates = {16000u, 24000u, 48000u};
  std::string failure;
  for (unsigned rate : rates) {
    const unsigned hop = rate / 100;
    itu::detail::OmlsaImcra tone_engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        adaptive_tuning(false));
    std::vector<float> tone_frame(hop);
    std::vector<float> tone_input;
    std::vector<float> tone_output;
    tone_input.reserve(180 * hop);
    tone_output.reserve(180 * hop);
    for (unsigned frame_index = 0; frame_index < 180; ++frame_index) {
      for (unsigned i = 0; i < hop; ++i) {
        const std::size_t n = frame_index * hop + i;
        tone_frame[i] = .12f * float(std::sin(
            2.0 * 3.14159265358979323846 * 440.0 * double(n) / rate));
      }
      tone_input.insert(tone_input.end(), tone_frame.begin(), tone_frame.end());
      // Synthetic valid capture evidence: zero means no current statistical
      // speech evidence. This is not an external-VAD accuracy claim.
      tone_engine.set_capture_evidence(0.f);
      tone_engine.process(tone_frame.data());
      require_finite(tone_frame, "setter cold-tone output");
      tone_output.insert(tone_output.end(), tone_frame.begin(), tone_frame.end());
    }
    const std::size_t early_begin = 10 * hop;
    const std::size_t early_count = 40 * hop;
    const std::size_t late_begin = 130 * hop;
    const std::size_t late_count = 40 * hop;
    const double early_ratio =
        rms(tone_output.data() + early_begin, early_count) /
        rms(tone_input.data() + early_begin, early_count);
    const double late_ratio =
        rms(tone_output.data() + late_begin, late_count) /
        rms(tone_input.data() + late_begin, late_count);

    itu::detail::OmlsaImcra weak_engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        adaptive_tuning(false));
    XorShift32 random(0x243f6a88u ^ rate);
    std::vector<float> weak_frame(hop);
    std::vector<float> weak_input;
    std::vector<float> weak_output;
    weak_input.reserve(180 * hop);
    weak_output.reserve(180 * hop);
    for (unsigned frame_index = 0; frame_index < 180; ++frame_index) {
      for (unsigned i = 0; i < hop; ++i) {
        const std::size_t n = frame_index * hop + i;
        const double time = double(n) / rate;
        float sample = .004f * random.bipolar();
        if (frame_index >= 30) {
          sample += .008f * float(std::sin(
              2.0 * 3.14159265358979323846 * 440.0 * time));
          sample += .004f * float(std::sin(
              2.0 * 3.14159265358979323846 * 880.0 * time));
        }
        weak_frame[i] = sample;
      }
      weak_input.insert(weak_input.end(), weak_frame.begin(), weak_frame.end());
      // Synthetic sustained weak-speech evidence after onset. This tests the
      // protection path, not the accuracy of any statistical/external VAD.
      weak_engine.set_capture_evidence(frame_index >= 30 ? .8f : 0.f);
      weak_engine.process(weak_frame.data());
      require_finite(weak_frame, "setter continuous weak-speech output");
      weak_output.insert(weak_output.end(), weak_frame.begin(), weak_frame.end());
    }
    const std::size_t weak_middle_begin = 60 * hop;
    const std::size_t weak_middle_count = 40 * hop;
    const std::size_t weak_late_begin = 120 * hop;
    const std::size_t weak_late_count = 60 * hop;
    const double weak_middle_ratio =
        rms(weak_output.data() + weak_middle_begin, weak_middle_count) /
        rms(weak_input.data() + weak_middle_begin, weak_middle_count);
    const double weak_late_ratio =
        rms(weak_output.data() + weak_late_begin, weak_late_count) /
        rms(weak_input.data() + weak_late_begin, weak_late_count);
    std::cout << "PROBE synthetic setter rate=" << rate
              << " tone_early=" << early_ratio
              << " tone_late=" << late_ratio
              << " weak_middle=" << weak_middle_ratio
              << " weak_late=" << weak_late_ratio << '\n';
    if ((early_ratio <= .7 || late_ratio <= .5) && failure.empty())
      failure = "setter cold-tone was severely attenuated at rate " +
                std::to_string(rate) + ": early=" +
                std::to_string(early_ratio) + " late=" +
                std::to_string(late_ratio);
    if (weak_late_ratio <= .25 && failure.empty())
      failure = "setter continuous weak speech was gated at rate " +
                std::to_string(rate) + ": late=" +
                std::to_string(weak_late_ratio);
  }
  require(failure.empty(), failure);
  std::cout << "PASS synthetic setter cold-tone and weak-speech protection\n";
}

void test_setter_capture_path() {
  constexpr unsigned rate = 24000;
  const unsigned hop = rate / 100;
  itu::detail::OmlsaImcra engine(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      adaptive_tuning(false));
  XorShift32 random(0x510e527fu);
  std::vector<float> frame(hop);
  std::vector<float> input;
  std::vector<float> output;
  input.reserve(85 * hop);
  output.reserve(85 * hop);
  for (unsigned frame_index = 0; frame_index < 85; ++frame_index) {
    for (unsigned i = 0; i < hop; ++i) {
      const std::size_t n = frame_index * hop + i;
      const double time = double(n) / rate;
      float sample = .012f * random.bipolar();
      if (frame_index >= 20 && frame_index < 40) {
        sample += .12f * float(std::sin(
            2.0 * 3.14159265358979323846 * 440.0 * time));
        sample += .06f * float(std::sin(
            2.0 * 3.14159265358979323846 * 1320.0 * time));
      } else if (frame_index >= 40 && frame_index < 55) {
        sample += .004f * float(std::sin(
            2.0 * 3.14159265358979323846 * 660.0 * time));
      }
      frame[i] = sample;
    }
    input.insert(input.end(), frame.begin(), frame.end());
    if (frame_index < 20)
      engine.set_capture_evidence(0.f);
    else if (frame_index < 40)
      engine.set_capture_evidence(1.f);
    else if (frame_index < 55)
      engine.set_capture_evidence(.55f);
    else
      engine.set_capture_evidence(0.f);
    engine.process(frame.data());
    require_finite(frame, "setter capture output frame " +
                              std::to_string(frame_index));
    output.insert(output.end(), frame.begin(), frame.end());
  }
  const std::size_t onset_begin = 20 * hop;
  const std::size_t onset_count = 5 * hop;
  const double onset_input_rms = rms(input.data() + onset_begin, onset_count);
  const double onset_output_rms = rms(output.data() + onset_begin, onset_count);
  const double onset_ratio = onset_output_rms / onset_input_rms;
  const std::size_t tail_begin = 40 * hop;
  const std::size_t tail_count = 15 * hop;
  const double tail_input_rms = rms(input.data() + tail_begin, tail_count);
  const double tail_output_rms = rms(output.data() + tail_begin, tail_count);
  const std::size_t pause_begin = 55 * hop;
  const std::size_t pause_count = 30 * hop;
  const double pause_input_rms = rms(input.data() + pause_begin, pause_count);
  const double pause_output_rms = rms(output.data() + pause_begin, pause_count);
  require(onset_ratio > .25,
          "setter speech onset was gated too early: ratio=" +
              std::to_string(onset_ratio));
  require(tail_output_rms > 0.,
          "setter weak-tail path produced no output energy");
  require(pause_output_rms <= pause_input_rms * 1.05,
          "setter pause noise was amplified: ratio=" +
              std::to_string(pause_output_rms / pause_input_rms));
  std::cout << "PASS setter path onset_ratio=" << onset_ratio
            << " tail_in/out=" << tail_input_rms << '/' << tail_output_rms
            << " pause_in/out=" << pause_input_rms << '/'
            << pause_output_rms << '\n';
}

struct EvidenceRun {
  std::vector<float> output;
  std::vector<float> mean_gain;
};

EvidenceRun run_tone_with_evidence(unsigned rate, bool set_evidence,
                                   float evidence) {
  const unsigned hop = rate / 100;
  itu::detail::OmlsaImcra engine(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      adaptive_tuning(false));
  const std::vector<float> input = tone_input(rate, 100);
  EvidenceRun result;
  result.output.reserve(input.size());
  result.mean_gain.reserve(input.size() / hop);
  std::vector<float> frame(hop);
  for (std::size_t offset = 0; offset < input.size(); offset += hop) {
    std::copy_n(input.data() + offset, hop, frame.begin());
    if (set_evidence) engine.set_capture_evidence(evidence);
    engine.process(frame.data());
    result.output.insert(result.output.end(), frame.begin(), frame.end());
    result.mean_gain.push_back(engine.mean_gain());
  }
  return result;
}

std::size_t differing_samples(const std::vector<float>& left,
                              const std::vector<float>& right) {
  require(left.size() == right.size(), "difference comparison size mismatch");
  std::size_t differing = 0;
  for (std::size_t i = 0; i < left.size(); ++i)
    differing += !same_bits(left[i], right[i]);
  return differing;
}

void test_capture_evidence_missing_sentinels() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const EvidenceRun never_set = run_tone_with_evidence(rate, false, 0.f);
    const EvidenceRun negative_one = run_tone_with_evidence(rate, true, -1.f);
    const EvidenceRun nan_value = run_tone_with_evidence(
        rate, true, std::numeric_limits<float>::quiet_NaN());
    const EvidenceRun valid_zero = run_tone_with_evidence(rate, true, 0.f);

    require_bitwise_equal(negative_one.output, never_set.output,
                          "setter -1 cold-tone PCM");
    require_bitwise_equal(negative_one.mean_gain, never_set.mean_gain,
                          "setter -1 cold-tone gain");
    require_bitwise_equal(nan_value.output, never_set.output,
                          "setter NaN cold-tone PCM");
    require_bitwise_equal(nan_value.mean_gain, never_set.mean_gain,
                          "setter NaN cold-tone gain");

    const std::size_t zero_output_differences =
        differing_samples(valid_zero.output, never_set.output);
    const std::size_t zero_gain_differences =
        differing_samples(valid_zero.mean_gain, never_set.mean_gain);
    require(zero_output_differences > 0,
            "positive zero evidence was indistinguishable from missing evidence");
    require(zero_gain_differences > 0,
            "positive zero evidence produced identical cold-tone gain trace");
    std::cout << "PASS missing-evidence sentinels rate=" << rate
              << " zero_output_diff=" << zero_output_differences
              << " zero_gain_diff=" << zero_gain_differences << '\n';
  }
}

void test_tone_only_cold_start() {
  const std::array<unsigned, 3> rates = {16000u, 24000u, 48000u};
  std::string failure;
  for (unsigned rate : rates) {
    itu::detail::OmlsaImcra engine(
        rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
        adaptive_tuning(false));
    const std::vector<float> input = tone_input(rate, 180);
    const EngineRun run = run_engine(engine, input);
    require_finite(run.output, "tone-only output");
    const unsigned hop = rate / 100;
    const std::size_t segment_begin = 10;
    const std::size_t segment_frames = 40;
    std::array<double, 4> ratios{};
    for (unsigned segment = 0; segment < ratios.size(); ++segment) {
      const std::size_t begin = (segment_begin + segment * segment_frames) * hop;
      const std::size_t count = segment_frames * hop;
      ratios[segment] = rms(run.output.data() + begin, count) /
                        rms(input.data() + begin, count);
    }
    std::cout << "PROBE tone-only rate=" << rate
              << " r10_49=" << std::setprecision(8) << ratios[0]
              << " r50_89=" << ratios[1]
              << " r90_129=" << ratios[2]
              << " r130_169=" << ratios[3]
              << " final_mean_gain=" << engine.mean_gain() << '\n';
    if (ratios[0] <= .7 && failure.empty())
      failure = "tone-only cold start severely attenuated at rate " +
                std::to_string(rate) + ": r10_49=" +
                std::to_string(ratios[0]);
  }
  require(failure.empty(), failure);
  std::cout << "PASS tone-only cold start is not severely attenuated\n";
}

#else  // OMLSA_ORIGINAL_GOLDEN

void hash_float(std::uint64_t& hash, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (unsigned shift = 0; shift < 32; shift += 8) {
    hash ^= (bits >> shift) & 0xffu;
    hash *= 1099511628211ULL;
  }
}

std::vector<float> golden_input(unsigned rate) {
  const unsigned hop = rate / 100;
  constexpr std::size_t frames = 130;
  std::vector<float> input(frames * hop);
  XorShift32 random(0x6a09e667u ^ rate);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const float noise_gain = frame < 110 ? .025f : .08f;
    for (unsigned i = 0; i < hop; ++i) {
      const std::size_t n = frame * hop + i;
      const double time = double(n) / rate;
      float sample = noise_gain * random.bipolar();
      if (frame >= 40 && frame < 80) {
        sample += .14f * float(std::sin(
            2.0 * 3.14159265358979323846 * 440.0 * time));
        sample += .08f * float(std::sin(
            2.0 * 3.14159265358979323846 * 1320.0 * time));
      } else if (frame >= 80 && frame < 95) {
        sample += .003f * float(std::sin(
            2.0 * 3.14159265358979323846 * 660.0 * time));
      }
      input[n] = sample;
    }
  }
  return input;
}

std::uint64_t golden_digest(unsigned rate, bool mild) {
  itu::detail::OmlsaImcra engine(
      rate, -18.f, true, itu::detail::SpectralGainRule::omlsa,
      mild ? mild_tuning() : itu::OmlsaTuningConfig{});
  const std::vector<float> input = golden_input(rate);
  const unsigned hop = engine.delay();
  std::vector<float> frame(hop);
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t offset = 0; offset < input.size(); offset += hop) {
    std::copy_n(input.data() + offset, hop, frame.begin());
    engine.process(frame.data());
    for (float sample : frame) hash_float(hash, sample);
    hash_float(hash, engine.mean_gain());
  }
  return hash;
}

#endif  // OMLSA_ORIGINAL_GOLDEN

}  // namespace

int main() {
#ifdef OMLSA_ORIGINAL_GOLDEN
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    std::cout << "ORIGINAL rate=" << rate
              << " default=0x" << std::hex << std::setw(16)
              << std::setfill('0') << golden_digest(rate, false)
              << " mild=0x" << std::setw(16) << golden_digest(rate, true)
              << std::dec << std::setfill(' ') << '\n';
  }
  return 0;
#else
  try {
    test_controller_pause_and_speech();
    test_controller_narrow_tone_continuity();
    test_off_config_bitwise();
    test_mixed_finite_passive_delay_and_allocation();
    test_direct_supergaussian_rejects_adaptive_flags();
    test_pause_floor_is_relative_to_constructor_floor();
    test_capture_evidence_missing_sentinels();
    test_legacy_step_recovery_after_pause();
    test_pause_psd_reopen_without_legacy_step();
    test_setter_capture_path();
    test_tone_only_cold_start();
    test_setter_coldstart_tone_and_continuous_weak_speech();
    std::cout << "ALL OMLSA ADAPTIVE TESTS PASSED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
#endif
}
