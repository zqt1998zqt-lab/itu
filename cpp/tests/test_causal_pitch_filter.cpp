#include "causal_pitch_filter.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using itu::detail::CausalPitchFilter;
using itu::detail::Fft;
namespace {
constexpr unsigned kRate = 16000, kHop = 160, kFft = 512, kBins = 257;
constexpr double kPi = 3.14159265358979323846;
using Spectrum = std::array<std::complex<float>, kBins>;
void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}
struct Noise {
  std::uint32_t state = 0x12345678u;
  float next() {
    state = state * 1664525u + 1013904223u;
    return (float((state >> 8) & 0xffffu) / 32767.5f) - 1.f;
  }
};
std::vector<float> make_signal(unsigned speech_frames, float pitch, float db,
                               float white = 0.f, float low = 0.f) {
  const unsigned history = 4 * kHop;
  std::vector<float> signal(history + speech_frames * kHop, 0.f);
  Noise noise;
  const float amplitude = std::pow(10.f, db / 20.f);
  for (unsigned n = history; n < signal.size(); ++n) {
    float harmonic = 0.f;
    for (unsigned h = 1; h * pitch < 4000.f; ++h)
      harmonic += std::sin(2. * kPi * h * pitch * n / kRate + .17 * h) / std::pow(float(h), .72f);
    signal[n] = amplitude * harmonic + white * noise.next() + low * std::sin(2. * kPi * 45. * n / kRate);
  }
  return signal;
}
void make_wet(const std::vector<float>& signal, unsigned frame, float gain,
              Fft& fft, Spectrum& wet) {
  std::array<std::complex<float>, kFft> spectrum{};
  const int start = int(frame * kHop) - int(kHop);
  for (unsigned i = 0; i < 320; ++i) {
    const int index = start + int(i);
    const float sample = index >= 0 ? signal[unsigned(index)] : 0.f;
    spectrum[i] = {sample * float(std::sin(kPi * (i + .5) / 320.)), 0.f};
  }
  fft.run(spectrum.data());
  for (unsigned k = 0; k < kBins; ++k) wet[k] = gain * spectrum[k];
}
std::array<float, kBins> make_gain(float value) {
  std::array<float, kBins> gain{}; gain.fill(value); return gain;
}
double harmonic_ratio(const Spectrum& spectrum, unsigned lag) {
  if (!lag) return 0.;
  double harmonic = 0., inter = 0.;
  for (unsigned h = 1; (h + .5) * kRate / lag < 4000.; ++h) {
    const unsigned hc = unsigned(std::lround(double(h) * kFft / lag));
    const unsigned ic = unsigned(std::lround(double(h + .5) * kFft / lag));
    for (int d = -1; d <= 1; ++d) {
      if (int(hc) + d >= 0 && int(hc) + d < int(kBins)) harmonic += std::norm(spectrum[hc + d]);
      if (int(ic) + d >= 0 && int(ic) + d < int(kBins)) inter += std::norm(spectrum[ic + d]);
    }
  }
  return harmonic / (inter + 1e-12);
}
struct RunResult {
  CausalPitchFilter filter; Spectrum last_wet{}, last_output{};
  std::vector<Spectrum> outputs;
  double before_ratio = 0., after_ratio = 0.;
  explicit RunResult(float strength) : filter(strength) {}
};
RunResult run(const std::vector<float>& signal, float strength, float gain_value,
              unsigned measure_from = 0) {
  RunResult result(strength);
  Fft fft(kFft);
  const auto gain = make_gain(gain_value);
  result.filter.reset();
  for (unsigned frame = 0; frame < signal.size() / kHop; ++frame) {
    result.filter.observe_input(signal.data() + frame * kHop);
    make_wet(signal, frame, gain_value, fft, result.last_wet);
    result.last_output = result.last_wet;
    result.filter.process(result.last_output.data(), gain.data());
    result.outputs.push_back(result.last_output);
    if (frame >= measure_from && result.filter.lag_samples()) {
      result.before_ratio = harmonic_ratio(result.last_wet, result.filter.lag_samples());
      result.after_ratio = harmonic_ratio(result.last_output, result.filter.lag_samples());
    }
  }
  return result;
}
bool bit_equal(const Spectrum& a, const Spectrum& b) {
  return std::memcmp(a.data(), b.data(), sizeof(a)) == 0;
}
void test_api_and_contract() {
  bool threw = false;
  try { CausalPitchFilter bad(-.01f); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, "negative strength accepted");
  threw = false;
  try { CausalPitchFilter bad(1.01f); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, "oversized strength accepted");
  CausalPitchFilter filter; Spectrum wet{}; auto gain = make_gain(1.f);
  require(filter.latency() == 0 && filter.strength() == .5f, "API defaults changed");
  std::array<float, kHop> silence{};
  filter.observe_input(silence.data()); filter.process(wet.data(), gain.data());
  require(filter.active_frames() == 0 && filter.fallback_frames() == 1, "history bypass");
  auto bad = silence; bad[kHop / 2] = std::numeric_limits<float>::quiet_NaN();
  filter.observe_input(bad.data()); filter.process(wet.data(), gain.data());
  require(filter.fallback_frames() == 2 && bit_equal(wet, Spectrum{}), "non-finite bypass");
  filter.reset();
  require(filter.fallback_frames() == 0 && filter.periodicity() == 0.f, "reset failed");
}
void test_voiced_and_negatives() {
  const auto voiced = make_signal(20, 180.f, -24.f, .035f, .05f);
  auto positive = run(voiced, .5f, .35f, 8);
  std::cerr << "positive p=" << positive.filter.periodicity() << " lag=" << positive.filter.lag_samples()
            << " active=" << positive.filter.active_frames() << " fallback=" << positive.filter.fallback_frames()
            << " ratio=" << positive.before_ratio << "->" << positive.after_ratio << '\n';
  require(positive.filter.active_frames() > 0, "periodic speech never activated");
  require(positive.after_ratio > positive.before_ratio * 1.01, "harmonic ratio did not improve");
  require(positive.filter.lag_samples() >= 40 && positive.filter.lag_samples() <= 228, "lag range");
  for (float gain : {1.f, .9999f}) {
    auto clean = run(voiced, .5f, gain, 8);
    require(bit_equal(clean.last_wet, clean.last_output), "clean gain was not transparent");
  }
  auto zero = run(voiced, 0.f, .35f, 8);
  require(bit_equal(zero.last_wet, zero.last_output), "strength zero changed wet");
  require(zero.filter.periodicity() > CausalPitchFilter::minimum_periodicity &&
              zero.filter.active_frames() == 0, "strength zero state was opaque");
  const auto noise = make_signal(20, 180.f, -100.f, .2f, 0.f);
  const auto low_noise = make_signal(20, 180.f, -100.f, .05f, .5f);
  require(run(noise, .5f, .35f).filter.active_frames() == 0, "white noise activated");
  require(run(low_noise, .5f, .35f).filter.active_frames() == 0, "low noise activated");
  for (float hz : {220.f, 1000.f, 1500.f}) {
    auto tone = make_signal(20, hz, -100.f);
    for (unsigned n = 4 * kHop; n < tone.size(); ++n) tone[n] = .12f * std::sin(2. * kPi * hz * n / kRate);
    auto result = run(tone, .5f, .35f);
    std::cerr << "tone " << hz << " p=" << result.filter.periodicity() << '\n';
    require(result.filter.periodicity() > CausalPitchFilter::minimum_periodicity && result.filter.active_frames() == 0,
            "steady sine passed harmonic guard");
  }
  auto two_tone = make_signal(20, 220.f, -100.f);
  for (unsigned n = 4 * kHop; n < two_tone.size(); ++n)
    two_tone[n] = .12f * std::sin(2. * kPi * 220. * n / kRate) + .08f * std::sin(2. * kPi * 440. * n / kRate);
  auto two_result = run(two_tone, .5f, .35f);
  std::cerr << "two-tone active=" << two_result.filter.active_frames() << '\n';
  require(std::isfinite(two_result.filter.periodicity()), "two-tone periodicity non-finite");
}
void test_onsets_amplitudes_and_jumps() {
  auto short_plosive = make_signal(1, 180.f, -18.f);
  short_plosive.resize(short_plosive.size() + 8 * kHop, 0.f);
  require(run(short_plosive, .5f, .35f).filter.active_frames() == 0, "short plosive active");
  for (float db : {-60.f, -45.f, -30.f, -15.f}) {
    const auto signal = make_signal(16, 150.f, db, std::pow(10.f, (db - 24.f) / 20.f), 0.f);
    auto result = run(signal, .5f, .35f, 8);
    require(result.filter.active_frames() > 0, "amplitude level failed to activate");
  }
  std::vector<float> jumping(4 * kHop + 18 * kHop, 0.f); Noise noise;
  for (unsigned n = 4 * kHop; n < jumping.size(); ++n) {
    const unsigned segment = (n - 4 * kHop) / (6 * kHop);
    const float pitch = segment == 0 ? 100.f : segment == 1 ? 260.f : 145.f;
    for (unsigned h = 1; h * pitch < 4000.f; ++h)
      jumping[n] += .08f * std::sin(2. * kPi * h * pitch * n / kRate + .11 * h) / h;
    jumping[n] += .002f * noise.next();
  }
  auto jump = run(jumping, .5f, .35f, 8);
  require(jump.filter.active_frames() > 0, "stable pitch segments never activated");
  require(jump.filter.periodicity() >= 0.f && jump.filter.periodicity() <= 1.f, "periodicity range");
}
void test_causal_determinism_and_energy() {
  const auto signal = make_signal(24, 170.f, -24.f, .04f, .04f);
  auto first = run(signal, .5f, .4f, 8), second = run(signal, .5f, .4f, 8);
  require(bit_equal(first.last_output, second.last_output) &&
              first.filter.active_frames() == second.filter.active_frames(), "chunking nondeterministic");
  require(bit_equal(first.last_wet, second.last_wet), "test fixture was nondeterministic");
  auto future = signal;
  for (unsigned n = 18 * kHop; n < future.size(); ++n) future[n] = -7.f * future[n];
  auto a = run(signal, .5f, .4f, 8), b = run(future, .5f, .4f, 8);
  for (unsigned frame = 0; frame < 18; ++frame)
    require(bit_equal(a.outputs[frame], b.outputs[frame]), "future changed prefix output");
  bool future_diff = false;
  for (unsigned frame = 18; frame < signal.size() / kHop; ++frame)
    future_diff = future_diff || !bit_equal(a.outputs[frame], b.outputs[frame]);
  require(future_diff, "future perturbation had no effect");
  double before = 0., after = 0.; bool finite = true;
  for (unsigned k = 0; k < kBins; ++k) {
    before += std::norm(first.last_wet[k]); after += std::norm(first.last_output[k]);
    const float hz = float(k) * kRate / kFft; require((hz >= 80.f && hz <= 4000.f) ||
        first.last_output[k] == first.last_wet[k], "out-of-band change");
    finite = finite && std::isfinite(first.last_output[k].real()) &&
             std::isfinite(first.last_output[k].imag());
  }
  require(finite && after > 0. && after <= 1.011 * before, "output energy/finite");
  require(first.last_output[0].imag() == 0.f && first.last_output[kBins - 1].imag() == 0.f, "DC/Nyquist complex");
}
}  // namespace
int main() {
  try {
    test_api_and_contract(); test_voiced_and_negatives();
    test_onsets_amplitudes_and_jumps(); test_causal_determinism_and_energy();
    std::cout << "PASS causal_pitch_filter: API, causal, voiced/noise/tone, onset, "
                 "amplitude, jumps, gain transparency, finite/energy bound\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL causal_pitch_filter: " << error.what() << '\n';
    return 1;
  }
}
