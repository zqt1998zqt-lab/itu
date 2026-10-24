#include "residual_noise_suppressor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
double rms(const std::vector<float>& x) {
  double sum = 0;
  for (float v : x) sum += double(v) * v;
  return std::sqrt(sum / std::max<std::size_t>(1, x.size()));
}
std::vector<float> noise(unsigned rate, unsigned frames) {
  std::mt19937 random(0x5eedu + rate);
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  std::vector<float> x(std::size_t(rate / 100) * frames);
  for (float& v : x) v = .04f * unit(random);
  return x;
}
std::vector<float> speech(unsigned rate, unsigned frames) {
  std::vector<float> x(std::size_t(rate / 100) * frames);
  for (unsigned n = 0; n < x.size(); ++n) {
    const double t = double(n) / rate;
    x[n] = .08f * std::sin(2. * 3.141592653589793 * 173. * t) +
           .04f * std::sin(2. * 3.141592653589793 * 317. * t) +
           .025f * std::sin(2. * 3.141592653589793 * 712. * t);
  }
  return x;
}
void rates_and_invalid_input() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    itu::ResidualNoiseSuppressor suppressor(rate, 9.f, .75f);
    const unsigned hop = rate / 100;
    std::vector<float> frame(hop, .01f);
    suppressor.process(frame.data(), hop);
    suppressor.process(nullptr, hop);
    suppressor.process(frame.data(), hop - 1);
    std::vector<float> after = frame;
    suppressor.process(after.data(), hop);
    for (float v : after) require(std::isfinite(v), "nonfinite output");
    std::cout << "rate=" << rate << " speech=" << suppressor.speech_probability()
              << " extra_db=" << suppressor.extra_attenuation_db() << '\n';
  }
}
void steady_noise_gets_extra_attenuation() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    itu::ResidualNoiseSuppressor suppressor(rate, 12.f, .75f);
    auto x = noise(rate, 360);
    const unsigned hop = rate / 100;
    std::vector<float> last(hop);
    for (unsigned frame = 0; frame < 360; ++frame) {
      std::copy_n(x.data() + frame * hop, hop, last.data());
      suppressor.process(last.data(), hop);
    }
    require(suppressor.extra_attenuation_db() > 2.f,
            "steady-noise negative control failed to activate");
    require(suppressor.speech_probability() < .35f,
            "random residual was classified as speech");
  }
}
void speech_guard_reduces_extra_attenuation() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    itu::ResidualNoiseSuppressor suppressor(rate, 12.f, .75f);
    const unsigned hop = rate / 100;
    auto background = noise(rate, 190);
    auto voice = speech(rate, 160);
    std::vector<float> frame(hop);
    for (unsigned n = 0; n < 190; ++n) {
      std::copy_n(background.data() + n * hop, hop, frame.data());
      suppressor.process(frame.data(), hop);
    }
    const float background_extra = suppressor.extra_attenuation_db();
    require(background_extra > 2.f, "background was not learned");
    float voice_probability = 0.f;
    float voice_extra = 0.f;
    unsigned voice_frames = 0;
    for (unsigned n = 0; n < 160; ++n) {
      std::copy_n(voice.data() + n * hop, hop, frame.data());
      suppressor.process(frame.data(), hop);
      const float probability = suppressor.speech_probability();
      if (probability > .45f) {
        voice_probability = std::max(voice_probability, probability);
        voice_extra = std::max(voice_extra, suppressor.extra_attenuation_db());
        ++voice_frames;
      }
    }
    require(voice_probability > .45f,
            "tonal speech negative control never detected");
    require(voice_frames >= 80, "speech evidence was not sustained");
    require(voice_extra < 6.f,
            "speech guard did not bound attenuation after detection");
  }
}
void snr_gate_is_effective_and_causal() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const unsigned hop = rate / 100;
    auto x = noise(rate, 260);
    itu::ResidualNoiseSuppressor low(rate, 6.f, .75f, -12.f, 3.f);
    itu::ResidualNoiseSuppressor high(rate, 6.f, .75f, -12.f, 3.f);
    float low_extra = 0.f, high_extra = 0.f;
    for (unsigned n = 0; n < 260; ++n) {
      std::vector<float> a(x.data() + n * hop, x.data() + (n + 1) * hop), b = a;
      low.process(a.data(), hop, -20.f);
      high.process(b.data(), hop, 5.f);
      for (float v : a) require(std::isfinite(v), "gated output nonfinite");
      for (float v : b) require(std::isfinite(v), "gated output nonfinite");
      low_extra = std::max(low_extra, low.extra_attenuation_db());
      high_extra = std::max(high_extra, high.extra_attenuation_db());
    }
    require(low_extra > 2.f, "low-SNR path never activated");
    require(high_extra < .01f, "high-SNR path violated the negative control");
  }
}
void zero_envelope_is_bit_exact() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    itu::ResidualNoiseSuppressor suppressor(rate, 0.f, .75f);
    auto x = noise(rate, 40);
    const auto reference = x;
    const unsigned hop = rate / 100;
    for (unsigned frame = 0; frame < 40; ++frame)
      suppressor.process(x.data() + frame * hop, hop);
    require(x == reference, "zero residual envelope changed PCM");
  }
}
void invalid_constructor_envelope() {
  for (unsigned rate : {8000u, 32000u}) {
    bool rejected = false;
    try { itu::ResidualNoiseSuppressor suppressor(rate, 6.f, .75f); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unsupported rate accepted");
  }
  bool rejected = false;
  try { itu::ResidualNoiseSuppressor suppressor(16000, 25.f, .75f); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "invalid attenuation accepted");
}
}
int main() {
  try {
    rates_and_invalid_input();
    steady_noise_gets_extra_attenuation();
    speech_guard_reduces_extra_attenuation();
    snr_gate_is_effective_and_causal();
    std::cout << "snr gate checks passed\n";
    zero_envelope_is_bit_exact();
    invalid_constructor_envelope();
    std::cout << "residual noise suppressor checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
