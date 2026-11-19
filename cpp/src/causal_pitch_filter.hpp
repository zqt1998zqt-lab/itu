#pragma once
#include "fft.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace itu::detail {
// Causal 16 kHz pitch comb filter. Thresholds are fixed engineering values.
class CausalPitchFilter {
 public:
  static constexpr unsigned sample_rate = 16000, frame_samples = 160;
  static constexpr unsigned fft_size = 512, bins = fft_size / 2 + 1;
  static constexpr unsigned window_samples = 320, min_lag = 40, max_lag = 228;
  static constexpr unsigned history_capacity = 1024;
  static constexpr unsigned minimum_history = window_samples + max_lag + 1;
  static constexpr unsigned bands = 64, latency_samples = 0;
  static constexpr float minimum_periodicity = .60f, gain_bypass = .9999f;
  static constexpr float epsilon = 1e-12f;
  explicit CausalPitchFilter(float strength = .5f)
      : fft_(fft_size), strength_(strength) {
    if (!std::isfinite(strength) || strength < 0.f || strength > 1.f) throw
        std::invalid_argument("pitch-filter strength must be finite in [0,1]");
    for (unsigned i = 0; i < window_samples; ++i)
      window_[i] = std::sin(3.14159265358979323846 * (i + .5) / window_samples);
    constexpr float nyquist = sample_rate * .5f;
    for (unsigned b = 0; b < bands; ++b)
      centers_[b] = b < 24 ? 1000.f * b / 23.f
          : b < 40 ? 1000.f + 3000.f * (b - 23) / 16.f
          : 4000.f + (nyquist - 4000.f) * (b - 39) / 24.f;
    for (unsigned k = 0; k < bins; ++k) {
      const float hz = float(k) * sample_rate / fft_size;
      unsigned band = 0;
      while (band + 1 < bands && hz > .5f * (centers_[band] + centers_[band + 1]))
        ++band;
      band_of_bin_[k] = band;
    }
  }
  // Call once per 160-sample frame before OM-LSA. It advances all causal state.
  void observe_input(const float* pcm160) noexcept {
    const float old_energy = frame_energy_, old_periodicity = periodicity_;
    ready_ = active_ = false; lag_ = 0; periodicity_ = 0.f;
    if (!pcm160) { stable_frames_ = 0; previous_energy_ = old_energy; return; }
    for (unsigned i = 0; i < frame_samples; ++i)
      if (!std::isfinite(pcm160[i])) {
        stable_frames_ = 0; previous_energy_ = old_energy; return;
      }
    for (unsigned i = 0; i < frame_samples; ++i) write_sample(pcm160[i]);
    total_samples_ += frame_samples; ready_ = true;
    double energy = 0.;
    for (unsigned i = 0; i < window_samples; ++i) {
      current_[i] = sample_from_newest(window_samples - 1 - i);
      energy += double(current_[i]) * current_[i];
    }
    frame_energy_ = float(energy);
    if (total_samples_ < minimum_history || frame_energy_ <= epsilon) {
      stable_frames_ = 0; previous_energy_ = frame_energy_; return;
    }
    double current_mean = 0.;
    for (float sample : current_) current_mean += sample;
    current_mean /= window_samples;
    std::array<float, window_samples> current_search{};
    double current_energy = 0.;
    for (unsigned i = 0; i < window_samples; ++i) {
      current_search[i] = float(current_[i] - current_mean) * window_[i];
      current_energy += double(current_search[i]) * current_search[i];
    }
    std::array<double, max_lag + 1> correlations{};
    for (unsigned lag = min_lag; lag <= max_lag; ++lag) {
      double delayed_mean = 0., cross = 0., delayed_energy = 0.;
      for (unsigned i = 0; i < window_samples; ++i)
        delayed_mean += sample_from_newest(window_samples - 1 - i + lag);
      delayed_mean /= window_samples;
      for (unsigned i = 0; i < window_samples; ++i) {
        const float delayed_raw = sample_from_newest(window_samples - 1 - i + lag);
        const float y = float(delayed_raw - delayed_mean) * window_[i];
        cross += double(current_search[i]) * y;
        delayed_energy += double(y) * y;
      }
      correlations[lag] = cross /
          std::sqrt(current_energy * delayed_energy + epsilon);
    }
    const auto global_best = std::max_element(correlations.begin() + min_lag,
                                               correlations.end());
    const double global_best_corr = *global_best;
    unsigned best_lag = unsigned(global_best - correlations.begin());
    for (unsigned lag = min_lag + 1; lag < max_lag; ++lag)
      if (correlations[lag] >= .92 * global_best_corr &&
          correlations[lag] >= correlations[lag - 1] &&
          correlations[lag] >= correlations[lag + 1]) {
        best_lag = lag;
        break;
      }
    lag_ = best_lag;
    periodicity_ = float(std::clamp(correlations[best_lag], 0., 1.));
    if (periodicity_ < minimum_periodicity) {
      stable_frames_ = 0; previous_energy_ = frame_energy_; return;
    }
    fill_window_fft(current_, raw_);
    if (!harmonic_evidence(raw_, best_lag)) {
      stable_frames_ = 0; previous_energy_ = frame_energy_; return;
    }
    for (unsigned i = 0; i < window_samples; ++i)
      delayed_input_[i] = sample_from_newest(window_samples - 1 - i + best_lag);
    fill_window_fft(delayed_input_, delayed_);
    const bool attack = old_energy > epsilon && frame_energy_ > 4.f * old_energy &&
                        old_periodicity < .5f;
    if (stable_frames_ > 0 &&
        std::abs(int(lag_) - int(stable_lag_)) <= 2) {
      ++stable_frames_;
    } else {
      stable_lag_ = lag_;
      stable_frames_ = 1;
    }
    if (attack) stable_frames_ = 0;
    active_ = stable_frames_ >= 2;
    if (active_) stable_lag_ = lag_;
    previous_energy_ = frame_energy_;
  }
  // wet257 is the current post-gain WOLA spectrum. On every fallback it is
  // left byte-for-byte untouched. No allocation, I/O, or locking occurs here.
  void process(std::complex<float>* wet257, const float* gain257) noexcept {
    if (!wet257 || !gain257 || !ready_ || !active_ || strength_ <= 0.f) {
      ++fallback_frames_; return;
    }
    bool finite = true, needs_effect = false;
    for (unsigned k = 0; k < bins; ++k) {
      finite = finite && std::isfinite(wet257[k].real()) &&
               std::isfinite(wet257[k].imag()) && std::isfinite(gain257[k]) &&
               gain257[k] >= 0.f && gain257[k] <= 1.001f;
      needs_effect = needs_effect || gain257[k] < gain_bypass;
    }
    if (!finite || !needs_effect) { ++fallback_frames_; return; }
    std::array<double, bands> raw_band{}, delayed_band{}, cross_band{};
    std::array<double, bands> gain_band{}, count_band{};
    std::array<float, bands> p_scale{}, alpha{};
    double wet_total = 0.;
    for (unsigned k = 0; k < bins; ++k) {
      wet_total += std::norm(wet257[k]); const float hz = float(k) * sample_rate / fft_size;
      if (hz < 80.f || hz > 4000.f) continue;
      const unsigned b = band_of_bin_[k];
      const double xp = std::norm(raw_[k]);
      const double pp = std::norm(delayed_[k]);
      raw_band[b] += xp;
      delayed_band[b] += pp;
      cross_band[b] += std::real(std::conj(raw_[k]) * delayed_[k]);
      gain_band[b] += gain257[k];
      count_band[b] += 1.;
    }
    for (unsigned b = 0; b < bands; ++b) {
      const double g = gain_band[b] / std::max(1., count_band[b]);
      const double pb = std::clamp(cross_band[b] /
          std::sqrt(raw_band[b] * delayed_band[b] + epsilon), 0., 1.);
      p_scale[b] = delayed_band[b] > epsilon
          ? float(std::sqrt(raw_band[b] / delayed_band[b])) : 0.f;
      if (pb <= 0. || g >= gain_bypass || raw_band[b] <= epsilon) {
        alpha[b] = 0.f;
        continue;
      }
      const double p2 = std::min(pb * pb, .999999), g2 = g * g;
      const double ratio = p2 * (1. - g2) /
          ((1. - p2) * g2 + double(epsilon));
      alpha[b] = strength_ * float(std::min(std::sqrt(std::max(0., ratio)), 1.));
    }
    std::array<std::complex<float>, bins> shaped{};
    std::array<double, bands> shaped_band{};
    double raw_total = 0., shaped_total = 0.; bool shaped_any = false;
    for (unsigned k = 0; k < bins; ++k) {
      const unsigned b = band_of_bin_[k];
      const float hz = float(k) * sample_rate / fft_size;
      const float a = hz >= 80.f && hz <= 4000.f ? alpha[b] : 0.f;
      shaped[k] = raw_[k] + a * p_scale[b] * delayed_[k]; shaped_any = shaped_any || a > 0.f;
      raw_total += std::norm(raw_[k]); shaped_total += std::norm(shaped[k]);
      if (hz >= 80.f && hz <= 4000.f) shaped_band[b] += std::norm(shaped[k]);
    }
    if (!shaped_any || !(raw_total > epsilon) || !(wet_total > epsilon) ||
        !std::isfinite(shaped_total) || shaped_total <= 0.) {
      ++fallback_frames_; return;
    }
    for (unsigned b = 0; b < bands; ++b) {
      if (shaped_band[b] > epsilon)
        for (unsigned k = 0; k < bins; ++k)
          if (band_of_bin_[k] == b && float(k) * sample_rate / fft_size >= 80.f &&
              float(k) * sample_rate / fft_size <= 4000.f)
            shaped[k] *= float(std::sqrt(raw_band[b] / shaped_band[b]));
    }
    shaped_band.fill(0.);
    for (unsigned k = 0; k < bins; ++k) {
      const float hz = float(k) * sample_rate / fft_size; if (hz >= 80.f && hz <= 4000.f)
        shaped_band[band_of_bin_[k]] += std::norm(shaped[k]);
    }
    for (unsigned b = 0; b < bands; ++b)
      if (raw_band[b] > epsilon && shaped_band[b] > 1.01 * raw_band[b]) {
        ++fallback_frames_; return;
      }
    shaped_total = 0.;
    for (const auto& value : shaped) shaped_total += std::norm(value);
    if (shaped_total > 1.01 * raw_total) {
      const float guard = float(std::sqrt(raw_total / shaped_total));
      for (auto& value : shaped) value *= guard;
    }
    std::array<std::complex<float>, bins> output{};
    double output_total = 0.;
    for (unsigned k = 0; k < bins; ++k) {
      const float hz = float(k) * sample_rate / fft_size;
      output[k] = hz >= 80.f && hz <= 4000.f ? shaped[k] * gain257[k] : wet257[k];
      output_total += std::norm(output[k]);
      finite = finite && std::isfinite(output[k].real()) &&
               std::isfinite(output[k].imag());
    }
    if (!finite) { ++fallback_frames_; return; }
    if (output_total > 1.01 * wet_total) {
      const float guard = float(std::sqrt(wet_total / output_total));
      for (auto& value : output) value *= guard;
    }
    output[0] = {output[0].real(), 0.f};
    output[bins - 1] = {output[bins - 1].real(), 0.f};
    for (unsigned k = 0; k < bins; ++k) wet257[k] = output[k];
    ++active_frames_;
  }
  void reset() noexcept {
    ring_.fill(0.f); current_.fill(0.f); delayed_input_.fill(0.f);
    raw_.fill({}); delayed_.fill({});
    ring_head_ = 0;
    total_samples_ = 0;
    frame_energy_ = previous_energy_ = periodicity_ = 0.f;
    lag_ = stable_lag_ = stable_frames_ = 0;
    active_frames_ = fallback_frames_ = 0;
    ready_ = active_ = false;
  }

  unsigned lag_samples() const noexcept { return lag_; }
  float periodicity() const noexcept { return periodicity_; }
  std::uint64_t active_frames() const noexcept { return active_frames_; }
  std::uint64_t fallback_frames() const noexcept { return fallback_frames_; }
  float strength() const noexcept { return strength_; }
  unsigned latency() const noexcept { return latency_samples; }
 private:
  void write_sample(float value) noexcept {
    ring_[ring_head_] = value;
    ring_head_ = (ring_head_ + 1) % history_capacity;
  }
  float sample_from_newest(unsigned age) const noexcept {
    return ring_[(ring_head_ + history_capacity - age - 1) % history_capacity];
  }
  void fill_window_fft(const std::array<float, window_samples>& input,
                       std::array<std::complex<float>, bins>& output) noexcept {
    std::array<std::complex<float>, fft_size> spectrum{};
    for (unsigned i = 0; i < window_samples; ++i)
      spectrum[i] = {input[i] * window_[i], 0.f};
    fft_.run(spectrum.data());
    std::copy_n(spectrum.begin(), bins, output.begin());
  }
  bool harmonic_evidence(const std::array<std::complex<float>, bins>& spectrum,
                         unsigned lag) const noexcept {
    double total = 0.;
    unsigned active_bins = 0;
    for (unsigned k = 0; k < bins; ++k) {
      const float hz = float(k) * sample_rate / fft_size;
      if (hz >= 80.f && hz <= 4000.f) total += std::norm(spectrum[k]), ++active_bins;
    }
    if (!(total > epsilon) || active_bins == 0) return false;
    const double mean = total / active_bins;
    double peak_sum = 0.; unsigned supported = 0;
    std::array<unsigned, bins> used_peak{};
    const unsigned harmonics = unsigned(4000. * lag / sample_rate);
    for (unsigned h = 1; h <= harmonics; ++h) {
      const unsigned center = unsigned(std::lround(double(h) * fft_size / lag));
      unsigned peak_bin = 0; double peak = 0.;
      for (int offset = -2; offset <= 2; ++offset) {
        const int k = int(center) + offset;
        if (k > 0 && k < int(bins) - 1 && std::norm(spectrum[k]) > peak)
          peak = std::norm(spectrum[k]), peak_bin = unsigned(k);
      }
      const float hz = float(peak_bin) * sample_rate / fft_size;
      const unsigned nearest = unsigned(std::lround(double(peak_bin) * lag / fft_size));
      if (hz < 80.f || hz > 4000.f || nearest != h || used_peak[peak_bin] ||
          peak <= 8. * mean || peak <= .015 * total) continue;
      used_peak[peak_bin] = 1; peak_sum += peak; ++supported;
    }
    return supported >= 2 && peak_sum / total >= .25;
  }
  Fft fft_; float strength_;
  std::array<float, window_samples> window_{}, current_{}, delayed_input_{};
  std::array<float, history_capacity> ring_{};
  std::array<std::complex<float>, bins> raw_{}, delayed_{};
  std::array<float, bands> centers_{};
  std::array<unsigned, bins> band_of_bin_{};
  std::uint64_t total_samples_ = 0, active_frames_ = 0, fallback_frames_ = 0;
  unsigned ring_head_ = 0, lag_ = 0, stable_lag_ = 0, stable_frames_ = 0;
  float frame_energy_ = 0.f, previous_energy_ = 0.f, periodicity_ = 0.f;
  bool ready_ = false, active_ = false;
};

static_assert(CausalPitchFilter::bins == 257);
static_assert(CausalPitchFilter::history_capacity >= 1024);

}  // namespace itu::detail
