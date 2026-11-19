#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace itu::detail {

// dpdf-frame-v2-db: the v1 causal 256 features plus a 64-band projection of
// the current OM-LSA base gain. The gain projection is linear before dB.
class ResidualFeatureBandsV2 {
 public:
  static constexpr unsigned sample_rate = 16000;
  static constexpr unsigned fft_size = 512;
  static constexpr unsigned hop_size = 160;
  static constexpr unsigned window_size = 320;
  static constexpr unsigned bins = fft_size / 2 + 1;
  static constexpr unsigned bands = 64;
  static constexpr unsigned v1_input_size = 4 * bands;
  static constexpr unsigned input_size = 5 * bands;
  static constexpr unsigned feature_version = 2;
  static constexpr float epsilon = 1e-8f;

  explicit ResidualFeatureBandsV2(unsigned rate = sample_rate,
                                  unsigned nfft = fft_size) {
    if (rate != sample_rate || nfft != fft_size)
      throw std::invalid_argument(
          "v2 residual features require mono 16 kHz FFT512");
    const float nyquist = static_cast<float>(rate) * .5f;
    for (unsigned band = 0; band < bands; ++band) {
      centers_[band] = band < 24
          ? 1000.f * band / 23.f
          : band < 40
              ? 1000.f + 3000.f * (band - 23) / 16.f
              : 4000.f + (nyquist - 4000.f) * (band - 39) / 24.f;
    }
    for (unsigned bin = 0; bin < bins; ++bin) {
      const float hz = static_cast<float>(bin) * rate / nfft;
      unsigned band = 0;
      while (band + 1 < bands - 1 && hz > centers_[band + 1]) ++band;
      left_band_[bin] = band;
      right_weight_[bin] = std::clamp(
          (hz - centers_[band]) / (centers_[band + 1] - centers_[band]),
          0.f, 1.f);
      normalization_[band] += 1.f - right_weight_[bin];
      normalization_[band + 1] += right_weight_[bin];
    }
  }

  bool process(const float* power, const float* noise,
               const float* base_gain) noexcept {
    if (!power || !noise || !base_gain) return false;
    for (unsigned bin = 0; bin < bins; ++bin) {
      if (!std::isfinite(power[bin]) || !std::isfinite(noise[bin]) ||
          power[bin] < 0.f || noise[bin] < 0.f ||
          !std::isfinite(base_gain[bin]) || base_gain[bin] < 0.f ||
          base_gain[bin] > 1.f) return false;
    }

    std::array<float, bands> next_power{};
    std::array<float, bands> next_noise{};
    std::array<float, bands> next_gain{};
    std::array<float, input_size> next_features{};
    project_into(power, next_power);
    project_into(noise, next_noise);
    project_into(base_gain, next_gain);
    for (unsigned band = 0; band < bands; ++band) {
      if (next_power[band] < 0.f || next_noise[band] < 0.f ||
          next_gain[band] < 0.f) return false;
      const float power_db =
          10.f * std::log10(std::max(next_power[band], epsilon));
      const float noise_db =
          10.f * std::log10(std::max(next_noise[band], epsilon));
      const float gain_db =
          20.f * std::log10(std::max(next_gain[band], epsilon));
      const float delta_db = initialized_
          ? power_db - previous_power_db_[band]
          : 0.f;
      if (!std::isfinite(power_db) || !std::isfinite(noise_db) ||
          !std::isfinite(gain_db) || !std::isfinite(delta_db)) return false;
      next_features[band] = power_db;
      next_features[bands + band] = noise_db;
      next_features[2 * bands + band] = power_db - noise_db;
      next_features[3 * bands + band] = delta_db;
      next_features[4 * bands + band] = gain_db;
    }

    projected_power_ = next_power;
    projected_noise_ = next_noise;
    projected_gain_ = next_gain;
    features_ = next_features;
    for (unsigned band = 0; band < bands; ++band) {
      previous_power_db_[band] = next_features[band];
      base_gain_db_[band] = next_features[4 * bands + band];
    }
    initialized_ = true;
    return true;
  }

  float interpolate(const float* values, unsigned bin) const noexcept {
    const unsigned band = left_band_[bin];
    const float weight = right_weight_[bin];
    return (1.f - weight) * values[band] + weight * values[band + 1];
  }

  void reset() noexcept {
    previous_power_db_.fill(0.f);
    initialized_ = false;
  }

  bool initialized() const noexcept { return initialized_; }
  const std::array<float, bands>& projected_power() const noexcept {
    return projected_power_;
  }
  const std::array<float, bands>& projected_noise() const noexcept {
    return projected_noise_;
  }
  const std::array<float, bands>& projected_gain() const noexcept {
    return projected_gain_;
  }
  const std::array<float, input_size>& features() const noexcept {
    return features_;
  }
  const std::array<float, bands>& base_gain_db() const noexcept {
    return base_gain_db_;
  }

 private:
  void project_into(const float* spectrum,
                    std::array<float, bands>& output) const noexcept {
    output.fill(0.f);
    for (unsigned bin = 0; bin < bins; ++bin) {
      const unsigned band = left_band_[bin];
      const float weight = right_weight_[bin];
      output[band] += (1.f - weight) * spectrum[bin];
      output[band + 1] += weight * spectrum[bin];
    }
    for (unsigned band = 0; band < bands; ++band)
      output[band] /= std::max(1e-12f, normalization_[band]);
  }

  std::array<float, bands> centers_{};
  std::array<float, bands> normalization_{};
  std::array<unsigned, bins> left_band_{};
  std::array<float, bins> right_weight_{};
  std::array<float, bands> projected_power_{};
  std::array<float, bands> projected_noise_{};
  std::array<float, bands> projected_gain_{};
  std::array<float, input_size> features_{};
  std::array<float, bands> previous_power_db_{};
  std::array<float, bands> base_gain_db_{};
  bool initialized_ = false;
};

static_assert(ResidualFeatureBandsV2::input_size == 320);

}  // namespace itu::detail
