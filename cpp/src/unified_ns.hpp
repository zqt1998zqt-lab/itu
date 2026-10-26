#pragma once

#include "fft.hpp"
#include "omlsa_imcra.hpp"
#include "super_gaussian.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace itu::detail {

// One native-rate, shared-spectrum noise suppressor with four synthesis rules.
// The experts share the STFT, IMCRA noise PSD, DD SNR and speech posterior.
// Fusion uses current/past capture evidence only: no labels, future PCM,
// external VAD or a separately trained model are consulted.
class UnifiedNs {
 public:
  explicit UnifiedNs(unsigned rate, float floor_db = -18.f,
                     float learning_rate = .08f)
      : rate_(check_rate(rate)),
        hop_(rate_ / 100),
        n_(next_power_two(2 * hop_)),
        bins_(n_ / 2 + 1),
        fft_(n_),
        noise_(bins_, rate_ == 16000 ? 1.519743f :
                       rate_ == 24000 ? 1.450226f : 1.448289f),
        window_(2 * hop_),
        past_(hop_),
        overlap_(hop_),
        power_(bins_),
        gamma_(bins_),
        xi_(bins_),
        previous_h1_snr_(bins_, 1.f),
        zeta_(bins_),
        gain_(bins_, 1.f),
        noise_used_(bins_),
        probability_(bins_),
        local_(kernel(1)),
        global_(kernel(std::max(
            1u, unsigned(std::lround(470. * n_ / rate_))))),
        spectrum_(n_),
        floor_gain_(std::pow(10.f, floor_db / 20.f)),
        learning_rate_(learning_rate) {
    if (!std::isfinite(floor_db) || floor_db > 0.f || floor_db < -40.f)
      throw std::invalid_argument("unified NS floor must be -40..0 dB");
    if (!std::isfinite(learning_rate) || learning_rate < 0.f ||
        learning_rate > 1.f)
      throw std::invalid_argument("unified NS learning rate must be 0..1");
    weights_ = {.34f, .30f, .23f, .13f};
    target_weights_ = weights_;
    for (unsigned i = 0; i < 2 * hop_; ++i)
      window_[i] =
          std::sin(3.14159265358979323846 * (i + .5) / (2 * hop_));
  }

  unsigned delay() const noexcept { return hop_; }
  unsigned fft_size() const noexcept { return n_; }
  float mean_probability() const noexcept { return mean_probability_; }
  float mean_gain() const noexcept { return mean_gain_; }
  float noise_power() const noexcept { return noise_power_; }
  float bootstrap_seed() const noexcept { return bootstrap_seed_; }
  const std::array<float, 4>& weights() const noexcept { return weights_; }
  const std::array<float, 4>& target_weights() const noexcept {
    return target_weights_;
  }
  float weight_entropy() const noexcept { return weight_entropy_; }
  float frame_snr_db() const noexcept { return frame_snr_db_; }
  float flatness() const noexcept { return flatness_; }
  float tonality() const noexcept { return tonality_; }
  float minima_confidence() const noexcept { return minima_confidence_; }
  float transient_evidence() const noexcept { return transient_evidence_; }
  float noise_step_evidence() const noexcept { return noise_step_evidence_; }
  std::size_t noise_updates() const noexcept { return noise_.noise_updates(); }
  unsigned minima_rotations() const noexcept { return noise_.rotations(); }

  // unity_gain is an explicit reconstruction test hook. Production processing
  // uses the default weighted spectral gain.
  void process(float* audio, bool unity_gain = false) noexcept {
    for (unsigned i = 0; i < hop_; ++i) {
      spectrum_[i] = {past_[i] * window_[i], 0.f};
      spectrum_[i + hop_] = {audio[i] * window_[i + hop_], 0.f};
      past_[i] = audio[i];
    }
    std::fill(spectrum_.begin() + 2 * hop_, spectrum_.end(),
              std::complex<float>{});
    fft_.run(spectrum_.data());
    for (unsigned k = 0; k < bins_; ++k)
      power_[k] = std::max(1e-16f, std::norm(spectrum_[k]));

    const float flat = spectral_flatness();
    if (!noise_.initialized()) {
      bootstrap_seed_ = flat < .15f ? .05f : 1.f;
      noise_.initialize(power_.data(), bootstrap_seed_);
    }

    double xi_sum = 0;
    for (unsigned k = 0; k < bins_; ++k) {
      gamma_[k] =
          std::min(1e8f, power_[k] / std::max(1e-16f, noise_.noise(k)));
      xi_[k] = std::clamp(
          .92f * previous_h1_snr_[k] +
              .08f * std::max(gamma_[k] - 1.f, 0.f),
          1e-3f, 1e8f);
      zeta_[k] = .7f * zeta_[k] + .3f * xi_[k];
      xi_sum += xi_[k];
    }
    const float frame_zeta =
        std::max(1e-8f, float(xi_sum / double(bins_)));
    float frame_probability = 0.f;
    if (frame_zeta > .1f) {
      if (frame_zeta > previous_frame_zeta_) {
        frame_probability = 1.f;
        peak_zeta_ = std::clamp(frame_zeta, 1.f, 10.f);
      } else {
        frame_probability = probability(frame_zeta / peak_zeta_);
      }
    }
    previous_frame_zeta_ = frame_zeta;

    mean_probability_ = 0.f;
    double minima_sum = 0;
    const unsigned low_bin = frequency_bin(150.f);
    const unsigned high_bin = std::min(
        bins_ - 1, frequency_bin(6000.f));
    for (unsigned k = low_bin; k <= high_bin; ++k)
      minima_sum += noise_.indicator(k);
    minima_confidence_ = high_bin >= low_bin ?
        float(minima_sum / double(high_bin - low_bin + 1)) : 0.f;
    flatness_ = flat;
    tonality_ = spectral_tonality(low_bin, high_bin, flat);

    mean_gain_ = 0.f;
    noise_power_ = 0.f;
    for (unsigned k = 0; k < bins_; ++k) {
      const float prior = std::max(
          .005f,
          probability(average(zeta_, k, local_)) *
              probability(average(zeta_, k, global_)) *
              frame_probability);
      const float p =
          omlsa_posterior(prior, xi_[k], gamma_[k]);
      probability_[k] = p;
      noise_used_[k] = noise_.noise(k);
      mean_probability_ += p;
      noise_power_ += noise_used_[k];
    }
    mean_probability_ /= bins_;
    noise_power_ /= bins_ * float(hop_);

    frame_snr_db_ = 10.f * std::log10(std::max(
        1e-12f, average(xi_, bins_ / 2, global_)));
    const float energy_db =
        10.f * std::log10(std::max(
            1e-20, double(
                std::accumulate(power_.begin(), power_.end(), 0.0)) /
                double(bins_)));
    const float rise_db = energy_db - short_energy_db_;
    transient_evidence_ = std::clamp(
        (rise_db - 3.f) / 9.f, 0.f, 1.f) *
        std::clamp((.65f - mean_probability_) / .65f, 0.f, 1.f);
    noise_step_evidence_ = std::clamp(
        (energy_db - slow_energy_db_ - 5.f) / 15.f, 0.f, 1.f) *
        std::max(minima_confidence_, .35f);
    short_energy_db_ += .20f * (energy_db - short_energy_db_);
    slow_energy_db_ += .02f * (energy_db - slow_energy_db_);

    update_weights();
    for (unsigned k = 0; k < bins_; ++k) {
      const float p = probability_[k];
      const float h1_gaussian =
          omlsa_h1_gain(xi_[k], gamma_[k]);
      const float h1_super =
          super_gaussian_h1_gain(xi_[k], gamma_[k]);
      const float gaussian = std::exp(
          p * std::log(std::max(h1_gaussian, floor_gain_)) +
          (1.f - p) * std::log(floor_gain_));
      const float super_gaussian = std::exp(
          p * std::log(std::max(h1_super, floor_gain_)) +
          (1.f - p) * std::log(floor_gain_));
      const float wiener = std::max(
          floor_gain_, xi_[k] / (1.f + xi_[k]));
      const float conservative = std::max(
          floor_gain_,
          std::sqrt(std::max(
              0.f, 1.f - 1.5f / std::max(gamma_[k], 1.f))));
      gain_[k] = std::clamp(
          weights_[0] * gaussian +
              weights_[1] * super_gaussian +
              weights_[2] * wiener +
              weights_[3] * conservative,
          floor_gain_, 1.f);
      previous_h1_snr_[k] =
          h1_gaussian * h1_gaussian * gamma_[k];
      mean_gain_ += gain_[k];
    }
    mean_gain_ /= bins_;

    // Estimate t+1 only after this frame has used the t PSD.
    noise_.update(power_.data(), xi_.data(), gamma_.data());
    if (unity_gain) std::fill(gain_.begin(), gain_.end(), 1.f);
    for (unsigned k = 0; k < bins_; ++k)
      spectrum_[k] *= gain_[k];
    for (unsigned k = 1; k < n_ / 2; ++k)
      spectrum_[n_ - k] = std::conj(spectrum_[k]);
    fft_.run(spectrum_.data(), true);
    for (unsigned i = 0; i < hop_; ++i) {
      audio[i] =
          overlap_[i] + spectrum_[i].real() * window_[i];
      overlap_[i] =
          spectrum_[i + hop_].real() * window_[i + hop_];
    }
  }

 private:
  static unsigned check_rate(unsigned rate) {
    if (rate != 16000 && rate != 24000 && rate != 48000)
      throw std::invalid_argument(
          "unified NS requires native 16/24/48 kHz");
    return rate;
  }

  static std::vector<float> kernel(unsigned radius) {
    std::vector<float> result(2 * radius + 1);
    float sum = 0.f;
    for (unsigned i = 0; i < result.size(); ++i) {
      result[i] = .5f - .5f * std::cos(
          6.283185307179586 * (i + 1) /
          (result.size() + 1));
      sum += result[i];
    }
    for (float& value : result) value /= sum;
    return result;
  }

  unsigned frequency_bin(float hz) const noexcept {
    return std::clamp(
        unsigned(std::lround(double(hz) * n_ / rate_)),
        1u, bins_ - 1);
  }

  float average(const std::vector<float>& values, unsigned k,
                const std::vector<float>& weights) const noexcept {
    float result = 0.f;
    const int radius = int(weights.size() / 2);
    for (unsigned i = 0; i < weights.size(); ++i) {
      result += weights[i] *
          values[unsigned(std::clamp(
              int(k) + int(i) - radius,
              0, int(bins_) - 1))];
    }
    return result;
  }

  static float probability(float snr) noexcept {
    if (snr <= .1f) return 0.f;
    if (snr >= .316227766f) return 1.f;
    return std::log(snr / .1f) / 1.1512925465f;
  }

  float spectral_flatness() const noexcept {
    const unsigned low = frequency_bin(150.f);
    const unsigned high = std::min(
        bins_ - 1, frequency_bin(6000.f));
    if (high < low) return 1.f;
    double log_sum = 0, sum = 0;
    for (unsigned k = low; k <= high; ++k) {
      const double value = std::max(1e-16f, power_[k]);
      log_sum += std::log(value);
      sum += value;
    }
    const double count = high - low + 1;
    return std::clamp(
        float(std::exp(log_sum / count) /
              std::max(1e-16, sum / count)),
        0.f, 1.f);
  }

  float spectral_tonality(unsigned low, unsigned high,
                          float flat) const noexcept {
    if (high < low) return 0.f;
    float peak = 0.f;
    double sum = 0;
    for (unsigned k = low; k <= high; ++k) {
      peak = std::max(peak, power_[k]);
      sum += power_[k];
    }
    const float mean =
        float(sum / double(high - low + 1));
    const float peak_ratio =
        peak / std::max(mean, 1e-16f);
    return std::clamp(
        .65f * std::clamp((peak_ratio - 4.f) / 16.f, 0.f, 1.f) +
            .35f * (1.f - flat),
        0.f, 1.f);
  }

  void update_weights() noexcept {
    const float speech = std::clamp(
        (mean_probability_ - .20f) / .65f, 0.f, 1.f);
    const float snr = std::clamp(
        (frame_snr_db_ + 3.f) / 18.f, 0.f, 1.f);
    const float events = std::max(
        transient_evidence_, noise_step_evidence_);
    std::array<float, 4> target{
        .34f + .15f * (1.f - speech) +
            .10f * minima_confidence_ - .08f * tonality_,
        .30f + .14f * (1.f - speech) + .15f * tonality_,
        .23f + .25f * speech * snr,
        .13f + .23f * events +
            .12f * (1.f - speech) * flatness_};
    float sum = 0.f;
    for (float& value : target) {
      value = std::clamp(value, .01f, 2.f);
      sum += value;
    }
    for (float& value : target) value /= sum;
    target_weights_ = target;
    if (learning_rate_ == 0.f) {
      update_weight_entropy();
      return;
    }
    for (unsigned i = 0; i < weights_.size(); ++i)
      weights_[i] += learning_rate_ * (target[i] - weights_[i]);
    sum = 0.f;
    for (float& value : weights_) {
      value = std::max(.01f, value);
      sum += value;
    }
    for (float& value : weights_) value /= sum;
    update_weight_entropy();
  }

  void update_weight_entropy() noexcept {
    weight_entropy_ = 0.f;
    for (float value : weights_)
      if (value > 0.f) weight_entropy_ -= value * std::log(value);
    weight_entropy_ /= std::log(4.f);
  }

  unsigned rate_, hop_, n_, bins_;
  Fft fft_;
  ImcraNoiseEstimator noise_;
  std::vector<float> window_, past_, overlap_, power_, gamma_, xi_;
  std::vector<float> previous_h1_snr_, zeta_, gain_, noise_used_;
  std::vector<float> probability_, local_, global_;
  std::vector<std::complex<float>> spectrum_;
  std::array<float, 4> weights_{}, target_weights_{};
  float floor_gain_, learning_rate_;
  float previous_frame_zeta_ = 0.f, peak_zeta_ = 1.f;
  float mean_probability_ = 0.f, mean_gain_ = 1.f;
  float noise_power_ = 0.f, bootstrap_seed_ = 1.f;
  float weight_entropy_ = 1.f, frame_snr_db_ = -60.f;
  float flatness_ = 1.f, tonality_ = 0.f;
  float minima_confidence_ = 0.f;
  float transient_evidence_ = 0.f, noise_step_evidence_ = 0.f;
  float short_energy_db_ = -200.f, slow_energy_db_ = -200.f;
};

}  // namespace itu::detail
