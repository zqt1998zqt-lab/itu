#pragma once

#include "ns_spectral_frame.hpp"
#include "residual_feature_bands.hpp"
#include "tiny_ns_residual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace itu::detail {

// Synthesis-only C64 residual control.  Statistical OM-LSA state is never
// touched: this observer reads the current frame, advances its own GRU state,
// and changes only gain[].  All per-frame storage is fixed-size; observe()
// performs no allocation, I/O, or locking.
class ResidualNsControl final : public NsSpectralControl {
 public:
  static constexpr unsigned sample_rate = 16000;
  static constexpr unsigned fft_size = 512;
  static constexpr unsigned bins = fft_size / 2 + 1;

  ResidualNsControl(unsigned rate, unsigned nfft, const std::string& model_path,
                    float floor_db = -9.f, float mix = 1.f)
      : bands_(rate, nfft), floor_gain_(std::pow(10.f, floor_db / 20.f)),
        mix_(mix) {
    if (!std::isfinite(floor_db) || floor_db > 0.f || floor_db < -40.f)
      throw std::invalid_argument("residual control floor must be -40..0 dB");
    if (!std::isfinite(mix) || mix < 0.f || mix > 1.f)
      throw std::invalid_argument("residual control mix must be 0..1");
    if (!model_path.empty()) model_.load(model_path);
  }

  void observe(NsSpectralFrame frame) noexcept override {
    applied_ = false;
    if (!frame.gain || !frame.power || !frame.noise || frame.rate != sample_rate ||
        frame.fft_size != fft_size || frame.bins != bins) {
      ++fallback_frames_;
      return;
    }
    for (unsigned bin = 0; bin < bins; ++bin) {
      last_base_gain_[bin] = frame.gain[bin];
      if (!std::isfinite(frame.gain[bin]) || frame.gain[bin] < 0.f ||
          frame.gain[bin] > 1.f || !std::isfinite(frame.power[bin]) ||
          !std::isfinite(frame.noise[bin])) {
        ++fallback_frames_;
        return;
      }
    }

    // Feature extraction and GRU forward run even when mix == 0.  This is the
    // exact disabled-gain negative control, not a skipped model path.
    if (!bands_.process(frame.power, frame.noise) || !model_.ready() ||
        !model_.process(bands_.features().data(), residual_.data())) {
      ++fallback_frames_;
      return;
    }
    ++inferences_;
    if (mix_ == 0.f) return;

    std::array<float, bins> gains{};
    for (unsigned bin = 0; bin < bins; ++bin) {
      const float delta_db = mix_ * bands_.interpolate(residual_.data(), bin);
      const float gain = frame.gain[bin] * std::pow(10.f, delta_db / 20.f);
      if (!std::isfinite(delta_db) || !std::isfinite(gain)) {
        ++fallback_frames_;
        return;
      }
      gains[bin] = std::clamp(gain, floor_gain_, 1.f);
    }
    for (unsigned bin = 0; bin < bins; ++bin) frame.gain[bin] = gains[bin];
    ++applied_frames_;
    applied_ = true;
  }

  bool model_ready() const noexcept { return model_.ready(); }
  bool applied() const noexcept { return applied_; }
  std::size_t inferences() const noexcept { return inferences_; }
  std::size_t fallback_frames() const noexcept { return fallback_frames_; }
  std::size_t applied_frames() const noexcept { return applied_frames_; }
  float floor_gain() const noexcept { return floor_gain_; }
  float mix() const noexcept { return mix_; }
  const std::array<float, ResidualFeatureBands::input_size>& features() const noexcept {
    return bands_.features();
  }
  const std::array<float, ResidualFeatureBands::bands>& projected_power() const noexcept {
    return bands_.projected_power();
  }
  const std::array<float, ResidualFeatureBands::bands>& projected_noise() const noexcept {
    return bands_.projected_noise();
  }
  const std::array<float, TinyNsResidual::output_size>& residual_db() const noexcept {
    return residual_;
  }
  float residual_db(unsigned bin) const noexcept {
    return bands_.interpolate(residual_.data(), bin);
  }
  const std::array<float, bins>& base_gain() const noexcept { return last_base_gain_; }
  const std::array<float, TinyNsResidual::hidden_size>& hidden() const noexcept {
    return model_.hidden();
  }

  void reset() noexcept {
    model_.reset();
    bands_.reset();
    residual_.fill(0.f);
  }

 private:
  ResidualFeatureBands bands_;
  TinyNsResidual model_;
  std::array<float, TinyNsResidual::output_size> residual_{};
  std::array<float, bins> last_base_gain_{};
  float floor_gain_;
  float mix_;
  std::size_t inferences_ = 0;
  std::size_t fallback_frames_ = 0;
  std::size_t applied_frames_ = 0;
  bool applied_ = false;
};

static_assert(ResidualNsControl::bins == 257,
              "C64 residual control requires 257 bins");

}  // namespace itu::detail
