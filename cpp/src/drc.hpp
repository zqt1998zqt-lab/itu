#pragma once

#include "itu/audio_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace itu {

// Real-time Dynamic Range Compressor (DRC) and Soft Limiter for automotive audio.
// Follows the Barrot VCP specification for both Uplink (TX) and Downlink (RX) paths:
// 1. Level tracking with decoupled attack/release smoothing.
// 2. Soft-knee compression curve avoiding sudden gain inflection.
// 3. Transparent soft-saturation ceiling preventing loudspeaker/codec clipping.
class DynamicRangeCompressor {
 public:
  explicit DynamicRangeCompressor(unsigned sample_rate = 16000,
                                  DrcConfig config = {})
      : sample_rate_(sample_rate ? sample_rate : 16000), config_(config) {
    update_coefficients();
    reset();
  }

  void configure(const DrcConfig& config) noexcept {
    config_ = config;
    update_coefficients();
  }

  void set_sample_rate(unsigned sample_rate) noexcept {
    sample_rate_ = sample_rate ? sample_rate : 16000;
    update_coefficients();
    reset();
  }

  void reset() noexcept {
    envelope_ = 1e-6f;
    gain_linear_ = 1.0f;
  }

  // In-place processing for 10ms (or arbitrary) buffer of float PCM [-1.0, 1.0]
  void process(float* data, std::size_t n) noexcept {
    if (!config_.enabled || !data || n == 0) return;

    const float ceiling_linear = std::pow(10.0f, config_.ceiling_dbfs / 20.0f);
    const float makeup_linear = std::pow(10.0f, config_.makeup_gain_db / 20.0f);
    const float half_knee = config_.knee_width_db * 0.5f;
    const float inv_ratio_minus_one = (1.0f / std::max(1.0f, config_.ratio)) - 1.0f;

    for (std::size_t i = 0; i < n; ++i) {
      const float x = data[i];
      const float abs_x = std::abs(x);

      // 1. Envelope detector
      if (abs_x > envelope_) {
        envelope_ = alpha_att_ * envelope_ + (1.0f - alpha_att_) * abs_x;
      } else {
        envelope_ = alpha_rel_ * envelope_ + (1.0f - alpha_rel_) * abs_x;
      }

      // Convert envelope to dBFS (clamped to avoid log(0))
      const float env_dbfs = 20.0f * std::log10(std::max(1e-6f, envelope_));

      // 2. Compute gain reduction in dB using soft-knee curve
      float gain_reduction_db = 0.0f;
      const float diff = env_dbfs - config_.threshold_dbfs;

      if (config_.knee_width_db > 0.0f && std::abs(diff) <= half_knee) {
        // Inside the soft-knee transition zone
        const float knee_factor = diff + half_knee;
        gain_reduction_db = inv_ratio_minus_one * (knee_factor * knee_factor) /
                            (2.0f * config_.knee_width_db);
      } else if (diff > half_knee) {
        // Above the knee: full compression
        gain_reduction_db = inv_ratio_minus_one * diff;
      }

      // Linear target compression gain
      const float target_gain = std::pow(10.0f, gain_reduction_db / 20.0f) * makeup_linear;

      // Smooth gain transitions
      gain_linear_ += 0.05f * (target_gain - gain_linear_);

      // Apply compression
      float y = x * gain_linear_;

      // 3. Transparent soft limiter / saturation near ceiling
      // Prevents loudspeaker clipping distortion and acoustic nonlinearities
      if (std::abs(y) > ceiling_linear * 0.85f) {
        const float threshold = ceiling_linear * 0.85f;
        const float headroom = ceiling_linear - threshold;
        const float excess = std::abs(y) - threshold;
        if (headroom > 1e-5f) {
          const float saturated = threshold + headroom * std::tanh(excess / headroom);
          y = std::copysign(saturated, y);
        } else {
          y = std::clamp(y, -ceiling_linear, ceiling_linear);
        }
      }

      data[i] = y;
    }
  }

  float current_gain_db() const noexcept {
    return 20.0f * std::log10(std::max(1e-6f, gain_linear_));
  }

  float current_envelope_dbfs() const noexcept {
    return 20.0f * std::log10(std::max(1e-6f, envelope_));
  }

 private:
  void update_coefficients() noexcept {
    const float att_time = std::max(0.0001f, config_.attack_ms * 0.001f);
    const float rel_time = std::max(0.001f, config_.release_ms * 0.001f);
    alpha_att_ = std::exp(-1.0f / (att_time * static_cast<float>(sample_rate_)));
    alpha_rel_ = std::exp(-1.0f / (rel_time * static_cast<float>(sample_rate_)));
  }

  unsigned sample_rate_ = 16000;
  DrcConfig config_;
  float alpha_att_ = 0.9f;
  float alpha_rel_ = 0.99f;
  float envelope_ = 1e-6f;
  float gain_linear_ = 1.0f;
};

} // namespace itu
