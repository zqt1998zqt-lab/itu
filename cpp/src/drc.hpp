#pragma once

#include "itu/audio_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

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
      : sample_rate_(sample_rate ? sample_rate : 16000), config_(config),
        delay_(lookahead_samples(config, sample_rate_), 0.0f) {
    update_coefficients();
    reset();
  }

  void configure(const DrcConfig& config) noexcept {
    config_ = config;
    const std::size_t next_delay = lookahead_samples(config, sample_rate_);
    if(next_delay != delay_.size()) {
      delay_.assign(next_delay, 0.0f);
      delay_position_ = 0;
    }
    update_coefficients();
  }

  void set_sample_rate(unsigned sample_rate) noexcept {
    sample_rate_ = sample_rate ? sample_rate : 16000;
    delay_.assign(lookahead_samples(config_, sample_rate_), 0.0f);
    delay_position_ = 0;
    update_coefficients();
    reset();
  }

  void reset() noexcept {
    envelope_ = 1e-6f;
    gain_linear_ = 1.0f;
    std::fill(delay_.begin(), delay_.end(), 0.0f);
    delay_position_ = 0;
    delay_primed_ = false;
    delayed_remaining_ = 0;
  }

  // In-place processing for 10ms (or arbitrary) buffer of float PCM. With
  // lookahead enabled, a bounded ring delays the signal while the gain
  // detector sees future peaks first; flush() drains the delayed tail.
  void process(float* data, std::size_t n) noexcept {
    if (!config_.enabled || !data || n == 0) return;
    if (delay_.empty()) {
      process_block(data, n);
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      const float future = data[i];
      data[i] = delayed_sample();
      process_sample(future);
      delay_[delay_position_] = future;
      delay_position_ = (delay_position_ + 1) % delay_.size();
      delay_primed_ = true;
      if (delayed_remaining_ < delay_.size()) ++delayed_remaining_;
    }
  }

  std::size_t delay_samples() const noexcept { return delay_.size(); }

  std::size_t memory_bytes() const noexcept {
    return sizeof(*this) + delay_.capacity() * sizeof(float);
  }

  // Returns one delayed sample per call; the caller inserts it after the input.
  bool flush_sample(float& sample) noexcept {
    if (!config_.enabled || delay_.empty() || !delay_primed_ ||
        delayed_remaining_ == 0) return false;
    sample = delayed_sample();
    delay_[delay_position_] = 0.0f;
    delay_position_ = (delay_position_ + 1) % delay_.size();
    --delayed_remaining_;
    return true;
  }

  bool lookahead_active() const noexcept {
    return !delay_.empty() && delayed_remaining_ != 0;
  }

 private:
  static std::size_t lookahead_samples(const DrcConfig& config,
                                       unsigned sample_rate) noexcept {
    if (!std::isfinite(config.lookahead_ms) || config.lookahead_ms <= 0.0f)
      return 0;
    return static_cast<std::size_t>(
        std::lround(config.lookahead_ms * 0.001f * sample_rate));
  }

  float delayed_sample() noexcept {
    if (delay_.empty()) return 0.0f;
    return delay_[delay_position_];
  }

  void process_block(float* data, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) data[i] = process_sample(data[i]);
  }

  float process_sample(float x) noexcept {
    const float ceiling_linear = std::pow(10.0f, config_.ceiling_dbfs / 20.0f);
    const float makeup_linear = std::pow(10.0f, config_.makeup_gain_db / 20.0f);
    const float half_knee = config_.knee_width_db * 0.5f;
    const float inv_ratio_minus_one = (1.0f / std::max(1.0f, config_.ratio)) - 1.0f;

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
    return y;
  }

 public:
  float current_gain_db() const noexcept {
    return 20.0f * std::log10(std::max(1e-6f, gain_linear_));
  }

  float current_envelope_dbfs() const noexcept {
    return 20.0f * std::log10(std::max(1e-6f, envelope_));
  }

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
  std::vector<float> delay_;
  std::size_t delay_position_ = 0;
  bool delay_primed_ = false;
  std::size_t delayed_remaining_ = 0;
};

} // namespace itu
