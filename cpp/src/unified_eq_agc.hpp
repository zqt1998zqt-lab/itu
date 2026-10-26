#pragma once

#include "eq_processor.hpp"
#include "parametric_eq.hpp"
#include "itu/audio_engine.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace itu {

// Causal fusion for the two outer tone/level stages. EQ combines the configured
// tonal curve with a phase-matched transparent expert. AGC combines the
// conservative physical-level estimator with the utterance estimator. Both
// decisions use current/past evidence only and are smoothed before the caller's
// existing gain slew limiter. With a flat EQ configuration this is transparent.
class UnifiedEqAgc {
 public:
  UnifiedEqAgc(const EqConfig& config, unsigned rate, unsigned frame_size)
      : frame_size_(frame_size), eq_enabled_(ParametricEq::enabled(config)) {
    if (!eq_enabled_) return;
    shaped_ = std::make_unique<EqProcessor>(config, rate);
    EqConfig flat = config;
    flat.preamp_db = 0.f;
    flat.bands = 0;
    flat_ = std::make_unique<EqProcessor>(flat, rate);
    shaped_buffer_.resize(frame_size_);
    flat_buffer_.resize(frame_size_);
  }

  float process_eq(float* pcm, unsigned count, float control_probability,
                   float unified_probability, float linear_probability,
                   float nonlinear_probability, bool far_active,
                   float near_confidence) noexcept {
    if (!eq_enabled_ || !pcm || count > frame_size_) {
      eq_weight_ = 0.f;
      return eq_weight_;
    }
    std::copy_n(pcm, count, shaped_buffer_.data());
    std::copy_n(pcm, count, flat_buffer_.data());
    shaped_->process(shaped_buffer_.data(), count);
    flat_->process(flat_buffer_.data(), count);
    const float speech = std::clamp(std::max({control_probability,
        unified_probability, linear_probability, nonlinear_probability}), 0.f, 1.f);
    const float confidence = std::clamp(near_confidence, 0.f, 1.f);
    const float target = (far_active ? .10f + .35f * speech
                                     : .55f + .45f * speech) *
        (.55f + .45f * confidence);
    eq_weight_ += std::clamp(.16f * (target - eq_weight_), -.15f, .15f);
    for (unsigned i = 0; i < count; ++i)
      pcm[i] = flat_buffer_[i] + eq_weight_ * (shaped_buffer_[i] - flat_buffer_[i]);
    return eq_weight_;
  }

  float fuse_gain(float conservative_db, float utterance_db, float maximum_db,
                  float control_probability, float unified_probability,
                  float linear_probability, float nonlinear_probability,
                  float near_confidence, bool far_active,
                  bool conservative_qualified,
                  bool utterance_qualified) noexcept {
    if (far_active) {
      agc_weight_ = 0.f;
      agc_gain_db_ = 0.f;
      return 0.f;
    }
    const float speech = std::clamp(std::max({control_probability,
        unified_probability, linear_probability, nonlinear_probability}), 0.f, 1.f);
    const float confidence = std::clamp(near_confidence, 0.f, 1.f);
    const float utterance_evidence = utterance_qualified || utterance_db > .01f
        ? .55f + .35f * speech : .05f * speech;
    const float conservative_evidence = conservative_qualified
        ? .35f + .25f * speech : .05f * speech;
    const float total = utterance_evidence + conservative_evidence;
    if (total <= 1e-6f) {
      agc_weight_ = 0.f;
      agc_gain_db_ = 0.f;
      return 0.f;
    }
    const float target_weight = std::clamp(
        utterance_evidence / total * (.65f + .35f * confidence), 0.f, 1.f);
    agc_weight_ += std::clamp(.18f * (target_weight - agc_weight_), -.15f, .15f);
    const float fused = (1.f - agc_weight_) * conservative_db +
        agc_weight_ * utterance_db;
    agc_gain_db_ = std::clamp(fused, 0.f, std::max(0.f, maximum_db));
    return agc_gain_db_;
  }

  bool eq_enabled() const noexcept { return eq_enabled_; }
  float eq_weight() const noexcept { return eq_weight_; }
  float agc_weight() const noexcept { return agc_weight_; }
  float agc_gain_db() const noexcept { return agc_gain_db_; }
  unsigned delay() const noexcept { return shaped_ ? shaped_->delay() : 0; }
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this) + shaped_buffer_.capacity() * sizeof(float) +
        flat_buffer_.capacity() * sizeof(float) +
        (shaped_ ? shaped_->memory_bytes() : 0) +
        (flat_ ? flat_->memory_bytes() : 0);
  }

 private:
  unsigned frame_size_ = 0;
  bool eq_enabled_ = false;
  std::unique_ptr<EqProcessor> shaped_, flat_;
  std::vector<float> shaped_buffer_, flat_buffer_;
  float eq_weight_ = 0.f;
  float agc_weight_ = .5f;
  float agc_gain_db_ = 0.f;
};

}  // namespace itu
