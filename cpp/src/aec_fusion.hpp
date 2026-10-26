#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace itu {

// A bounded, causal short-tail NLMS shadow estimator. It never owns the
// primary AEC clock: the caller decides when to adapt and how much residual
// to blend. All vectors are sized in the constructor so process() is callback
// safe and performs no allocation.
struct AecFusionFrame {
  float shadow_weight = 0.f;
  float primary_error_power = 0.f;
  float shadow_error_power = 0.f;
  float reference_confidence = 0.f;
  bool double_talk = false;
  unsigned updates = 0;
};

class AecFusion {
 public:
  AecFusion(unsigned sample_rate, unsigned frame_size, float learning_rate,
            unsigned shadow_taps_ms)
      : frame_size_(frame_size),
        taps_(std::clamp((sample_rate * shadow_taps_ms) / 1000u, 16u, 1024u)),
        learning_rate_(std::clamp(learning_rate, .005f, .5f)),
        coefficients_(taps_, 0.f), history_(taps_, 0.f), shadow_error_(frame_size, 0.f) {}

  void process(const float* capture, const float* render, bool adapt) noexcept {
    double ref_power = 0.;
    unsigned updates = 0;
    for (unsigned i = 0; i < frame_size_; ++i) {
      history_[head_] = render[i];
      double estimate = 0., norm = 1e-7;
      for (unsigned tap = 0; tap < taps_; ++tap) {
        const float value = history_[(head_ + taps_ - tap) % taps_];
        estimate += double(coefficients_[tap]) * value;
        norm += double(value) * value;
      }
      const float error = capture[i] - static_cast<float>(estimate);
      shadow_error_[i] = error;
      ref_power += double(render[i]) * render[i];
      if (adapt && norm > 1e-6) {
        const float step = learning_rate_ * error / static_cast<float>(norm);
        for (unsigned tap = 0; tap < taps_; ++tap) {
          const float value = history_[(head_ + taps_ - tap) % taps_];
          coefficients_[tap] = std::clamp(coefficients_[tap] + step * value, -2.f, 2.f);
        }
        ++updates;
      }
      head_ = (head_ + 1) % taps_;
    }
    reference_power_ = static_cast<float>(ref_power / std::max(1u, frame_size_));
    updates_ = updates;
  }

  void blend(float* primary, bool far_active, unsigned reference_state,
             float reference_confidence, float incoherent_ratio,
             float nonlinear_speech_probability, unsigned near_hold) noexcept {
    double primary_power = 0., shadow_power = 0.;
    for (unsigned i = 0; i < frame_size_; ++i) {
      primary_power += double(primary[i]) * primary[i];
      shadow_power += double(shadow_error_[i]) * shadow_error_[i];
    }
    primary_power_ = static_cast<float>(primary_power / std::max(1u, frame_size_));
    shadow_power_ = static_cast<float>(shadow_power / std::max(1u, frame_size_));
    reference_confidence_ = std::clamp(reference_confidence, 0.f, 1.f);
    double_talk_ = near_hold > 0 || reference_state == 2 ||
        (far_active && incoherent_ratio > .45f && nonlinear_speech_probability > .55f);

    float target = .02f;
    if (far_active && reference_state == 1 && !double_talk_ && reference_power_ > 1e-7f) {
      const float advantage = std::clamp(
          (primary_power_ - shadow_power_) /
              std::max(primary_power_ + shadow_power_, 1e-8f), -1.f, 1.f);
      const float confidence = std::clamp(.35f + .65f * reference_confidence_, .35f, 1.f);
      target = confidence * (.08f + .72f * std::clamp(.5f + .5f * advantage, 0.f, 1.f));
      target = std::clamp(target, .05f, .85f);
    }
    // Enter faster when the short-tail estimator is better; leave slowly on
    // near speech so the adaptive shadow cannot pump a syllable valley.
    const float delta = target - shadow_weight_;
    shadow_weight_ += std::clamp(delta, -.12f, .08f);
    shadow_weight_ = std::clamp(shadow_weight_, .02f, .85f);
    for (unsigned i = 0; i < frame_size_; ++i)
      primary[i] = (1.f - shadow_weight_) * primary[i] + shadow_weight_ * shadow_error_[i];
  }

  AecFusionFrame frame() const noexcept {
    return {shadow_weight_, primary_power_, shadow_power_, reference_confidence_,
            double_talk_, updates_};
  }
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this) + (coefficients_.capacity() + history_.capacity() +
                             shadow_error_.capacity()) * sizeof(float);
  }

 private:
  unsigned frame_size_;
  unsigned taps_;
  float learning_rate_;
  std::vector<float> coefficients_;
  std::vector<float> history_;
  std::vector<float> shadow_error_;
  unsigned head_ = 0;
  unsigned updates_ = 0;
  float reference_power_ = 0.f;
  float shadow_weight_ = .02f;
  float primary_power_ = 0.f;
  float shadow_power_ = 0.f;
  float reference_confidence_ = 0.f;
  bool double_talk_ = false;
};

}  // namespace itu
