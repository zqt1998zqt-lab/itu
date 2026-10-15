#pragma once

#include "itu/audio_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace itu {

// Context-bounded online control. The caller supplies the product mode and
// sample rate; this controller only adapts state inside that mode's envelope.
class AdaptiveController {
 public:
  struct Context {
    ProcessingMode mode = ProcessingMode::general;
    unsigned sample_rate = 16000;
    unsigned frame_size = 160;
    unsigned microphones = 1;
    unsigned reference_channels = 1;
  };
  struct Observation {
    float level = 0;
    float spectral_excess = 0;
    float linear_speech = 0;
    float nonlinear_speech = 0;
    float residual_ratio = 1;
    float coherence = 1;
    float wind = 0;
    bool far_active = false;
    bool near_hold = false;
    bool reference_reliable = false;
  };
  struct Decision {
    float noise_floor = 1e-12f;
    float near_confidence = 0;
    float far_confidence = 0;
    float preserve_mix = 0;
    float gain_limit_db = 0;
    unsigned state = 0; // 0 uncertain, 1 noise, 2 near speech, 3 double talk.
  };

  explicit AdaptiveController(Context context) : context_(context), policy_(policy_for(context)) {
    reset();
  }

  void reset() noexcept {
    frames_ = 0;
    noise_floor_ = 1e-8;
    preserve_mix_ = 0;
    gain_limit_db_ = policy_.max_gain_db;
    step_hold_ = 0;
    state_ = 0;
  }

  Decision observe(const Observation& input) noexcept {
    ++frames_;
    const double power = std::max(1e-14, double(input.level) * input.level);
    const float speech = std::clamp(std::max({input.spectral_excess,
                                               .65f * input.linear_speech,
                                               .75f * input.nonlinear_speech}), 0.f, 1.f);
    const bool quiet_candidate = !input.far_active && !input.near_hold && speech < .18f;
    const double ratio = power / std::max(noise_floor_, 1e-12);
    if (quiet_candidate) {
      if (frames_ > 12 && ratio > 4.) {
        step_hold_ = std::min(50u, step_hold_ + 1);
        if (step_hold_ == 1) noise_floor_ = power;
      }
      const double alpha = step_hold_ ? .12 : (frames_ < 30 ? 1. / double(frames_) : .025);
      noise_floor_ += alpha * (power - noise_floor_);
    } else if (step_hold_) {
      --step_hold_;
    }

    const float snr_conf = static_cast<float>(std::clamp(
        (10. * std::log10(std::max(ratio, 1e-6)) - policy_.snr_start_db) /
            std::max(1.f, policy_.snr_full_db - policy_.snr_start_db),
        0., 1.));
    float near = std::clamp(.82f * speech + .18f * snr_conf, 0.f, 1.f);
    if (frames_ < 30) near *= .35f; // Do not mistake cold-start noise for speech.
    if (input.near_hold) near = std::max(near, .75f);

    float far = input.far_active ? (input.reference_reliable ? .82f : .55f) : 0.f;
    if (input.far_active && input.reference_reliable)
      far *= std::clamp(.65f + .35f * (1.f - input.coherence), .55f, 1.f);
    const float double_talk = near * far;
    const float far_only = far * (1.f - near);
    const float wind_penalty = std::clamp(input.wind, 0.f, 1.f) * policy_.wind_penalty;

    float target = policy_.minimum_preserve + policy_.near_boost * near +
        policy_.double_talk_boost * double_talk - policy_.far_only_penalty * far_only - wind_penalty;
    target = std::clamp(target, 0.f, 1.f);
    const float step = target >= preserve_mix_ ? policy_.attack_step : policy_.release_step;
    preserve_mix_ += std::clamp(target - preserve_mix_, -step, step);

    const float noise_penalty = std::clamp((snr_conf - .2f) * policy_.noise_gain_penalty, 0.f, policy_.noise_gain_penalty);
    const float gain_target = std::clamp(policy_.max_gain_db - noise_penalty + policy_.near_gain_boost * near -
                                             policy_.far_gain_penalty * far_only,
                                         policy_.min_gain_db, policy_.max_gain_db);
    const float gain_step = gain_target >= gain_limit_db_ ? policy_.gain_attack_db : policy_.gain_release_db;
    gain_limit_db_ += std::clamp(gain_target - gain_limit_db_, -gain_step, gain_step);

    state_ = frames_ < 30 ? 0 : double_talk > .35f ? 3 : near > .45f ? 2 : far > .45f ? 1 : 1;
    return {static_cast<float>(noise_floor_), near, far, preserve_mix_, gain_limit_db_, state_};
  }

  const Context& context() const noexcept { return context_; }

 private:
  struct Policy {
    float minimum_preserve = 0;
    float near_boost = .85f;
    float double_talk_boost = .2f;
    float far_only_penalty = .8f;
    float wind_penalty = .1f;
    float snr_start_db = 3.f;
    float snr_full_db = 15.f;
    float max_gain_db = 18.f;
    float min_gain_db = 0;
    float near_gain_boost = 1.5f;
    float far_gain_penalty = 4.f;
    float noise_gain_penalty = 6.f;
    float attack_step = .12f;
    float release_step = .035f;
    float gain_attack_db = .8f;
    float gain_release_db = .25f;
  };

  static Policy policy_for(const Context& c) noexcept {
    Policy p;
    switch (c.mode) {
      case ProcessingMode::call:
        p.minimum_preserve=.10f; p.near_boost=.95f; p.double_talk_boost=.3f;
        p.far_only_penalty=.7f; p.max_gain_db=12.f; p.noise_gain_penalty=4.f;
        break;
      case ProcessingMode::ecnr:
        p.minimum_preserve=.03f; p.near_boost=.82f; p.double_talk_boost=.25f;
        p.far_only_penalty=.95f; p.max_gain_db=18.f; p.noise_gain_penalty=7.f;
        break;
      case ProcessingMode::vehicle_stream:
        p.minimum_preserve=.05f; p.near_boost=.88f; p.double_talk_boost=.28f;
        p.far_only_penalty=.9f; p.wind_penalty=.16f; p.max_gain_db=18.f;
        break;
      case ProcessingMode::referenced_radio:
        p.minimum_preserve=.08f; p.near_boost=.9f; p.far_only_penalty=.82f;
        p.max_gain_db=15.f;
        break;
      case ProcessingMode::general:
      default:
        break;
    }
    // The controller runs on a 10 ms frame at every I/O rate. Higher-rate
    // input gets slightly more weight from the high-band evidence without
    // changing the output clock or the state update cadence.
    if (c.sample_rate >= 48000) p.near_boost += .03f;
    else if (c.sample_rate == 24000) p.near_boost += .015f;
    if (c.microphones > 1) p.double_talk_boost += .03f;
    if (c.reference_channels > 1) p.far_only_penalty += .03f;
    return p;
  }

  Context context_;
  Policy policy_;
  std::uint64_t frames_ = 0;
  double noise_floor_ = 1e-8;
  float preserve_mix_ = 0;
  float gain_limit_db_ = 0;
  unsigned step_hold_ = 0;
  unsigned state_ = 0;
};

} // namespace itu
