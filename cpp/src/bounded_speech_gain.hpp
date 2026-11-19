#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace itu::detail {

// A fixed-hop speech-gain envelope. process() is called once per 10 ms frame
// and retains only scalar control state; it does not inspect or delay audio.
class BoundedSpeechGain {
 public:
  struct Settings {
    float max_db = 3.f;
    float rise_db_per_second = 3.f;
    float fall_db_per_second = 6.f;
    unsigned hold_ms = 120;
  };

  BoundedSpeechGain() { validate_settings(settings_); }

  explicit BoundedSpeechGain(Settings settings) : settings_(settings) {
    validate_settings(settings_);
  }

  float process(float requested_db, bool speech_qualified,
                bool far_active) noexcept {
    if (far_active) {
      reset();
      return gain_db_;
    }

    const float target = std::isfinite(requested_db)
                             ? std::clamp(requested_db, 0.f, settings_.max_db)
                             : 0.f;
    const bool refresh_hold = speech_qualified && target > 0.f;
    const bool had_hold = hold_frames_ > 0;
    const float held_value = held_target_db_;

    if (refresh_hold) {
      held_target_db_ = target;
      hold_frames_ = frames_for_hold(settings_.hold_ms);
    } else if (hold_frames_ > 0) {
      --hold_frames_;
      if (hold_frames_ == 0) held_target_db_ = 0.f;
    } else {
      held_target_db_ = 0.f;
    }

    // Only an unqualified silence consumes the held target. Positive requests
    // are always bounded by the current request, but cannot renew hold time.
    const float effective_target =
        (!speech_qualified && target <= 0.f && had_hold) ? held_value : target;

    if (effective_target > gain_db_) {
      gain_db_ = std::min(effective_target,
                          gain_db_ + settings_.rise_db_per_second * .01f);
    } else if (effective_target < gain_db_) {
      gain_db_ = std::max(effective_target,
                          gain_db_ - settings_.fall_db_per_second * .01f);
    }
    gain_db_ = std::clamp(gain_db_, 0.f, settings_.max_db);
    return gain_db_;
  }

  void reset() noexcept {
    gain_db_ = 0.f;
    held_target_db_ = 0.f;
    hold_frames_ = 0;
  }

  [[nodiscard]] float gain_db() const noexcept { return gain_db_; }
  [[nodiscard]] float held_target_db() const noexcept {
    return held_target_db_;
  }
  [[nodiscard]] unsigned hold_frames() const noexcept { return hold_frames_; }

 private:
  void validate_settings(const Settings& settings) const {
    if (!std::isfinite(settings.max_db) || settings.max_db < 0.f ||
        settings.max_db > 18.f ||
        !std::isfinite(settings.rise_db_per_second) ||
        settings.rise_db_per_second < .1f ||
        settings.rise_db_per_second > 20.f ||
        !std::isfinite(settings.fall_db_per_second) ||
        settings.fall_db_per_second < .1f ||
        settings.fall_db_per_second > 30.f || settings.hold_ms > 300u) {
      throw std::invalid_argument("invalid BoundedSpeechGain settings");
    }
  }

  [[nodiscard]] static unsigned frames_for_hold(unsigned hold_ms) noexcept {
    return (hold_ms + 9u) / 10u;
  }

  Settings settings_{};
  float gain_db_ = 0.f;
  float held_target_db_ = 0.f;
  unsigned hold_frames_ = 0;
};

}  // namespace itu::detail
