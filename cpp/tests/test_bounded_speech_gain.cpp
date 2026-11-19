#include "bounded_speech_gain.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using itu::detail::BoundedSpeechGain;

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

bool near(float actual, float expected, float tolerance = 1e-5f) {
  return std::abs(actual - expected) <= tolerance;
}

void expect_invalid(BoundedSpeechGain::Settings settings, const char* label) {
  try {
    BoundedSpeechGain gain(settings);
    (void)gain;
  } catch (const std::invalid_argument&) {
    return;
  }
  std::cerr << label << '\n';
  std::exit(1);
}

void test_rise_cap_and_steady_state() {
  BoundedSpeechGain gain;
  float previous = gain.process(8.f, true, false);
  require(near(previous, .03f), "first qualified +8 dB frame must rise one step");
  require(near(gain.held_target_db(), 3.f), "+8 dB request must clamp held target to max");
  require(gain.hold_frames() == 12u, "120 ms hold must be 12 ten-ms frames");

  for (unsigned frame = 0; frame < 200; ++frame) {
    const float next = gain.process(8.f, true, false);
    require(next >= previous, "rise toward a fixed target must be monotonic");
    require(next - previous <= .030001f, "rise step exceeded 3 dB/s at 10 ms");
    require(next >= 0.f && next <= 3.f, "gain escaped [0, max_db] while rising");
    previous = next;
  }
  require(near(gain.gain_db(), 3.f), "sustained +8 dB request must settle at 3 dB");
}

void test_hold_and_release() {
  BoundedSpeechGain gain;
  for (unsigned frame = 0; frame < 120; ++frame) gain.process(8.f, true, false);
  require(near(gain.gain_db(), 3.f), "setup must reach the capped target");

  for (unsigned frame = 0; frame < 12; ++frame) {
    const unsigned before = gain.hold_frames();
    require(near(gain.process(0.f, false, false), 3.f),
            "finite hold must preserve the prior target");
    require(gain.hold_frames() + 1u == before,
            "each silent hold frame must consume exactly one hold frame");
  }
  require(gain.hold_frames() == 0u, "hold must expire after its configured duration");

  float previous = gain.gain_db();
  for (unsigned frame = 0; frame < 50; ++frame) {
    const float next = gain.process(0.f, false, false);
    require(next <= previous, "release must be monotonic");
    require(previous - next <= .060001f, "release exceeded 6 dB/s at 10 ms");
    require(next >= 0.f && next <= 3.f, "gain escaped [0, max_db] while releasing");
    previous = next;
  }
  require(near(gain.gain_db(), 0.f), "release must return to unity gain");
  require(near(gain.held_target_db(), 0.f), "expired hold target must be cleared");
}

void test_unqualified_positive_does_not_refresh_hold() {
  BoundedSpeechGain gain;
  for (unsigned frame = 0; frame < 100; ++frame) gain.process(3.f, true, false);
  require(gain.hold_frames() == 12u, "qualified request must initialize hold");
  require(near(gain.gain_db(), 3.f), "hold-refresh test needs a settled gain");

  for (unsigned frame = 0; frame < 8; ++frame) {
    const unsigned before = gain.hold_frames();
    const float out = gain.process(3.f, false, false);
    require(out >= 0.f && out <= 3.f, "unqualified positive request escaped gain bounds");
    require(gain.hold_frames() <= before,
            "unqualified positive request must not refresh hold");
  }
  require(gain.hold_frames() == 4u,
          "unqualified frames must not extend the qualified hold deadline");

  for (unsigned frame = 0; frame < 4; ++frame) {
    require(near(gain.process(0.f, false, false), 3.f),
            "remaining hold must retain the last qualified target");
  }
  require(gain.hold_frames() == 0u, "hold must expire despite intervening requests");
  const float before_release = gain.gain_db();
  const float released = gain.process(0.f, false, false);
  require(released < before_release && before_release - released <= .060001f,
          "expired hold must release instead of becoming permanent");
}

void test_far_end_is_immediate_and_leaves_no_stale_state() {
  BoundedSpeechGain gain;
  for (unsigned frame = 0; frame < 100; ++frame) gain.process(3.f, true, false);
  require(gain.gain_db() > 2.9f, "far-end test needs an active boost");
  require(gain.process(18.f, true, true) == 0.f,
          "far-end frame must immediately remove boost");
  require(gain.gain_db() == 0.f && gain.held_target_db() == 0.f &&
              gain.hold_frames() == 0u,
          "far-end frame must clear all envelope state");
  require(gain.process(0.f, false, false) == 0.f,
          "far-end state must not restore a stale held target");

  const float restarted = gain.process(1.f, true, false);
  require(near(restarted, .03f),
          "post-far qualified speech must restart with one bounded rise step");
}

void test_reset_is_deterministic() {
  BoundedSpeechGain first;
  BoundedSpeechGain second;
  for (unsigned frame = 0; frame < 60; ++frame) first.process(3.f, true, false);
  second.process(1.f, true, false);
  second.process(0.f, false, false);
  first.reset();
  second.reset();
  require(first.gain_db() == 0.f && first.held_target_db() == 0.f &&
              first.hold_frames() == 0u,
          "reset must clear every envelope field");
  require(first.process(2.f, true, false) == second.process(2.f, true, false),
          "reset instances must evolve identically");
}

void test_zero_max_is_unity() {
  BoundedSpeechGain::Settings settings;
  settings.max_db = 0.f;
  BoundedSpeechGain gain(settings);
  for (unsigned frame = 0; frame < 1000; ++frame) {
    require(gain.process(18.f, true, false) == 0.f,
            "max_db=0 must remain at unity gain");
  }
  require(gain.held_target_db() == 0.f,
          "a request clamped to zero must not establish a held boost");
}

void test_legal_setting_boundaries() {
  BoundedSpeechGain::Settings minimum;
  minimum.max_db = 0.f;
  minimum.rise_db_per_second = .1f;
  minimum.fall_db_per_second = .1f;
  minimum.hold_ms = 0u;
  BoundedSpeechGain low(minimum);
  require(low.process(0.f, true, false) == 0.f,
          "minimum legal settings must be usable");

  BoundedSpeechGain::Settings maximum;
  maximum.max_db = 18.f;
  maximum.rise_db_per_second = 20.f;
  maximum.fall_db_per_second = 30.f;
  maximum.hold_ms = 300u;
  BoundedSpeechGain high(maximum);
  require(near(high.process(30.f, true, false), .2f),
          "maximum legal rise must still use exactly one 10 ms step");
  require(high.hold_frames() == 30u,
          "maximum legal hold must round to 30 frames");
  require(near(high.held_target_db(), 18.f),
          "maximum legal request must establish the held target");
  const float held = high.process(0.f, false, false);
  require(held > .2f && held <= .400001f,
          "maximum legal hold must retain its target under the rise bound");
  const float still_held = high.process(0.f, false, false);
  require(still_held > held && still_held <= .600001f,
          "hold frame accounting must continue toward the held target");

  BoundedSpeechGain::Settings fractional;
  fractional.max_db = 1.f;
  fractional.rise_db_per_second = 1.f;
  fractional.fall_db_per_second = 2.f;
  fractional.hold_ms = 1u;
  BoundedSpeechGain rounded(fractional);
  rounded.process(1.f, true, false);
  require(rounded.hold_frames() == 1u, "hold_ms must round up to whole 10 ms frames");
}

void test_invalid_settings_are_rejected() {
  BoundedSpeechGain::Settings settings;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  settings = {};
  settings.max_db = -0.01f;
  expect_invalid(settings, "negative max_db must be rejected");
  settings = {};
  settings.max_db = 18.01f;
  expect_invalid(settings, "max_db above 18 must be rejected");
  settings = {};
  settings.rise_db_per_second = .09f;
  expect_invalid(settings, "rise below 0.1 dB/s must be rejected");
  settings = {};
  settings.rise_db_per_second = 20.01f;
  expect_invalid(settings, "rise above 20 dB/s must be rejected");
  settings = {};
  settings.fall_db_per_second = .09f;
  expect_invalid(settings, "fall below 0.1 dB/s must be rejected");
  settings = {};
  settings.fall_db_per_second = 30.01f;
  expect_invalid(settings, "fall above 30 dB/s must be rejected");
  settings = {};
  settings.hold_ms = 301u;
  expect_invalid(settings, "hold above 300 ms must be rejected");

  settings = {};
  settings.max_db = nan;
  expect_invalid(settings, "NaN max_db must be rejected");
  settings = {};
  settings.rise_db_per_second = inf;
  expect_invalid(settings, "infinite rise must be rejected");
  settings = {};
  settings.fall_db_per_second = nan;
  expect_invalid(settings, "NaN fall must be rejected");
}

void test_nonfinite_request_is_zero_without_hold_refresh() {
  BoundedSpeechGain gain;
  gain.process(3.f, true, false);
  const unsigned initial_hold = gain.hold_frames();
  require(gain.process(std::numeric_limits<float>::quiet_NaN(), false, false) ==
              gain.gain_db(),
          "nonfinite request must be treated as a silent target");
  require(gain.hold_frames() < initial_hold,
          "nonfinite request must not refresh hold");
  const unsigned after_nan = gain.hold_frames();
  gain.process(std::numeric_limits<float>::infinity(), true, false);
  require(gain.hold_frames() < after_nan,
          "qualified nonfinite request must also be treated as zero");
}

void test_long_random_sequence_is_rate_bounded() {
  BoundedSpeechGain::Settings settings;
  settings.max_db = 18.f;
  settings.rise_db_per_second = 20.f;
  settings.fall_db_per_second = 30.f;
  settings.hold_ms = 300u;
  BoundedSpeechGain gain(settings);

  std::uint32_t state = 0x12345678u;
  unsigned rises = 0;
  unsigned falls = 0;
  float previous = gain.gain_db();
  for (unsigned frame = 0; frame < 10000u; ++frame) {
    state = state * 1664525u + 1013904223u;
    const float requested = static_cast<float>(state % 2601u) / 100.f - 4.f;
    const bool qualified = (state & 0x10000u) != 0u;
    const float next = gain.process(requested, qualified, false);
    require(next >= 0.f && next <= settings.max_db,
            "random sequence escaped [0, max_db]");
    if (next > previous) {
      ++rises;
      require(next - previous <= .200001f,
              "random rise exceeded configured per-frame bound");
    } else if (next < previous) {
      ++falls;
      require(previous - next <= .300001f,
              "random fall exceeded configured per-frame bound");
    }
    previous = next;
  }
  require(rises > 100u && falls > 100u,
          "rate-bound test must exercise both directions");
  require(gain.gain_db() > 0.f,
          "rate-bound test must produce nontrivial gain state");
}

void test_long_silence_has_no_bootstrap_boost() {
  BoundedSpeechGain gain;
  for (unsigned frame = 0; frame < 10000u; ++frame) {
    const bool qualified = (frame & 1u) != 0u;
    require(gain.process(0.f, qualified, false) == 0.f,
            "long requested silence must never bootstrap gain");
  }
  require(gain.gain_db() == 0.f && gain.held_target_db() == 0.f &&
              gain.hold_frames() == 0u,
          "silent requests must leave no latent boost state");
}

void test_instances_are_independent() {
  BoundedSpeechGain active;
  BoundedSpeechGain silent;
  for (unsigned frame = 0; frame < 100; ++frame) {
    active.process(8.f, true, false);
    silent.process(0.f, false, false);
  }
  require(near(active.gain_db(), 3.f), "active instance must reach its target");
  require(silent.gain_db() == 0.f && silent.held_target_db() == 0.f &&
              silent.hold_frames() == 0u,
          "one instance must not affect another");
}

}  // namespace

int main() {
  test_rise_cap_and_steady_state();
  test_hold_and_release();
  test_unqualified_positive_does_not_refresh_hold();
  test_far_end_is_immediate_and_leaves_no_stale_state();
  test_reset_is_deterministic();
  test_zero_max_is_unity();
  test_legal_setting_boundaries();
  test_invalid_settings_are_rejected();
  test_nonfinite_request_is_zero_without_hold_refresh();
  test_long_random_sequence_is_rate_bounded();
  test_long_silence_has_no_bootstrap_boost();
  test_instances_are_independent();
  std::cout << "bounded speech gain tests passed\n";
}
