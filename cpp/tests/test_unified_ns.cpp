#include "unified_ns.hpp"
#include "unified_eq_agc.hpp"
#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include "itu/vehicle_ns.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

float sample(unsigned index, unsigned rate) {
  const float t = float(index) / float(rate);
  return .08f * std::sin(2.f * 3.14159265358979323846f * 211.f * t) +
         .015f * std::sin(2.f * 3.14159265358979323846f * 1373.f * t);
}

void reconstruction() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const unsigned hop = rate / 100;
    const unsigned samples = hop * 80;
    std::mt19937 rng(rate + 17);
    std::uniform_real_distribution<float> random(-.35f, .35f);
    std::vector<float> source(samples + hop), rendered(samples + hop), frame(hop);
    for (unsigned i = 0; i < samples; ++i) source[i] = random(rng);
    itu::detail::UnifiedNs ns(rate, 0.f, 0.f);
    require(ns.delay() == hop, "unified delay is not one hop");
    for (unsigned offset = 0; offset < samples + hop; offset += hop) {
      std::copy_n(source.data() + offset, hop, frame.data());
      ns.process(frame.data(), true);
      std::copy(frame.begin(), frame.end(), rendered.data() + offset);
    }
    float error = 0.f;
    for (unsigned i = 0; i < samples; ++i)
      error = std::max(error, std::abs(source[i] - rendered[i + hop]));
    require(error < 3e-5f, "unified unity WOLA reconstruction mismatch");
    std::cout << "unified reconstruction rate=" << rate << " error=" << error << '\n';
  }
}

void weights_and_causality() {
  const std::array<float, 4> initial{.34f, .30f, .23f, .13f};
  itu::detail::UnifiedNs adaptive(16000, -18.f, .08f);
  itu::detail::UnifiedNs frozen(16000, -18.f, 0.f);
  const unsigned hop = 160;
  std::vector<float> a(hop), b(hop);
  bool changed = false;
  for (unsigned frame = 0; frame < 180; ++frame) {
    for (unsigned i = 0; i < hop; ++i) {
      const unsigned index = frame * hop + i;
      a[i] = sample(index, 16000) + (frame < 60 ? .025f : 0.f);
      b[i] = a[i];
    }
    adaptive.process(a.data());
    frozen.process(b.data());
    for (float value : a) require(std::isfinite(value), "unified produced nonfinite PCM");
    const auto weights = adaptive.weights();
    const auto targets = adaptive.target_weights();
    float sum = 0.f;
    for (unsigned i = 0; i < weights.size(); ++i) {
      require(std::isfinite(weights[i]) && weights[i] >= .01f && weights[i] <= 1.f,
              "unified weight out of bounds");
      require(std::isfinite(targets[i]) && targets[i] >= .01f && targets[i] <= 1.f,
              "unified target out of bounds");
      sum += weights[i];
      changed |= std::abs(weights[i] - initial[i]) > 1e-6f;
    }
    require(std::abs(sum - 1.f) < 2e-5f, "unified weights are not normalized");
    require(std::isfinite(adaptive.weight_entropy()) && adaptive.weight_entropy() >= 0.f &&
                adaptive.weight_entropy() <= 1.f,
            "unified entropy invalid");
    require(std::isfinite(adaptive.frame_snr_db()) && std::isfinite(adaptive.flatness()) &&
                std::isfinite(adaptive.tonality()), "unified evidence invalid");
  }
  require(changed, "nonzero unified learning rate never changed weights");
  for (unsigned i = 0; i < initial.size(); ++i)
    require(frozen.weights()[i] == initial[i], "zero learning rate changed weights");
  require(adaptive.noise_updates() > 0, "unified noise estimator never updated");

  itu::detail::UnifiedNs first(16000, -18.f, .08f);
  itu::detail::UnifiedNs second(16000, -18.f, .08f);
  unsigned suffix_difference = 0;
  for (unsigned frame = 0; frame < 40; ++frame) {
    for (unsigned i = 0; i < hop; ++i) {
      const unsigned index = frame * hop + i;
      a[i] = sample(index, 16000);
      b[i] = frame < 20 ? a[i] : (i & 1 ? -.25f : .25f);
    }
    first.process(a.data());
    second.process(b.data());
    if (frame < 20) {
      for (unsigned i = 0; i < hop; ++i)
        require(a[i] == b[i], "unified output changed before causal suffix");
    } else {
      for (unsigned i = 0; i < hop; ++i)
        suffix_difference += std::abs(a[i] - b[i]) > 1e-6f;
    }
  }
  require(suffix_difference > 0, "unified causal negative control was vacuous");
}

void minima_history_and_slew() {
  const unsigned hop = 160, frames = 193;
  std::vector<float> a(hop), b(hop);
  itu::detail::UnifiedNs implicit_zero(16000, -18.f, .08f);
  itu::detail::UnifiedNs explicit_zero(16000, -18.f, .08f, 0.f, 8u, 8u);
  itu::detail::UnifiedNs shallow(16000, -18.f, .08f, 0.f, 4u, 8u);
  itu::detail::UnifiedNs conditional_shallow(16000, -18.f, .08f, 0.f, 8u, 4u);
  require(implicit_zero.minima_history_depth() == 8u &&
              implicit_zero.conditional_minima_history_depth() == 8u &&
              shallow.minima_history_depth() == 4u &&
              conditional_shallow.conditional_minima_history_depth() == 4u,
          "unified history depth configuration was not retained");
  float psd_after_first = 0.f, psd_before_rotation = 0.f;
  for (unsigned frame = 0; frame < frames; ++frame) {
    for (unsigned i = 0; i < hop; ++i)
      a[i] = b[i] = sample(frame * hop + i, 16000);
    implicit_zero.process(a.data());
    explicit_zero.process(b.data());
    for (unsigned i = 0; i < hop; ++i)
      require(a[i] == b[i], "zero minima slew changed PCM");
    require(std::isfinite(implicit_zero.noise_psd_band_db()) &&
                std::isfinite(implicit_zero.smooth_band_db()) &&
                std::isfinite(implicit_zero.conditional_band_db()) &&
                std::isfinite(implicit_zero.dd_prior_band_db()) &&
                std::isfinite(implicit_zero.xi_band_db()) &&
                std::isfinite(implicit_zero.gamma_band_db()) &&
                std::isfinite(implicit_zero.gain_band_db()),
            "unified diagnostic state is nonfinite");
    if (frame == 1) psd_after_first = implicit_zero.noise_psd_band_db();
    if (frame == 12) psd_before_rotation = implicit_zero.noise_psd_band_db();
  }
  require(implicit_zero.minima_rotations() == frames / 12,
          "unified minima rotation count mismatch");
  require(std::isfinite(psd_after_first) && psd_after_first > -199.f,
          "noise PSD telemetry did not update after initialization");
  require(std::isfinite(psd_before_rotation) && psd_before_rotation > -199.f,
          "noise PSD telemetry stopped after first frame");
  require(implicit_zero.minima_rebuild_jump_db() == 0.f &&
              implicit_zero.conditional_minima_rebuild_jump_db() == 0.f,
          "rebuild jump telemetry leaked across non-rotation frames");

  const unsigned bins = 32;
  std::vector<float> low(bins, 1e-6f), high(bins, 1.f);
  std::vector<float> xi(bins, 1.f), gamma(bins, 1.f);
  itu::detail::ImcraNoiseEstimator legacy(bins);
  itu::detail::ImcraNoiseEstimator bounded(bins, 1.66f, 1.f);
  legacy.initialize(low.data());
  bounded.initialize(low.data());
  for (unsigned frame = 0; frame < 204; ++frame) {
    const float* power = frame < 96 ? low.data() : high.data();
    legacy.update(power, xi.data(), gamma.data());
    bounded.update(power, xi.data(), gamma.data());
  }
  require(legacy.rotations() == 17 && bounded.rotations() == 17,
          "IMCRA test did not reach the expected rotations");
  const float legacy_jump = legacy.minimum_rebuild_jump_db(0);
  const float bounded_jump = bounded.minimum_rebuild_jump_db(0);
  require(std::isfinite(legacy_jump) && legacy_jump > 5.f,
          "legacy minima rebuild did not expose an upward jump");
  require(std::isfinite(bounded_jump) && bounded_jump <= 1.001f &&
              bounded_jump > 0.f,
          "minima rebuild slew did not cap the upward jump");

  bool rejected_low = false, rejected_high = false, rejected_nan = false;
  bool rejected_depth_low = false, rejected_depth_high = false;
  try { itu::detail::UnifiedNs invalid(16000, -18.f, .08f, -1.f); }
  catch (const std::invalid_argument&) { rejected_low = true; }
  try { itu::detail::UnifiedNs invalid(16000, -18.f, .08f, 25.f); }
  catch (const std::invalid_argument&) { rejected_high = true; }
  try { itu::detail::UnifiedNs invalid(16000, -18.f, .08f,
                                        std::numeric_limits<float>::quiet_NaN()); }
  catch (const std::invalid_argument&) { rejected_nan = true; }
  try { itu::detail::UnifiedNs invalid(16000, -18.f, .08f, 0.f, 0u, 8u); }
  catch (const std::invalid_argument&) { rejected_depth_low = true; }
  try { itu::detail::UnifiedNs invalid(16000, -18.f, .08f, 0.f, 8u, 9u); }
  catch (const std::invalid_argument&) { rejected_depth_high = true; }
  require(rejected_low && rejected_high && rejected_nan &&
              rejected_depth_low && rejected_depth_high,
          "invalid minima slew or history depth was accepted");
}
void eq_agc_fusion() {
  itu::EqConfig eq;
  eq.preamp_db=-3.f;
  eq.bands=1;
  eq.band[0]={itu::EqKind::peak,1000.f,3.f,.70710678f};
  itu::UnifiedEqAgc fusion(eq,16000,160);
  std::vector<float> frame(160,.01f);
  const float speech_weight=fusion.process_eq(frame.data(),160,1.f,.8f,.7f,.7f,false,.9f);
  require(speech_weight>.1f && speech_weight<=1.f,"unified EQ did not follow speech evidence");
  const float far_weight=fusion.process_eq(frame.data(),160,0.f,0.f,0.f,0.f,true,0.f);
  require(far_weight<speech_weight,"unified EQ did not leave far/uncertain mode");
  const float gain=fusion.fuse_gain(0.f,8.f,18.f,1.f,.8f,.7f,.7f,.9f,false,true,true);
  require(gain>0.f && gain<=18.f,"unified AGC fusion rejected qualified gain");
  require(fusion.fuse_gain(0.f,8.f,18.f,1.f,.8f,.7f,.7f,.9f,true,true,true)==0.f,
          "unified AGC applied gain during far activity");
}

void high_snr_bypass() {
  itu::VehicleNsConfig c;
  c.backend = itu::VehicleNsBackend::unified;
  c.strength = 1;
  c.dry_max = .75f;
  c.noise_gain = 1.f;
  c.policy = "legacy";
  c.linear_rumble = false;
  c.unified_raw_budget = true;
  c.unified_high_snr_bypass_db = .01f;
  c.unified_high_snr_probability = 0.f;
  itu::VehicleNsProcessor processor(16000, c);
  std::vector<float> frame(160);
  for (unsigned n = 0; n < 100; ++n) {
    for (unsigned i = 0; i < frame.size(); ++i)
      frame[i] = n < 10 ? 0.f :
          sample(n * frame.size() + i, 16000);
    require(processor.process(frame.data(), frame.size()),
            "high-SNR bypass wrapper rejected frame");
  }
  const auto stats = processor.statistics();
  require(stats.unified_high_snr_bypass,
          "high-SNR bypass evidence was not exported");
  require(stats.raw_mix_budget == 1.f,
          "high-SNR bypass did not expand the raw budget");

  itu::VehicleNsConfig invalid;
  invalid.backend = itu::VehicleNsBackend::unified;
  invalid.unified_high_snr_bypass_db = 1.f;
  bool rejected = false;
  try { itu::VehicleNsProcessor processor(16000, invalid); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "high-SNR bypass accepted without a raw donor");
}
void wrappers_and_engine() {
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    const unsigned hop = rate / 100;
    itu::VehicleNsConfig c;
    c.backend = itu::VehicleNsBackend::unified;
    c.strength = 1;
    c.dry_max = 0.f;
    c.noise_gain = 1.f;
    c.policy = "legacy";
    c.linear_rumble = false;
    c.native_24 = rate == 24000;
    itu::VehicleNsProcessor processor(rate, c);
    require(processor.core_rate() == rate, "unified did not stay at native rate");
    require(processor.core_delay_samples() == hop && processor.delay_samples() == hop,
            "unified wrapper delay mismatch");
    std::vector<float> frame(hop);
    for (unsigned n = 0; n < 160; ++n) {
      for (unsigned i = 0; i < hop; ++i) frame[i] = sample(n * hop + i, rate);
      require(processor.process(frame.data(), frame.size()), "unified wrapper rejected frame");
      for (float value : frame) require(std::isfinite(value), "unified wrapper nonfinite output");
    }
    const auto stats = processor.statistics();
    require(stats.unified_weights[0] > 0.f && stats.unified_weights[1] > 0.f,
            "unified wrapper statistics were not exported");
    require(stats.unified_noise_updates > 0 && std::isfinite(stats.unified_weight_entropy),
            "unified wrapper telemetry stayed at defaults");
    require(stats.unified_minima_rotations == 13 &&
                std::isfinite(stats.unified_noise_psd_band_db) &&
                stats.unified_noise_psd_band_db > -199.f,
            "unified wrapper minima diagnostics missing");

    itu::Config config;
    config.sample_rate = rate;
    config.frame_size = hop;
    config.aec = false;
    config.noise_suppression = true;
    config.automatic_gain = false;
    config.limiter = false;
    config.noise_level = 1;
    config.vehicle_ns_control = true;
    config.noise_backend = itu::NoiseBackend::unified;
    config.vehicle_ns = c;
    itu::AudioEngine engine(config);
    itu::FrameInfo info;
    std::vector<float> output(hop);
    for (unsigned n = 0; n < 80; ++n) {
      for (unsigned i = 0; i < hop; ++i) frame[i] = sample(n * hop + i, rate);
      require(engine.process(frame.data(), frame.size(), nullptr, 0, output.data(), output.size(), info) == itu::Status::ok,
              "unified AudioEngine rejected frame");
    }
    require(info.unified_ns_weights[0] > 0.f && std::isfinite(info.unified_ns_weight_entropy),
            "unified FrameInfo telemetry missing");
    require(info.unified_ns_minima_rotations == 6 &&
                std::isfinite(info.unified_ns_noise_psd_band_db),
            "unified FrameInfo minima diagnostics missing");
  }
  const itu::Config preset = itu::barrot_vcp_unified_config();
  require(preset.vehicle_ns_control && preset.noise_backend == itu::NoiseBackend::unified,
          "unified VCP preset did not select unified backend");
  require(preset.aec && preset.aec_backend == itu::AecBackend::speex_fused &&
              preset.speex_fused_shadow_taps_ms == 32,
          "unified VCP preset did not select fused Speex AEC");
  require(preset.vehicle_ns.unified_raw_budget &&
              preset.vehicle_ns.dry_max == .75f &&
              preset.vehicle_ns.snr_threshold_db == 24.f &&
              preset.vehicle_ns.noise_gain == 1.f,
          "unified VCP preset does not keep the bounded 75 percent raw donor");
}

void c_api() {
  static_assert(int(itu::NoiseBackend::unified) == ITU_AUDIO_NS_UNIFIED);
  for (unsigned rate : {16000u, 24000u, 48000u}) {
    itu_audio_config config;
    itu_audio_default_config(&config);
    config.sample_rate = rate;
    config.frame_size = rate / 100;
    config.aec = config.automatic_gain = config.limiter = 0;
    config.noise_suppression = 1;
    char error[256]{};
    itu_audio_engine* engine = itu_audio_create_with_backend(
        &config, ITU_AUDIO_PROFILE_GENERAL, ITU_AUDIO_NS_UNIFIED,
        error, sizeof(error));
    require(engine != nullptr, error[0] ? error : "C API unified create failed");
    std::vector<float> input(config.frame_size, .02f), output(config.frame_size);
    itu_audio_frame_info info{};
    require(itu_audio_process(engine, input.data(), input.size(), nullptr, 0,
                              output.data(), output.size(), &info, nullptr, nullptr, nullptr) == ITU_AUDIO_OK,
            "C API unified process failed");
    itu_audio_destroy(engine);
    itu_audio_engine* profiled = itu_audio_create_with_profile(
        &config, ITU_AUDIO_PROFILE_VCP_UNIFIED, error, sizeof(error));
    require(profiled != nullptr, error[0] ? error : "C API unified profile create failed");
    require(itu_audio_processing_profile(profiled) == ITU_AUDIO_PROFILE_VCP_UNIFIED,
            "C API unified profile identity was not retained");
    itu_audio_destroy(profiled);
    require(itu_audio_create_with_backend(&config, ITU_AUDIO_PROFILE_GENERAL, 7,
                                          error, sizeof(error)) == nullptr,
            "invalid C API backend accepted after unified enum addition");
  }
}
}

int main() {
  try {
    reconstruction();
    weights_and_causality();
    minima_history_and_slew();
    high_snr_bypass();
    eq_agc_fusion();
    wrappers_and_engine();
    c_api();
    std::cout << "Unified NS checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}