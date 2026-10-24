#pragma once

#include "drc.hpp"
#include "parametric_eq.hpp"
#include "itu/audio_engine.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace itu {

// Barrot VCP Downlink (RX / Playback) Processing Chain:
// Downlink In -> Noise Reduction (NR) -> Equalizer (EQ) -> AGC / DRC -> Loudspeaker (Echo Reference)
//
// Key engineering goals:
// 1. Cleans far-end background/street noise before playing into the cabin.
// 2. Compensates for loudspeaker cabin frequency response (EQ).
// 3. Dynamically compresses high peaks and soft-limits amplitude (DRC) to prevent
//    loudspeaker physical clipping. Non-linear speaker distortion is the #1 cause
//    of AEC failure during high playback volumes; this stage enforces acoustic linearity.
class DownlinkPipeline {
 public:
  explicit DownlinkPipeline(unsigned sample_rate = 16000,
                            unsigned frame_size = 160,
                            DownlinkConfig config = {})
      : sample_rate_(sample_rate ? sample_rate : 16000),
        frame_size_(frame_size ? frame_size : 160),
        config_(config),
        drc_(sample_rate_, config.drc),temp_buf_(frame_size_) {
    if (ParametricEq::enabled(config_.eq)) {
      eq_ = std::make_unique<ParametricEq>(config_.eq, sample_rate_);
    }
    reset();
  }

  void configure(const DownlinkConfig& config) {
    config_ = config;
    drc_.configure(config_.drc);
    if (ParametricEq::enabled(config_.eq)) {
      eq_ = std::make_unique<ParametricEq>(config_.eq, sample_rate_);
    } else {
      eq_.reset();
    }
  }

  void reset() noexcept {
    drc_.reset();
    if (ParametricEq::enabled(config_.eq)) {
      try {
        eq_ = std::make_unique<ParametricEq>(config_.eq, sample_rate_);
      } catch (...) {
        eq_.reset();
      }
    }
    noise_floor_estimate_ = 1e-4f;
    smoothed_gain_ = 1.0f;
  }

  // Process a frame of downlink audio:
  // - in: raw incoming far-end audio (frame_size samples)
  // - out_speaker: audio to send to vehicle amplifier/loudspeaker
  // - out_aec_ref: audio tapped to feed into Acoustic Echo Canceller
  // Note: out_speaker and out_aec_ref can point to the same buffer or in-place with in.
  void process(const float* in, float* out_speaker, float* out_aec_ref, std::size_t n) noexcept {
    if (!in || !out_speaker || n == 0) return;

    temp_buf_.resize(n);
    std::copy_n(in, n, temp_buf_.data());

    if (config_.enabled) {
      // 1. Downlink Noise Reduction (NR)
      if (config_.enable_nr) {
        process_downlink_nr(temp_buf_.data(), n);
      }

      // 2. Downlink Loudspeaker Frequency Response Equalization (EQ)
      if (eq_) {
        eq_->process(temp_buf_.data(), static_cast<unsigned>(n));
      }

      // 3. Downlink Dynamic Range Compression & Soft Limiting (AGC / DRC)
      if (config_.drc.enabled) {
        drc_.process(temp_buf_.data(), n);
      }
    }

    std::copy_n(temp_buf_.data(), n, out_speaker);

    if (out_aec_ref) {
      std::copy_n(temp_buf_.data(), n, out_aec_ref);
    }
  }

  const DrcConfig& drc_config() const noexcept { return config_.drc; }
  float current_drc_gain_db() const noexcept { return drc_.current_gain_db(); }
  // Includes inline DRC state; opaque WebRTC biquad storage is excluded.
  std::size_t wrapper_memory_bytes() const noexcept {
    return sizeof(*this)+temp_buf_.capacity()*sizeof(float)+
        drc_.memory_bytes()+(eq_ ? sizeof(ParametricEq) : 0);
  }

 private:
  void process_downlink_nr(float* data, std::size_t n) noexcept {
    float frame_power = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
      frame_power += data[i] * data[i];
    }
    frame_power /= static_cast<float>(n);

    // Track minimum background noise floor with fast attack, slow decay
    if (frame_power < noise_floor_estimate_) {
      noise_floor_estimate_ = 0.85f * noise_floor_estimate_ + 0.15f * frame_power;
    } else {
      noise_floor_estimate_ = 0.999f * noise_floor_estimate_ + 0.001f * frame_power;
    }

    // Compute Wiener-style suppression gain
    const float snr = (frame_power + 1e-12f) / (noise_floor_estimate_ + 1e-12f);
    const float min_gain = std::pow(10.0f, -config_.nr_max_suppression_db / 20.0f);
    const float target_gain = std::clamp(1.0f - (1.0f / (snr + 1.0f)), min_gain, 1.0f);

    smoothed_gain_ += 0.1f * (target_gain - smoothed_gain_);

    for (std::size_t i = 0; i < n; ++i) {
      data[i] *= smoothed_gain_;
    }
  }

  unsigned sample_rate_ = 16000;
  unsigned frame_size_ = 160;
  DownlinkConfig config_;
  DynamicRangeCompressor drc_;
  std::unique_ptr<ParametricEq> eq_;
  std::vector<float> temp_buf_;
  float noise_floor_estimate_ = 1e-4f;
  float smoothed_gain_ = 1.0f;
};

} // namespace itu
