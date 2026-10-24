#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace itu::detail {
// Current-frame values, captured before the t+1 noise update. The observer may
// change only the synthesis gain; statistical/DD state remains independent.
struct NsSpectralFrame {
  unsigned rate, fft_size, bins;
  const float *power, *noise, *xi, *gamma, *probability;
  float* gain;
};
class NsSpectralControl {
 public:
  virtual ~NsSpectralControl() = default;
  virtual void observe(NsSpectralFrame frame) noexcept = 0;
};

class NsSpectralChain final : public NsSpectralControl {
 public:
  NsSpectralChain(std::initializer_list<NsSpectralControl*> controls)
      : controls_(controls) {}
  NsSpectralChain(NsSpectralControl* first, NsSpectralControl* second)
      : controls_({first, second}) {}
  NsSpectralChain(NsSpectralControl* first, NsSpectralControl* second,
                  NsSpectralControl* third)
      : controls_({first, second, third}) {}
  void observe(NsSpectralFrame frame) noexcept override {
    for (NsSpectralControl* control : controls_)
      if (control) control->observe(frame);
  }
 private:
  std::vector<NsSpectralControl*> controls_;
};

// Optional synthesis-only protection for the lower speech band. It uses only
// the current frame's statistical evidence and never changes the estimator.
class SpeechBandProtection final : public NsSpectralControl {
 public:
  SpeechBandProtection(unsigned rate, unsigned nfft, float recovery)
      : recovery_(recovery), rate_(rate), nfft_(nfft),
        low_bin_(unsigned(std::ceil(80.0 * nfft / rate))),
        high_bin_(unsigned(std::ceil(1000.0 * nfft / rate))),
        anchor_low_bin_(unsigned(std::ceil(1000.0 * nfft / rate))),
        anchor_high_bin_(unsigned(std::ceil(3400.0 * nfft / rate))) {
    if (!std::isfinite(recovery) || recovery < 0 || recovery > 1)
      throw std::invalid_argument("speech-band recovery must be 0..1");
  }
  void observe(NsSpectralFrame frame) noexcept override {
    frames_ = 0;
    last_floor_ = 1;
    last_delta_ = 0;
    last_evidence_ = 0;
    last_mid_probability_ = 0;
    last_anchor_probability_ = 0;
    last_anchor_gain_ = 0;
    if (recovery_ <= 0 || low_bin_ >= high_bin_ ||
        anchor_low_bin_ >= anchor_high_bin_ ||
        high_bin_ > frame.bins || anchor_high_bin_ > frame.bins)
      return;
    double mid_probability = 0, anchor_probability = 0, anchor_gain = 0;
    for (unsigned k = low_bin_; k < high_bin_; ++k) {
      mid_probability += frame.probability[k];
    }
    mid_probability /= high_bin_ - low_bin_;
    for (unsigned k = anchor_low_bin_; k < anchor_high_bin_; ++k) {
      anchor_probability += frame.probability[k];
      anchor_gain += frame.gain[k];
    }
    anchor_probability /= anchor_high_bin_ - anchor_low_bin_;
    anchor_gain /= anchor_high_bin_ - anchor_low_bin_;
    // The upper band is a causal speech anchor: its gain has already been
    // computed by OM-LSA. Both probability and gain must support speech.
    const float probability_evidence = std::clamp(
        (float(std::min(mid_probability, anchor_probability)) - .55f) / .25f,
        0.f, 1.f);
    const float gain_evidence = std::clamp(
        (float(anchor_gain) - .55f) / .20f, 0.f, 1.f);
    const float evidence = probability_evidence * gain_evidence;
    if (evidence <= 0) return;
    // The floor is the explicitly bounded experiment parameter. Its gain
    // change scales with causal speech evidence, while the parameter itself
    // spans the full linear gain range for paired screening.
    float delta = 0;
    float minimum_floor = 1.f;
    for (unsigned k = low_bin_; k < high_bin_; ++k) {
      const float frequency = float(k) * rate_ / nfft_;
      const float lower_edge = std::clamp((frequency - 80.f) / 40.f, 0.f, 1.f);
      const float upper_edge = std::clamp((1000.f - frequency) / 150.f, 0.f, 1.f);
      const float band_weight = lower_edge * upper_edge;
      const float original = frame.gain[k];
      // Full recovery retains the measured 0 dB target-protection floor.
      // Intermediate recovery values interpolate between the estimator gain
      // and the evidence-scaled floor without changing statistical state.
      constexpr float target_floor_gain = 1.f;
      const float protected_floor =
          target_floor_gain + (1.f - target_floor_gain) * (1.f - evidence);
      const float floor_gain =
          original + recovery_ * band_weight * (protected_floor - original);
      frame.gain[k] = std::max(original, floor_gain);
      delta += frame.gain[k] - original;
      minimum_floor = std::min(minimum_floor, floor_gain);
    }
    if (delta > 0) {
      ++frames_;
      last_floor_ = minimum_floor;
      last_delta_ = delta / (high_bin_ - low_bin_);
    }
    last_evidence_ = evidence;
    last_mid_probability_ = float(mid_probability);
    last_anchor_probability_ = float(anchor_probability);
    last_anchor_gain_ = float(anchor_gain);
  }
  std::size_t frames() const noexcept { return frames_; }
  float last_floor() const noexcept { return last_floor_; }
  float last_delta() const noexcept { return last_delta_; }
  float last_evidence() const noexcept { return last_evidence_; }
  float last_mid_probability() const noexcept { return last_mid_probability_; }
  float last_anchor_probability() const noexcept { return last_anchor_probability_; }
  float last_anchor_gain() const noexcept { return last_anchor_gain_; }
 private:
  float recovery_;
  unsigned rate_, nfft_;
  unsigned low_bin_, high_bin_, anchor_low_bin_, anchor_high_bin_;
  std::size_t frames_ = 0;
  float last_floor_ = 1, last_delta_ = 0, last_evidence_ = 0;
  float last_mid_probability_ = 0, last_anchor_probability_ = 0, last_anchor_gain_ = 0;
};
}
