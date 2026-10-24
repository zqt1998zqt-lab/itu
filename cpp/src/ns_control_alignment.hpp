#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace itu {
// Align input-derived control to the backend's existing output sample time.
// This delays controls only: no audio samples, lookahead, or latency are added.
class NsControlAlignment {
 public:
  NsControlAlignment(unsigned delay, unsigned hop, float slew_ms)
      : hop_(hop), mix_delay_(std::max(1u, delay)),
        gain_delay_(std::max(1u, delay), 1.f), delay_(delay),
        alpha_(slew_ms > 0 ? 1.f - std::exp(-10.f / slew_ms) : 1.f) {}
  void begin(float mix, float gain) noexcept {
    start_mix_ = mix_;
    start_gain_ = gain_;
    mix_ += alpha_ * (mix - mix_);
    gain_ += alpha_ * (gain - gain_);
  }
  void sample(unsigned i, float& mix, float& gain) noexcept {
    const float f = float(i + 1) / hop_;
    mix = start_mix_ + f * (mix_ - start_mix_);
    gain = start_gain_ + f * (gain_ - start_gain_);
    if (delay_) {
      const float m = mix_delay_[head_], g = gain_delay_[head_];
      mix_delay_[head_] = mix;
      gain_delay_[head_] = gain;
      mix = m;
      gain = g;
      head_ = (head_ + 1) % delay_;
    }
  }
  float mix() const noexcept { return mix_; }
 private:
  unsigned hop_, delay_, head_ = 0;
  std::vector<float> mix_delay_, gain_delay_;
  float alpha_, mix_ = 0, gain_ = 1, start_mix_ = 0, start_gain_ = 1;
};
}
