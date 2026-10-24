#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace itu {
// Align input-derived control to the backend's existing output sample time.
// This delays controls only: no audio samples, lookahead, or latency are added.
class NsRepairAlignment {
 public:
  NsRepairAlignment(unsigned delay, unsigned hop, float slew_ms, bool fast_attack=false)
      : hop_(hop), mix_delay_(std::max(1u, delay)),
        gain_delay_(std::max(1u, delay), 1.f), delay_(delay),
        fast_attack_(fast_attack), alpha_(slew_ms > 0 ? 1.f - std::exp(-10.f / slew_ms) : 1.f) {}
  void begin(float mix, float gain) noexcept {
    if (fast_attack_ && !initialized_) {
      mix_=mix; gain_=gain;
      std::fill(mix_delay_.begin(),mix_delay_.end(),mix);
      std::fill(gain_delay_.begin(),gain_delay_.end(),gain);
    }
    initialized_=true;
    start_mix_ = mix_;
    start_gain_ = gain_;
    mix_ += (fast_attack_ && mix>mix_ ? 1.f : alpha_) * (mix - mix_);
    gain_ += (fast_attack_ && gain>gain_ ? 1.f : alpha_) * (gain - gain_);
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
  bool fast_attack_, initialized_=false;
  float alpha_, mix_ = 0, gain_ = 1, start_mix_ = 0, start_gain_ = 1;
};
}
