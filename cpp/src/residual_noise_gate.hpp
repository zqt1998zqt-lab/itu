#pragma once
#include <algorithm>
#include <cmath>
namespace itu {
// One scalar update per 10 ms callback. No PCM buffering or signal delay.
class ResidualNoiseGate {
 public:
  float observe(float probability,bool eligible,bool broadband) noexcept {
    const bool quiet=eligible && std::isfinite(probability) && probability<.1f;
    quiet_frames_=quiet ? std::min(1000u,quiet_frames_+1) : 0;
    const float target=quiet_frames_>=100 && broadband ? .50118723f : 1.f;
    previous_=gain_;
    gain_+=std::clamp(target-gain_,-.05f,.5f);
    return gain_;
  }
  float sample(unsigned index,unsigned frame_size) const noexcept {
    return previous_+(gain_-previous_)*float(index+1)/frame_size;
  }
  float gain() const noexcept {return gain_;}
  unsigned quiet_frames() const noexcept {return quiet_frames_;}
 private:
  unsigned quiet_frames_=0;
  float gain_=1.f,previous_=1.f;
};
}
