#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace itu {
// Input-only, 10 ms control. Never consumes clean PCM, SNR or scene labels.
// Suppressors continue learning while their output is blended with aligned dry PCM.
class NoiseTransition {
 public:
  float observe(float level,float speech,bool far,bool near) noexcept {
    ++frames_;
    const double p=double(level)*level;
    const bool noise=!far && !near && speech<.2f && p>1e-12;
    bool jump=false;
    if(noise) {
      if(qualified_>=20 && p>noise_*4) jump=++innovation_>=3;
      else innovation_=0;
      if(jump) {hold_=50; state_=2; noise_=p; innovation_=0; ++events_;}
      else noise_=qualified_ ? .98*noise_+.02*p : p;
      ++qualified_;
    } else innovation_=0;
    if(hold_) --hold_;
    if(!hold_) state_=qualified_>=20 ? 1 : 0;
    const bool protect=(!far || near) && (near || speech>.35f);
    // Startup uncertainty is distinct from a measured high-SNR estimate.
    // An unqualified noise floor is not evidence of high SNR. The former 80%
    // dry path persisted indefinitely without quiet frames and reintroduced
    // startup noise. Keep bootstrap protection bounded in time and amplitude;
    // use the larger floor only after observing noise and a >10 dB power gap.
    const bool high_snr=qualified_>=20 && p>10*std::max(noise_,1e-12);
    // VAD may itself be untrained at startup. A bounded soft floor protects
    // onset consonants without requiring a speech decision from that VAD.
    const float target=!far && frames_<=50 ? .5f :
        protect ? (high_snr ? .8f : hold_ ? .1f : 0.f) : 0.f;
    mix_+=std::clamp(target-mix_,-.05f,.2f);
    return mix_;
  }
  unsigned state() const noexcept {return state_;}
  unsigned events() const noexcept {return events_;}
  unsigned observations() const noexcept {return qualified_;}
 private:
  std::uint64_t frames_=0;
  unsigned qualified_=0,innovation_=0,hold_=0,state_=0,events_=0;
  double noise_=0;
  float mix_=0;
};
}
