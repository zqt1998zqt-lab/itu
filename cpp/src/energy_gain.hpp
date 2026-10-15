#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace itu {
// Noise-relative energy calibration. Fixed wall-time onset observation,
// all input frames contribute level evidence; no external speech classifier.
class EnergyGain {
 public:
  float observe(float level,float excess,bool far,float maximum) noexcept {
    if(far) {age_=quiet_=votes_=0;peak_=0;return applied_=0;}
    quiet_=excess<.15f ? std::min(500u,quiet_+1) : 0;
    if(quiet_>=500) {learned_=false;target_=0;}
    if(!learned_ && !age_ && excess>.65f) {age_=1;peak_=level;votes_=1;}
    else if(age_) {
      ++age_;peak_=std::max(peak_,level);votes_+=excess>.65f;
      if(age_>=60) {
        target_=votes_>=8 && peak_<.012589254f ? std::min(8.f,maximum) : 0.f;
        learned_=true;age_=0;
      }
    }
    const float desired=quiet_>=10 ? 0.f : learned_ ? target_ :
        age_>=12 && votes_>=8 && peak_<.012589254f ? std::min(5.f,maximum) : 0.f;
    applied_+=std::clamp(desired-applied_,-.5f,.5f);
    return applied_;
  }
  bool qualified() const noexcept {return learned_;}
  unsigned state() const noexcept {return learned_ ? 2u : age_ ? 1u : 0u;}
 private:
  unsigned age_=0,quiet_=0,votes_=0;
  float peak_=0,target_=0,applied_=0;
  bool learned_=false;
};
}
