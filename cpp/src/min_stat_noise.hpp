#pragma once
#include <algorithm>
namespace itu {
// Broadband minimum-statistics noise floor with a speech-gated extra gain.
// Module 1 of the hybrid NS design: the estimator is replaced, the gain rule is
// still a closed form, and nothing here trains or generates audio.
class MinStatNoise {
 public:
  float observe(float frame_power,float speech_probability) noexcept {
    const float p=std::max(frame_power,1e-12f);
    sub_=std::min(sub_,p);
    if(++sub_count_>=50) {
      if(glob_count_==0 || sub_<glob_) { glob_=sub_; glob_count_=1; }
      else if(++glob_count_>=3) { glob_=sub_; glob_count_=1; }
      sub_=p; sub_count_=0;
    }
    const float noise=std::max(1.5f*glob_,1e-12f);
    if(speech_probability>0.5f) return 1.f;
    const float snr=p/noise;
    if(snr>=4.f) return 1.f;
    return std::max(std::sqrt(std::clamp(snr/4.f,0.f,1.f)),0.25f);
  }
  float noise() const noexcept {return glob_;}
 private:
  float sub_=1e9f,glob_=1e9f;
  unsigned sub_count_=0,glob_count_=0;
};
}
