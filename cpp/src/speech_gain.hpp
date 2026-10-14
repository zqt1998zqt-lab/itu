#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace itu {
// All observations are one 10 ms frame, independent of the PCM sample rate.
class SpeechGain {
 public:
  float observe(float level,float probability,bool far,float maximum,float release_rate,
                double shared_noise_power=-1, double qualification_power=-1) noexcept {
    const bool coordinated=shared_noise_power>=0;
    const double power=qualification_power>=0 ? qualification_power : double(level)*level;
    if(!far && probability<.1f && (coordinated || !calibrated_ || quiet_frames_>=99)) {
      const double rate=noise_frames_<30 ? 1./(++noise_frames_) : .05;
      noise_power_+=(power-noise_power_)*rate;
    }
    effective_noise_power_=std::max(noise_power_,std::max(shared_noise_power,1e-12));
    const unsigned quiet_limit=100u;
    const bool speech=!far && probability>.8f && noise_frames_>=10 &&
        power>4*effective_noise_power_;
    qualified_=speech;
    inactive_frames_=speech ? 0 : std::min(500u,inactive_frames_+1);
    if(inactive_frames_==500) cached_=false;
    if(speech) {
      quiet_frames_=0;
      if(cached_) applied_=target_;
      if(!calibrated_) {
        levels_[speech_frames_++]=level;
        if(speech_frames_==levels_.size()) {
          auto sorted=levels_;
          std::sort(sorted.begin(),sorted.end());
          target_=sorted[sorted.size()/2]<.017782794f ? std::min(8.f,maximum) : 0.f;
          calibrated_=true;
          cached_=true;
          // AGC2 interpolates fixed gain over the next frame. Then lock the
          // utterance gain; a seconds-long attack would modulate weak speech.
          applied_=target_;
        }
      }
    } else if(probability<.1f || far || (noise_frames_>=10 && power<=4*effective_noise_power_)) {
      quiet_frames_=std::min(quiet_limit,quiet_frames_+1);
      if(quiet_frames_==quiet_limit) {
        calibrated_=false;
        speech_frames_=0;
      }
    } else {
      quiet_frames_=0;
    }
    if(far) return applied_=0;
    if(quiet_frames_>=quiet_limit)
      applied_+=std::clamp(-applied_,-release_rate*.01f,release_rate*.01f);
    return applied_;
  }
  double noise_power() const noexcept {return effective_noise_power_;}
  bool qualified() const noexcept {return qualified_;}
 private:
  std::array<float,30> levels_{};
  unsigned noise_frames_=0,quiet_frames_=0,speech_frames_=0,inactive_frames_=0;
  double noise_power_=0,effective_noise_power_=1e-12;
  bool qualified_=false;
  bool calibrated_=false,cached_=false;
  float target_=0,applied_=0;
};
}
