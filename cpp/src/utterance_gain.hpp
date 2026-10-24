#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace itu {
// One observation per 10 ms. Only control state is retained; no PCM buffer.
// Acquire a scalar utterance gain from 600 ms of physical input, then hold it
// through syllables and short pauses. AGC2 interpolates initial acquisition over one output frame. Only
// sustained noise releases the gain; a new utterance reuses the cached target.
// It changes future gain only: the output sample clock and all taps stay intact.
class UtteranceGain {
 public:
  enum State : unsigned { observing=0,onset=1,locked=2,quiet=3,conservative=4,far_end=5 };
  float observe(float level,float probability,bool far,float maximum,
                double shared_noise,float conservative_gain,float release_db_per_second=6.f,
                bool noise_domain_release=false) noexcept {
    const double power=double(level)*level;
    recent_power_[head_]=far ? 0. : power;
    head_=(head_+1)%recent_power_.size();
    count_=std::min(unsigned(recent_power_.size()),count_+1);
    if(!far && probability<.1f && (noise_frames_<10 || power<=4*std::max(noise_,1e-14))) {
      const double rate=noise_frames_<30 ? 1./(++noise_frames_) : .03;
      noise_+=(power-noise_)*rate;
    }
    effective_noise_=std::max({noise_,shared_noise,1e-14});
    qualified_=!far && (noise_frames_>=10 || shared_noise>=0) && probability>.8f && power>4*effective_noise_;
    const bool quiet_evidence=!far && (probability<.1f || power<=4*effective_noise_);
    quiet_frames_=quiet_evidence ? std::min(500u,quiet_frames_+1) : 0;
    if(far) {
      // An interrupted acquisition never bridges the far-end interval.
      if(!learned_) onset_frames_=votes_=0;
      state_=far_end;return applied_=0;
    }
    if(!learned_ && (onset_frames_ || qualified_)) {
      if(onset_frames_==0) onset_power_=0;
      ++onset_frames_;votes_+=qualified_;onset_power_+=power;
      if(onset_frames_>=60) {
        if(votes_>=3) {
          // Do not dilute the onset level with preceding silence or far-end
          // zeros. Several syllables distinguish a quiet onset from a quiet
          // utterance; one isolated peak cannot permanently veto weak speech.
          const double mean=onset_power_/onset_frames_;
          target_=mean<.014*.014 ? std::min(8.f,maximum) : 0.f;
          learned_=true;applied_=target_;
        }
        // An uncertain observation expires; it cannot calibrate future speech.
        onset_frames_=votes_=0;
      }
    }
    if(learned_ && qualified_) applied_=target_;
    // A weak utterance can remain learned while the gain is being applied to
    // a later noise-only interval. Release from sustained physical evidence
    // only after the speech target is already active; this preserves the
    // qualified low-level onset and short syllable gaps.
    if(noise_domain_release && learned_ && applied_>0.f &&
        quiet_evidence && power<=4*effective_noise_) {
      const float step=std::max(.0001f,release_db_per_second*.01f);
      applied_=std::max(0.f,applied_-step);
    }
    if(quiet_frames_>=150) {
      const float step=std::max(.0001f,release_db_per_second*.01f);
      applied_=std::max(0.f,applied_-step);
    }
    if(quiet_frames_>=500) {learned_=false;target_=0;onset_frames_=votes_=0;}
    state_=onset_frames_ ? onset : learned_ ? (quiet_frames_>=150 ? quiet : locked) : observing;
    return applied_=std::clamp(applied_,0.f,maximum);
  }
  float applied() const noexcept {return applied_;}
  bool qualified() const noexcept {return qualified_;}
  double noise() const noexcept {return effective_noise_;}
  unsigned state() const noexcept {return state_;}
  unsigned votes() const noexcept {return votes_;}
  float target() const noexcept {return target_;}
 private:
  std::array<double,200> recent_power_{};unsigned head_=0,count_=0;
  unsigned noise_frames_=0,onset_frames_=0,votes_=0,quiet_frames_=0,state_=observing;
  bool learned_=false,qualified_=false;
  float target_=0,applied_=0;double noise_=0,effective_noise_=0,onset_power_=0;
};
}
