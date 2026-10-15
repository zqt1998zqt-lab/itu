#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace itu {
// One observation per 10 ms. Only control state is retained; no PCM buffer.
// The 600 ms onset observation is wall time, never scattered voiced frames.
// A 500 ms scalar power history guards weak consonants after stronger speech.
// It changes future gain only: the output sample clock and all taps stay intact.
class UtteranceGain {
 public:
  enum State : unsigned { observing=0,onset=1,locked=2,quiet=3,conservative=4,far_end=5 };
  float observe(float level,float probability,bool far,float maximum,
                double shared_noise,float conservative_gain) noexcept {
    const double power=double(level)*level;
    recent_levels_[recent_head_]=far ? 0.f : level;
    recent_voice_[recent_head_]=!far && probability>.8f;
    recent_head_=(recent_head_+1)%recent_levels_.size();
    if(!far && probability<.1f) {
      const double rate=noise_frames_<30 ? 1./(++noise_frames_) : .03;
      noise_+=(power-noise_)*rate;
    }
    effective_noise_=std::max({noise_,shared_noise,1e-14});
    qualified_=!far && noise_frames_>=10 && probability>.8f && power>4*effective_noise_;
    const bool quiet_evidence=!far && (probability<.1f || power<=4*effective_noise_);
    quiet_frames_=quiet_evidence ? std::min(500u,quiet_frames_+1) : 0;
    if(far) {
      // Interrupted observations are discarded: their age never bridges a
      // far-end interval. Completed utterance targets remain cached.
      if(!learned_) {onset_frames_=votes_=0;peak_=0;conservative_=provisional_=false;recent_levels_.fill(0);recent_voice_.fill(false);fresh_onset_=false;}
      state_=far_end;return applied_=0;
    }
    if(!learned_ && onset_frames_==0 && qualified_) {
      onset_frames_=1;peak_=*std::max_element(recent_levels_.begin(),recent_levels_.end());votes_=1;
      // The qualifier may lag VAD under changing noise. Do not treat a late
      // qualified frame in an already voiced episode as a fresh onset.
      fresh_onset_=std::count(recent_voice_.begin(),recent_voice_.end(),true)<=5;
    } else if(onset_frames_) {
      ++onset_frames_;peak_=std::max(peak_,level);votes_+=qualified_;
      // Low-confidence routing uses the first physical 120 ms. A later loud
      // noise burst must not revoke the existing projection safeguard.
      if(onset_frames_==12) {
        conservative_=peak_*peak_<10*effective_noise_;
        provisional_=fresh_onset_ && !conservative_ && votes_>=8 && peak_<.012589254f;
      }
      if(onset_frames_>=60) {
        // Signal below this confidence boundary keeps the existing conservative
        // controller. NS can attenuate weak speech here; unity would erase
        // pre-existing projection protection. The route is locked per episode.
        conservative_=conservative_ || peak_*peak_<10*effective_noise_;
        // Restricted causal onset gain removes the late 0 -> 8 dB jump.
        // If later normal-level evidence appears, retain a small fixed target
        // rather than switching from the provisional gain all the way to zero.
        target_=votes_>=3 && peak_<.012589254f ? std::min(8.f,maximum) :
            provisional_ ? std::min(3.f,maximum) : 0;
        learned_=active_=true;onset_frames_=0;
      }
    }
    if(learned_ && qualified_) active_=true;
    if(quiet_frames_>=60) active_=false;
    if(quiet_frames_>=500) {learned_=false;target_=0;conservative_=provisional_=false;}
    state_=conservative_ && learned_ ? conservative : onset_frames_ ? onset :
        active_ ? locked : learned_ ? quiet : observing;
    applied_=conservative_ && learned_ ? conservative_gain : active_ ? target_ :
        onset_frames_>=12 && provisional_ ? std::min(5.f,maximum) : 0;
    // Keep the utterance target across unvoiced gaps, but do not apply a large
    // boost to observed noise. Twenty quiet frames cap the output at 3 dB;
    // this is a noise-domain cap, not a new speech calibration or PCM gate.
    if(quiet_frames_>=20 && !conservative_) applied_=std::min(applied_,3.f);
    return applied_;
  }
  float applied() const noexcept {return applied_;}
  bool qualified() const noexcept {return qualified_;}
  double noise() const noexcept {return effective_noise_;}
  unsigned state() const noexcept {return state_;}
 private:
  std::array<float,50> recent_levels_{};std::array<bool,50> recent_voice_{};unsigned recent_head_=0;
  unsigned noise_frames_=0,onset_frames_=0,votes_=0,quiet_frames_=0,state_=observing;
  bool learned_=false,active_=false,qualified_=false,conservative_=false,provisional_=false,fresh_onset_=false;
  float peak_=0,target_=0,applied_=0;double noise_=0,effective_noise_=0;
};
}
