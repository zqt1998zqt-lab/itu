#pragma once
#include <algorithm>
#include <cmath>
namespace itu {
// G1 v2: causal bounded state. Only the ORIGINAL strict dual decision with
// strong nonlinear VAD can arm/renew the latch. Weak strict evidence resets
// echo continuity but cannot refresh age. Entry conditions remain in F.
class NearProtectionGap {
 public:
  static constexpr unsigned kMaximumStrictAge=100;
  static constexpr unsigned kProtectedEchoFrames=40;
  static constexpr float kStrongNonlinearVad=.95f;
  void reset() noexcept {confirmed_=false; age_=101;echo_frames_=startup_frames_=0;}
  void observe(bool eligible,bool original_strict_confirmed,float nonlinear_vad,
               bool startup_echo,bool original_confirmed_echo) noexcept {
    if(!eligible) {reset();return;}
    const bool strong=original_strict_confirmed && std::isfinite(nonlinear_vad) &&
        nonlinear_vad>=kStrongNonlinearVad;
    if(strong) {confirmed_=true;age_=0;}
    else if(confirmed_) {age_=std::min(101u,age_+1);if(age_>100) confirmed_=false;}
    // Every original strict confirmation breaks continuity, including weak
    // confirmations that cannot arm/renew the new confidence-qualified latch.
    echo_frames_=original_strict_confirmed ? 0 :
        startup_echo || original_confirmed_echo ? std::min(40u,echo_frames_+1) : 0;
    // Retained only for the accurately named v1 telemetry column.
    startup_frames_=startup_echo ? std::min(40u,startup_frames_+1) : 0;
  }
  bool protects(unsigned existing_hold) const noexcept {
    return confirmed_ && age_<=100 && existing_hold>0;
  }
  bool defer_clear(unsigned existing_hold) const noexcept {
    return protects(existing_hold) && echo_frames_<40;
  }
  unsigned strict_age() const noexcept {return age_;}
  unsigned echo_frames() const noexcept {return echo_frames_;}
  unsigned startup_frames() const noexcept {return startup_frames_;}
  unsigned threshold(unsigned existing_hold) const noexcept {return protects(existing_hold)?40u:0u;}
 private:
  bool confirmed_=false;
  unsigned age_=101,echo_frames_=0,startup_frames_=0;
};
}
