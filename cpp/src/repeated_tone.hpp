#pragma once
#include "common_audio/real_fourier_ooura.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>

namespace itu {
// Causal repeated narrowband burst observer. Only the observer retains input
// history; the signal path has no FIFO, lookahead, or added sample offset.
class RepeatedTone {
 public:
  explicit RepeatedTone(unsigned rate):fft_(9),rate_(rate) {
    for(unsigned i=0;i<512;++i) window_[i]=.5f-.5f*std::cos(6.283185307179586f*i/511);
  }
  void observe(const float* input,bool eligible) noexcept {
    ++frames_;
    std::move(history_.begin()+160,history_.end(),history_.begin());
    std::copy_n(input,160,history_.end()-160);
    for(unsigned i=0;i<512;++i) time_[i]=history_[i]*window_[i];
    fft_.Forward(time_.data(),spectrum_.data());
    double total=0;
    unsigned peak=10;
    for(unsigned k=10;k<244;++k) {
      power_[k]=std::norm(spectrum_[k]);total+=power_[k];
      if(power_[k]>power_[peak]) peak=k;
    }
    double narrow=0;
    for(unsigned k=peak>13?peak-3:10;k<=std::min(243u,peak+3);++k) narrow+=power_[k];
    concentration_=total>1e-14 ? narrow/total : 0;
    const bool burst=eligible && peak>=16 && concentration_>.93f && total>1e-10;
    const float left=std::log(std::max(power_[peak-1],1e-20f));
    const float center=std::log(std::max(power_[peak],1e-20f));
    const float right=std::log(std::max(power_[peak+1],1e-20f));
    const float denom=left-2*center+right;
    const float offset=std::abs(denom)>1e-8f ? std::clamp(.5f*(left-right)/denom,-.5f,.5f) : 0.f;
    const float hz=(peak+offset)*31.25f;
    if(burst) {
      if(!burst_frames_) {event_start_=frames_;event_hz_=hz;event_peak_=narrow;}
      ++burst_frames_;event_peak_=std::max(event_peak_,narrow);
    } else if(burst_frames_) {
      if(burst_frames_>=2 && burst_frames_<=12 && narrow<event_peak_*.35 && eligible) {
        const auto interval=event_start_-last_event_;
        if(last_event_ && std::abs(event_hz_-candidate_hz_)<62.5f && interval>=15 && interval<=200) {
          ++repeats_;candidate_hz_=.8f*candidate_hz_+.2f*event_hz_;
        } else {repeats_=1;candidate_hz_=event_hz_;}
        last_event_=event_start_;
        if(repeats_>=3 && !learned_) {learned_=true;frequency_=candidate_hz_;set_notch();}
      }
      burst_frames_=0;
    }
    if(learned_) {
      const int bin=std::lround(frequency_/31.25f);
      double tone=0;
      for(int k=std::max(10,bin-3);k<=std::min(243,bin+3);++k) tone+=power_[k];
      // Cross-band energy vetoes the filter on frames dominated by other
      // content. The frequency comes from repeated input events, never labels.
      const float fraction=total>1e-14 ? tone/total : 0.f;
      const float target=eligible ? std::clamp((fraction-.02f)/.08f,0.f,1.f) : 0.f;
      previous_=mix_;mix_+=std::clamp(target-mix_,-.5f,.5f);
    } else previous_=mix_=0;
  }
  void process(float* audio,unsigned size) noexcept {
    if(!learned_) return;
    for(unsigned i=0;i<size;++i) {
      const double x=audio[i];
      const double y=b0_*x+z1_;z1_=b1_*x-a1_*y+z2_;z2_=b2_*x-a2_*y;
      const float weight=previous_+(mix_-previous_)*float(i+1)/size;
      audio[i]=float(x+weight*(y-x));
    }
  }
  bool learned() const noexcept {return learned_;}
  unsigned repeats() const noexcept {return repeats_;}
  float frequency() const noexcept {return frequency_;}
  float mix() const noexcept {return mix_;}
  float concentration() const noexcept {return concentration_;}
  std::size_t memory_bytes() const noexcept {return sizeof(*this);}
 private:
  void set_notch() noexcept {
    const double angle=6.283185307179586*frequency_/rate_;
    const double alpha=std::sin(angle)/24.; // Q=12; no pure transport delay.
    b0_=b2_=1/(1+alpha);b1_=-2*std::cos(angle)/(1+alpha);
    a1_=b1_;a2_=(1-alpha)/(1+alpha);
  }
  webrtc::RealFourierOoura fft_;
  unsigned rate_,burst_frames_=0,repeats_=0;
  std::uint64_t frames_=0,event_start_=0,last_event_=0;
  std::array<float,512> history_{},time_{},window_{};
  std::array<float,257> power_{};
  std::array<std::complex<float>,257> spectrum_{};
  double event_peak_=0,b0_=1,b1_=0,b2_=0,a1_=0,a2_=0,z1_=0,z2_=0;
  float event_hz_=0,candidate_hz_=0,frequency_=0,concentration_=0,mix_=0,previous_=0;
  bool learned_=false;
};
}
