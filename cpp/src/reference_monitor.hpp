#pragma once
#include "common_audio/real_fourier_ooura.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace itu {
// One-second GCC-PHAT history. FFT work is split across four 10 ms calls.
// This observes capture/render only, never an evaluation target or scene label.
class ReferenceMonitor {
 public:
  explicit ReferenceMonitor(bool retain_state_on_silence=false):fft_(15),mic_(32768),ref_(32768),time_(32768),render_time_(32768),corr_(32768),
      x_(16385),y_(16385),cross_(16385),retain_state_on_silence_(retain_state_on_silence) {}
  void push(const float* capture,const float* render,unsigned channels) noexcept {
    for(unsigned i=0;i<160;++i) {
      mic_[head_]=capture[i*channels]; ref_[head_]=render[i];
      head_=(head_+1)%16000;
    }
    if(++frames_<100) return;
    const unsigned phase=(frames_-100)%50;
    if(phase==0) {
      prepare(ref_); std::copy(time_.begin(),time_.end(),render_time_.begin());
      prepare(mic_); fft_.Forward(time_.data(),x_.data());
      double power=0;
      for(unsigned i=0;i<16000;++i) power+=double(render_time_[i])*render_time_[i];
      render_excited_=power/16000>1e-7;
    } else if(phase==1) {
      fft_.Forward(render_time_.data(),y_.data());
    } else if(phase==2) {
      for(unsigned i=0;i<cross_.size();++i) {
        const auto product=x_[i]*std::conj(y_[i]);
        cross_[i]=product/std::max(std::abs(product),1e-12f);
      }
      cross_[0]=0; fft_.Inverse(cross_.data(),corr_.data());
    } else if(phase==3) {
      if(!render_excited_) {
        confidence_=0;
        consistent_=weak_windows_=0;
        if(!retain_state_on_silence_) reliable_=unreliable_=false;
        return;
      }
      double p=0; float peak=0; unsigned lag=0;
      for(unsigned i=0;i<=8000;++i) {
        p+=double(corr_[i])*corr_[i];
        if(std::abs(corr_[i])>peak) {peak=std::abs(corr_[i]); lag=i;}
      }
      confidence_=peak/std::sqrt(p/8001+1e-20);
      if(confidence_>=13.f) {
        weak_windows_=0;
        if(consistent_ && std::abs(int(lag)-int(candidate_))<=64) ++consistent_;
        else consistent_=1;
        candidate_=lag;
        if(consistent_>=2) {delay_=lag; reliable_=has_delay_=true; unreliable_=false;}
      } else {
        consistent_=0;
        // A once-valid playback reference can be replaced during a live stream.
        if(++weak_windows_>=4) {reliable_=has_delay_=false; unreliable_=true;}
      }
    }
  }
  bool reliable() const noexcept {return reliable_;}
  // Silence stops confidence updates but does not change the acoustic path.
  bool delay_available() const noexcept {return has_delay_ && !unreliable_;}
  bool unreliable() const noexcept {return unreliable_;}
  unsigned delay_samples() const noexcept {return delay_;}
  float confidence() const noexcept {return confidence_;}
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this)+(mic_.capacity()+ref_.capacity()+time_.capacity()+render_time_.capacity()+corr_.capacity())*sizeof(float)+
        (x_.capacity()+y_.capacity()+cross_.capacity())*sizeof(std::complex<float>)+
        184u*sizeof(std::size_t)+16385u*sizeof(float);
  }
 private:
  void prepare(const std::vector<float>& source) noexcept {
    for(unsigned i=0;i<16000;++i) time_[i]=source[(head_+i)%16000];
    std::fill(time_.begin()+16000,time_.end(),0.f);
  }
  webrtc::RealFourierOoura fft_;
  std::vector<float> mic_,ref_,time_,render_time_,corr_;
  std::vector<std::complex<float>> x_,y_,cross_;
  unsigned head_=0,frames_=0,candidate_=0,consistent_=0,delay_=0,weak_windows_=0;
  float confidence_=0;
  bool reliable_=false,unreliable_=false,render_excited_=false,has_delay_=false;
  bool retain_state_on_silence_=false;
};
}
