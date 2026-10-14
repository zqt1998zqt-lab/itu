#pragma once
#include "common_audio/real_fourier_ooura.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

namespace itu {
class EchoCoherence {
 public:
  EchoCoherence():fft_(11) {
    for(unsigned i=0;i<2048;++i) window_[i]=.5f-.5f*std::cos(2*3.14159265358979323846f*i/2048);
  }
  // Includes the Ooura work arrays allocated behind RealFourierOoura.
  std::size_t memory_bytes() const noexcept {
    constexpr std::size_t length = 1u << 11;
    constexpr std::size_t work_ip = 2u + 46u;  // ceil(sqrt(2048))
    constexpr std::size_t complex_length = length / 2u + 1u;
    return sizeof(*this) + work_ip * sizeof(std::size_t) +
        complex_length * sizeof(float);
  }
  void push(const float* capture,const float* render,unsigned channels,unsigned delay,bool reset_on_delay_change=false) noexcept {
    std::copy(mic_.begin()+160,mic_.end(),mic_.begin());
    std::copy(ref_.begin()+160,ref_.end(),ref_.begin());
    delay=std::min(delay,8000u);
    if(reset_on_delay_change && delay!=delay_) {
      mic_.fill(0); ref_.fill(0);
      pxx_.fill(0); pyy_.fill(0); pxy_.fill({});
      delay_=delay;
    }
    for(unsigned i=0;i<160;++i) {
      render_history_[head_]=render[i];
      mic_[1888+i]=capture[i*channels];
      ref_[1888+i]=render_history_[(head_+8192-delay)%8192];
      head_=(head_+1)%8192;
    }
    for(unsigned i=0;i<2048;++i) time_[i]=mic_[i]*window_[i];
    fft_.Forward(time_.data(),x_.data());
    for(unsigned i=0;i<2048;++i) time_[i]=ref_[i]*window_[i];
    fft_.Forward(time_.data(),y_.data());
    double unexplained=0,total=0;
    for(unsigned i=26;i<=448;++i) {
      pxx_[i]=.95f*pxx_[i]+.05f*std::norm(x_[i]);
      pyy_[i]=.95f*pyy_[i]+.05f*std::norm(y_[i]);
      pxy_[i]=.95f*pxy_[i]+.05f*x_[i]*std::conj(y_[i]);
      const double coherence=std::clamp(double(std::norm(pxy_[i]))/std::max(double(pxx_[i])*pyy_[i],1e-20),0.,1.);
      unexplained+=(1-coherence)*pxx_[i]; total+=pxx_[i];
    }
    ratio_=total>1e-10 ? unexplained/total : 0;
  }
  float ratio() const noexcept {return ratio_;}
 private:
  webrtc::RealFourierOoura fft_;
  std::array<float,8192> render_history_{};
  std::array<float,2048> mic_{},ref_{},time_{},window_{};
  std::array<std::complex<float>,1025> x_{},y_{},pxy_{};
  std::array<float,1025> pxx_{},pyy_{};
  unsigned head_=0,delay_=~0u;
  float ratio_=0;
};
}
