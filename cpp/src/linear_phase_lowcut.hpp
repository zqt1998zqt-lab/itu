#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <memory>
#include <complex>
#include "common_audio/real_fourier_ooura.h"

namespace itu {
// Shared symmetric coefficients; independent, fixed-size histories per channel.
// All allocation and coefficient design happen on the control thread.
class LinearPhaseLowCut {
 public:
  explicit LinearPhaseLowCut(unsigned channels,float cutoff_hz,unsigned rate=16000):
      delay_(rate*16/1000),length_(2*delay_+1),coefficients_(length_),
      history_(channels,std::vector<float>(2*length_)),head_(channels) {
    constexpr double pi=3.14159265358979323846;
    double sum=0;
    for(unsigned i=0;i<length_;++i) {
      const int x=int(i)-int(delay_);
      const double sinc=x==0 ? 2.*cutoff_hz/rate : std::sin(2*pi*cutoff_hz*x/rate)/(pi*x);
      coefficients_[i]=static_cast<float>(sinc*(.54-.46*std::cos(2*pi*i/(length_-1))));
      sum+=coefficients_[i];
    }
    for(unsigned i=0;i<length_;++i)
      coefficients_[i]=static_cast<float>((i==delay_ ? 1. : 0.)-coefficients_[i]/sum);
    const unsigned hop=rate/100;
    while(fft_size_<length_-1+hop) fft_size_*=2;
    unsigned order=0; for(unsigned n=fft_size_;n>1;n/=2) ++order;
    fft_=std::make_unique<webrtc::RealFourierOoura>(order);
    time_.resize(fft_size_); filter_.resize(fft_size_/2+1); spectrum_.resize(filter_.size());
    std::copy(coefficients_.begin(),coefficients_.end(),time_.begin());
    fft_->Forward(time_.data(),filter_.data());
  }
  float process(float value,unsigned channel) noexcept {
    auto& history=history_[channel]; auto& head=head_[channel];
    history[head]=history[head+length_]=value;
    head=(head+1)%length_;
    // The FIR is symmetric. Pairing mirrored taps halves the multiply/add work
    // while preserving the causal 513-sample response.
    std::array<float,4> accum{};
    for(unsigned i=0;i<delay_;i+=4) for(unsigned j=0;j<4;++j)
      accum[j]+=coefficients_[i+j]*(history[head+i+j]+history[head+length_-1-i-j]);
    return ((accum[0]+accum[1])+(accum[2]+accum[3]))+coefficients_[delay_]*history[head+delay_];
  }
  // Overlap-save of the identical causal FIR. All FFT workspaces are allocated
  // in the constructor. Gather a complete frame before writing aliasing output.
  void process_frame(const float* input,float* output,unsigned count,unsigned channel) noexcept {
    auto& history=history_[channel]; auto& head=head_[channel];
    std::copy_n(history.data()+head+1,length_-1,time_.data());
    std::copy_n(input,count,time_.data()+length_-1);
    std::fill(time_.begin()+length_-1+count,time_.end(),0.f);
    for(unsigned i=0;i<count;++i) {
      history[head]=history[head+length_]=input[i]; head=(head+1)%length_;
    }
    fft_->Forward(time_.data(),spectrum_.data());
    for(unsigned i=0;i<spectrum_.size();++i) spectrum_[i]*=filter_[i];
    fft_->Inverse(spectrum_.data(),time_.data());
    std::copy_n(time_.data()+length_-1,count,output);
  }
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this)+(coefficients_.capacity()+history_.size()*2*length_+time_.capacity())*sizeof(float)+head_.capacity()*sizeof(unsigned)+
        (filter_.capacity()+spectrum_.capacity())*sizeof(std::complex<float>)+
        (2+unsigned(std::ceil(std::sqrt(fft_size_))))*sizeof(std::size_t)+(fft_size_/2+1)*sizeof(float);
  }
 private:
  unsigned delay_,length_;
  std::vector<float> coefficients_;
  std::vector<std::vector<float>> history_;
  std::vector<unsigned> head_;
  unsigned fft_size_=1;
  std::unique_ptr<webrtc::RealFourierOoura> fft_;
  std::vector<float> time_;
  std::vector<std::complex<float>> filter_,spectrum_;
};
}
