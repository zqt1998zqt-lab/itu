#pragma once
#include "common_audio/real_fourier_ooura.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace itu {
// Multiple coherence: fraction of microphone power outside the joint playback
// spectral subspace. Regularized Cholesky supports dependent stereo channels.
// Histories/workspace are allocated at construction, never in push().
class EchoSubspace {
 public:
  explicit EchoSubspace(unsigned channels):channels_(channels),fft_(11),
      history_((channels+1)*2048),mic_power_(1025),render_history_(channels*8192),
      spectra_((channels+1)*1025),covariance_(1025*channels*channels),cross_(1025*channels) {
    for(unsigned i=0;i<2048;++i) window_[i]=.5f-.5f*std::cos(2*3.14159265358979323846f*i/2048);
  }
  // Reset spectral evidence after a playback-path change without allocating
  // on the real-time thread. A caller can align each render independently.
  void reset() noexcept {
    std::fill(history_.begin(),history_.end(),0.f);
    std::fill(mic_power_.begin(),mic_power_.end(),0.f);
    std::fill(render_history_.begin(),render_history_.end(),0.f);
    std::fill(covariance_.begin(),covariance_.end(),std::complex<float>{});
    std::fill(cross_.begin(),cross_.end(),std::complex<float>{});
    frames_=head_=delay_=0; ratio_=1.f;
  }
  void push(const float* capture,const float* render,unsigned mics,bool excited,unsigned delay) noexcept {
    delay=std::min(delay,8000u);
    if(std::abs(int(delay)-int(delay_))>16) {
      std::fill(covariance_.begin(),covariance_.end(),std::complex<float>{});
      std::fill(cross_.begin(),cross_.end(),std::complex<float>{});
      std::fill(mic_power_.begin(),mic_power_.end(),0.f);
      std::fill(history_.begin(),history_.end(),0.f); frames_=0; ratio_=1.f;
    }
    delay_=delay;
    for(unsigned i=0;i<160;++i) for(unsigned ch=0;ch<channels_;++ch)
      render_history_[((head_+i)%8192)*channels_+ch]=render[i*channels_+ch];
    for(unsigned ch=0;ch<=channels_;++ch) {
      float* history=history_.data()+ch*2048;
      std::move(history+160,history+2048,history);
      for(unsigned i=0;i<160;++i) history[1888+i]=ch ?
        render_history_[((head_+i+8192-delay)%8192)*channels_+ch-1] : capture[i*mics];
    }
    head_=(head_+160)%8192;
    if(!excited) return;
    for(unsigned ch=0;ch<=channels_;++ch) {
      for(unsigned i=0;i<2048;++i) time_[i]=history_[ch*2048+i]*window_[i];
      fft_.Forward(time_.data(),spectra_.data()+ch*1025);
    }
    ++frames_;
    double unexplained=0,total=0;
    for(unsigned bin=26;bin<=448;++bin) {
      const auto y=spectra_[bin];
      mic_power_[bin]=.97f*mic_power_[bin]+.03f*std::norm(y);
      auto* covariance=covariance_.data()+bin*channels_*channels_;
      auto* cross=cross_.data()+bin*channels_;
      double trace=0;
      for(unsigned i=0;i<channels_;++i) {
        const auto x=spectra_[(i+1)*1025+bin];
        cross[i]=.97f*cross[i]+.03f*x*std::conj(y);
        for(unsigned j=0;j<channels_;++j)
          covariance[i*channels_+j]=.97f*covariance[i*channels_+j]+.03f*x*std::conj(spectra_[(j+1)*1025+bin]);
        trace+=covariance[i*channels_+i].real();
      }
      const double loading=std::max(1e-12,trace*.003/channels_);
      std::array<std::complex<double>,64> lower{};
      std::array<std::complex<double>,8> v{},solution{};
      bool valid=true;
      for(unsigned i=0;i<channels_ && valid;++i) for(unsigned j=0;j<=i;++j) {
        std::complex<double> value=covariance[i*channels_+j];
        if(i==j) value+=loading;
        for(unsigned k=0;k<j;++k) value-=lower[i*8+k]*std::conj(lower[j*8+k]);
        if(i==j) {
          if(value.real()<=0 || !std::isfinite(value.real())) {valid=false;break;}
          lower[i*8+j]=std::sqrt(value.real());
        } else lower[i*8+j]=value/lower[j*8+j];
      }
      double explained=0;
      if(valid) {
        for(unsigned i=0;i<channels_;++i) {
          std::complex<double> value=cross[i];
          for(unsigned j=0;j<i;++j) value-=lower[i*8+j]*v[j];
          v[i]=value/lower[i*8+i];
        }
        for(int i=int(channels_)-1;i>=0;--i) {
          auto value=v[i];
          for(unsigned j=i+1;j<channels_;++j) value-=std::conj(lower[j*8+i])*solution[j];
          solution[i]=value/std::conj(lower[i*8+i]);
          explained+=(std::conj(std::complex<double>(cross[i]))*solution[i]).real();
        }
      }
      unexplained+=std::max(0.,double(mic_power_[bin])-std::clamp(explained,0.,double(mic_power_[bin])));
      total+=mic_power_[bin];
    }
    ratio_=total>1e-10 ? float(unexplained/total) : 1.f;
  }
  float ratio() const noexcept {return ratio_;}
  unsigned frames() const noexcept {return frames_;}
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this)+(history_.capacity()+mic_power_.capacity()+render_history_.capacity())*sizeof(float)+
      (spectra_.capacity()+covariance_.capacity()+cross_.capacity())*sizeof(std::complex<float>);
  }
 private:
  unsigned channels_,frames_=0,head_=0,delay_=0;
  webrtc::RealFourierOoura fft_;
  std::vector<float> history_,mic_power_,render_history_;
  std::vector<std::complex<float>> spectra_,covariance_,cross_;
  std::array<float,2048> time_{},window_{};
  float ratio_=1;
};
}
