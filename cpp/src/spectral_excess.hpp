#pragma once
#include "common_audio/real_fourier_ooura.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
namespace itu {
// Minimum statistics, without a learned speech classifier. The score measures
// excess above the causal per-band noise floor; it is NOT a VAD probability.
class SpectralExcess {
 public:
  SpectralExcess():fft_(8) {
    for(unsigned i=0;i<256;++i) window_[i]=.5f-.5f*std::cos(6.283185307179586f*i/255);
    block_min_.fill(std::numeric_limits<float>::max());
  }
  float observe(const float* audio,bool eligible) noexcept {
    std::move(history_.begin()+160,history_.end(),history_.begin());
    std::copy_n(audio,160,history_.end()-160);
    for(unsigned i=0;i<256;++i) time_[i]=history_[i]*window_[i];
    fft_.Forward(time_.data(),spectrum_.data());
    std::array<float,16> bands{};
    for(unsigned k=2;k<128;++k) bands[std::min(15u,(k-2)/8)]+=std::norm(spectrum_[k]);
    if(!eligible) return score_=0;
    ++frames_;
    double sum=0,log_sum=0,floor_sum=0;
    for(unsigned b=0;b<16;++b) {
      const double p=std::max(double(bands[b]),1e-20);
      sum+=p; log_sum+=std::log(p); floor_sum+=floor_[b];
    }
    flatness_=static_cast<float>(sum>1e-14 ? std::exp(log_sum/16.)/(sum/16.) : 1.);
    if(frames_<=30) {
      // Do not let an early voiced onset overwrite the initial ambient floor,
      // while still tracking downward minima and broad ambient noise startups.
      if(floor_frames_==0 || sum<=2.5*std::max(floor_sum,1e-18) || flatness_>.38f) {
        ++floor_frames_;
        for(unsigned b=0;b<16;++b) floor_[b]+=(bands[b]-floor_[b])/floor_frames_;
      } else {
        for(unsigned b=0;b<16;++b) floor_[b]=std::min(floor_[b],bands[b]);
      }
    }
    for(unsigned b=0;b<16;++b) block_min_[b]=std::min(block_min_[b],bands[b]);
    if(frames_%20==0) {
      minima_[head_]=block_min_;head_=(head_+1)%10;blocks_=std::min(10u,blocks_+1);
      const unsigned valid_blocks=std::clamp(blocks_,1u,10u);
      for(unsigned b=0;b<16;++b) {
        std::array<float,10> sorted;
        sorted.fill(std::numeric_limits<float>::max());
        for(unsigned j=0;j<valid_blocks;++j) sorted[j]=minima_[j][b];
        std::sort(sorted.begin(),sorted.end());
        const float estimated=std::max(1e-18f,2.f*sorted[valid_blocks/3]);
        floor_[b]+=.2f*(estimated-floor_[b]);
      }
      block_min_.fill(std::numeric_limits<float>::max());
    }
    double excess=0,total=0;
    unsigned active=0;
    for(unsigned b=0;b<16;++b) {
      total+=bands[b];excess+=std::max(0.f,bands[b]-2.f*floor_[b]);
      active+=bands[b]>4.f*std::max(floor_[b],1e-18f);
    }
    fraction_=total>1e-14 ? excess/total : 0;
    const float broad=std::clamp((float(active)-1)/4.f,0.f,1.f);
    const float measured=std::clamp((fraction_-.35f)/.5f,0.f,1.f)*broad;
    score_=frames_>=30 ? measured : std::max(.5f,measured);
    return score_;
  }
  float score() const noexcept {return score_;}
  float fraction() const noexcept {return fraction_;}
  float flatness() const noexcept {return flatness_;}
 private:
  webrtc::RealFourierOoura fft_;
  std::array<float,256> history_{},time_{},window_{};
  std::array<std::complex<float>,129> spectrum_{};
  std::array<float,16> floor_{},block_min_{};
  std::array<std::array<float,16>,10> minima_{};
  unsigned frames_=0,floor_frames_=0,head_=0,blocks_=0;
  float score_=0,fraction_=0,flatness_=1.f;
};
}
