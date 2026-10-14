#pragma once
#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace itu {
// Causal WOLA operating on already delay-calibrated microphones. All three
// protection branches share weights, phase and delay. No per-callback allocation.
class SpatialFrontend {
 public:
  SpatialFrontend(unsigned rate,unsigned hop,unsigned microphones,bool mvdr,bool wind,bool fixed_steering=false,bool continuous_covariance=false);
  void process(const float* capture,const float* linear,const float* lowband,
               float* output,float* linear_output,float* lowband_output,
               bool noise_qualified) noexcept;
  unsigned delay() const noexcept {return size_-hop_;}
  unsigned updates() const noexcept {return updates_;}
  unsigned noise_updates() const noexcept {return noise_updates_;}
  float wind_score() const noexcept {return wind_score_;}
  unsigned fallbacks() const noexcept {return fallbacks_;}
  std::size_t memory_bytes() const noexcept;
 private:
  using C=std::complex<float>;
  void fft(C* x,bool inverse) noexcept;
  bool weights(unsigned bin,std::array<C,8>& w) noexcept;
  unsigned rate_,hop_,m_,size_=1,updates_=0,noise_updates_=0,fallbacks_=0;
  bool mvdr_,wind_,fixed_steering_,continuous_covariance_;
  float wind_score_=0;
  std::vector<float> window_,history_,overlap_,normalizer_;
  std::vector<C> spectrum_,covariance_,speech_covariance_,scratch_,roots_;
};
}
