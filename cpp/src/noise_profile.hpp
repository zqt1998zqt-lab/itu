#pragma once
#include "common_audio/real_fourier_ooura.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

namespace itu {
class NoiseProfile {
 public:
  explicit NoiseProfile(bool dynamic=false):fft_(8),dynamic_(dynamic) {
    for(unsigned i=0;i<256;++i) window_[i]=.5f-.5f*std::cos(2*3.14159265358979323846f*i/256);
  }
  // Includes the Ooura work arrays allocated behind RealFourierOoura.
  std::size_t memory_bytes() const noexcept {
    constexpr std::size_t length = 1u << 8;
    constexpr std::size_t work_ip = 2u + 16u;  // ceil(sqrt(256))
    constexpr std::size_t complex_length = length / 2u + 1u;
    return sizeof(*this) + work_ip * sizeof(std::size_t) +
        complex_length * sizeof(float);
  }
  void observe(const float* input,bool eligible) noexcept {
    std::copy(history_.begin()+160,history_.end(),history_.begin());
    std::copy_n(input,160,history_.begin()+96);
    if(!dynamic_) {
      // Keep the original frozen estimator exactly for paired baseline runs.
      if(!eligible || count_>=40) return;
    } else {
      // A 256-sample FFT spans two 10 ms observations. Do not let an
      // ineligible speech/render tail enter the next qualified noise window.
      eligible_frames_=eligible ? std::min(2u,eligible_frames_+1) : 0;
      if(eligible_frames_<2) {advance_mix(); return;}
    }
    for(unsigned i=0;i<256;++i) time_[i]=history_[i]*window_[i];
    fft_.Forward(time_.data(),spectrum_.data());
    double arithmetic=0,logarithmic=0;
    for(unsigned i=2;i<128;++i) {
      const double p=std::max(double(std::norm(spectrum_[i])),1e-20);
      arithmetic+=p; logarithmic+=std::log(p);
    }
    if(arithmetic>1e-9) {
      const double value=std::exp(logarithmic/126)/(arithmetic/126);
      if(!dynamic_) {
        flatness_sum_+=value;
        ++count_;
      } else {
        if(count_<40) {
          flatness_sum_+=value;
          ++count_;
          online_flatness_=flatness_sum_/count_;
        } else {
          online_flatness_+=.025*(value-online_flatness_);
          if(count_!=~0u) ++count_;
        }
        if(count_>=20) {
          if(!white_ && online_flatness_>.48) white_=true;
          else if(white_ && online_flatness_<.42) white_=false;
        }
      }
    }
    if(dynamic_) advance_mix();
  }
  float flatness() const noexcept {
    return dynamic_ ? online_flatness_ : count_ ? flatness_sum_/count_ : 0;
  }
  bool white() const noexcept {return dynamic_ ? white_ : count_>=20 && flatness()>.45f;}
  // Candidate control changes at most 0.05 per 10 ms frame, including frames
  // where classification is held. The frozen baseline keeps its old switch.
  float white_mix() const noexcept {return dynamic_ ? white_mix_ : white() ? 1.f : 0.f;}
  unsigned observations() const noexcept {return count_;}
 private:
  void advance_mix() noexcept {
    const float target=white_ ? 1.f : 0.f;
    white_mix_+=std::clamp(target-white_mix_,-.05f,.05f);
  }
  webrtc::RealFourierOoura fft_;
  std::array<float,256> history_{},time_{},window_{};
  std::array<std::complex<float>,129> spectrum_{};
  unsigned count_=0,eligible_frames_=0;
  double flatness_sum_=0,online_flatness_=0;
  bool dynamic_=false,white_=false;
  float white_mix_=0;
};
}
