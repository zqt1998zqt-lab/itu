#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace itu {
// Subtractive adaptive filters can diverge transiently. Reduce the entire
// hop's prediction (one coefficient) toward real capture to respect the
// downstream PCM contract. No sample clipping, labels or clean reference.
// The filter keeps adapting; restoration is immediate, recovery is gradual.
class AecSafetyMix {
 public:
  bool process(const float* capture,float* residual,std::size_t n) noexcept {
    float bound=1;
    bool finite=true;
    for(std::size_t i=0;i<n;++i) {
      // Rate conversion may overshoot even when source PCM was valid.
      // An invalid fallback cannot satisfy the downstream PCM contract.
      if(!std::isfinite(capture[i]) || std::abs(capture[i])>1.f) return false;
      if(!std::isfinite(residual[i])) {finite=false;continue;}
      peak_=std::max(peak_,std::abs(residual[i]));
      const float delta=residual[i]-capture[i];
      if(delta>0) bound=std::min(bound,(1.f-capture[i])/delta);
      else if(delta<0) bound=std::min(bound,(-1.f-capture[i])/delta);
    }
    if(!finite) {
      // Do not evaluate 0 * Inf/NaN. Use the entire real capture hop,
      // and retain a fault count so recovery cannot masquerade as acceptance.
      std::copy_n(capture,n,residual);
      mix_=minimum_=0.f;++protected_;++nonfinite_;
      return true;
    }
    bound=std::clamp(bound,0.f,1.f);
    // Leave numerical headroom only during protection, not on normal hops.
    if(bound<1.f) {bound*=.9999f;++protected_;}
    mix_=std::min(bound,mix_+.05f);
    minimum_=std::min(minimum_,mix_);
    for(std::size_t i=0;i<n;++i) residual[i]=capture[i]+mix_*(residual[i]-capture[i]);
    return true;
  }
  float mix() const noexcept {return mix_;}
  float minimum_mix() const noexcept {return minimum_;}
  float raw_peak() const noexcept {return peak_;}
  unsigned protected_hops() const noexcept {return protected_;}
  unsigned nonfinite_hops() const noexcept {return nonfinite_;}
 private:float mix_=1,minimum_=1,peak_=0;unsigned protected_=0,nonfinite_=0;
};
}
