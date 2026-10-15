#pragma once
#include <algorithm>
#include <array>
#include <cstddef>

namespace itu {
// 160-sample input callbacks -> 256-sample causal context -> 160 samples.
// Each callback reuses the previous 96 samples as context and emits only the
// current 160-sample region of the model result. The first context is zero
// padded, so startup has no fabricated output prefix and no FIFO bloat.
// State and scratch are fixed arrays. The callback is an injected streaming model.
class FixedHopAdapter {
 public:
  template<class Process>
  bool process(const float* mic,const float* ref,float* out,Process&& model) {
    std::copy(context_mic_.begin(),context_mic_.end(),mic_.begin());
    std::copy(context_ref_.begin(),context_ref_.end(),ref_.begin());
    std::copy_n(mic,160,mic_.begin()+96);
    std::copy_n(ref,160,ref_.begin()+96);
    if(!model(mic_.data(),ref_.data(),block_.data())) return false;
    std::copy_n(block_.begin()+96,160,out);
    std::copy(mic_.begin()+160,mic_.end(),context_mic_.begin());
    std::copy(ref_.begin()+160,ref_.end(),context_ref_.begin());
    ++inferences_;return true;
  }
  unsigned inferences() const noexcept {return inferences_;}
  unsigned pending() const noexcept {return 0;}
  unsigned queued() const noexcept {return 0;}
 private:
  std::array<float,256> mic_{},ref_{},block_{};
  std::array<float,96> context_mic_{},context_ref_{};
  unsigned inferences_=0;
};
}
