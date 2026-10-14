#pragma once
#include <algorithm>
#include <array>
#include <cstddef>

namespace itu {
// 160-sample input callbacks -> causal 256-sample model hops -> 160 samples.
// A fixed 256-sample output prefix prevents variable-latency underflow. State
// and scratch are fixed arrays. The callback is an injected streaming model.
class FixedHopAdapter {
 public:
  template<class Process>
  bool process(const float* mic,const float* ref,float* out,Process&& model) {
    for(unsigned i=0;i<160;++i) {
      mic_[pending_]=mic[i];ref_[pending_]=ref[i];
      if(++pending_==256) {
        if(!model(mic_.data(),ref_.data(),block_.data())) return false;
        if(queued_+256>queue_.size()) return false;
        for(float v:block_) queue_[(head_+queued_++)%queue_.size()]=v;
        pending_=0;++inferences_;
      }
    }
    if(queued_<160) return false;
    for(unsigned i=0;i<160;++i) {out[i]=queue_[head_];head_=(head_+1)%queue_.size();}
    queued_-=160;return true;
  }
  unsigned inferences() const noexcept {return inferences_;}
  unsigned pending() const noexcept {return pending_;}
  unsigned queued() const noexcept {return queued_;}
 private:
  std::array<float,256> mic_{},ref_{},block_{};
  std::array<float,1024> queue_{};
  unsigned pending_=0,head_=0,queued_=256,inferences_=0;
};
}
