#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

namespace itu {
// Compatibility name retained for the public/configuration surface. The old
// symmetric FIR introduced a fixed 16 ms look-ahead-equivalent group delay.
// This implementation is a causal second-order Butterworth high-pass. Its
// phase is frequency dependent, so it has no fixed samples to add to the
// stream alignment contract; the measured signal remains sample-causal.
class MinimumDelayHighPass {
 public:
  explicit MinimumDelayHighPass(unsigned channels,float cutoff_hz,unsigned rate=16000)
      : channels_(channels),rate_(rate),cutoff_hz_(cutoff_hz),states_(channels) {
    constexpr double pi=3.14159265358979323846;
    constexpr double q=0.7071067811865475244; // 2nd-order Butterworth Q.
    const double omega=2.*pi*static_cast<double>(cutoff_hz_)/static_cast<double>(rate_);
    const double cosine=std::cos(omega),sine=std::sin(omega);
    const double alpha=sine/(2.*q);
    const double a0=1.+alpha;
    b0_=(1.+cosine)/(2.*a0);
    b1_=-(1.+cosine)/a0;
    b2_=(1.+cosine)/(2.*a0);
    a1_=-2.*cosine/a0;
    a2_=(1.-alpha)/a0;
  }

  float process(float value,unsigned channel) noexcept {
    auto& state=states_[channel];
    const double y=b0_*value+state.z1;
    state.z1=b1_*value-a1_*y+state.z2;
    state.z2=b2_*value-a2_*y;
    return static_cast<float>(y);
  }

  // The filter is causal and supports in-place frame processing.
  void process_frame(const float* input,float* output,unsigned count,unsigned channel) noexcept {
    for(unsigned i=0;i<count;++i) output[i]=process(input[i],channel);
  }

  unsigned delay() const noexcept {return 0;}

  std::size_t memory_bytes() const noexcept {
    return sizeof(*this)+states_.capacity()*sizeof(State);
  }

 private:
  struct State {double z1=0.,z2=0.;};
  unsigned channels_,rate_;
  float cutoff_hz_;
  double b0_=1.,b1_=0.,b2_=0.,a1_=0.,a2_=0.;
  std::vector<State> states_;
};

// Source compatibility for existing callers and diagnostic tools. New code
// should use MinimumDelayHighPass when it needs to name the actual topology.
using LinearPhaseLowCut = MinimumDelayHighPass;
}
