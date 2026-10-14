#include "noise_profile.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void require(bool value,const char* message) {
  if(!value) throw std::runtime_error(message);
}
// Original implementation is the negative control, independent of the new
// estimator's mode branch. Exact float comparisons protect the frozen baseline.
class FrozenNoiseProfile {
 public:
  FrozenNoiseProfile():fft_(8) {
    for(unsigned i=0;i<256;++i) window_[i]=.5f-.5f*std::cos(2*3.14159265358979323846f*i/256);
  }
  void observe(const float* input,bool eligible) {
    std::copy(history_.begin()+160,history_.end(),history_.begin());
    std::copy_n(input,160,history_.begin()+96);
    if(!eligible || count_>=40) return;
    for(unsigned i=0;i<256;++i) time_[i]=history_[i]*window_[i];
    fft_.Forward(time_.data(),spectrum_.data());
    double arithmetic=0,logarithmic=0;
    for(unsigned i=2;i<128;++i) {
      const double p=std::max(double(std::norm(spectrum_[i])),1e-20);
      arithmetic+=p; logarithmic+=std::log(p);
    }
    if(arithmetic>1e-9) {
      flatness_sum_+=std::exp(logarithmic/126)/(arithmetic/126);
      ++count_;
    }
  }
  float flatness() const {return count_ ? flatness_sum_/count_ : 0;}
  bool white() const {return count_>=20 && flatness()>.45f;}
 private:
  webrtc::RealFourierOoura fft_;
  std::array<float,256> history_{},time_{},window_{};
  std::array<std::complex<float>,129> spectrum_{};
  unsigned count_=0;
  double flatness_sum_=0;
};
struct NoiseSource {
  std::mt19937 random{20261011};
  std::normal_distribution<float> sample{0.f,.02f};
  float previous=0;
  void fill(std::array<float,160>& frame,bool white) {
    for(float& value:frame) {
      const float v=sample(random);
      previous=.97f*previous+v;
      value=white ? v : previous*.24310492f;
    }
  }
};
void frozen_baseline() {
  itu::NoiseProfile default_mode,explicit_frozen(false);
  FrozenNoiseProfile original;
  NoiseSource noise;
  std::array<float,160> frame{};
  for(unsigned i=0;i<600;++i) {
    noise.fill(frame,i>=150 && i<400);
    const bool eligible=i%11!=0;
    default_mode.observe(frame.data(),eligible);
    explicit_frozen.observe(frame.data(),eligible);
    original.observe(frame.data(),eligible);
    require(default_mode.flatness()==original.flatness(),"default baseline flatness changed");
    require(explicit_frozen.flatness()==original.flatness(),"explicit baseline flatness changed");
    require(default_mode.white()==original.white(),"baseline classification changed");
    require(default_mode.white_mix()==(original.white()?1.f:0.f),"baseline switch changed");
  }
  require(default_mode.observations()==40,"baseline no longer freezes at 40");
}
void transitions_and_qualification() {
  itu::NoiseProfile online(true),frozen;
  NoiseSource noise;
  std::array<float,160> frame{};
  float last_mix=0;
  auto observe=[&](bool eligible) {
    online.observe(frame.data(),eligible);
    frozen.observe(frame.data(),eligible);
    require(std::abs(online.white_mix()-last_mix)<=.050001f,"classification blend jumped");
    last_mix=online.white_mix();
  };
  for(unsigned i=0;i<160;++i) {noise.fill(frame,false);observe(true);}
  require(!online.white() && online.flatness()<.3f,"colored startup misclassified");
  const float colored=online.flatness();
  for(unsigned i=0;i<240;++i) {noise.fill(frame,true);observe(true);}
  require(online.white() && online.white_mix()>.999f,"online white noise not acquired");
  require(!frozen.white() && frozen.flatness()<.3f,"frozen negative control did not expose stale profile");
  const float white=online.flatness();
  const unsigned observations=online.observations();
  for(unsigned i=0;i<250;++i) {
    for(unsigned j=0;j<160;++j) frame[j]=.8f*std::sin(2*3.14159265358979323846*(160*i+j)*997/16000);
    observe(false);
    require(online.flatness()==white && online.white(),"ineligible speech changed noise estimate");
    require(online.observations()==observations,"ineligible speech counted as noise");
  }
  noise.fill(frame,false);observe(true);
  require(online.flatness()==white && online.observations()==observations,"speech tail entered resumed FFT");
  for(unsigned i=0;i<240;++i) {noise.fill(frame,false);observe(true);}
  require(!online.white() && online.flatness()<.3f && online.white_mix()<.001f,"colored noise not reacquired");
  std::cout<<"flatness colored="<<colored<<" white="<<white<<" returned="<<online.flatness()
           <<" observations="<<online.observations()<<" frozen="<<frozen.flatness()<<'\n';
}
void silence_and_hold() {
  itu::NoiseProfile silent(true),held(true);
  std::array<float,160> frame{};
  for(unsigned i=0;i<300;++i) silent.observe(frame.data(),true);
  require(silent.observations()==0 && silent.flatness()==0 && !silent.white() && silent.white_mix()==0,
          "silence produced a white classification");
  NoiseSource noise;
  while(!held.white()) {
    require(held.observations()<300,"white readiness missing");
    noise.fill(frame,true);
    held.observe(frame.data(),true);
  }
  const float classification=held.flatness();
  const unsigned observations=held.observations();
  for(unsigned i=0;i<25;++i) held.observe(frame.data(),false);
  require(held.white_mix()>.999f && held.flatness()==classification && held.observations()==observations,
          "held classification did not finish bounded blend");
  frame.fill(0);
  held.observe(frame.data(),true);
  held.observe(frame.data(),true);
  const float flushed_flatness=held.flatness();
  const unsigned flushed_observations=held.observations();
  for(unsigned i=0;i<300;++i) held.observe(frame.data(),true);
  require(held.flatness()==flushed_flatness && held.observations()==flushed_observations && held.white(),
          "continuous silence updated the held noise profile");
}
}
int main() {
  try {
    frozen_baseline();
    transitions_and_qualification();
    silence_and_hold();
    std::cout<<"ECNR online noise-profile contracts passed\n";
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
  }
}
