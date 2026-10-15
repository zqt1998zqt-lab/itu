#include "linear_phase_lowcut.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace {
void check_filter(unsigned rate,float cutoff) {
  itu::MinimumDelayHighPass filter(2,cutoff,rate);
  itu::MinimumDelayHighPass scalar(2,cutoff,rate);
  std::array<float,160> input{},output{},reference{};
  input[0]=1.f;
  filter.process_frame(input.data(),output.data(),160,0);
  for(unsigned i=0;i<160;++i) reference[i]=scalar.process(input[i],0);
  for(unsigned i=0;i<160;++i) {
    if(!std::isfinite(output[i]) || std::abs(output[i]-reference[i])>1e-6f)
      throw std::runtime_error("IIR frame/scalar mismatch");
  }
  const float impulse0=output[0];
  if(filter.delay()!=0 || std::abs(impulse0)<.1f)
    throw std::runtime_error("minimum-delay impulse contract failed");

  std::fill(input.begin(),input.end(),1.f);
  double tail=0;
  for(unsigned frame=0;frame<100;++frame) {
    filter.process_frame(input.data(),output.data(),160,0);
    if(frame>=50) for(float v:output) tail+=double(v)*v;
  }
  if(cutoff>=60.f && tail>1e-3) throw std::runtime_error("high-pass DC rejection failed");
  std::cout<<"lowcut rate="<<rate<<" cutoff="<<cutoff
           <<" delay_samples="<<filter.delay()<<" impulse0="<<impulse0
           <<" tail_power="<<tail<<'\n';
}

template<class Filter> void measure(const char* label,unsigned channels,unsigned rate) {
  Filter filter(channels,60.f,rate);
  std::vector<float> input(4000*rate/100),output(rate/100);
  std::uint32_t seed=7; double checksum=0;
  std::vector<double> times;
  for(auto& value:input) {seed=seed*1664525u+1013904223u;value=(int(seed>>16)-32768)/32768.f*.5f;}
  for(unsigned frame=0;frame<4000;++frame) {
    const auto begin=std::chrono::steady_clock::now();
    for(unsigned i=0;i<rate/100;++i) checksum+=filter.process(input[frame*rate/100+i],0);
    const auto end=std::chrono::steady_clock::now();
    if(frame>=100) times.push_back(std::chrono::duration<double,std::micro>(end-begin).count());
  }
  std::sort(times.begin(),times.end());
  std::cout<<label<<" channels="<<channels<<" rate="<<rate
           <<" p50_us="<<times[times.size()/2]
           <<" p95_us="<<times[static_cast<std::size_t>(std::ceil(.95*times.size()))-1]
           <<" checksum="<<checksum<<'\n';
}
}

int main() {
  try {
    for(unsigned rate:{16000u,24000u,48000u}) for(float cutoff:{1.f,60.f,120.f}) check_filter(rate,cutoff);
    for(unsigned rate:{16000u,24000u,48000u}) measure<itu::MinimumDelayHighPass>("iir",1,rate);
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n'; return 1;
  }
}
