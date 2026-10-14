#include "linear_phase_lowcut.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
// Retained direct convolution is the numerical and performance control.
template<bool Symmetric> class ProbeFilter {
 public:
  ProbeFilter(unsigned channels,float cutoff):history_(channels),heads_(channels) {
    itu::LinearPhaseLowCut impulse(1,cutoff);
    for(unsigned i=0;i<513;++i) coefficients_[512-i]=impulse.process(i==0 ? 1.f : 0.f,0);
  }
  float process(float input,unsigned channel) {
    auto& h=history_[channel]; auto& head=heads_[channel];
    h[head]=h[head+513]=input;
    head=(head+1)%513;
    float sum=0;
    if constexpr(Symmetric) {
      std::array<float,4> accum{};
      for(unsigned i=0;i<256;i+=4) for(unsigned j=0;j<4;++j)
        accum[j]+=coefficients_[i+j]*(h[head+i+j]+h[head+512-i-j]);
      sum=((accum[0]+accum[1])+(accum[2]+accum[3]))+coefficients_[256]*h[head+256];
    } else for(unsigned i=0;i<513;++i) sum+=coefficients_[i]*h[head+i];
    return sum;
  }
 private:
  std::array<float,513> coefficients_{};
  std::vector<std::array<float,1026>> history_;
  std::vector<unsigned> heads_;
};
template<class Filter> void measure(const char* label,unsigned channels,const std::vector<float>& input) {
  Filter filter(channels,60.f);
  std::vector<double> times;
  double checksum=0;
  for(unsigned frame=0;frame<4000;++frame) {
    const auto begin=std::chrono::steady_clock::now();
    for(unsigned i=0;i<160;++i) for(unsigned ch=0;ch<channels;++ch)
      checksum+=filter.process(input[frame*160+i],ch);
    const auto end=std::chrono::steady_clock::now();
    if(frame>=100) times.push_back(std::chrono::duration<double,std::micro>(end-begin).count());
  }
  std::sort(times.begin(),times.end());
  std::cout<<label<<" channels="<<channels<<" p50_us="<<times[times.size()/2]
           <<" p95_us="<<times[static_cast<std::size_t>(std::ceil(.95*times.size()))-1]
           <<" checksum="<<checksum<<'\n';
}
}
int main() {
  std::vector<float> input(4000*160); std::uint32_t seed=7;
  for(auto& v:input) {seed=seed*1664525u+1013904223u; v=(int(seed>>16)-32768)/32768.f*.5f;}
  double max_error=0,negative_error=0;
  for(float cutoff:{1.f,60.f,120.f}) {
    itu::LinearPhaseLowCut actual(2,cutoff); ProbeFilter<false> direct(2,cutoff); ProbeFilter<true> candidate(2,cutoff);
    for(unsigned i=0;i<10000;++i) for(unsigned ch=0;ch<2;++ch) {
      const float x=ch==0 ? input[i] : (i==0 ? 1.f : 0.f);
      const float a=actual.process(x,ch),d=direct.process(x,ch),b=candidate.process(x,ch);
      max_error=std::max({max_error,double(std::abs(a-b)),double(std::abs(a-d))});
      negative_error=std::max(negative_error,double(std::abs(a-x)));
    }
  }
  std::cout<<"max_absolute_error="<<max_error<<" bypass_negative_error="<<negative_error<<'\n';
  if(max_error>2e-6 || negative_error<.1) return 1;
  for(unsigned repeat=0;repeat<3;++repeat) for(unsigned channels:{1u,2u}) {
    measure<itu::LinearPhaseLowCut>("actual",channels,input);
    measure<ProbeFilter<true>>("symmetric_candidate",channels,input);
    measure<ProbeFilter<false>>("retained_direct",channels,input);
  }
}
