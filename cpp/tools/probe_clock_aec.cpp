#include "itu/audio_engine.hpp"
#include "itu/reference_clock.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
double playback(double t) {
  if(t<0) return 0;
  double x=0;
  for(unsigned k=0;k<19;++k) {
    const double hz=157+k*143;
    x+=.013*std::sin(2*3.141592653589793*(hz*t+13*std::sin(.19*t+k)));
  }
  return x;
}
}
int main(int argc,char** argv) {
  const bool disabled=argc==2 && std::string_view(argv[1])=="--disable-correction";
  if(argc>1 && !disabled) {std::cerr<<"usage: itu_clock_aec_probe [--disable-correction]\n"; return 2;}
  unsigned failures=0;
  double uncorrected=0;
  for(unsigned rate:{16000,24000,48000}) for(double ppm:{-300.,300.}) for(bool correction:{false,true}) {
    itu::Config config; config.sample_rate=rate; config.frame_size=rate/100;
    config.noise_suppression=config.automatic_gain=config.limiter=false;
    config.linear_phase_low_cut=config.reference_guard=config.linear_speech_protection=false;
    config.correlation_delay=false; config.native_fullband=true;
    itu::AudioEngine engine(config); itu::ReferenceClock clock(rate); itu::FrameInfo info;
    const unsigned h=rate/100; const double rr=rate*(1+ppm*1e-6);
    std::vector<float> input(h),capture(h),reference(h),output(h);
    std::uint64_t written=0; double before=0,after=0;
    for(unsigned k=0;k<9000;++k) {
      const double t=k*.01;
      if(correction && !disabled && k%50==0) {
        const auto ns=std::int64_t(1000000000LL+std::llround(t*1e9));
        if(!clock.observe_capture({std::uint64_t(std::llround(t*rate)),ns}) ||
           !clock.observe_render({std::uint64_t(std::llround(t*rr)),ns})) return 2;
      }
      const auto available=std::uint64_t(std::floor((t+.08)*rr));
      while(written+h<=available) {
        for(unsigned i=0;i<h;++i) input[i]=float(playback((written+i)/rr));
        if(!clock.push(input.data(),h)) return 3;
        written+=h;
      }
      if(!clock.pull(reference.data(),h)) return 4;
      for(unsigned i=0;i<h;++i) {
        const double time=double(k*h+i)/rate;
        capture[i]=float(.6*playback(time-.08)+.18*playback(time-.085));
      }
      if(engine.process(capture.data(),h,reference.data(),h,output.data(),h,info)!=itu::Status::ok) return 5;
      if(k>=7000 && k<8500) for(unsigned i=0;i<h;++i) {
        before+=double(capture[i])*capture[i]; after+=double(output[i])*output[i];
      }
    }
    const auto ci=clock.info();
    const double attenuation=10*std::log10(before/std::max(after,1e-30));
    std::cout<<"{\"rate\":"<<rate<<",\"injected_ppm\":"<<ppm<<",\"correction\":"<<(correction?"true":"false")
      <<",\"correction_applied\":"<<(correction && !disabled?"true":"false")
      <<",\"estimated_ppm\":"<<ci.drift_ppm<<",\"echo_attenuation_db\":"<<attenuation
      <<",\"underruns\":"<<ci.underruns<<",\"overruns\":"<<ci.overruns<<"}"<<std::endl;
    if(!std::isfinite(attenuation) || before<=0 || ci.underruns || ci.overruns) ++failures;
    if(!correction) uncorrected=attenuation;
    else if(!ci.locked || std::abs(ci.drift_ppm-ppm)>=4 || attenuation<50 || attenuation-uncorrected<20) {
      std::cerr<<"FAIL clock/AEC rate="<<rate<<" ppm="<<ppm<<" attenuation="<<attenuation
        <<" improvement="<<attenuation-uncorrected<<" ppm_error="<<std::abs(ci.drift_ppm-ppm)<<'\n';
      ++failures;
    }
  }
  return failures ? 1 : 0;
}
