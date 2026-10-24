#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include "linear_phase_eq.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void require(bool b,const char* why) {if(!b) throw std::runtime_error(why);}
void response() {
  for(unsigned rate:{16000,24000,48000}) for(unsigned microphones:{1,2,4}) {
    itu::Config base; base.sample_rate=rate; base.frame_size=rate/100; base.microphones=microphones;
    base.aec=base.noise_suppression=base.automatic_gain=base.limiter=false;
    base.beamforming=microphones>1;
    auto tuned=base; tuned.output_eq.bands=1;
    tuned.output_eq.band[0]={itu::EqKind::peak,1000,-6,1};
    auto flat=base; flat.output_eq.bands=1; flat.output_eq.band[0].gain_db=0;
    itu::AudioEngine a(base),b(tuned),identity(flat);
    itu::FrameInfo f,g,j;
    std::vector<float> x(base.frame_size*microphones),u(base.frame_size),v(base.frame_size),z(base.frame_size),tap(x.size()),control(x.size());
    double before=0,after=0;
    for(unsigned k=0;k<150;++k) {
      for(unsigned i=0;i<base.frame_size;++i) for(unsigned ch=0;ch<microphones;++ch)
        x[i*microphones+ch]=.1f*std::sin(2*3.141592653589793*1000*(k*base.frame_size+i)/rate);
      require(a.process(x.data(),x.size(),nullptr,0,u.data(),u.size(),f,control.data())==itu::Status::ok,"EQ baseline process");
      require(b.process(x.data(),x.size(),nullptr,0,v.data(),v.size(),g,tap.data())==itu::Status::ok,"EQ process");
      require(identity.process(x.data(),x.size(),nullptr,0,z.data(),z.size(),j)==itu::Status::ok,"flat EQ process");
      require(z==u && tap==control,"EQ identity or AEC tap changed");
      require(f.latency_samples==g.latency_samples,"EQ introduced buffered delay");
      if(k>50) for(unsigned i=0;i<base.frame_size;++i) {before+=double(u[i])*u[i]; after+=double(v[i])*v[i];}
    }
    const double db=10*std::log10(after/before);
    require(std::abs(db+6)<.03,"EQ applied at wrong rate or more than once");
    std::cout<<"EQ rate="<<rate<<" mics="<<microphones<<" center_gain_db="<<db<<'\n';
    b.reset(); itu::AudioEngine fresh(tuned);
    require(b.process(x.data(),x.size(),nullptr,0,v.data(),v.size(),g)==itu::Status::ok,"EQ reset process");
    require(fresh.process(x.data(),x.size(),nullptr,0,z.data(),z.size(),j)==itu::Status::ok && z==v,"EQ reset state leakage");
  }
}
void validation() {
  for(unsigned rate:{16000,24000,48000}) for(unsigned bad=0;bad<5;++bad) {
    itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.output_eq.bands=1;
    if(bad==0) c.output_eq.band[0].frequency_hz=rate*.49f;
    if(bad==1) c.output_eq.band[0].gain_db=NAN;
    if(bad==2) c.output_eq.band[0].gain_db=3;
    if(bad==3) c.output_eq.band[0].q=0;
    if(bad==4) c.output_eq.bands=9;
    bool rejected=false; try {itu::AudioEngine e(c);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid EQ accepted");
  }
  itu_audio_config c; itu_audio_default_config(&c);
  c.aec=c.noise_suppression=c.automatic_gain=0;
  itu_audio_eq_config eq{}; eq.struct_size=sizeof(eq); eq.bands=1;
  eq.band[0]={ITU_AUDIO_EQ_PEAK,1000,-6,1}; char error[256];
  auto e=itu_audio_create_with_tuning(&c,ITU_AUDIO_PROFILE_VEHICLE_STREAM,ITU_AUDIO_NS_WEBRTC,&eq,error,sizeof(error));
  require(e && !error[0],"C EQ creation");
  itu_audio_destroy(e); eq.band[0].gain_db=6;
  require(!itu_audio_create_with_tuning(&c,0,0,&eq,error,sizeof(error)) && error[0],"C EQ boost lacks headroom");
}
}
void linear_phase() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu::Config c;
    c.sample_rate=rate;c.frame_size=rate/100;
    c.aec=c.noise_suppression=c.automatic_gain=c.limiter=false;
    c.output_eq.phase=itu::EqPhase::linear;
    c.output_eq.bands=1;
    c.output_eq.band[0]={itu::EqKind::peak,1000,-6,1};
    itu::LinearPhaseEq filter(c.output_eq,rate);
    const auto& taps=filter.coefficients();
    for(unsigned i=0;i<taps.size()/2;++i)
      require(std::abs(taps[i]-taps[taps.size()-1-i])<1e-7f,
              "linear-phase coefficients are asymmetric");
    require(filter.delay()==(taps.size()-1)/2 &&
        itu::LinearPhaseEq::delay_for(c.output_eq,rate)==filter.delay(),
        "linear-phase delay contract mismatch");
    auto minimum=c;minimum.output_eq.phase=itu::EqPhase::minimum;
    itu::AudioEngine iir(minimum),fir(c);
    itu::FrameInfo fi,ff;
    std::vector<float> x(rate/100),a(rate/100),b(rate/100);
    std::vector<float> iir_all,fir_all;
    std::uint32_t seed=913;
    for(unsigned frame=0;frame<300;++frame) {
      for(float& v:x) {
        seed=seed*1664525u+1013904223u;
        v=(float(seed>>8)/float(1u<<24)-.5f)*.2f;
      }
      require(iir.process(x.data(),x.size(),nullptr,0,a.data(),a.size(),fi)==
              itu::Status::ok,"minimum EQ process");
      require(fir.process(x.data(),x.size(),nullptr,0,b.data(),b.size(),ff)==
              itu::Status::ok,"linear EQ process");
      iir_all.insert(iir_all.end(),a.begin(),a.end());
      fir_all.insert(fir_all.end(),b.begin(),b.end());
    }
    require(ff.latency_samples-fi.latency_samples==filter.delay(),
            "linear EQ latency not published");
    const unsigned d=filter.delay();
    double cross=0,left=0,right=0;
    for(unsigned i=1000;i+d+2000<iir_all.size();++i) {
      cross+=double(iir_all[i])*fir_all[i+d];
      left+=double(iir_all[i])*iir_all[i];
      right+=double(fir_all[i+d])*fir_all[i+d];
    }
    const double correlation=cross/std::sqrt(left*right+1e-30);
    std::cout<<"linear EQ rate="<<rate<<" taps="<<taps.size()
             <<" delay="<<d<<" correlation="<<correlation<<'\n';
    require(correlation>.96,"linear FIR target response diverged");
  }
  for(unsigned taps:{32u,34u,2051u}) {
    itu::Config c;c.sample_rate=16000;c.frame_size=160;c.output_eq.phase=itu::EqPhase::linear;
    c.output_eq.linear_phase_taps=taps;
    bool rejected=false;
    try {itu::LinearPhaseEq filter(c.output_eq,c.sample_rate);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid linear-phase tap count accepted");
    std::cout<<"linear EQ rejected taps="<<taps<<'\n';
  }
}
int main() {
  try {response(); validation(); linear_phase(); return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
