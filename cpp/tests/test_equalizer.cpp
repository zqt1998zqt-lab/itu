#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
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
int main() {
  try {response(); validation(); return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
