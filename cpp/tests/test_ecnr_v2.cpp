#include "itu/audio_engine.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void trace_parity(unsigned rate,bool eq) {
  auto c=itu::vehicle_stream_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.aec=false;c.noise_suppression=false;c.automatic_gain=false;c.limiter=false;
  if(eq) {c.output_eq.preamp_db=-6;c.output_eq.bands=1;c.output_eq.band[0]={itu::EqKind::peak,1500,3,.70710678f};}
  itu::AudioEngine plain(c);itu::DiagnosticStageTaps taps;c.diagnostic_stages=&taps;itu::AudioEngine traced(c);
  std::vector<float> input(c.frame_size),a(c.frame_size),b(c.frame_size);itu::FrameInfo fa,fb;
  double changed=0;
  for(unsigned k=0;k<120;++k) {
    for(unsigned i=0;i<c.frame_size;++i) {
      const double t=double(k*c.frame_size+i)/rate;
      input[i]=float(.008*std::sin(2*3.141592653589793*1513*t)+.002*std::sin(2*3.141592653589793*331*t));
    }
    require(plain.process(input.data(),input.size(),nullptr,0,a.data(),a.size(),fa)==itu::Status::ok,"plain process");
    require(traced.process(input.data(),input.size(),nullptr,0,b.data(),b.size(),fb)==itu::Status::ok,"traced process");
    require(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,"trace changed output");
    require(fa.latency_samples==fb.latency_samples,"trace changed delay");
    require(taps.eq_sample_rate==rate && taps.eq_frame_size==rate/100,"actual I/O EQ rate");
    require(taps.core_sample_rate==(rate==24000?48000:rate),"actual core tap rate");
    require(taps.eq_delay_samples==fb.latency_samples,"EQ delay metadata");
    require(!taps.nr_selected_available,"bypassed NR marked active");
    require(std::memcmp(taps.post_eq.data(),b.data(),b.size()*sizeof(float))==0,"EQ-after is not pre-limiter output");
    for(unsigned i=0;i<c.frame_size;++i) changed+=std::abs(taps.pre_eq[i]-taps.post_eq[i]);
  }
  require(eq ? changed>.001 : changed==0,"EQ trace did not distinguish actual filter");
}
void fallback_parity() {
  auto c=itu::vehicle_stream_config();c.ecnr_coordinated_control=true;
  itu::AudioEngine v1(c);c.ecnr_v2_control=true;itu::AudioEngine v2(c);
  std::array<float,160> mic{},ref{},a{},b{};itu::FrameInfo fa,fb;
  for(unsigned k=0;k<300;++k) {
    for(unsigned i=0;i<160;++i) mic[i]=float(.001*std::sin((k*160+i)*.157));
    require(v1.process(mic.data(),160,ref.data(),160,a.data(),160,fa)==itu::Status::ok,"v1 fallback");
    require(v2.process(mic.data(),160,ref.data(),160,b.data(),160,fb)==itu::Status::ok,"v2 fallback");
    require(std::memcmp(a.data(),b.data(),sizeof(a))==0,"v2 changed unexcited reference fallback");
  }
}
}
int main() {
  try {
    itu::Config defaults;require(!defaults.ecnr_v2_control && !defaults.diagnostic_stages,"default enabled experiment");
    for(unsigned rate:{16000u,24000u,48000u}) for(bool eq:{false,true}) trace_parity(rate,eq);
    fallback_parity();
    std::cout<<"ECNR v2: trace output parity, actual-rate EQ, fallback parity PASS\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
