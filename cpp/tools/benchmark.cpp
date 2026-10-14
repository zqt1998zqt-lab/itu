#include "itu/audio_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
  try {
    itu::Config c; unsigned count=2000;
    for(int i=1;i<argc;++i) {
      std::string a=argv[i];
      if ((a=="--frames" || a=="--channels") && i+1<argc) {
        unsigned v=std::stoul(argv[++i]); if(a=="--frames") count=v; else c.microphones=v;
      } else throw std::runtime_error("usage: itu_audio_bench [--frames 2000] [--channels 1..8]");
    }
    if(count<10 || count>100000) throw std::runtime_error("frames must be 10..100000");
    c.beamforming=c.microphones>1;
    itu::AudioEngine e(c); itu::FrameInfo info;
    std::vector<float> mic(c.frame_size*c.microphones),ref(c.frame_size),out(c.frame_size);
    std::vector<double> times(count); std::uint32_t seed=12831; unsigned misses=0;
    double checksum=0; const double frame_ms=1000.*c.frame_size/c.sample_rate;
    for(unsigned k=0;k<count+200;++k) {
      for(unsigned i=0;i<c.frame_size;++i) {
        seed=seed*1664525u+1013904223u; ref[i]=(int(seed>>16)-32768)/32768.f*0.2f;
        for(unsigned ch=0;ch<c.microphones;++ch) mic[i*c.microphones+ch]=ref[i]*(0.6f+ch*0.02f);
      }
      auto start=std::chrono::steady_clock::now();
      auto status=e.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info);
      auto stop=std::chrono::steady_clock::now();
      if(status!=itu::Status::ok) return 2;
      if(k>=200) { auto ms=std::chrono::duration<double,std::milli>(stop-start).count(); times[k-200]=ms; misses+=ms>frame_ms; checksum+=out[k%c.frame_size]; }
    }
    double sum=0; for(double t:times) sum+=t; std::sort(times.begin(),times.end());
    auto percentile=[&](double q){return times[static_cast<std::size_t>(std::ceil(q*count))-1];};
    std::cout<<"{\"schema_version\":2,\"webrtc_revision\":\""<<itu::kWebRtcRevision<<"\",\"input_kind\":\"deterministic_synthetic_far_only\",\"timing_scope\":\"C++ process only, unpaced\","
      <<"\"sample_rate\":"<<c.sample_rate<<",\"frame_size\":"<<c.frame_size<<",\"microphones\":"<<c.microphones
      <<",\"frames\":"<<count<<",\"p50_ms\":"<<percentile(.5)<<",\"p95_ms\":"<<percentile(.95)<<",\"p99_ms\":"<<percentile(.99)
      <<",\"max_ms\":"<<times.back()<<",\"mean_ms\":"<<sum/count<<",\"rtf\":"<<sum/(count*frame_ms)<<",\"deadline_misses\":"<<misses
      <<",\"adaptation_frames\":null,\"double_talk_frames\":null,\"dsp_delay_samples\":"<<info.latency_samples
      <<",\"wrapper_memory_bytes\":"<<e.working_memory_bytes()<<",\"apm_memory_bytes\":null,\"checksum\":"<<checksum<<"}\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
