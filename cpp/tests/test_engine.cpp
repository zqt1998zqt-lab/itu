#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include "wav.hpp"
#include "denormal_scope.hpp"
#include "reference_monitor.hpp"
#include "speech_gain.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
namespace { bool watch=false; std::size_t allocations=0; }
void* operator new(std::size_t n) {if(watch) ++allocations; if(void* p=std::malloc(n?n:1)) return p; throw std::bad_alloc();}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
namespace {
void require(bool b,const char* msg) {if(!b) throw std::runtime_error(msg);}
itu::Config bypass() {itu::Config c; c.aec=c.beamforming=c.noise_suppression=c.automatic_gain=c.limiter=false; return c;}
float random_sample(std::uint32_t& seed) {seed=seed*1664525u+1013904223u; return (int(seed>>16)-32768)/32768.f;}
void run(itu::AudioEngine& e,const std::vector<float>& mic,const std::vector<float>& ref,std::vector<float>& out,itu::FrameInfo& f,float* tap=nullptr,float* linear=nullptr) {
  require(e.process(mic.data(),mic.size(),ref.empty()?nullptr:ref.data(),ref.size(),out.data(),out.size(),f,tap,nullptr,linear)==itu::Status::ok,"process status");
}
void validation() {
  itu::AudioEngine e(bypass()); itu::FrameInfo f; std::vector<float> x(160,.3f),o(160);
  require(e.process(nullptr,160,nullptr,0,o.data(),160,f)==itu::Status::invalid_pointer,"null mic");
  require(e.process(x.data(),159,nullptr,0,o.data(),160,f)==itu::Status::invalid_length,"short input");
  for(float bad:{std::numeric_limits<float>::quiet_NaN(),1.01f,-1.01f}) {
    x[159]=bad; require(e.process(x.data(),160,nullptr,0,o.data(),160,f)==itu::Status::non_finite,"invalid PCM accepted");
  }
  x[159]=.3f; run(e,x,{},o,f); require(o==x && f.frame_index==0 && !f.statistics_valid && !f.linear_aec_available,"bypass/state");
  auto c=bypass(); c.sample_rate=44100; bool rejected=false;
  try {itu::AudioEngine unsupported(c);} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"unsupported rate accepted");
}
void latency() {
  for(unsigned mode=0;mode<4;++mode) {
    auto c=bypass(); c.aec=mode&1; c.noise_suppression=mode&2;
    itu::AudioEngine e(c); itu::FrameInfo f; std::vector<float> x(160),r(160),o(160),l(160),source,after,linear;
    std::uint32_t seed=113;
    for(unsigned k=0;k<200;++k) {
      for(float& v:x) v=random_sample(seed)*.3f;
      source.insert(source.end(),x.begin(),x.end()); run(e,x,c.aec?r:std::vector<float>{},o,f,nullptr,l.data());
      after.insert(after.end(),o.begin(),o.end());
      linear.insert(linear.end(),l.begin(),l.end());
    }
    double best=-1; unsigned lag=0;
    for(unsigned d=0;d<600;++d) {
      double cross=0,a=0,b=0;
      for(unsigned i=1600;i+600<source.size();++i) {cross+=double(source[i])*after[i+d]; a+=double(source[i])*source[i]; b+=double(after[i+d])*after[i+d];}
      double corr=cross/std::sqrt(a*b+1e-30); if(corr>best) {best=corr; lag=d;}
    }
    std::cout<<"latency mode="<<mode<<" measured="<<lag<<" reported="<<f.latency_samples<<" correlation="<<best<<'\n';
    require(lag==f.latency_samples && best>.7,"measured latency mismatch");
    if(c.aec) {
      best=-1; lag=0;
      for(unsigned d=0;d<600;++d) {
        double cross=0,a=0,b=0;
        for(unsigned i=1600;i+600<source.size();++i) {cross+=double(source[i])*linear[i+d]; a+=double(source[i])*source[i]; b+=double(linear[i+d])*linear[i+d];}
        double corr=cross/std::sqrt(a*b+1e-30); if(corr>best) {best=corr; lag=d;}
      }
      std::cout<<"linear AEC mode="<<mode<<" measured="<<lag<<" correlation="<<best<<'\n';
      require(lag==f.linear_aec_delay_samples && best>.7,"linear AEC measured latency");
    }
  }
}
void aec_effect_doubletalk() {
  auto c=bypass(); c.aec=true; c.stream_delay_ms=40;
  itu::AudioEngine e(c),off(bypass()); itu::FrameInfo f,fn;
  std::vector<float> mic(160),ref(160),out(160),negative(160),linear(160),history(240000),near(240000),result;
  std::uint32_t seed=193; double input=0,residual=0,linpower=0;
  for(unsigned k=0;k<1500;++k) {
    for(unsigned i=0;i<160;++i) {
      unsigned n=k*160+i; ref[i]=history[n]=random_sample(seed)*.15f;
      float envelope=.4f+.6f*std::pow(std::sin(n*.00073f),2);
      float phase=n*.052f+.4f*std::sin(n*.0003f);
      near[n]=k>=1200 ? envelope*(.18f*std::sin(phase)+.09f*std::sin(2*phase)+.05f*std::sin(3*phase)+.03f*std::sin(5*phase)):0;
      mic[i]=(n>=647 ? history[n-647]*.6f:0)+(n>=700 ? history[n-700]*.2f:0)+near[n];
    }
    run(e,mic,ref,out,f,nullptr,linear.data()); run(off,mic,{},negative,fn);
    require(negative==mic,"AEC off negative control");
    require(f.linear_aec_available,"linear AEC unavailable");
    result.insert(result.end(),out.begin(),out.end());
    if(k>=900 && k<1100) for(unsigned i=0;i<160;++i) {input+=double(mic[i])*mic[i]; residual+=double(out[i])*out[i]; linpower+=double(linear[i])*linear[i];}
  }
  double attenuation=10*std::log10(input/std::max(residual,1e-30));
  double linear_reduction=10*std::log10(input/std::max(linpower,1e-30));
  double correlation=-1,gain=0; unsigned near_lag=0;
  for(unsigned d=0;d<400;++d) {
    double dot=0,a=0,b=0;
    for(unsigned n=1300*160;n+400<result.size();++n) {dot+=double(near[n])*result[n+d]; a+=double(near[n])*near[n]; b+=double(result[n+d])*result[n+d];}
    double corr=dot/std::sqrt(a*b+1e-30);
    if(corr>correlation) {correlation=corr; gain=dot/a; near_lag=d;}
  }
  std::cout<<"AEC synthetic attenuation="<<attenuation<<" linear="<<linear_reduction<<" doubletalk near corr="<<correlation<<" gain="<<gain<<" lag="<<near_lag<<'\n';
  require(attenuation>15 && linear_reduction>10,"AEC3 did not cancel delayed echo");
  require(correlation>.7 && gain>.4 && gain<1.5,"AEC3 damaged doubletalk near end");
  require(f.statistics_valid & itu::erle_valid,"AEC statistics missing");
}
void beamforming() {
  auto c=bypass(); c.microphones=2; c.beamforming=true; c.beam_delays_samples[0]=163;
  itu::AudioEngine e(c); itu::FrameInfo f; std::vector<float> x(320),o(160); x[0]=1;
  run(e,x,{},o,f); require(std::all_of(o.begin(),o.end(),[](float v){return v==0;}),"BF early output");
  std::fill(x.begin(),x.end(),0); x[7]=1; run(e,x,{},o,f);
  require(o[3]==1 && f.latency_samples==163,"BF alignment");
  c=bypass(); c.beamforming=true; c.beam_delays_samples[0]=.5f; itu::AudioEngine half(c);
  x.assign(160,0); x[0]=1; run(half,x,{},o,f); require(o[0]==.5f && o[1]==.5f,"fractional BF");
}
void ns_effect() {
  auto c=bypass(); c.noise_suppression=true;
  itu::AudioEngine e(c),off(bypass()); itu::FrameInfo f,fn; std::vector<float> x(160),o(160),negative(160);
  std::uint32_t seed=21; double a=0,b=0;
  for(unsigned k=0;k<500;++k) {
    for(float& v:x) v=random_sample(seed)*.05f; run(e,x,{},o,f); run(off,x,{},negative,fn);
    require(x==negative,"NS off negative control");
    if(k>=300) for(unsigned i=0;i<160;++i) {a+=double(x[i])*x[i]; b+=double(o[i])*o[i];}
  }
  double reduction=10*std::log10(a/b); std::cout<<"WebRTC NS stationary noise reduction="<<reduction<<" dB\n";
  require(reduction>6 && reduction<40,"WebRTC NS effect absent or invalid");
}
void agc_limiter() {
  auto c=bypass(); c.automatic_gain=true;
  c.speech_gain_policy=false; c.fixed_gain_db=8;
  itu::AudioEngine e(c),off(bypass()); itu::FrameInfo f,fn; std::vector<float> x(160),o(160),negative(160);
  double a=0,b=0;
  for(unsigned k=0;k<20;++k) {
    for(unsigned i=0;i<160;++i) x[i]=.01f*std::sin((k*160+i)*.2f);
    run(e,x,{},o,f); run(off,x,{},negative,fn); require(x==negative,"AGC off negative control");
    for(unsigned i=0;i<160;++i) {a+=double(x[i])*x[i]; b+=double(o[i])*o[i];}
  }
  double gain=10*std::log10(b/a); std::cout<<"AGC2 initial digital gain="<<gain<<" dB\n"; require(gain>6,"AGC2 absent");
  c=bypass(); c.limiter=true; itu::AudioEngine limit(c); std::fill(x.begin(),x.end(),1.f);
  run(limit,x,{},o,f); require(f.limited_samples==160 && o[0]==c.output_ceiling,"limiter absent");
}
void taps_and_multichannel() {
  auto c=bypass(); c.aec=true; c.microphones=4;
  auto full=c; full.noise_suppression=full.automatic_gain=full.beamforming=true;
  itu::AudioEngine before(c),after(full); itu::FrameInfo f,g;
  std::vector<float> mic(640),ref(160),o(160),p(160),t1(640),t2(640),l1(640),l2(640);
  std::uint32_t seed=122;
  for(unsigned k=0;k<400;++k) {
    for(unsigned i=0;i<160;++i) {
      ref[i]=random_sample(seed)*.1f;
      for(unsigned ch=0;ch<4;++ch) mic[4*i+ch]=ref[i]*.5f+random_sample(seed)*(.01f+ch*.005f);
    }
    run(before,mic,ref,o,f,t1.data(),l1.data()); run(after,mic,ref,p,g,t2.data(),l2.data());
    require(t1==t2 && l1==l2,"downstream processing changed AEC measurement taps");
    require(f.linear_aec_available && g.linear_aec_available,"multichannel linear output absent");
    require(std::all_of(l1.begin(),l1.end(),[](float v){return std::isfinite(v) && std::abs(v)<=1;}),"linear PCM scale");
  }
}
void reset_allocations() {
  itu::Config c; c.microphones=8; c.beamforming=true;
  itu::AudioEngine e(c); itu::FrameInfo f;
  std::vector<float> x(1280,.04f),ref(160,.02f),o(160),first;
  for(unsigned k=0;k<100;++k) {run(e,x,ref,o,f); first.insert(first.end(),o.begin(),o.end());}
  e.reset(); bool same=true;
  for(unsigned k=0;k<100;++k) {run(e,x,ref,o,f); for(unsigned i=0;i<160;++i) same &= first[k*160+i]==o[i];}
  require(same,"reset not repeatable");
  for(unsigned k=0;k<200;++k) run(e,x,ref,o,f);
  allocations=0; watch=true; bool valid=true;
  for(unsigned k=0;k<300;++k) valid &= e.process(x.data(),x.size(),ref.data(),ref.size(),o.data(),o.size(),f)==itu::Status::ok;
  watch=false; require(valid,"allocation probe process failed");
  std::cout<<"steady-state C++ new/new[] allocations (8 mic, 300 frames)="<<allocations<<"; malloc/aligned alloc not intercepted\n";
  auto count=allocations; watch=true; void* p=::operator new(19); watch=false; ::operator delete(p);
  require(allocations==count+1,"allocation detector negative control");
}
void denormal_mode_restoration() {
#if defined(__SSE2__)
  const unsigned original=_mm_getcsr();
  _mm_setcsr(original & ~0x8040u);
  const unsigned caller=_mm_getcsr();
  {
    itu::DenormalScope scope;
    require((_mm_getcsr() & 0x8040u)==0x8040u,"FTZ/DAZ not active inside scope");
    {itu::DenormalScope nested;}
    require((_mm_getcsr() & 0x8040u)==0x8040u,"nested scope changed FTZ/DAZ");
  }
  const bool restored=_mm_getcsr()==caller;
  _mm_setcsr(original);
  require(restored,"denormal scope leaked caller floating-point mode");
#else
  require(webrtc::DenormalDisabler::IsSupported(),"denormal guard unavailable on target architecture");
#endif
}
void reference_monitor_transitions() {
  itu::ReferenceMonitor monitor;
  std::vector<float> capture(160),render(160),history(128);
  unsigned head=0;
  std::uint32_t seed=32,wrong_seed=87;
  auto feed=[&](unsigned count,bool mismatch,bool silent) {
    for(unsigned k=0;k<count;++k) {
      for(unsigned i=0;i<160;++i) {
        const float source=random_sample(seed)*.1f;
        capture[i]=silent ? 0.f : history[head]*.5f;
        history[head]=source; head=(head+1)%history.size();
        render[i]=silent ? 0.f : mismatch ? random_sample(wrong_seed)*.1f : source;
      }
      monitor.push(capture.data(),render.data(),1);
    }
  };
  feed(500,false,true);
  require(!monitor.reliable() && !monitor.unreliable() && !monitor.delay_available(),"silent playback incorrectly marked mismatched");
  feed(500,false,false);
  require(monitor.reliable() && !monitor.unreliable() && monitor.delay_samples()==128,"valid playback not acquired");
  feed(500,true,false);
  require(!monitor.reliable() && monitor.unreliable() && !monitor.delay_available(),"old reliable state survived wrong playback reference");
  feed(500,false,false);
  require(monitor.reliable() && !monitor.unreliable(),"corrected playback reference not reacquired");
  feed(500,false,true);
  require(!monitor.reliable() && !monitor.unreliable(),"stopped playback retained reliability state");
  require(monitor.delay_available() && monitor.delay_samples()==128,"silence discarded learned acoustic delay");
  itu::ReferenceMonitor reset;
  require(!reset.delay_available(),"new reference epoch retained old delay");
}
void speech_gain_state() {
  itu::SpeechGain gain;
  for(unsigned i=0;i<30;++i) require(gain.observe(.001f,0,false,18,6)==0,"noise gained during calibration");
  for(unsigned i=0;i<29;++i) require(gain.observe(.005f,.95f,false,18,6)==0,"speech calibration prematurely applied");
  require(gain.observe(.005f,.95f,false,18,6)==8,"weak speech gain not calibrated");
  for(unsigned pause=0;pause<40;++pause) {
    for(unsigned i=0;i<20;++i) require(gain.observe(.001f,0,false,18,6)==8,"short pause released utterance gain");
    require(gain.observe(.005f,.95f,false,18,6)==8,"utterance gain recalibrated");
  }
  for(unsigned i=0;i<99;++i) require(gain.observe(.001f,0,false,18,6)==8,"nonconsecutive quiet counter");
  require(std::abs(gain.observe(.001f,0,false,18,6)-7.94f)<1e-5,"long quiet did not release gain");
  for(unsigned i=0;i<200;++i) gain.observe(.001f,0,false,18,6);
  require(gain.observe(.001f,0,false,18,6)==0,"gain persisted in long noise interval");
  for(unsigned i=0;i<100;++i) require(gain.observe(.0011f,.95f,false,18,6)==0,"VAD false positive calibrated noise");
  for(unsigned i=0;i<600;++i) gain.observe(.001f,0,false,18,6);
  require(gain.observe(.005f,.95f,false,18,6)==0,"expired utterance cache restored stale gain");
  itu::SpeechGain no_floor;
  for(unsigned i=0;i<100;++i) require(no_floor.observe(.005f,.95f,false,18,6)==0,"unobserved noise floor accepted as zero");
  itu::SpeechGain bounded;
  for(unsigned i=0;i<30;++i) bounded.observe(.001f,0,false,3,6);
  for(unsigned i=0;i<30;++i) bounded.observe(.005f,.95f,false,3,6);
  require(bounded.observe(.005f,.95f,false,3,6)==3,"gain maximum ignored");
  require(bounded.observe(.005f,.95f,true,3,6)==0,"far-only reference gained");
  require(bounded.observe(.005f,.95f,false,3,6)==3,"gain not restored after playback");
}
void c_abi() {
  itu_audio_config c; itu_audio_default_config(&c); char error[128];
  c.microphones=0; require(!itu_audio_create(&c,error,sizeof(error)) && error[0],"invalid config");
  itu_audio_default_config(&c); c.abi_version=1; require(!itu_audio_create(&c,error,sizeof(error)),"legacy ABI accepted");
  itu_audio_default_config(&c); c.aec=2; require(!itu_audio_create(&c,error,sizeof(error)),"invalid boolean");
  itu_audio_default_config(&c); c.aec=c.noise_suppression=c.automatic_gain=0;
  auto* e=itu_audio_create(&c,error,sizeof(error)); require(e,"C create");
  std::vector<float> x(160,.1f),o(160); itu_audio_frame_info f;
  require(itu_audio_process(e,x.data(),160,nullptr,0,o.data(),160,&f,nullptr,nullptr,nullptr)==0 && o==x,"C process");
  require(itu_audio_reset(e)==0,"C reset"); itu_audio_destroy(e); itu_audio_destroy(nullptr);
}
void vehicle_profile() {
  const auto preset=itu::vehicle_stream_config();
  require(preset.continuous_vehicle_policy && !preset.adaptive_noise_policy &&
          preset.stable_echo_coherence && preset.speech_preserving_low_cut,"vehicle policies not selected");
  itu_audio_config config; itu_audio_default_config(&config); char error[128];
  require(!itu_audio_create_with_profile(&config,99,error,sizeof(error)) && error[0],"unknown profile accepted");
  require(itu_audio_processing_profile(nullptr)==-1,"null profile query");
  for(unsigned microphones:{1u,2u,4u}) for(bool aec:{false,true}) {
    auto expected=preset; expected.aec=aec; expected.microphones=microphones; expected.beamforming=microphones>1;
    config.aec=aec; config.microphones=microphones; config.beamforming=expected.beamforming;
    for(unsigned ch=0;ch<microphones;++ch)
      expected.beam_delays_samples[ch]=config.beam_delays_samples[ch]=float(ch)*.5f;
    auto* engine=itu_audio_create_with_profile(&config,ITU_AUDIO_PROFILE_VEHICLE_STREAM,error,sizeof(error));
    require(engine && itu_audio_processing_profile(engine)==ITU_AUDIO_PROFILE_VEHICLE_STREAM,"vehicle C profile missing");
    itu::AudioEngine paired(expected); itu::FrameInfo info; itu_audio_frame_info ci;
    std::vector<float> mic(160*microphones),ref(160),actual(160),control(160);
    std::uint32_t seed=481;
    for(unsigned k=0;k<400;++k) {
      for(unsigned i=0;i<160;++i) {
        ref[i]=k>=100 && k<300 ? random_sample(seed)*.08f : 0;
        for(unsigned ch=0;ch<microphones;++ch)
          mic[i*microphones+ch]=.08f*std::sin((k*160+i+ch)*.12f)+ref[i]*.5f;
      }
      require(itu_audio_process(engine,mic.data(),mic.size(),aec?ref.data():nullptr,aec?160:0,
              actual.data(),160,&ci,nullptr,nullptr,nullptr)==0,"vehicle C process failed");
      run(paired,mic,aec?ref:std::vector<float>{},control,info);
      require(ci.frame_index==k && actual==control && ci.latency_samples==info.latency_samples,
              "vehicle C/C++ profile diverged");
      require(std::all_of(actual.begin(),actual.end(),[](float v){return std::isfinite(v);}),"vehicle output nonfinite");
    }
    require(itu_audio_reset(engine)==0 && itu_audio_processing_profile(engine)==ITU_AUDIO_PROFILE_VEHICLE_STREAM,
            "reset lost vehicle profile");
    itu_audio_destroy(engine);
  }
}
void radio_protection() {
  for(bool guard:{false,true}) for(bool ns:{false,true}) for(unsigned mics:{1u,2u}) {
    auto c=itu::referenced_radio_config();
    c.reference_guard=guard; c.noise_suppression=ns; c.automatic_gain=false;
    c.microphones=mics; c.beamforming=mics>1;
    itu::AudioEngine e(c); itu::FrameInfo f;
    std::vector<float> x(160*mics),ref(160),out(160),source,after;
    for(unsigned k=0;k<150;++k) {
      for(unsigned i=0;i<160;++i) for(unsigned ch=0;ch<mics;++ch)
        x[i*mics+ch]=.1f*std::sin((k*160+i)*2*3.14159265358979323846*85/16000);
      for(unsigned i=0;i<160;++i) source.push_back(x[i*mics]);
      run(e,x,ref,out,f); after.insert(after.end(),out.begin(),out.end());
      require(std::all_of(out.begin(),out.end(),[](float v){return std::isfinite(v);}),"radio combination nonfinite output");
    }
    const unsigned delay=f.latency_samples;
    require(delay==128+(ns?96:0),"radio latency budget");
    if(guard && !ns) {
      double output_energy=0,energy=0;
      for(unsigned i=1600;i+delay<source.size();++i) {
        output_energy+=double(after[i+delay])*after[i+delay]; energy+=double(source[i])*source[i];
      }
      require(output_energy>energy*.8,"radio near-only loses low frequencies without NS");
    }
    e.reset(); std::fill(x.begin(),x.end(),0); run(e,x,ref,out,f);
    require(f.frame_index==0 && f.fullband_ns_mix==0 && f.fullband_noise_power==0,"radio reset leaks protection state");
  }
}
void wav() {
  std::vector<float> x={-1.f,-.3f,0.f,.3f,.999f}; const char* path="itu-core-roundtrip-test.wav";
  itu::write_wav(path,16000,1,x); auto w=itu::read_wav(path); std::remove(path);
  require(w.samples.size()==x.size() && w.sample_rate==16000,"WAV shape");
  for(unsigned i=0;i<x.size();++i) require(std::abs(w.samples[i]-x[i])<1.f/32768,"WAV roundtrip");
}
}
int main() {
  unsigned passed=0;
  try {
    for(auto test:{validation,latency,aec_effect_doubletalk,beamforming,ns_effect,agc_limiter,taps_and_multichannel,reset_allocations,denormal_mode_restoration,reference_monitor_transitions,speech_gain_state,c_abi,vehicle_profile,radio_protection,wav}) {test(); ++passed;}
    std::cout<<"PASS "<<passed<<" groups; WebRTC "<<itu::kWebRtcRevision<<'\n';
  } catch(const std::exception& e) {watch=false; std::cerr<<"FAIL after "<<passed<<" groups: "<<e.what()<<'\n'; return 1;}
}
