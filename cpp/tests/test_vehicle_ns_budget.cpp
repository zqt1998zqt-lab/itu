#include "itu/vehicle_ns.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
namespace { bool watching=false; std::size_t allocations=0; }
void* operator new(std::size_t n) {if(watching) ++allocations;if(void* p=std::malloc(n?n:1)) return p;throw std::bad_alloc();}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
float noise(unsigned& state,float amount) {
  state=1664525u*state+1013904223u;
  return amount*(float(state>>8)/16777216.f-.5f);
}
float speech(unsigned n,unsigned rate,float amount) {
  const double t=double(n)/rate;
  return amount*(std::sin(2*3.141592653589793*173*t)+.4*std::sin(2*3.141592653589793*419*t)
    +.3*std::sin(2*3.141592653589793*811*t)+.25*std::sin(2*3.141592653589793*1631*t))
    *(.6+.4*std::sin(2*3.141592653589793*3.7*t));
}
float quantize(float x) {return std::lround(std::clamp(x*32768.f,-32768.f,32767.f))/32768.f;}
itu::VehicleNsConfig config(float cap) {
  itu::VehicleNsConfig c;c.backend=itu::VehicleNsBackend::super_gaussian;c.strength=1;
  c.dry_max=cap;c.unified_raw_budget=true;c.rumble_hz=0;c.startup_ms=100;
  c.slew_ms=50;c.noise_gain=1;c.policy="parallel_start";return c;
}
void caps_and_timebase() {
  for(unsigned rate:{16000u,24000u,48000u}) for(float cap:{0.f,.25f,1.f}) {
    auto c=config(cap);if(cap==0) c.weak_speech_recovery=.5f;
    itu::VehicleNsProcessor mixed(rate,c);
    c.unified_raw_budget=false;c.dry_max=0;c.weak_speech_recovery=0;
    itu::VehicleNsProcessor wet(rate,c);
    const unsigned hop=rate/100,delay=mixed.delay_samples(),frames=420;
    require(delay==wet.delay_samples() && delay==hop,"unified raw mix changed declared latency");
    std::vector<float> source(frames*hop),a(hop),b(hop);
    unsigned random=71+rate;std::size_t checked=0,smooth_checked=0;float previous_weight=0;
    bool previous_valid=false;float peak_mix=0;double peak_step=0;
    for(unsigned frame=0;frame<frames;++frame) {
      for(unsigned i=0;i<hop;++i) {
        const unsigned n=frame*hop+i;
        const bool voice=frame<50 || (frame>=100 && frame<250) || frame>=350;
        source[n]=quantize(noise(random,voice?.0006f:.03f)+
          (voice?speech(n,rate,.065f):0.f));
        a[i]=b[i]=source[n];
      }
      const auto before=allocations;watching=true;
      const bool ok=mixed.process(a.data(),hop) && wet.process(b.data(),hop);watching=false;
      require(ok && before==allocations,"unified callback failed or allocated");
      const auto s=mixed.statistics();
      require(s.unified_raw_budget && s.raw_mix_budget==cap,"raw budget telemetry does not describe configuration");
      require(s.raw_mix_target>=0 && s.raw_mix_target<=cap+1e-7f,"merged raw target exceeded total budget");
      require(s.raw_mix_applied_min>=0 && s.raw_mix_applied_max<=cap+1e-7f &&
        s.raw_mix_applied_min<=s.raw_mix_applied_max,"actual sample raw weight exceeded total budget");
      peak_mix=std::max(peak_mix,s.raw_mix_applied_max);
      for(unsigned i=0;i<hop;++i) {
        const unsigned n=frame*hop+i;const float raw=n>=delay?source[n-delay]:0;
        const float endpoint=b[i]+cap*(raw-b[i]);
        require(a[i]>=std::min(b[i],endpoint)-2e-7f && a[i]<=std::max(b[i],endpoint)+2e-7f,
          "PCM escaped the same-delay wet/raw budget segment");
        const float denominator=raw-b[i];
        if(std::abs(denominator)>1e-4f) {
          const float weight=(a[i]-b[i])/denominator;
          require(weight>=s.raw_mix_applied_min-2e-3f && weight<=s.raw_mix_applied_max+2e-3f,
            "reported applied extrema do not bound weights inferred from PCM");
          if(previous_valid) {
            peak_step=std::max(peak_step,double(std::abs(weight-previous_weight)));
            require(std::abs(weight-previous_weight)<.02f,"sample raw mix jumped across frame/startup transition");
            ++smooth_checked;
          }
          previous_weight=weight;previous_valid=true;++checked;
        } else previous_valid=false;
      }
    }
    require(checked>hop*20 && smooth_checked>hop*10,"raw-mix PCM oracle was vacuous");
    if(cap>0) require(peak_mix>.8f*cap,"startup/high-SNR raw requests never exercised the budget");
    else require(peak_mix==0,"cap zero enabled a hidden quiet/startup bypass");
    std::cout<<"caps rate="<<rate<<" cap="<<cap<<" peak="<<peak_mix
      <<" inferred_samples="<<checked<<" max_sample_step="<<peak_step<<" allocations=0\n";
  }
}
void pure_raw_impulse() {
  for(auto backend:{itu::VehicleNsBackend::super_gaussian,itu::VehicleNsBackend::bypass,
      itu::VehicleNsBackend::filterbank,itu::VehicleNsBackend::webrtc,itu::VehicleNsBackend::speex})
    for(unsigned rate:{16000u,24000u,48000u}) {
    // Exact PCM oracle applies where no SRC changes the impulse response.
    // Native statistical paths cover 24k; bridged 24k is an integration check.
    if(rate==24000 && backend!=itu::VehicleNsBackend::super_gaussian) continue;
    auto c=config(1);c.backend=backend;c.slew_ms=0;c.startup_ms=500;c.policy="fixedblend";
    if(backend==itu::VehicleNsBackend::speex) c.strength=-12;
    itu::VehicleNsProcessor p(rate,c);const unsigned hop=rate/100;
    std::vector<float> data(hop),source(40*hop),output(source.size());
    const unsigned impulse=10*hop+hop/3;source[impulse]=.125f;
    for(unsigned frame=0;frame<40;++frame) {
      std::copy_n(source.data()+frame*hop,hop,data.data());
      require(p.process(data.data(),hop),"raw impulse process failed");
      std::copy(data.begin(),data.end(),output.begin()+frame*hop);
    }
    for(unsigned i=3*hop;i<output.size();++i)
      require(output[i]==(i==impulse+p.delay_samples()?.125f:0.f),
        "raw donor phase/gain differs from the declared common delay");
  }
}
void quiet_is_not_a_bypass() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    auto c=config(.25f);c.policy="aligned";c.startup_ms=0;c.snr_threshold_db=30;
    c.quiet_floor_db=-50;c.slew_ms=30;
    itu::VehicleNsProcessor p(rate,c);c.unified_raw_budget=false;
    itu::VehicleNsProcessor legacy(rate,c);std::vector<float> a(rate/100),b(a.size());
    unsigned random=191;bool independent_quiet=false;float qmax=0,rawmax=0,legacymax=0;
    for(unsigned frame=0;frame<600;++frame) {
      const float amplitude=frame<200?0.f:frame<300?.008f:frame<400?.016f:frame<500?.032f:.064f;
      for(unsigned i=0;i<a.size();++i)
        a[i]=b[i]=noise(random,.004f)+speech(frame*a.size()+i,rate,amplitude);
      require(p.process(a.data(),a.size()) && legacy.process(b.data(),b.size()),"quiet controller process failed");
      const auto s=p.statistics();qmax=std::max(qmax,s.quiet_mix_target);
      rawmax=std::max(rawmax,s.raw_mix_applied_max);
      legacymax=std::max(legacymax,legacy.statistics().raw_mix_applied_max);
      independent_quiet|=s.quiet_mix_target>.15f && s.dry_mix_target<.01f;
      require(s.raw_mix_applied_max<=.250001f,"quiet request bypassed the total raw budget");
    }
    require(qmax>.5f && independent_quiet && rawmax>.05f && legacymax>.5f,"quiet path/independent request and legacy negative control did not activate");
    std::cout<<"quiet rate="<<rate<<" quiet_peak="<<qmax<<" actual_raw_peak="<<rawmax<<" legacy_raw_peak="<<legacymax<<"\n";
  }
}
void incompatible_policies() {
  for(auto backend:{itu::VehicleNsBackend::webrtc,itu::VehicleNsBackend::speex})
    for(const char* policy:{"parallel","parallel_start","dynamic","steady_dynamic"}) {
      auto c=config(.25f);c.backend=backend;c.policy=policy;
      if(backend==itu::VehicleNsBackend::speex) c.strength=-12;
      bool rejected=false;
      try {itu::VehicleNsProcessor p(16000,c);}
      catch(const std::invalid_argument& e) {
        rejected=std::string(e.what()).find("use aligned/fixedblend/legacy")!=std::string::npos;
      }
      require(rejected,"unified mode silently reinterpreted a processed-donor policy");
      c.unified_raw_budget=false;
      if(backend==itu::VehicleNsBackend::speex || std::string(policy).find("dynamic")==std::string::npos) {
        itu::VehicleNsProcessor historical(16000,c);
      }
    }
}
void noise_protection() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    auto c=config(1);c.startup_ms=0;c.policy="aligned";
    itu::VehicleNsProcessor p(rate,c);c.dry_max=0;c.unified_raw_budget=false;
    itu::VehicleNsProcessor wet(rate,c);
    std::vector<float>a(rate/100),b(a.size());unsigned random=772;
    double input_power=0,output_power=0,wet_power=0;float late_raw=0;
    for(unsigned frame=0;frame<700;++frame) {
      for(unsigned i=0;i<a.size();++i) a[i]=b[i]=noise(random,.04f);
      if(frame>=500) for(float x:a) input_power+=x*x;
      require(p.process(a.data(),a.size()) && wet.process(b.data(),b.size()),"noise process failed");
      if(frame>=500) {
        late_raw=std::max(late_raw,p.statistics().raw_mix_applied_max);
        for(unsigned i=0;i<a.size();++i) {output_power+=a[i]*a[i];wet_power+=b[i]*b[i];}
      }
    }
    require(late_raw<.02f,"noise-only evidence retained a large raw request");
    require(output_power<=1.08*wet_power,"raw controller erased steady-noise backend benefit");
    require(output_power<.8*input_power,"steady noise attenuation test was vacuous");
    std::cout<<"noise rate="<<rate<<" late_raw="<<late_raw<<" power_vs_wet="<<output_power/wet_power
      <<" attenuation_db="<<10*std::log10(input_power/output_power)<<"\n";
  }
}
}
int main() {
  try {caps_and_timebase();pure_raw_impulse();quiet_is_not_a_bypass();noise_protection();incompatible_policies();
    std::cout<<"unified raw budget behavior passed\n";return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
