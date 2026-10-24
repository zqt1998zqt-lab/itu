#include "itu/audio_engine.hpp"
#include "itu/vehicle_ns.hpp"
#include "vehicle_rumble_filter.hpp"
#include "vehicle_ns_controller.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
namespace { bool watching=false; std::size_t allocations=0; }
void* operator new(std::size_t n) { if(watching) ++allocations; if(void* p=std::malloc(n?n:1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
namespace {
void require(bool b,const char* message) {if(!b) throw std::runtime_error(message);}
float input(unsigned n,unsigned r) {
  const double t=double(n)/r;
  const float x=.07*std::sin(2*3.141592653589793*173*t)+.03*std::sin(2*3.141592653589793*390*t)+.01*std::sin(2*3.141592653589793*27*t);
  return std::round(x*32768)/32768;
}
void filter() {
  constexpr double pi=3.141592653589793;
  for(unsigned rate:{16000u,24000u,48000u}) {
    const unsigned d=rate/50,n=2*d+1;
    itu::LinearRumbleFilter f(rate,30),off(rate,0),adaptive(rate,30);
    std::vector<float> impulse(3*n),original(3*n);impulse[0]=original[0]=1;
    off.process(original.data(),original.size());require(original[0]==1 && original[1]==0,"disabled FIR changed PCM");
    f.process(impulse.data(),impulse.size());require(f.delay()==d,"FIR declared delay");
    double scale=0;std::vector<double> expected(n);
    for(unsigned j=0;j<n;++j) {
      const int m=int(j)-int(d);
      const double lp=m ? std::sin(2*pi*30*m/rate)/(pi*m):60./rate;
      expected[j]=(m==0?1.:0.)-lp*(.54-.46*std::cos(2*pi*j/(n-1)));
      scale+=expected[j]*(j%2?-1:1);
    }
    for(unsigned j=0;j<n;++j) require(std::abs(impulse[j]-expected[j]/scale)<2e-7,"FIR coefficients/reference mismatch");
    require(unsigned(std::max_element(impulse.begin(),impulse.end())-impulse.begin())==d,"FIR impulse peak/delay mismatch");
    for(unsigned j=n;j<impulse.size();++j) require(impulse[j]==0,"FIR failed to flush");
    std::vector<float> raw(3*n),zero(3*n);raw[0]=1;
    adaptive.process(raw.data(),raw.size(),zero.data());
    for(unsigned j=0;j<raw.size();++j) require(raw[j]==(j==d?1.f:0.f),"inactive adaptive FIR changed raw PCM/timebase");
    itu::LinearRumbleFilter retuned(rate,30);
    retuned.retune(60);
    std::vector<float> high(3*n);high[0]=1;
    retuned.process(high.data(),high.size());
    scale=0;
    for(unsigned j=0;j<n;++j) {
      const int m=int(j)-int(d);
      const double lp=m ? std::sin(2*pi*60*m/rate)/(pi*m):120./rate;
      expected[j]=(m==0?1.:0.)-lp*(.54-.46*std::cos(2*pi*j/(n-1)));
      scale+=expected[j]*(j%2?-1:1);
    }
    for(unsigned j=0;j<n;++j)
      require(std::abs(high[j]-expected[j]/scale)<2e-7,"retuned FIR coefficients/reference mismatch");
  }
}
void causal_and_allocation() {
  for(bool experimental:{false,true}) for(auto backend:{itu::VehicleNsBackend::webrtc,itu::VehicleNsBackend::speex,itu::VehicleNsBackend::omlsa_imcra,itu::VehicleNsBackend::super_gaussian}) for(unsigned rate:{16000u,24000u,48000u}) {
    for(bool native:{false,true}) {
      if(native && (rate!=24000 || backend==itu::VehicleNsBackend::webrtc)) continue;
      itu::VehicleNsConfig c;c.backend=backend;c.strength=backend==itu::VehicleNsBackend::speex?-12:1;
      c.mild_strength=backend==itu::VehicleNsBackend::speex?-6:0;c.rumble_hz=30;c.native_24=native;c.adaptive_rumble=true;
      if(experimental) {
        if(backend==itu::VehicleNsBackend::speex) c.policy="steady_dynamic";
        else c.repair_mode=4;
        if(backend==itu::VehicleNsBackend::omlsa_imcra || backend==itu::VehicleNsBackend::super_gaussian) c.spectral_dry_guard=1;
      }
      itu::VehicleNsProcessor a(rate,c),b(rate,c);
      std::vector<float> x(rate/100),y(x.size());unsigned differences=0,guard_frames=0;
      require(!a.process(x.data(),x.size()-1),"short frame accepted");
      x.back()=std::numeric_limits<float>::quiet_NaN();require(!a.process(x.data(),x.size()),"NaN accepted");
      for(unsigned frame=0;frame<100;++frame) {
        for(unsigned i=0;i<x.size();++i) {x[i]=input(frame*x.size()+i,rate);y[i]=frame<40?x[i]:-x[i];}
        const auto before=allocations;watching=true;
        const bool aa=a.process(x.data(),x.size()),bb=b.process(y.data(),y.size());watching=false;
        require(aa && bb,"stream processing rejected");
        require(allocations==before,"callback C++ allocation");
        guard_frames+=a.statistics().spectral_guard_applied<.99f;
        if(frame<40) require(x==y,"future suffix changed prefix / invalid call advanced state");
        else for(unsigned i=0;i<x.size();++i) differences+=x[i]!=y[i];
        for(float v:x) require(std::isfinite(v) && std::abs(v)<1,"nonfinite or clipped stream");
      }
      require(differences>0,"causality negative control was vacuous");
      if(c.spectral_dry_guard>0) require(guard_frames>0,"spectral dry guard never activated");
      const unsigned expected=(backend!=itu::VehicleNsBackend::webrtc?rate/100:rate==48000?336:96)+rate/50;
      if(rate!=24000 || native || backend==itu::VehicleNsBackend::omlsa_imcra || backend==itu::VehicleNsBackend::super_gaussian) require(a.delay_samples()==expected,"native stream declared delay");
      std::cout<<"stream rate="<<rate<<" backend="<<int(backend)<<" native24="<<native<<" delay="<<a.delay_samples()<<" allocations=0\n";
    }
  }
}
void engine() {
  for(bool vcp:{false,true}) for(bool experimental:{false,true}) for(auto backend:{itu::NoiseBackend::webrtc,itu::NoiseBackend::speex,itu::NoiseBackend::omlsa_imcra,itu::NoiseBackend::super_gaussian}) for(unsigned rate:{16000u,24000u,48000u}) {
    itu::Config c=vcp?itu::barrot_vcp_config():itu::Config{};
    c.sample_rate=rate;c.frame_size=rate/100;c.aec=false;c.noise_suppression=true;
    c.tx_drc.enabled=false;
    c.automatic_gain=c.limiter=c.linear_phase_low_cut=false;c.vehicle_ns_control=true;c.noise_backend=backend;
    c.noise_level=1;c.speex_suppression_db=-12;c.vehicle_ns.rumble_hz=30;c.vehicle_ns.adaptive_rumble=true;
    c.vehicle_ns.mild_strength=backend==itu::NoiseBackend::speex?-6:0;
    if(backend==itu::NoiseBackend::speex) {
      c.vehicle_ns.rumble_hz=60;c.vehicle_ns.repair_mode=2;c.vehicle_ns.snr_threshold_db=3;
      c.vehicle_ns.startup_ms=60;c.vehicle_ns.policy="dynamic";c.vehicle_ns.release_ms=80;c.vehicle_ns.floor_guard=2;
    }
    if(experimental) {
      if(backend==itu::NoiseBackend::speex) c.vehicle_ns.policy="steady_dynamic";
      else c.vehicle_ns.repair_mode=4;
      if(backend==itu::NoiseBackend::omlsa_imcra || backend==itu::NoiseBackend::super_gaussian) c.vehicle_ns.spectral_dry_guard=1;
      c.vehicle_ns.rumble_recovery=1;
      c.vehicle_ns.weak_speech_recovery=1;
      c.vehicle_ns.noise_step_recovery=.5f;
    }
    itu::AudioEngine e(c);auto options=c.vehicle_ns;
    options.backend=backend==itu::NoiseBackend::super_gaussian?itu::VehicleNsBackend::super_gaussian:backend==itu::NoiseBackend::omlsa_imcra?itu::VehicleNsBackend::omlsa_imcra:backend==itu::NoiseBackend::speex?itu::VehicleNsBackend::speex:itu::VehicleNsBackend::webrtc;
    options.strength=backend==itu::NoiseBackend::speex?-12:1;
    itu::VehicleNsProcessor direct(rate,options);
    std::vector<float> x(c.frame_size),y(x.size()),z(x.size());itu::FrameInfo f;
    float maximum=0;unsigned nonzero_tail=0;
    for(unsigned frame=0;frame<104;++frame) {
      for(unsigned i=0;i<x.size();++i)
        x[i]=frame<100 ? input(frame*x.size()+i,rate)+(i%2?.37f:-.37f)/32768.f : 0.f;
      y=x;require(direct.process(y.data(),y.size()),"direct failed");
      require(e.process(x.data(),x.size(),nullptr,0,z.data(),z.size(),f)==itu::Status::ok,"integrated engine failed");
      for(unsigned i=0;i<x.size();++i) maximum=std::max(maximum,std::abs(y[i]-z[i]));
      if(frame>=100) for(float value:z) nonzero_tail+=value!=0.f;
      require(f.external_vad_calls==0,"external VAD enabled");
    }
    require(f.latency_samples==direct.delay_samples(),"engine and shared module latency differ");
    require(maximum==0.f,"engine differs from shared module");
    require(nonzero_tail>0,"silent input tail check was vacuous");
    std::cout<<"engine vcp="<<vcp<<" rate="<<rate<<" delay="<<f.latency_samples<<" max_difference="<<maximum<<'\n';
  }
}
void steady_parallel_alignment() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu::VehicleNsConfig c;c.backend=itu::VehicleNsBackend::speex;
    c.strength=c.mild_strength=c.steady_suppression_db=-12;c.startup_ms=0;
    c.policy="dynamic";c.quiet_floor_db=-300;c.repair_mode=2;c.rumble_hz=60;c.adaptive_rumble=true;
    itu::VehicleNsProcessor baseline(rate,c);
    c.policy="steady_dynamic";itu::VehicleNsProcessor equal(rate,c);
    c.steady_suppression_db=-20;itu::VehicleNsProcessor stronger(rate,c);
    std::vector<float> a(rate/100),b(a.size()),d(a.size());
    unsigned random=937;float maximum=0,negative=0,trigger=0;
    for(unsigned frame=0;frame<350;++frame) {
      for(unsigned i=0;i<a.size();++i) {
        random=1664525u*random+1013904223u;
        a[i]=b[i]=d[i]=.05f*(float(random>>8)/16777216.f-.5f);
      }
      require(baseline.process(a.data(),a.size()) && equal.process(b.data(),b.size()) && stronger.process(d.data(),d.size()),"steady alignment process failed");
      for(unsigned i=0;i<a.size();++i) {
        maximum=std::max(maximum,std::abs(a[i]-b[i]));
        negative=std::max(negative,std::abs(a[i]-d[i]));
      }
      trigger=std::max(trigger,equal.statistics().steady_mix_target);
    }
    require(trigger>.1f,"steady parallel path never activated");
    require(maximum<1e-7f,"parallel Speex path changed timebase or primary state");
    require(negative>1e-5f,"stronger parallel path negative control was vacuous");
    require(equal.delay_samples()==baseline.delay_samples(),"steady parallel path added delay");
    std::cout<<"steady parallel rate="<<rate<<" matched_state_error="<<maximum<<" negative_control="<<negative<<" trigger="<<trigger<<'\n';
  }
}
void invalid_envelopes() {
  itu::VehicleNsConfig c;c.adaptive_rumble=true;
  bool rejected=false;try {itu::VehicleNsProcessor e(16000,c);} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"adaptive rumble with disabled filter accepted");
  c.adaptive_rumble=false;c.native_24=true;
  rejected=false;try {itu::VehicleNsProcessor e(24000,c);} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"native WebRTC 24k silently accepted");
  itu::Config engine_config;engine_config.sample_rate=24000;engine_config.frame_size=240;
  engine_config.aec=false;engine_config.vehicle_ns_control=true;engine_config.noise_backend=itu::NoiseBackend::speex;
  engine_config.vehicle_ns.native_24=true;
  rejected=false;try {itu::AudioEngine e(engine_config);} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"AudioEngine silently replaced explicit native24 request with bridge");
}
void residual_stream_invariants() {
  unsigned cases=0;
  for(unsigned rate:{16000u,24000u,48000u})
    for(auto backend:{itu::VehicleNsBackend::webrtc,itu::VehicleNsBackend::speex,
                     itu::VehicleNsBackend::omlsa_imcra,itu::VehicleNsBackend::super_gaussian})
      for(bool gated:{false,true}) {
        itu::VehicleNsConfig c;c.backend=backend;
        c.strength=backend==itu::VehicleNsBackend::speex?-12:1;
        c.mild_strength=backend==itu::VehicleNsBackend::speex?-6:0;
        c.native_24=rate==24000 && backend!=itu::VehicleNsBackend::webrtc;
        itu::VehicleNsProcessor off(rate,c);
        c.residual_max_db=3;
        if(gated) c.residual_gate_snr_db=-12;
        itu::VehicleNsProcessor a(rate,c),b(rate,c);
        require(a.delay_samples()==off.delay_samples(),"residual stage changed declared delay");
        require(a.core_rate()==off.core_rate(),"residual stage changed core rate");
        const unsigned hop=rate/100;
        std::vector<float> x(hop),y(hop),raw(hop);
        unsigned differences=0;unsigned random=rate;
        for(unsigned frame=0;frame<120;++frame) {
          for(unsigned i=0;i<hop;++i) {
            random=1664525u*random+1013904223u;
            raw[i]=.04f*(float(random>>8)/16777216.f-.5f);
            x[i]=raw[i];y[i]=frame<60?raw[i]:-raw[i];
          }
          const auto before=allocations;watching=true;
          const bool oa=off.process(raw.data(),hop),aa=a.process(x.data(),hop),bb=b.process(y.data(),hop);
          watching=false;
          require(oa && aa && bb,"residual-enabled stream failed");
          require(allocations==before,"residual-enabled callback allocated");
          if(frame<60) require(x==y,"residual output depends on future PCM");
          for(unsigned i=0;i<hop;++i) {
            require(std::isfinite(x[i]),"residual stream output nonfinite");
            differences+=x[i]!=raw[i];
          }
          const auto stats=a.statistics();
          require(stats.residual_attenuation_db>=0 && stats.residual_attenuation_db<=3.001f,
                  "residual attenuation exceeded envelope");
        }
        require(differences>0,"residual-enabled stream never changed noise output");
        ++cases;
      }
  require(cases==24,"incomplete residual rate/backend/gate matrix");
  std::cout<<"residual rate/backend/gate invariant cases="<<cases<<'\n';
}
void rumble_recovery_invariants() {
  unsigned cases=0;
  for(unsigned rate:{16000u,24000u,48000u})
    for(auto backend:{itu::VehicleNsBackend::webrtc,itu::VehicleNsBackend::speex,
                     itu::VehicleNsBackend::omlsa_imcra,itu::VehicleNsBackend::super_gaussian}) {
      itu::VehicleNsConfig c;c.backend=backend;
      c.strength=backend==itu::VehicleNsBackend::speex?-12:1;
      c.mild_strength=backend==itu::VehicleNsBackend::speex?-6:0;
      c.native_24=rate==24000 && backend!=itu::VehicleNsBackend::webrtc;
      c.rumble_hz=60;c.adaptive_rumble=true;
      itu::VehicleNsProcessor off(rate,c);
      c.rumble_recovery=1;
      c.weak_speech_recovery=.75f;c.noise_step_recovery=1;
      itu::VehicleNsProcessor a(rate,c),b(rate,c);
      require(a.delay_samples()==off.delay_samples(),"rumble recovery added delay");
      require(a.core_rate()==off.core_rate(),"rumble recovery changed core rate");
      std::vector<float> x(rate/100),y(x.size()),raw(x.size());
      unsigned random=817;unsigned changed=0;float triggered=0;
      for(unsigned frame=0;frame<180;++frame) {
        for(unsigned i=0;i<x.size();++i) {
          random=1664525u*random+1013904223u;
          const double t=double(frame*x.size()+i)/rate;
          raw[i]=.03f*std::sin(2*3.141592653589793*37*t)+.02f*(float(random>>8)/16777216.f-.5f);
          x[i]=raw[i];y[i]=frame<90?raw[i]:-raw[i];
        }
        const auto before=allocations;watching=true;
        const bool aa=a.process(x.data(),x.size()),bb=b.process(y.data(),y.size()),oo=off.process(raw.data(),raw.size());
        watching=false;
        require(aa && bb && oo,"rumble recovery processing failed");
        require(allocations==before,"rumble recovery allocated in callback");
        if(frame<90) require(x==y,"rumble recovery depends on future PCM");
        for(unsigned i=0;i<x.size();++i) {
          require(std::isfinite(x[i]),"rumble recovery produced nonfinite PCM");
          changed+=x[i]!=raw[i];
        }
        const auto stats=a.statistics();
        require(stats.rumble_mix_applied>=0 && stats.rumble_mix_applied<=1.00001f,"rumble mix outside envelope");
        triggered=std::max(triggered,stats.rumble_mix_target-off.statistics().rumble_mix_target);
      }
      require(changed>0 && triggered>0,"rumble recovery negative control did not trigger");
      ++cases;
    }
  require(cases==12,"rumble recovery rate/backend matrix incomplete");
  itu::VehicleNsConfig bad;bad.rumble_recovery=.5f;
  bool rejected=false;try {itu::VehicleNsProcessor e(16000,bad);} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"rumble recovery silently accepted without adaptive FIR");
  itu::VehicleNsConfig invalid_corner;invalid_corner.rumble_hz=30;
  invalid_corner.adaptive_rumble=true;
  invalid_corner.rumble_hz=0;
  invalid_corner.rumble_adaptive_hz=60;
  bool invalid_corner_rejected=false;
  try {itu::VehicleNsProcessor e(16000,invalid_corner);}
  catch(const std::invalid_argument&) {invalid_corner_rejected=true;}
  require(invalid_corner_rejected,"adaptive corner accepted without adaptive FIR");
  std::cout<<"rumble recovery invariant cases="<<cases<<'\n';
}
void noise_step_evidence() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu::VehicleNsController noise(rate,1,15,1,0,0,-65,1,4,0,1);
    itu::VehicleNsController tone(rate,1,15,1,0,0,-65,1,4,0,1);
    std::vector<float> x(rate/100),y(x.size());unsigned random=19;
    float noise_peak=0,tone_peak=0;
    unsigned active=0;std::vector<unsigned> recent(150);
    for(unsigned frame=0;frame<250;++frame) {
      const float level=frame<100?.01f:.05f;
      for(unsigned i=0;i<x.size();++i) {
        random=1664525u*random+1013904223u;
        x[i]=level*(float(random>>8)/16777216.f-.5f);
        y[i]=level*std::sin(2*3.141592653589793*173*(frame*x.size()+i)/rate);
      }
      noise.observe(x.data());tone.observe(y.data());
      noise_peak=std::max(noise_peak,noise.noise_step_probability());
      tone_peak=std::max(tone_peak,tone.noise_step_probability());
      active-=recent[frame%150];recent[frame%150]=noise.noise_step_probability()>0;
      active+=recent[frame%150];
      require(active<=25,"step relearning exceeded its budget or cooldown");
    }
    require(noise_peak>.2f,"persistent broadband noise step was not detected");
    require(tone_peak<.1f,"tonal onset was learned as a broadband noise step");
    std::cout<<"step evidence rate="<<rate<<" noise/tone="<<noise_peak<<'/'<<tone_peak<<'\n';
  }
}
void speech_band_invariants() {
  unsigned cases=0;
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu::VehicleNsConfig c;c.backend=itu::VehicleNsBackend::omlsa_imcra;
    c.strength=0;c.native_24=rate==24000;c.rumble_hz=30;
    c.adaptive_rumble=true;c.rumble_recovery=1;c.noise_step_recovery=.5f;
    itu::VehicleNsProcessor off(rate,c);
    c.speech_band_recovery=1;
    itu::VehicleNsProcessor a(rate,c),b(rate,c);
    require(a.delay_samples()==off.delay_samples(),"speech-band recovery added delay");
    require(a.core_rate()==off.core_rate(),"speech-band recovery changed core rate");
    std::vector<float>x(rate/100),y(x.size()),raw(x.size());
    unsigned random=4111,changed=0;float evidence=0,delta=0;
    for(unsigned frame=0;frame<180;++frame) {
      for(unsigned i=0;i<x.size();++i) {
        random=1664525u*random+1013904223u;
        const double t=double(frame*x.size()+i)/rate;
        raw[i]=.025f*std::sin(2*3.141592653589793*173*t)+
            .02f*(float(random>>8)/16777216.f-.5f);
        x[i]=raw[i];y[i]=frame<90?raw[i]:-raw[i];
      }
      const auto before=allocations;watching=true;
      const bool aa=a.process(x.data(),x.size()),bb=b.process(y.data(),y.size()),
          oo=off.process(raw.data(),raw.size());
      watching=false;
      require(aa && bb && oo,"speech-band recovery processing failed");
      require(allocations==before,"speech-band recovery allocated in callback");
      if(frame<90) require(x==y,"speech-band recovery depends on future PCM");
      for(unsigned i=0;i<x.size();++i) {
        require(std::isfinite(x[i]),"speech-band recovery produced nonfinite PCM");
        changed+=x[i]!=raw[i];
      }
      const auto stats=a.statistics();
      require(stats.speech_band_floor>=0 && stats.speech_band_floor<=1,
              "speech-band floor outside gain range");
      require(stats.speech_band_evidence>=0 && stats.speech_band_evidence<=1,
              "speech-band evidence outside probability range");
      evidence=std::max(evidence,stats.speech_band_evidence);
      delta=std::max(delta,stats.speech_band_gain_delta);
    }
    require(changed>0 && evidence>0 && delta>0,
            "speech-band recovery negative control was vacuous");
    ++cases;
  }
  require(cases==3,"speech-band rate matrix incomplete");
  itu::VehicleNsConfig invalid;invalid.speech_band_recovery=1;
  bool rejected=false;try {itu::VehicleNsProcessor e(16000,invalid);}
  catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"speech-band recovery accepted without OM-LSA");
  std::cout<<"speech-band recovery cases="<<cases<<" allocations=0\n";
}
void transient_guard_invariants() {
  unsigned controller_cases=0;unsigned guard_rates=0;
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu::VehicleNsController indicator(rate,1,6,1,0,0,-65,1,4,0,0,1);
    itu::VehicleNsController wind(rate,1,6,1,0,0,-65,1,4,0,0,1);
    std::vector<float> x(rate/100);
    unsigned random=0x51f15e5du;
    double low_state=0;
    float indicator_evidence=0,indicator_period=0,wind_evidence=0,wind_period=0;
    for(unsigned frame=0;frame<180;++frame) {
      const bool indicator_dropout=frame>=60 && frame%50<10;
      for(unsigned i=0;i<x.size();++i) {
        random=1664525u*random+1013904223u;
        const float white=float(random>>8)/16777216.f-.5f;
        x[i]=(indicator_dropout?.0005f:.02f)*white;
      }
      indicator.observe(x.data());
      indicator_evidence=std::max(indicator_evidence,indicator.transient_noise_evidence());
      indicator_period=std::max(indicator_period,indicator.transient_periodicity_500ms());

      const bool wind_dropout=frame>=60 && frame%10<1;
      for(unsigned i=0;i<x.size();++i) {
        random=1664525u*random+1013904223u;
        const float white=float(random>>8)/16777216.f-.5f;
        low_state=.995*low_state+.005*white;
        x[i]=float(low_state)*(wind_dropout?.002f:.02f);
      }
      wind.observe(x.data());
      wind_evidence=std::max(wind_evidence,wind.transient_noise_evidence());
      wind_period=std::max(wind_period,wind.transient_periodicity_100ms());
    }
    require(indicator_period>.55f && indicator_evidence>.5f,
            "periodic indicator positive control did not trigger");
    require(wind_period>.1f,
            "periodic low-band power positive control did not trigger");

    itu::VehicleNsConfig c;c.backend=itu::VehicleNsBackend::omlsa_imcra;
    c.strength=0;c.native_24=rate==24000;c.rumble_hz=30;
    c.adaptive_rumble=true;c.rumble_recovery=1;
    c.noise_step_recovery=.5f;c.speech_band_recovery=1;
    itu::VehicleNsProcessor off(rate,c);
    c.transient_dry_guard=1;
    itu::VehicleNsProcessor a(rate,c),b(rate,c);
    require(a.delay_samples()==off.delay_samples(),
            "transient guard changed declared delay");
    require(a.core_rate()==off.core_rate(),
            "transient guard changed core rate");
    std::vector<float>left(rate/100),right(left.size()),raw(left.size());
    unsigned guard_frames=0;float peak_guard=1;
    for(unsigned frame=0;frame<180;++frame) {
      const bool indicator_dropout=frame>=60 && frame%50<10;
      for(unsigned i=0;i<left.size();++i) {
        random=1664525u*random+1013904223u;
        const float white=float(random>>8)/16777216.f-.5f;
        raw[i]=(indicator_dropout?.0005f:.02f)*white;
        left[i]=raw[i];right[i]=frame<90?raw[i]:-raw[i];
      }
      const auto before=allocations;watching=true;
      const bool aa=a.process(left.data(),left.size());
      const bool bb=b.process(right.data(),right.size());
      const bool oo=off.process(raw.data(),raw.size());
      watching=false;
      require(aa && bb && oo,"transient guard processing failed");
      require(allocations==before,"transient guard allocated in callback");
      if(frame<90) require(left==right,"transient guard depends on future PCM");
      const auto stats=a.statistics();
      require(stats.transient_noise_evidence>=0 && stats.transient_noise_evidence<=1 &&
              stats.transient_guard_applied>=0 && stats.transient_guard_applied<=1,
              "transient evidence or guard outside probability range");
      peak_guard=std::min(peak_guard,stats.transient_guard_applied);
      guard_frames+=stats.transient_guard_applied<.999f;
      for(unsigned i=0;i<left.size();++i) {
        require(std::isfinite(left[i]),"transient guard produced nonfinite PCM");
      }
    }
    require(guard_frames>0 && peak_guard<.9f,
            "transient guard positive control did not reduce dry mix");
    ++controller_cases;
    ++guard_rates;
    std::cout<<"transient guard rate="<<rate
             <<" indicator="<<indicator_evidence
             <<" wind_corr="<<wind_period
             <<" wind_evidence="<<wind_evidence
             <<" peak_guard="<<peak_guard
             <<" allocations=0\n";

    itu::VehicleNsConfig spectral_config=c;
    spectral_config.transient_spectral_guard_db=0;
    itu::VehicleNsProcessor spectral_off(rate,spectral_config);
    spectral_config.transient_spectral_guard_db=18;
    itu::VehicleNsProcessor spectral_on(rate,spectral_config);
    itu::VehicleNsProcessor spectral_future(rate,spectral_config);
    require(spectral_on.delay_samples()==spectral_off.delay_samples(),
            "transient spectral guard changed declared delay");
    require(spectral_on.core_rate()==spectral_off.core_rate(),
            "transient spectral guard changed core rate");
    float peak_spectral=1;
    for(unsigned frame=0;frame<180;++frame) {
      const bool indicator_dropout=frame>=60 && frame%50<10;
      for(unsigned i=0;i<left.size();++i) {
        random=1664525u*random+1013904223u;
        const float white=float(random>>8)/16777216.f-.5f;
        raw[i]=(indicator_dropout?.0005f:.02f)*white;
        left[i]=raw[i];right[i]=frame<90?raw[i]:-raw[i];
      }
      const auto before=allocations;watching=true;
      const bool oo=spectral_off.process(raw.data(),raw.size());
      const bool on=spectral_on.process(left.data(),left.size());
      const bool future=spectral_future.process(right.data(),right.size());
      watching=false;
      require(oo && on && future,"transient spectral guard processing failed");
      require(allocations==before,
              "transient spectral guard allocated in callback");
      if(frame<90) require(left==right,
          "transient spectral guard depends on future PCM");
      const auto stats=spectral_on.statistics();
      require(stats.transient_spectral_guard_applied>=0 &&
              stats.transient_spectral_guard_applied<=1,
              "transient spectral guard outside gain range");
      peak_spectral=std::min(
          peak_spectral,stats.transient_spectral_guard_applied);
      for(float value:left) require(std::isfinite(value),
          "transient spectral guard produced nonfinite PCM");
    }
    require(peak_spectral<.9f,
            "transient spectral guard positive control did not attenuate");
  }
  require(controller_cases==3 && guard_rates==3,
          "transient guard rate matrix incomplete");
  itu::VehicleNsConfig invalid;invalid.backend=itu::VehicleNsBackend::webrtc;
  invalid.transient_dry_guard=1;
  bool rejected=false;
  try {itu::VehicleNsProcessor e(16000,invalid);}
  catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"transient guard accepted without OM-LSA");
  invalid.transient_dry_guard=0;
  invalid.transient_spectral_guard_db=18;
  rejected=false;
  try {itu::VehicleNsProcessor e(16000,invalid);}
  catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"transient spectral guard accepted without OM-LSA");
}
void aec_bf_integration() {
  unsigned cases=0;
  for(unsigned rate:{16000u,24000u,48000u}) for(unsigned microphones:{1u,2u,4u})
    for(auto aec:{itu::AecBackend::webrtc,itu::AecBackend::speex})
      for(auto ns:{itu::NoiseBackend::webrtc,itu::NoiseBackend::speex,itu::NoiseBackend::omlsa_imcra,itu::NoiseBackend::super_gaussian}) {
        itu::Config c;c.sample_rate=rate;c.frame_size=rate/100;c.microphones=microphones;
        c.beamforming=microphones>1;c.aec_backend=aec;c.noise_backend=ns;
        c.automatic_gain=c.limiter=c.linear_phase_low_cut=false;c.vehicle_ns_control=true;
        c.noise_level=1;c.speex_suppression_db=-12;c.vehicle_ns.rumble_hz=30;c.vehicle_ns.adaptive_rumble=true;
        c.vehicle_ns.mild_strength=ns==itu::NoiseBackend::speex?-6:0;
        itu::AudioEngine engine(c);itu::FrameInfo info;
        std::vector<float> mic(c.frame_size*microphones),reference(c.frame_size),out(c.frame_size);
        for(unsigned frame=0;frame<20;++frame) {
          for(unsigned i=0;i<c.frame_size;++i) {
            reference[i]=input(frame*c.frame_size+i,rate);
            for(unsigned ch=0;ch<microphones;++ch) mic[i*microphones+ch]=.5f*reference[i];
          }
          require(engine.process(mic.data(),mic.size(),reference.data(),reference.size(),out.data(),out.size(),info)==itu::Status::ok,"AEC/BF/vehicle NS path failed");
          for(float v:out) require(std::isfinite(v),"AEC/BF integration nonfinite output");
          require(!info.external_vad_calls,"AEC/BF path enabled external VAD");
        }
        const unsigned delay=ns!=itu::NoiseBackend::webrtc ? rate*30/1000 : rate==16000?416:rate==24000?648:1296;
        require(info.ns_delay_samples==delay,"AEC/BF wrapper lost vehicle NS delay");
        if(c.beamforming) require(info.beamforming_before_aec && info.aec_capture_channels==1,"BF/AEC topology changed");
        ++cases;
      }
  require(cases==72,"incomplete AEC/BF integration matrix");
  std::cout<<"AEC/BF/rate/backend integration cases="<<cases<<'\n';
}
}
int main() {
  try {invalid_envelopes();filter();causal_and_allocation();engine();steady_parallel_alignment();residual_stream_invariants();rumble_recovery_invariants();noise_step_evidence();speech_band_invariants();transient_guard_invariants();aec_bf_integration();std::cout<<"vehicle NS checks passed\n";return 0;}
  catch(const std::exception& e) {watching=false;std::cerr<<e.what()<<'\n';return 1;}
}
