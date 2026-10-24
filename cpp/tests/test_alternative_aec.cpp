#include "itu/audio_engine.hpp"
#include "echo_subspace.hpp"
#include "speex_multirender_context.hpp"
#include "fixed_hop_adapter.hpp"
#include "aec_safety_mix.hpp"
#include "reference_monitor.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

void require(bool value,const char* reason) {if(!value) throw std::runtime_error(reason);}
void aec_safety() {
  itu::AecSafetyMix guard;
  float mic[]={.1f,-.1f,.3f,-.3f},residual[]={1.5f,-1.2f,.5f,-.6f};
  const float original[]={1.5f,-1.2f,.5f,-.6f};
  require(guard.process(mic,residual,4),"valid capture rejected");
  require(guard.protected_hops()==1 && guard.mix()<1,"AEC divergence not detected");
  for(unsigned i=0;i<4;++i) {
    require(std::abs(residual[i])<=1.f,"AEC guarded PCM exceeds contract");
    require(std::abs((residual[i]-mic[i])/(original[i]-mic[i])-guard.mix())<1e-5,"AEC guard clips samples instead of reducing prediction");
  }
  for(unsigned hop=0;hop<30;++hop) {
    for(unsigned i=0;i<4;++i) residual[i]=.5f*mic[i];
    require(guard.process(mic,residual,4),"valid capture rejected during recovery");
  }
  require(guard.mix()==1.f && residual[0]==.5f*mic[0],"AEC guard fails to recover transparency");
  for(float invalid:{std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity()}) {
    residual[0]=.2f;residual[1]=invalid;residual[2]=.4f;residual[3]=-.2f;
    require(guard.process(mic,residual,4),"valid fallback rejected");
    for(unsigned i=0;i<4;++i) require(residual[i]==mic[i],"nonfinite residual leaked through whole-hop fallback");
    require(guard.mix()==0.f && std::isfinite(guard.raw_peak()),"nonfinite residual not recorded safely");
  }
  require(guard.nonfinite_hops()==3 && guard.protected_hops()==4,"nonfinite hop fault count lost");
  for(unsigned i=0;i<4;++i) residual[i]=.5f*mic[i];
  require(guard.process(mic,residual,4) && std::abs(guard.mix()-.05f)<1e-7,"nonfinite fallback recovery is not gradual");
  const float bad_capture[]={.1f,1.01f,.3f,-.3f};
  require(!guard.process(bad_capture,residual,4),"invalid resampled capture accepted as safe fallback");
}
void hop_adapter() {
  itu::FixedHopAdapter adapter;
  std::array<float,160> mic{},ref{},out{};unsigned calls=0;
  for(unsigned frame=0;frame<1000;++frame) {
    for(unsigned i=0;i<160;++i) {mic[i]=float(frame*160+i+1);ref[i]=-mic[i];}
    require(adapter.process(mic.data(),ref.data(),out.data(),[&](const float* m,const float* r,float* o){
      ++calls;for(unsigned i=0;i<256;++i) {require(m[i]+r[i]==0,"render block mismatched");o[i]=m[i];}return true;
    }),"hop adapter overflow/underflow");
    for(unsigned i=0;i<160;++i) {
      // The injected identity model sees a zero-padded first block. Its first
      // callback is therefore emitted immediately; the padded tail is exposed
      // on the next callback before block 1 becomes available.
      if(frame==0) require(out[i]==float(i+1),"cold-start output was delayed or altered");
    }
  }
  require(calls==1000 && adapter.inferences()==calls && adapter.pending()==0 && adapter.queued()==0,"hop count mismatch");
  itu::FixedHopAdapter rejected;
  require(!rejected.process(mic.data(),ref.data(),out.data(),[](const float*,const float*,float*){return false;}),"cold-start model error ignored");
}
void subspace(bool dependent,unsigned mics) {
  std::vector<std::unique_ptr<itu::EchoSubspace>> detectors;
  for(unsigned ch=0;ch<mics;++ch) detectors.emplace_back(std::make_unique<itu::EchoSubspace>(2));
  std::mt19937 random(17); std::normal_distribution<float> draw(0,.04f);
  std::vector<float> ref(320),mic(160*mics),first(8192),second(8192);
  unsigned head=0; float echo_ratio=1;
  for(unsigned frame=0;frame<500;++frame) {
    for(unsigned i=0;i<160;++i) {
      ref[i*2]=draw(random); ref[i*2+1]=dependent ? ref[i*2] : draw(random);
      first[head]=ref[i*2]; second[head]=ref[i*2+1];
      const float echo=.6f*first[(head+8192-480)%8192]+.3f*second[(head+8192-480)%8192];
      for(unsigned ch=0;ch<mics;++ch)
        mic[i*mics+ch]=echo+(frame>=300 && ch==mics-1 ? draw(random) : 0.f);
      head=(head+1)%8192;
    }
    for(unsigned ch=0;ch<mics;++ch) detectors[ch]->push(mic.data()+ch,ref.data(),mics,true,480);
    if(frame==299) {
      echo_ratio=0;for(auto& d:detectors) echo_ratio=std::max(echo_ratio,d->ratio());
    }
  }
  std::cout<<"subspace dependent="<<dependent<<" mics="<<mics<<" echo="<<echo_ratio<<" double_talk="<<detectors.back()->ratio()<<'\n';
  require(echo_ratio<.02,"joint render echo not recognized");
  require(detectors.back()->ratio()>echo_ratio+.3,"last-mic near-speech negative control not detected");
  if(mics>1) require(detectors.front()->ratio()<.02,"near speech leaked into another microphone statistic");
  for(unsigned ch=0;ch<mics;++ch) {
    detectors[ch]->push(mic.data()+ch,ref.data(),mics,false,960);
    require(detectors[ch]->frames()==0 && detectors[ch]->ratio()==1.f,"delay reset retained stale confidence");
  }
}
void tap_delays(unsigned rate,bool lowcut,bool res,bool protect,bool ns=false) {
  itu::Config c;c.sample_rate=rate;c.frame_size=rate/100;c.reference_channels=2;
  c.aec_backend=itu::AecBackend::speex;c.speex_residual_suppression=res;
  c.linear_phase_low_cut=lowcut;c.speech_preserving_low_cut=protect;
  c.noise_suppression=ns;c.automatic_gain=c.limiter=false;
  itu::AudioEngine engine(c);itu::FrameInfo info;
  const unsigned h=c.frame_size,onset=rate/2;
  std::vector<float> mic(h),ref(h*2),out(h),at(h),bt(h),it(h),all,aa,bb,ii;
  for(unsigned f=0;f<100;++f) {
    std::fill(mic.begin(),mic.end(),0.f);
    if(f*h<=onset && onset<(f+1)*h) mic[onset-f*h]=.2f;
    require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info,at.data(),bt.data(),nullptr,it.data())==itu::Status::ok,"MDF impulse failed");
    all.insert(all.end(),out.begin(),out.end());aa.insert(aa.end(),at.begin(),at.end());
    bb.insert(bb.end(),bt.begin(),bt.end());ii.insert(ii.end(),it.begin(),it.end());
  }
  auto peak=[](const std::vector<float>& v) {
    return unsigned(std::max_element(v.begin(),v.end(),[](float a,float b){return std::abs(a)<std::abs(b);})-v.begin());
  };
  const auto output=peak(all)-onset,aec=peak(aa)-onset,beam=peak(bb)-onset,input=peak(ii)-onset;
  std::cout<<"MDF impulse rate="<<rate<<" lowcut="<<lowcut<<" res="<<res<<" protect="<<protect
           <<" input="<<input<<" aec="<<aec<<" bf="<<beam<<" output="<<output<<'\n';
  // NS can attenuate an impulse, but its synthesis center must retain the
  // declared causal delay. No fitted output alignment is used here.
  // Band splitting + nonlinear NS can shift the impulse maximum within the
  // synthesis lobe (48k: two samples; resampled 24k: one sample). The exact
  // unsuppressed tap delays above still have a zero-sample tolerance.
  const unsigned peak_tolerance=ns ? (rate==48000 ? 2 : rate==24000 ? 1 : 0) : 0;
  require(std::abs(int(output)-int(info.latency_samples))<=int(peak_tolerance) && aec==info.aec_output_delay_samples,"measured MDF/RES/NS output delay differs from report");
  require(input==info.aec_input_delay_samples && beam==aec,"MDF conditioned/beam tap delay differs from report");
  require(info.latency_samples-info.ns_delay_samples-info.post_low_cut_delay_samples==beam,"BF export length includes post conditioning");
  require(info.aec_input_delay_samples==0,"minimum-delay low-cut added an input offset");
  if(protect && rate!=24000 && !ns) require(std::abs(all[onset+output]-.2f)<1e-6,"silent-render near-speech protection changes PCM");
}
void mdf(unsigned rate,unsigned mics) {
  itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.microphones=mics;
  c.reference_channels=2; c.aec_backend=itu::AecBackend::speex;
  c.noise_suppression=c.automatic_gain=c.limiter=c.linear_phase_low_cut=false;
  c.beamforming=mics>1;
  itu::AudioEngine engine(c); itu::FrameInfo info;
  const unsigned h=c.frame_size,delay=rate*17/1000;
  std::vector<float> first(rate),second(rate),ref(h*2),mic(h*mics),out(h),tap(h*mics),linear(160*mics,7.f);
  std::mt19937 random(31); std::normal_distribution<float> draw(0,.04f);
  unsigned head=0; double before=0,after=0,cross=0,clean_power=0;
  for(unsigned frame=0;frame<500;++frame) {
    for(unsigned i=0;i<h;++i) {
      ref[i*2]=draw(random); ref[i*2+1]=draw(random);
      first[head]=ref[i*2]; second[head]=ref[i*2+1];
      float echo=.6f*first[(head+rate-delay)%rate]+.3f*second[(head+rate-delay)%rate];
      float near=frame>=350 ? .04f*std::sin(2*3.141592653589793*997*(frame*h+i)/rate) : 0.f;
      for(unsigned ch=0;ch<mics;++ch) mic[i*mics+ch]=echo+near;
      head=(head+1)%rate;
    }
    require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info,tap.data(),nullptr,linear.data())==itu::Status::ok,"MDF processing failed");
    require(info.frame_index==frame && info.latency_samples==(rate==24000 ? 160u : 0u) && info.linear_aec_available,"MDF clock/delay/tap contract");
    for(unsigned i=0;i<h;++i) {
      require(std::isfinite(out[i]),"MDF non-finite output");
      if(frame>=250 && frame<300) {before+=mic[i*mics]*mic[i*mics];after+=out[i]*out[i];}
      if(frame>=400) {
        const double clean=.04*std::sin(2*3.141592653589793*997*(frame*h+i-info.latency_samples)/rate);
        cross+=clean*out[i]; clean_power+=clean*clean;
      }
    }
  }
  const double attenuation=10*std::log10(before/after),gain=cross/clean_power;
  std::cout<<"MDF rate="<<rate<<" mics="<<mics<<" echo_attenuation="<<attenuation<<" near_gain="<<gain<<'\n';
  require(attenuation>6,"AEC disabled negative control would also pass");
  require(gain>.8 && gain<1.2,"MDF double-talk target removed");
  double linear_power=0;
  for(float v:linear) {require(std::isfinite(v),"MDF linear tap is nonfinite");linear_power+=double(v)*v;}
  require(linear_power>1e-4,"an all-zero fake MDF tap would pass");
  engine.reset();
  require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok && info.frame_index==0,"MDF reset");
}
void mdf_linear_tap(unsigned rate) {
  itu::Config c;c.sample_rate=rate;c.frame_size=rate/100;
  c.aec_backend=itu::AecBackend::speex;
  c.noise_suppression=c.automatic_gain=c.limiter=c.linear_phase_low_cut=false;
  c.speech_preserving_low_cut=true;
  itu::AudioEngine plain(c);
  c.speex_residual_suppression=true;
  itu::AudioEngine res(c);
  std::vector<float> capture(c.frame_size),render(c.frame_size),out(c.frame_size),tap(160),res_tap(160);
  std::vector<float> all;
  itu::FrameInfo info,ri;
  for(unsigned frame=0;frame<100;++frame) {
    std::fill(capture.begin(),capture.end(),0);
    if(frame==20)capture[0]=.2f;
    require(plain.process(capture.data(),capture.size(),render.data(),render.size(),out.data(),out.size(),info,
          nullptr,nullptr,tap.data())==itu::Status::ok,"MDF impulse processing failed");
    require(res.process(capture.data(),capture.size(),render.data(),render.size(),out.data(),out.size(),ri,
          nullptr,nullptr,res_tap.data())==itu::Status::ok,"MDF/RES impulse processing failed");
    require(info.linear_aec_available && ri.linear_aec_available,"raw MDF tap unavailable");
    require(info.linear_aec_delay_samples==ri.linear_aec_delay_samples,"RES changed raw MDF tap delay");
    for(unsigned i=0;i<160;++i) require(std::abs(tap[i]-res_tap[i])<1e-7,"raw MDF tap was taken after RES/dry blend");
    all.insert(all.end(),tap.begin(),tap.end());
  }
  auto peak=std::max_element(all.begin(),all.end(),[](float a,float b){return std::abs(a)<std::abs(b);});
  const auto measured=unsigned(peak-all.begin());
  std::cout<<"MDF raw tap rate="<<rate<<" measured impulse="<<measured<<" declared="<<3200+info.linear_aec_delay_samples<<'\n';
  require(std::abs(*peak)>.02f,"zero MDF tap passes the delay test");
  require(std::abs(int(measured)-int(3200+info.linear_aec_delay_samples))<=1,"MDF linear tap delay differs from impulse");
}
void mdf_reference_guard(unsigned rate,bool wrong,bool silent,bool context=false) {
  itu::Config c;c.sample_rate=rate;c.frame_size=rate/100;
  c.aec_backend=itu::AecBackend::speex;c.speex_native_conditioning=true;
  c.speex_post_context=context;
  c.speex_echo_tail_ms=500;c.noise_suppression=c.automatic_gain=c.limiter=false;
  itu::AudioEngine engine(c);
  std::vector<float> mic(c.frame_size),ref(c.frame_size),out(c.frame_size),history(rate);
  std::mt19937 random(819);std::normal_distribution<float> draw(0,.03f);
  unsigned head=0,unreliable=0,reliable=0;double before=0,after=0;
  itu::FrameInfo info;
  for(unsigned frame=0;frame<450;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      const float playback=draw(random),independent=draw(random);
      history[head]=playback;
      ref[i]=silent ? 0.f : playback;
      mic[i]=wrong ? independent : .6f*history[(head+rate-rate/50)%rate];
      head=(head+1)%rate;
    }
    require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok,"MDF reference monitor failed");
    require(info.post_aec_context_applied==context,"MDF state did not reach the post-AEC core");
    require(!info.external_vad_active && !info.external_vad_calls,"MDF context enabled external VAD");
    if(frame>=350) {
      unreliable+=info.reference_state==2;reliable+=info.reference_state==1;
      for(unsigned i=0;i<c.frame_size;++i) {before+=double(mic[i])*mic[i];after+=double(out[i])*out[i];}
    }
  }
  require(before>1e-3 && after>1e-12,"silent output makes reference checks vacuous");
  std::cout<<"MDF reference rate="<<rate<<" wrong="<<wrong<<" silent="<<silent
           <<" unreliable="<<unreliable<<" reliable="<<reliable<<" dry_mix="<<info.nearend_mix<<'\n';
  if(silent) require(unreliable==0 && reliable==0,"silent render falsely classified as wrong reference");
  else if(wrong) require(unreliable>=90 && info.nearend_mix>.99f,"unrelated reference did not protect real capture");
  else require(unreliable==0 && reliable>=90 && 10*std::log10(before/after)>15,"valid echo path failed reference negative control");
}
void mdf_native_conditioning(unsigned rate,bool aec) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.aec=aec;c.aec_backend=itu::AecBackend::speex;c.speex_native_conditioning=true;
  c.speex_echo_tail_ms=500;
  itu::AudioEngine native(c);
  c.speex_native_conditioning=false;
  if(aec)c.linear_phase_low_cut=false;
  itu::AudioEngine explicit_chain(c);
  std::vector<float> capture(c.frame_size),render(c.frame_size),a(c.frame_size),b(c.frame_size);
  itu::FrameInfo ai,bi;
  double energy=0;
  for(unsigned frame=0;frame<100;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      const double t=double(frame*c.frame_size+i)/rate;
      render[i]=.03f*std::sin(2*3.141592653589793*397*t);
      capture[i]=.01f*std::sin(2*3.141592653589793*997*t)+render[i]*.5f;
    }
    require(native.process(capture.data(),capture.size(),render.data(),render.size(),a.data(),a.size(),ai)==itu::Status::ok,"native MDF configuration failed");
    require(explicit_chain.process(capture.data(),capture.size(),render.data(),render.size(),b.data(),b.size(),bi)==itu::Status::ok,"MDF comparison failed");
    require(ai.latency_samples==bi.latency_samples,"native conditioning changed the output time base");
    for(unsigned i=0;i<c.frame_size;++i) {
      require(std::abs(a[i]-b[i])<1e-7,"native conditioning changed an unrelated NS path or retained a redundant HPF");
      energy+=double(a[i])*a[i];
    }
  }
  require(energy>.001,"zero output makes the conditioning comparison vacuous");
}
void mdf_post_stages(unsigned rate,bool residual=false) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.aec_backend=itu::AecBackend::speex;c.speex_native_conditioning=true;
  c.speex_post_context=true;c.automatic_gain=false;c.tx_drc.enabled=false;
  c.speex_residual_suppression=residual;
  itu::DiagnosticStageTaps stages;c.diagnostic_stages=&stages;
  itu::AudioEngine engine(c);
  std::vector<float> capture(c.frame_size),render(c.frame_size),out(c.frame_size);
  itu::FrameInfo info;double selected=0,protected_power=0;
  for(unsigned frame=0;frame<120;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      const double t=double(frame*c.frame_size+i)/rate;
      capture[i]=.02f*std::sin(2*3.141592653589793*997*t);
    }
    require(engine.process(capture.data(),capture.size(),render.data(),render.size(),out.data(),out.size(),info)==itu::Status::ok,"MDF diagnostic stages failed");
    require(info.post_aec_context_applied,"native post core dropped AEC context");
    const auto core_rate=rate==24000 ? 48000u : rate;
    require(stages.core_sample_rate==core_rate && stages.core_frame_size==core_rate/100 && stages.nr_selected_available,"MDF stage rate/extent mismatch");
    require(stages.core_delay_samples==unsigned(std::lround(double(info.latency_samples)*core_rate/rate))-(rate==24000 ? 160u : 0u),"MDF stage alignment differs from native conversion contract");
    for(unsigned i=0;i<stages.core_frame_size;++i) {
      require(std::isfinite(stages.nr_selected[i]) && std::isfinite(stages.nr_protected[i]),"MDF stage PCM is nonfinite");
      selected+=double(stages.nr_selected[i])*stages.nr_selected[i];
      protected_power+=double(stages.nr_protected[i])*stages.nr_protected[i];
    }
  }
  require(selected>.001 && protected_power>.001,"zero diagnostic buffers pass post-AEC test");
}
void reference_silence_memory() {
  itu::ReferenceMonitor retained(true),legacy;
  std::mt19937 random(271);std::normal_distribution<float> draw(0,.03f);
  std::array<float,160> capture{},render{};
  for(unsigned frame=0;frame<700;++frame) {
    for(unsigned i=0;i<160;++i) {
      render[i]=frame>=300 && frame<450 ? 0.f : draw(random);
      capture[i]=frame>=450 ? .6f*render[i] : draw(random);
    }
    retained.push(capture.data(),render.data(),1);legacy.push(capture.data(),render.data(),1);
    if(frame==299) require(retained.unreliable() && legacy.unreliable(),"wrong-reference negative control was not excited");
    if(frame==449) require(retained.unreliable() && !legacy.unreliable(),"silence erased retained reference identity or changed default policy");
  }
  require(retained.reliable() && !retained.unreliable(),"new valid reference cannot replace retained wrong-reference verdict");
}
void mdf_res_context_clock(unsigned rate) {
  for(bool residual:{false,true}) {
    auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
    c.aec_backend=itu::AecBackend::speex;c.speex_post_context=true;
    c.speex_native_conditioning=true;c.speex_residual_suppression=residual;
    c.reference_guard=false;c.automatic_gain=false;
    itu::AudioEngine engine(c);itu::FrameInfo info;
    std::vector<float> mic(c.frame_size),ref(c.frame_size),out(c.frame_size);
    double energy=0;
    for(unsigned frame=0;frame<60;++frame) {
      for(unsigned i=0;i<c.frame_size;++i) {
        const double t=double(frame*c.frame_size+i)/rate;
        ref[i]=frame==10 ? .1f*std::sin(2*3.141592653589793*997*t) : 0.f;
        mic[i]=.5f*ref[i];
      }
      require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok,
              "RES context alignment processing failed");
      if(frame==10) require(info.far_active==!residual,"render context is not aligned to the RES output block");
      if(frame==11) require(info.far_active,"delayed render context never reaches the post core");
      for(float v:out) energy+=double(v)*v;
    }
    require(energy>1e-8,"zero RES output makes context clock test vacuous");
  }
}
void mdf_unrelated_silence(unsigned rate,bool residual,bool wrong) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.aec_backend=itu::AecBackend::speex;c.speex_post_context=true;
  c.speex_native_conditioning=true;c.speex_residual_suppression=residual;
  c.speech_preserving_low_cut=false;
  c.noise_suppression=c.automatic_gain=c.limiter=false;
  itu::AudioEngine engine(c);itu::FrameInfo info;
  std::mt19937 random(731);std::normal_distribution<float> draw(0,.03f);
  std::vector<float> mic(c.frame_size),ref(c.frame_size),out(c.frame_size);
  double energy=0;unsigned idle_checks=0;
  for(unsigned frame=0;frame<450;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      ref[i]=frame<300 ? draw(random) : 0.f;
      mic[i]=frame<300 && !wrong ? .6f*ref[i] : draw(random);
    }
    require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok,
            "wrong-reference silence processing failed");
    if(frame==299) require(info.reference_state==(wrong ? 2u : 1u),
                          "reference classification was not excited before silence");
    if(frame>=400) {
      require(!info.far_active,"render tail did not drain before silence assertion");
      require(wrong ? info.nearend_mix>.99f : info.nearend_mix<.01f,
              "dry fallback lost its independent reference verdict after render silence");
      ++idle_checks;
      for(float v:out) {require(std::isfinite(v),"nonfinite silence fallback");energy+=double(v)*v;}
    }
  }
  require(idle_checks==50 && energy>1e-5,"empty or silent fallback makes the test vacuous");
}
void mdf_gain_slew(unsigned rate) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.aec_backend=itu::AecBackend::speex;c.speex_native_conditioning=c.speex_post_context=true;
  itu::AudioEngine engine(c);itu::FrameInfo info;
  std::mt19937 random(910);std::normal_distribution<float> noise(0,.00003f);
  std::vector<float> mic(c.frame_size),ref(c.frame_size),out(c.frame_size);
  float previous=0,maximum=0;
  for(unsigned frame=0;frame<400;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      const double t=double(frame*c.frame_size+i)/rate;
      const double phase=2*3.141592653589793*(177*t+40*std::sin(t*9));
      const double envelope=.3+.7*std::pow(std::sin(t*11),2);
      const float speech=frame>=50 && frame<300 ? .001f*envelope*(std::sin(phase)+.7f*std::sin(phase*3)+.4f*std::sin(phase*7)) : 0.f;
      mic[i]=speech+noise(random);
    }
    require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok,"MDF gain slew processing failed");
    require(std::abs(info.applied_gain_db-previous)<=c.max_gain_change_db_per_second/100.f+1e-5f,"post-AEC AGC bypasses configured slew limit");
    previous=info.applied_gain_db;maximum=std::max(maximum,previous);
  }
  require(maximum>.1f,"untriggered AGC makes slew test vacuous");
}
void mdf_coherence_residual_candidate() {
  itu::Config c;
  c.sample_rate=16000;c.frame_size=160;c.reference_channels=1;c.aec_backend=itu::AecBackend::speex;
  c.speex_native_conditioning=true;c.speex_post_context=true;
  c.noise_suppression=c.automatic_gain=c.limiter=false;
  itu::AudioEngine baseline(c);
  require(!baseline.config().speex_coherence_residual_suppression,
          "coherence residual suppression is not disabled by default");
  c.speex_coherence_residual_suppression=true;
  itu::AudioEngine candidate(c);
  itu::FrameInfo bi,ci;
  std::vector<float> mic(160),ref(160),bo(160),co(160);
  std::mt19937 random(441);std::normal_distribution<float> draw(0,.08f);
  double baseline_power=0,candidate_power=0;unsigned changed=0;
  for(unsigned frame=0;frame<420;++frame) {
    for(unsigned i=0;i<160;++i) {
      const float playback=std::clamp(draw(random),-.25f,.25f);
      ref[i]=playback;
      const float nonlinear=playback*std::abs(playback);
      mic[i]=.62f*playback+.45f*nonlinear;
    }
    require(baseline.process(mic.data(),mic.size(),ref.data(),ref.size(),bo.data(),bo.size(),bi)==itu::Status::ok,
            "baseline MDF candidate probe failed");
    require(candidate.process(mic.data(),mic.size(),ref.data(),ref.size(),co.data(),co.size(),ci)==itu::Status::ok,
            "coherence residual candidate probe failed");
    for(unsigned i=0;i<160;++i) {
      require(std::isfinite(co[i]),"coherence residual candidate emitted nonfinite PCM");
      if(frame>=300) {
        baseline_power+=double(bo[i])*bo[i];candidate_power+=double(co[i])*co[i];
        changed+=std::abs(bo[i]-co[i])>1e-8f;
      }
    }
  }
  require(baseline_power>1e-6 && candidate_power>1e-6,"AEC residual probe became silent");
  require(changed>32,"coherence residual switch did not change a measured far-end residual");
  require(candidate_power<baseline_power*.98,"coherence residual switch did not suppress the measured coherent residual");
}
static void speex_multirender_state_sequence() {
  itu::SpeexMultiRenderNearState detector;
  using D=itu::SpeexMultiRenderNearState::Decision;
  D d;
  // Unobserved evidence cannot be used as measured non-near speech.
  for(unsigned f=0;f<300;++f) {
    d=detector.update(true,false,false,false,.15f,1.f,.99f,.99f);
    require(d.state==0 && d.hold==0,"missing coherence became echo-only or near");
  }
  for(unsigned f=0;f<100;++f) d=detector.update(true,true,true,false,.15f,.18f,.98f,.98f);
  require(d.state==1 && !d.hold,"echo-only never converged despite high speech probabilities");
  const float echo=d.echo_floor,coherence=d.coherence_floor;
  // Ordinary far-envelope fluctuations must not enter near.
  for(unsigned f=0;f<200;++f) {
    d=detector.update(true,true,true,false,f%2 ? .18f : .14f,f%2 ? .20f : .16f,.99f,.99f);
    require(d.state==1 && !d.hold,"far-envelope modulation triggered near");
  }
  for(unsigned f=0;f<15;++f) d=detector.update(true,true,true,false,.31f,.51f,.95f,.95f);
  require(d.state==2 && d.hold,"double talk not independently confirmed");
  require(std::abs(d.echo_floor-echo)<.02 && std::abs(d.coherence_floor-coherence)<.03,
          "near evidence trained itself into the echo baseline");
  for(unsigned f=0;f<8;++f) d=detector.update(true,true,true,false,.16f,.21f,.2f,.2f);
  require(d.state==2,"a syllable valley erased the near latch");
  for(unsigned f=0;f<80;++f) d=detector.update(true,true,true,false,.14f,.16f,.99f,.99f);
  require(d.state==1 && !d.hold,"echo-only did not recover after near speech ended");
  d=detector.update(true,true,true,true,.3f,.5f,.99f,.99f);
  require(d.state==0,"path change retained a stale echo-only verdict");
  for(unsigned f=0;f<100;++f) d=detector.update(true,true,true,false,.10f,.12f,.99f,.99f);
  require(d.state==1,"new path could not relearn echo-only");
  d=detector.update(false,false,false,false,1.f,1.f,.99f,.99f);
  require(d.state==0 && !d.hold,"render silence retained a false echo verdict");
  // Moderate cold double talk is not sufficient initial echo-only proof,
  // even when both speech estimators confidently report speech.
  for(const auto& evidence:std::array<std::array<float,2>,2>{{{{.6f,.6f}},{{.4f,.45f}}}}) {
    itu::SpeexMultiRenderNearState moderate;
    for(unsigned f=0;f<100;++f) {
      d=moderate.update(true,true,true,false,evidence[0],evidence[1],.99f,.99f);
      require(d.state==0,"moderate cold double talk trained an echo-only baseline");
    }
  }
  // Initial echo proof must be forty consecutive trusted, excited frames.
  for(unsigned gap=0;gap<3;++gap) {
    itu::SpeexMultiRenderNearState interrupted;
    for(unsigned f=0;f<20;++f) d=interrupted.update(true,true,true,false,.8f,.008f,.99f,.99f);
    d=interrupted.update(gap!=0,gap!=1,gap!=2,false,.8f,.008f,.99f,.99f);
    for(unsigned f=0;f<20;++f) {
      d=interrupted.update(true,true,true,false,.8f,.008f,.99f,.99f);
      require(d.state==0,"interrupted initialization counted nonconsecutive echo evidence");
    }
    for(unsigned f=0;f<20;++f) d=interrupted.update(true,true,true,false,.8f,.008f,.99f,.99f);
    require(d.state==1,"continuous echo evidence could not finish initialization");
  }
  itu::SpeexMultiRenderNearState slow_mdf;
  for(unsigned f=0;f<100;++f) d=slow_mdf.update(true,true,true,false,.8f,.008f,.99f,.99f);
  require(d.state==1,"strong joint echo evidence cannot recover a slow MDF path");
  for(unsigned f=0;f<15;++f) d=slow_mdf.update(true,true,true,false,.85f,.35f,.99f,.99f);
  require(d.state==2,"recovered high-error echo baseline masks renewed independent near speech");
  for(unsigned f=0;f<80;++f) d=slow_mdf.update(true,true,true,false,.75f,.008f,.99f,.99f);
  require(d.state==1 && !d.hold,"high-error joint baseline cannot recover after near ends");
  itu::SpeexMultiRenderNearState cold_near;
  for(unsigned f=0;f<100;++f) {
    d=cold_near.update(true,true,true,false,.95f,.9f,.99f,.99f);
    require(d.state==0,"cold unmodelled near speech authorized echo-only suppression");
  }
}
static void speex_multirender_reference_sequence(int dependency) {
  itu::SpeexMultiRenderReference observer(2);
  std::mt19937 rng(712);std::normal_distribution<float> draw(0,.04f);
  std::vector<float> first(8192),second(8192);
  std::array<float,320> render{};
  std::array<float,160> capture{};
  unsigned head=0,near_frames=0,valid_echo=0,recovery=0;
  float echo_ratio=1,near_ratio=0;
  itu::SpeexMultiRenderReference::Evidence evidence;
  for(unsigned frame=0;frame<1100;++frame) {
    const unsigned delay=frame>=800 ? 960u : 480u;
    for(unsigned i=0;i<160;++i) {
      render[2*i]=draw(rng);
      render[2*i+1]=dependency ? float(dependency)*render[2*i] : draw(rng);
      first[head]=render[2*i];second[head]=render[2*i+1];
      const float echo=.60f*first[(head+8192-delay)%8192]+.30f*second[(head+8192-delay-32)%8192];
      capture[i]=echo+(frame>=500 && frame<650 ? draw(rng) : 0.f);
      head=(head+1)%8192;
    }
    evidence=observer.observe(capture.data(),render.data(),true);
    require(std::isfinite(evidence.incoherent_ratio),"joint subspace emitted NaN for dependent stereo");
    if(frame>=400 && frame<450) {
      valid_echo+=evidence.valid;echo_ratio=std::min(echo_ratio,evidence.incoherent_ratio);
    }
    if(frame>=580 && frame<640) {
      near_frames+=evidence.valid;near_ratio=std::max(near_ratio,evidence.incoherent_ratio);
    }
    if(frame>=1050) recovery+=evidence.valid;
  }
  std::cout<<"dependency="<<dependency<<" echo="<<echo_ratio<<" near="<<near_ratio
           <<" valid_echo="<<valid_echo<<" recovery="<<recovery<<'\n';
  require(valid_echo>=40 && echo_ratio<.12f,"joint playback path not identified");
  require(near_frames>=50 && near_ratio>echo_ratio+.3f,"independent near speech not outside joint render space");
  require(recovery>=20 && evidence.incoherent_ratio<.15f,"changed path never re-converged");
  evidence=observer.observe(capture.data(),render.data(),false);
  require(!evidence.valid,"unexcited stale evidence declared valid");
}

static itu::Config speex_multirender_configuration(unsigned rate,unsigned refs,bool flag) {
  auto c=itu::barrot_vcp_config();c.sample_rate=rate;c.frame_size=rate/100;
  c.reference_channels=refs;c.aec_backend=itu::AecBackend::speex;
  c.speex_native_conditioning=c.speex_post_context=true;
  c.speex_multirender_context=flag;c.speex_coherence_residual_suppression=false;
  c.automatic_gain=false;c.tx_drc.enabled=false;
  return c;
}
static void speex_multirender_mono_compatibility(unsigned rate) {
  auto c=speex_multirender_configuration(rate,1,false);itu::AudioEngine original(c);
  c.speex_multirender_context=true;itu::AudioEngine enabled(c);
  std::vector<float> mic(c.frame_size),render(c.frame_size),a(c.frame_size),b(c.frame_size);
  std::mt19937 rng(511);std::normal_distribution<float> draw(0,.03f);
  itu::FrameInfo ai,bi;double energy=0;
  for(unsigned frame=0;frame<200;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      render[i]=draw(rng);
      mic[i]=.6f*render[i]+.02f*std::sin(2*3.141592653589793*997*(frame*c.frame_size+i)/rate);
    }
    require(original.process(mic.data(),mic.size(),render.data(),render.size(),a.data(),a.size(),ai)==itu::Status::ok,"mono flag-off failed");
    require(enabled.process(mic.data(),mic.size(),render.data(),render.size(),b.data(),b.size(),bi)==itu::Status::ok,"mono flag-on failed");
    require(a==b,"multi-render opt-in changed mono PCM");
    require(!bi.post_aec_multirender_frames && !bi.post_aec_near_state && !bi.post_aec_reference_evidence_valid,
            "mono stream created multi-render evidence");
    for(float v:a) energy+=double(v)*v;
  }
  require(energy>1e-4,"mono compatibility had no audio");
  std::cout<<"mono bit-identical rate="<<rate<<'\n';
}
static void speex_multirender_unknown_neutral(unsigned rate,bool silence) {
  auto c=speex_multirender_configuration(rate,2,true);
  itu::DiagnosticStageTaps taps;c.diagnostic_stages=&taps;
  itu::AudioEngine engine(c);
  std::mt19937 rng(733);std::normal_distribution<float> draw(0,.03f);
  std::vector<float> mic(c.frame_size),render(c.frame_size*2),out(c.frame_size);
  itu::FrameInfo info;unsigned checked=0;double selected=0,protected_power=0;
  for(unsigned frame=0;frame<100;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      render[2*i]=silence ? 0.f : draw(rng);
      render[2*i+1]=silence ? 0.f : draw(rng);
      const double t=double(frame*c.frame_size+i)/rate;
      mic[i]=.025f*(std::sin(2*3.141592653589793*173*t)+.7f*std::sin(2*3.141592653589793*519*t))+.0001f*draw(rng);
    }
    require(engine.process(mic.data(),mic.size(),render.data(),render.size(),out.data(),out.size(),info)==itu::Status::ok,
            "unknown-reference processing failed");
    require(info.post_aec_near_state==0,"unmeasured independent references became echo-only");
    require(!info.post_aec_reference_evidence_valid,"reference became valid before monitor warmup");
    if(frame>=20 && (info.far_active || silence)) {
      ++checked;
      for(unsigned i=0;i<taps.core_frame_size;++i) {
        selected+=double(taps.nr_selected[i])*taps.nr_selected[i];
        protected_power+=double(taps.nr_protected[i])*taps.nr_protected[i];
      }
    }
  }
  std::cout<<"unknown rate="<<rate<<" silent="<<silence<<" selected="<<selected
           <<" protected="<<protected_power<<" checks="<<checked<<'\n';
  require(checked>=70 && selected>1e-3,"neutrality comparison was not excited");
  require(protected_power>=selected*.98,"unknown reference applied the old extra .24 far floor");
}
static void speex_multirender_stream_sequence(unsigned rate,bool wrong) {
  auto c=speex_multirender_configuration(rate,2,true);c.noise_suppression=false;
  itu::AudioEngine engine(c);
  std::mt19937 rng(921);std::normal_distribution<float> draw(0,.04f);
  std::vector<float> first(rate),second(rate),render(c.frame_size*2),mic(c.frame_size),out(c.frame_size);
  unsigned head=0,echo_valid=0,near=0,recovered=0,unrelated=0;
  itu::FrameInfo info;double before=0,after=0;
  for(unsigned frame=0;frame<1000;++frame) {
    for(unsigned i=0;i<c.frame_size;++i) {
      render[2*i]=draw(rng);render[2*i+1]=draw(rng);
      first[head]=render[2*i];second[head]=render[2*i+1];
      const float echo=.6f*first[(head+rate-rate*3/100)%rate]+.3f*second[(head+rate-rate*32/1000)%rate];
      const double t=double(frame*c.frame_size+i)/rate;
      const double phase=2*3.141592653589793*(167*t+13*std::sin(t*4));
      const double envelope=.3+.7*std::pow(std::sin(t*9),2);
      const float speech=frame>=500 && frame<700 ? .045f*envelope*(std::sin(phase)+.7f*std::sin(3*phase)+.4f*std::sin(7*phase)) : 0.f;
      mic[i]=wrong ? .04f*std::sin(phase)+draw(rng) : echo+speech;
      if(frame>=350 && frame<450) before+=double(mic[i])*mic[i];
      head=(head+1)%rate;
    }
    require(engine.process(mic.data(),mic.size(),render.data(),render.size(),out.data(),out.size(),info)==itu::Status::ok,
            "multi-reference stream failed");
    for(float v:out) require(std::isfinite(v),"multi-reference stream emitted nonfinite PCM");
    if(frame>=350 && frame<450) {
      echo_valid+=info.post_aec_near_state==1 && info.post_aec_reference_evidence_valid;
      for(float v:out) after+=double(v)*v;
    }
    if(frame>=580 && frame<680) near+=info.post_aec_near_state==2 && info.near_protection_hold_frames>0;
    if(frame>=850 && frame<950) recovered+=info.post_aec_near_state==1;
    if(frame>=850 && wrong) unrelated+=info.reference_state==2 && info.nearend_mix>.99f;
  }
  std::cout<<"stream rate="<<rate<<" wrong="<<wrong<<" echo="<<echo_valid<<" near="<<near
           <<" recovery="<<recovered<<" unrelated="<<unrelated<<" attenuation="<<10*std::log10(before/after)<<'\n';
  if(wrong) require(!echo_valid && !recovered && unrelated>=100,"unrelated playback reference became echo-only");
  else {
    require(echo_valid>=80,"valid multi-render echo-only never converged");
    require(near>=50,"actual AudioEngine never propagated confirmed multi-render near");
    require(recovered>=80,"actual AudioEngine did not return to far-only suppression");
    require(before>1e-3 && after>1e-12 && 10*std::log10(before/after)>6,"far-only AEC suppression lost");
  }
}
static void speex_multirender_path_change_during_near() {
  auto c=speex_multirender_configuration(16000,2,true);c.noise_suppression=false;
  itu::AudioEngine engine(c);
  std::mt19937 rng(617);std::normal_distribution<float> draw(0,.05f);
  std::array<float,160> capture{},out{};
  std::array<float,320> render{};
  std::array<float,8192> first{},second{};
  itu::FrameInfo info;
  unsigned head=0,previous_frames=0,near_before_change=0,resets_during_near=0,recovered=0,near_again=0;
  for(unsigned frame=0;frame<1500;++frame) {
    const unsigned delay=frame>=500 ? 1120u : 480u;
    for(unsigned i=0;i<160;++i) {
      render[2*i]=draw(rng);render[2*i+1]=draw(rng);
      first[head]=render[2*i];second[head]=render[2*i+1];
      const double t=double(frame*160+i)/16000;
      const double phase=2*3.141592653589793*(181*t+11*std::sin(t*5));
      const float near=((frame>=400 && frame<750) || (frame>=1200 && frame<1350)) ?
          .025f*(std::sin(phase)+.7f*std::sin(3*phase)+.4f*std::sin(7*phase)) : 0.f;
      capture[i]=.6f*first[(head+8192-delay)%8192]+.3f*second[(head+8192-delay-32)%8192]+near;
      head=(head+1)%8192;
    }
    require(engine.process(capture.data(),capture.size(),render.data(),render.size(),out.data(),out.size(),info)==itu::Status::ok,
            "path-during-near stream failed");
    if(frame>=470 && frame<500) near_before_change+=info.post_aec_near_state==2;
    if(frame>=500 && frame<750 && info.post_aec_multirender_frames<previous_frames) ++resets_during_near;
    previous_frames=info.post_aec_multirender_frames;
    if(frame>=1270 && frame<1330) near_again+=info.post_aec_near_state==2;
    if(frame>=1050 && frame<1150) recovered+=info.post_aec_near_state==1;
  }
  std::cout<<"path during near confirmed_before="<<near_before_change
           <<" measured_resets="<<resets_during_near<<" recovered="<<recovered<<" near_again="<<near_again<<'\n';
  require(near_before_change>=20,"path-change negative control never entered near");
  require(resets_during_near>0,"confirmed near blocked a newly measured playback path indefinitely");
  require(recovered>=80,"new playback path did not recover echo-only suppression");
  require(near_again>=40,"strong-joint recovery baseline masked the next near onset");
}


static void speex_multirender_unknown_after_echo_floor() {
  auto c=speex_multirender_configuration(16000,2,true);
  itu::DiagnosticStageTaps taps;c.diagnostic_stages=&taps;
  itu::AudioEngine engine(c);
  std::mt19937 rng(521);std::normal_distribution<float> draw(0,.05f);
  std::array<float,160> capture{},out{};
  std::array<float,320> render{};
  std::array<float,8192> first{},second{};
  unsigned head=0,echo_frames=0,unknown_frames=0;
  double selected=0,protected_power=0;
  itu::FrameInfo info;
  for(unsigned frame=0;frame<900;++frame) {
    const unsigned delay=480u;
    for(unsigned i=0;i<160;++i) {
      render[2*i]=draw(rng);render[2*i+1]=draw(rng);
      first[head]=render[2*i];second[head]=render[2*i+1];
      capture[i]=.6f*first[(head+8192-delay)%8192]+.3f*second[(head+8192-delay-32)%8192];
      head=(head+1)%8192;
    }
    // A playback/reference gap is an immediate loss of evidence after
    // confirmed echo-only. Keep capture excited so a stale floor is audible.
    if(frame>=400) render.fill(0.f);
    require(engine.process(capture.data(),capture.size(),render.data(),render.size(),out.data(),out.size(),info)==itu::Status::ok,
            "unknown-after-echo stream failed");
    if(frame>=300 && frame<400) echo_frames+=info.post_aec_near_state==1;
    if(frame>=400 && echo_frames>=80 && info.post_aec_near_state==0 && unknown_frames<5) {
      ++unknown_frames;
      for(unsigned i=0;i<taps.core_frame_size;++i) {
        selected+=double(taps.nr_selected[i])*taps.nr_selected[i];
        protected_power+=double(taps.nr_protected[i])*taps.nr_protected[i];
      }
    }
  }
  std::cout<<"unknown after echo floor echo_frames="<<echo_frames<<" unknown_frames="<<unknown_frames
           <<" protected/selected="<<protected_power/std::max(selected,1e-20)<<'\n';
  require(echo_frames>=80 && unknown_frames==5 && selected>1e-5,"old-floor transition test was not excited");
  require(protected_power>=selected*.90,"unknown path inherited a stale echo-only floor");
}

int main() {
  try {
    speex_multirender_unknown_after_echo_floor();
    speex_multirender_path_change_during_near();
    for(unsigned rate:{16000,24000,48000}) {
      speex_multirender_mono_compatibility(rate);
      speex_multirender_unknown_neutral(rate,false);
      speex_multirender_unknown_neutral(rate,true);
      speex_multirender_stream_sequence(rate,false);
      speex_multirender_stream_sequence(rate,true);
    }
    speex_multirender_state_sequence();
    for(int dependent:{0,1,-1}) speex_multirender_reference_sequence(dependent);
    hop_adapter();
    aec_safety();
    reference_silence_memory();
    for(unsigned rate:{16000,24000,48000}) mdf_linear_tap(rate);
    for(unsigned rate:{16000,24000,48000}) {mdf_reference_guard(rate,false,false);mdf_reference_guard(rate,true,false);mdf_reference_guard(rate,true,true);}
    for(unsigned rate:{16000,24000,48000}) {mdf_reference_guard(rate,false,false,true);mdf_reference_guard(rate,true,false,true);mdf_reference_guard(rate,true,true,true);mdf_post_stages(rate);}
    for(unsigned rate:{16000,24000,48000}) mdf_gain_slew(rate);
    for(unsigned rate:{16000,24000,48000}) {mdf_res_context_clock(rate);mdf_post_stages(rate,true);}
    for(unsigned rate:{16000,24000,48000}) for(bool res:{false,true})
      for(bool wrong:{false,true}) mdf_unrelated_silence(rate,res,wrong);
    for(unsigned rate:{16000,24000,48000}) {mdf_native_conditioning(rate,true);mdf_native_conditioning(rate,false);}
    for(unsigned m:{1,2,4}) {subspace(false,m);subspace(true,m);}
    for(unsigned rate:{16000,24000,48000}) for(bool lowcut:{false,true}) for(bool res:{false,true}) {
      tap_delays(rate,lowcut,res,false);tap_delays(rate,lowcut,res,true);
      tap_delays(rate,lowcut,res,true,true);
    }
    for(unsigned rate:{16000,24000,48000}) for(unsigned m:{1,2,4}) mdf(rate,m);
    mdf_coherence_residual_candidate();
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
