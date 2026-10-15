#include "itu/audio_engine.hpp"
#include "echo_subspace.hpp"
#include "fixed_hop_adapter.hpp"
#include "aec_safety_mix.hpp"
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
    require(info.frame_index==frame && info.latency_samples==(rate==24000 ? 160u : 0u) && !info.linear_aec_available,"MDF clock/delay/tap contract");
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
  for(float v:linear) require(v==0,"unavailable linear tap not cleared");
  engine.reset();
  require(engine.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),info)==itu::Status::ok && info.frame_index==0,"MDF reset");
}
int main() {
  try {
    hop_adapter();
    aec_safety();
    for(unsigned m:{1,2,4}) {subspace(false,m);subspace(true,m);}
    for(unsigned rate:{16000,24000,48000}) for(bool lowcut:{false,true}) for(bool res:{false,true}) {
      tap_delays(rate,lowcut,res,false);tap_delays(rate,lowcut,res,true);
      tap_delays(rate,lowcut,res,true,true);
    }
    for(unsigned rate:{16000,24000,48000}) for(unsigned m:{1,2,4}) mdf(rate,m);
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
