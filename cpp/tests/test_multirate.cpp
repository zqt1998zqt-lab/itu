#include "itu/audio_engine.hpp"
#include "itu/reference_clock.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool b,const char* why) {if(!b) throw std::runtime_error(why);}
void rates() {
  for(unsigned rate:{16000,24000,48000}) for(unsigned m:{1,2,4}) {
    auto c=itu::vehicle_stream_config(); c.sample_rate=rate; c.frame_size=rate/100;
    c.native_fullband=true; c.microphones=m;
    c.aec=c.noise_suppression=c.automatic_gain=c.limiter=false;
    c.beamforming=m>1;
    itu::AudioEngine e(c); itu::FrameInfo f;
    std::vector<float> x(c.frame_size*m),o(c.frame_size),tap(c.frame_size*m),source,after;
    for(unsigned k=0;k<100;++k) {
      for(unsigned i=0;i<c.frame_size;++i) {
        const float v=.1f*std::sin(2*3.141592653589793*997*(k*c.frame_size+i)/rate);
        source.push_back(v);
        for(unsigned ch=0;ch<m;++ch) x[i*m+ch]=v;
      }
      require(e.process(x.data(),x.size(),nullptr,0,o.data(),o.size(),f,tap.data())==itu::Status::ok,"rate processing");
      require(f.aec_output_delay_samples==f.latency_samples,"AEC-off conversion tap delay");
      for(unsigned i=0;i<c.frame_size;++i) for(unsigned ch=0;ch<m;++ch)
        require(std::abs(tap[i*m+ch]-o[i])<1e-6f,"AEC-off conversion tap waveform");
      after.insert(after.end(),o.begin(),o.end());
    }
    double best_snr=-1; unsigned best_lag=0;
    for(unsigned d=0;d<rate/10;++d) {
      double p=0,error=0;
      for(unsigned i=rate/10;i+100<source.size() && i+d<after.size();++i) {
        p+=double(source[i])*source[i]; error+=std::pow(after[i+d]-source[i],2);
      }
      const double snr=10*std::log10(p/std::max(error,1e-30));
      if(snr>best_snr) {best_snr=snr; best_lag=d;}
    }
    std::cout<<"rate="<<rate<<" mics="<<m<<" conversion_snr="<<best_snr<<" best_lag="<<best_lag<<" reported="<<f.latency_samples<<'\n';
    require(best_snr>25,"resampled waveform/frame continuity");
    e.reset(); require(e.process(x.data(),x.size(),nullptr,0,o.data(),o.size(),f)==itu::Status::ok && f.frame_index==0,"rate reset");
    x.back()=NAN; require(e.process(x.data(),x.size(),nullptr,0,o.data(),o.size(),f)==itu::Status::non_finite,"rate invalid sample");
  }
}
void latency() {
  unsigned mismatches=0;
  for(unsigned rate:{16000,24000,48000}) for(unsigned mode=0;mode<13;++mode) {
#ifndef ITU_TEST_RNNOISE
    if(mode==4) continue;
#endif
#ifndef ITU_TEST_SPEEXDSP
    if(mode>=5) continue;
#endif
    itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.native_fullband=true;
    c.linear_phase_low_cut=false;
    c.aec=mode==1 || mode==3 || mode==6 || mode==8 || mode==10 || mode==12; c.noise_suppression=mode>=2; c.automatic_gain=c.limiter=false;
    if(mode==4) c.noise_backend=itu::NoiseBackend::rnnoise;
    if(mode==5 || mode==6) c.noise_backend=itu::NoiseBackend::speex;
    if(mode>=7) {
      c.hybrid_speech_ns=c.speech_preserving_low_cut=true;
      c.fullband_near_ns=mode==7;
      c.linear_phase_low_cut=mode<9 || mode>=11;
      c.low_cut_hz=1;
      if(mode>=11) c.high_snr_dry_mix=.85f;
    }
    itu::AudioEngine e(c); itu::FrameInfo f;
    std::vector<float> x(c.frame_size),r(c.frame_size),o(c.frame_size),source,after;
    std::uint32_t seed=17; float smooth=0;
    for(unsigned k=0;k<180;++k) {
      for(float& v:x) {seed=seed*1664525u+1013904223u; smooth=.85f*smooth+.15f*(int(seed>>16)-32768)/32768.f; v=smooth*.5f;}
      if(mode==4) for(unsigned i=0;i<c.frame_size;++i) {
        const double t=double(k*c.frame_size+i)/rate;
        const double phase=2*3.141592653589793*(170*t+3*std::sin(3*t));
        x[i]=float((.12+.06*std::sin(13*t))*(std::sin(phase)+.4*std::sin(2*phase)+.2*std::sin(3*phase)));
      }
      source.insert(source.end(),x.begin(),x.end());
      require(e.process(x.data(),x.size(),c.aec?r.data():nullptr,c.aec?r.size():0,o.data(),o.size(),f)==itu::Status::ok,"latency processing");
      after.insert(after.end(),o.begin(),o.end());
    }
    double best=-1; unsigned lag=0;
    for(unsigned d=0;d<rate/10;++d) {
      double cross=0,p=0,q=0;
      for(unsigned i=rate/2;i+rate/10<source.size();++i) {cross+=double(source[i])*after[i+d]; p+=double(source[i])*source[i]; q+=double(after[i+d])*after[i+d];}
      const double corr=cross/std::sqrt(p*q+1e-30); if(corr>best) {best=corr; lag=d;}
    }
    std::cout<<"latency rate="<<rate<<" mode="<<mode<<" measured="<<lag<<" reported="<<f.latency_samples<<" corr="<<best<<std::endl;
    if(mode>=7) {
      const unsigned expected=mode<9 || mode>=11 ? rate/100 : rate==16000 ? 96 : rate==24000 ? 168 : 336;
      require(f.ns_delay_samples==expected,"hybrid NS delay in I/O units");
      require(std::isfinite(f.fullband_ns_mix) && f.fullband_ns_mix>=0 && f.fullband_ns_mix<=1,
              "hybrid NS branch mix out of range");
      double cross=0,p=0,q=0;
      for(unsigned i=rate/2;i+rate/20<source.size();++i) {
        cross+=double(source[i])*after[i]; p+=double(source[i])*source[i]; q+=double(after[i])*after[i];
      }
      require(cross/std::sqrt(p*q+1e-30)<best-.2,"zero-delay negative control insensitive");
    }
    // Fullband APM includes frequency-dependent post-filter phase. Require
    // the peak within 0.1 ms of the fixed budget, rather than waveform identity.
    mismatches+=std::abs(int(lag)-int(f.latency_samples))>int(std::ceil(rate*.0001)) || best<=.3;
  }
  require(!mismatches,"native algorithmic delay mismatch");
}
void module_contracts() {
  for(unsigned rate:{16000,24000,48000}) for(unsigned m:{1,2,4}) {
    itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.microphones=m;
    c.native_fullband=true; c.beamforming=m>1; c.fixed_gain_db=8;
    c.noise_suppression=false; c.speech_gain_policy=false; c.limiter=false;
    auto off=c; off.automatic_gain=false;
    itu::AudioEngine e(c),negative(off); itu::FrameInfo f,g;
    const unsigned h=c.frame_size;
    std::vector<float> x(h*m),r(h),o(h),p(h),aec(h*m+2,7),bf(h+2,7),linear(160*m+2,7),conditioned(h*m+2,7);
    double a=0,b=0;
    for(unsigned k=0;k<80;++k) {
      for(unsigned i=0;i<h;++i) for(unsigned ch=0;ch<m;++ch)
        x[i*m+ch]=.01f*(1+ch*.1f)*std::sin(2*3.141592653589793*997*(k*h+i)/rate);
      require(e.process(x.data(),x.size(),r.data(),r.size(),o.data(),o.size(),f,
                        aec.data()+1,bf.data()+1,linear.data()+1,conditioned.data()+1)==itu::Status::ok,"module taps process");
      require(negative.process(x.data(),x.size(),r.data(),r.size(),p.data(),p.size(),g)==itu::Status::ok,"AGC control");
      if(k>=20) for(unsigned i=0;i<h;++i) {a+=double(p[i])*p[i]; b+=double(o[i])*o[i];}
    }
    for(const auto* tap:{&aec,&bf,&linear,&conditioned}) {
      require(tap->front()==7 && tap->back()==7,"tap buffer overrun");
      require(std::all_of(tap->begin()+1,tap->end()-1,[](float v){return std::isfinite(v) && v!=7;}),"tap incomplete");
    }
    const double gain=10*std::log10(b/a);
    std::cout<<"modules rate="<<rate<<" mics="<<m<<" agc_gain_db="<<gain<<std::endl;
    require(gain>7.9 && gain<8.1 && f.linear_aec_available,"AGC/tap contract");
    require(f.linear_aec_sample_rate==16000,"linear tap sample rate");
  }
}
void beam_delays() {
  {
    itu::Config c; c.sample_rate=24000; c.frame_size=240; c.beamforming=true;
    c.beam_delays_samples[0]=4096;
    c.aec=c.noise_suppression=c.automatic_gain=c.limiter=false;
    itu::AudioEngine e(c); itu::FrameInfo f;
    std::vector<float> input(240),output(240),after;
    for(unsigned k=0;k<25;++k) {
      input[0]=k==0 ? .5f : 0.f;
      require(e.process(input.data(),240,nullptr,0,output.data(),240,f)==itu::Status::ok,"maximum I/O BF delay rejected");
      after.insert(after.end(),output.begin(),output.end());
    }
    const auto peak=unsigned(std::max_element(after.begin(),after.end())-after.begin());
    require(f.latency_samples==4256 && peak==4256,"maximum BF delay scaled incorrectly");
  }
  for(unsigned rate:{16000,24000,48000}) for(unsigned m:{2,4}) {
    itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.microphones=m;
    c.native_fullband=c.beamforming=true;
    c.aec=c.noise_suppression=c.automatic_gain=c.limiter=false;
    const double step=rate*.00015,maximum=(m-1)*step;
    for(unsigned ch=0;ch<m;++ch) c.beam_delays_samples[ch]=float(std::max(0.,maximum-ch*step));
    auto wrong=c; wrong.beam_delays_samples={};
    itu::AudioEngine e(c),negative(wrong); itu::FrameInfo f,g;
    std::vector<float> x(c.frame_size*m),o(c.frame_size),p(c.frame_size),after,bad;
    for(unsigned k=0;k<100;++k) {
      for(unsigned i=0;i<c.frame_size;++i) for(unsigned ch=0;ch<m;++ch)
        x[i*m+ch]=.1f*std::sin(2*3.141592653589793*997*(k*c.frame_size+i-ch*step)/rate);
      require(e.process(x.data(),x.size(),nullptr,0,o.data(),o.size(),f)==itu::Status::ok,"BF fractional process");
      require(negative.process(x.data(),x.size(),nullptr,0,p.data(),p.size(),g)==itu::Status::ok,"BF control");
      after.insert(after.end(),o.begin(),o.end()); bad.insert(bad.end(),p.begin(),p.end());
    }
    double power=0,error=0,wrong_error=0;
    const unsigned conversion=g.latency_samples;
    for(unsigned n=rate/10;n+conversion<after.size();++n) {
      const double target=.1*std::sin(2*3.141592653589793*997*(n-maximum)/rate);
      power+=target*target; error+=std::pow(after[n+conversion]-target,2);
      wrong_error+=std::pow(bad[n+conversion]-target,2);
    }
    const double snr=10*std::log10(power/error),negative_snr=10*std::log10(power/wrong_error);
    std::cout<<"BF rate="<<rate<<" mics="<<m<<" snr="<<snr<<" wrong_delays_snr="<<negative_snr<<std::endl;
    require(snr>28 && snr>negative_snr+15,"nonzero fractional BF alignment");
    require(f.latency_samples==conversion+unsigned(std::ceil(maximum)),"BF delay units");
  }
}
void highband() {
  for(unsigned rate:{16000,24000,48000}) for(bool aec:{false,true}) {
    itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.native_fullband=true;
    c.aec=aec; c.noise_suppression=c.automatic_gain=c.limiter=false;
    c.linear_phase_low_cut=false; c.speech_preserving_low_cut=true;
    itu::AudioEngine e(c); itu::FrameInfo f;
    std::vector<float> x(c.frame_size),o(c.frame_size),r(c.frame_size),source,after;
    const unsigned hz=rate==48000 ? 18000 : rate==24000 ? 10000 : 6000;
    for(unsigned k=0;k<100;++k) {
      for(unsigned i=0;i<c.frame_size;++i) x[i]=float(.1*std::sin(2*3.141592653589793*hz*(k*c.frame_size+i)/rate));
      source.insert(source.end(),x.begin(),x.end());
      require(e.process(x.data(),x.size(),aec?r.data():nullptr,aec?r.size():0,o.data(),o.size(),f)==itu::Status::ok,"highband process");
      after.insert(after.end(),o.begin(),o.end());
    }
    double p=0,q=0;
    for(unsigned n=rate/10;n+f.latency_samples<after.size();++n) {
      p+=double(source[n])*source[n]; q+=double(after[n+f.latency_samples])*after[n+f.latency_samples];
    }
    const double gain=10*std::log10(q/p);
    std::cout<<"highband rate="<<rate<<" aec="<<aec<<" hz="<<hz<<" gain_db="<<gain<<std::endl;
    require(gain>-.5 && gain<.5,"highband silently discarded");
  }
}
void reference_protection() {
  for(unsigned rate:{16000,24000,48000}) {
    auto c=itu::vehicle_stream_config(); c.sample_rate=rate; c.frame_size=rate/100;
    c.linear_phase_low_cut=c.noise_suppression=c.automatic_gain=c.limiter=false;
    c.linear_speech_protection=c.analyze_linear_noise=false;
    auto off=c; off.reference_guard=false;
    auto bypass=c; bypass.aec=false;
    itu::AudioEngine e(c),negative(off),conversion(bypass); itu::FrameInfo f,g,q;
    std::vector<float> x(c.frame_size),r(c.frame_size),o(c.frame_size),p(c.frame_size),v(c.frame_size),source,after;
    std::uint32_t a=17,b=43;
    unsigned unreliable=0,recovered=0; double difference=0;
    for(unsigned k=0;k<1000;++k) {
      for(unsigned i=0;i<c.frame_size;++i) {
        a=a*1664525u+1013904223u; b=b*1664525u+1013904223u;
        x[i]=.15f*(int(a>>16)-32768)/32768.f;
        r[i]=k<500 ? .15f*(int(b>>16)-32768)/32768.f : x[i];
      }
      require(conversion.process(x.data(),x.size(),nullptr,0,v.data(),v.size(),q)==itu::Status::ok,"reference fallback target");
      source.insert(source.end(),v.begin(),v.end());
      require(e.process(x.data(),x.size(),r.data(),r.size(),o.data(),o.size(),f)==itu::Status::ok,"reference guard process");
      require(negative.process(x.data(),x.size(),r.data(),r.size(),p.data(),p.size(),g)==itu::Status::ok,"reference guard control");
      after.insert(after.end(),o.begin(),o.end());
      if(k>=350 && k<450) {
        unreliable+=f.reference_state==2;
        for(unsigned i=0;i<c.frame_size;++i) difference+=std::pow(o[i]-p[i],2);
      }
      if(k>=850) recovered+=f.reference_state==1;
    }
    double power=0,error=0;
    for(unsigned n=rate*4;n<rate*4.4;++n) {
      power+=double(source[n+q.latency_samples])*source[n+q.latency_samples];
      error+=std::pow(after[n+f.latency_samples]-source[n+q.latency_samples],2);
    }
    const double snr=10*std::log10(power/std::max(error,1e-30));
    std::cout<<"protection rate="<<rate<<" unreliable="<<unreliable<<" recovered="<<recovered<<" fallback_snr="<<snr<<" control_difference="<<difference<<std::endl;
    require(unreliable==100 && recovered==150,"reference mismatch/recovery policy missing");
    require(snr>25 && difference>1e-5,"reference fallback not applied to audio");
  }
}
void intermittent_taps() {
  auto c=itu::vehicle_stream_config(); c.sample_rate=24000; c.frame_size=240;
  itu::AudioEngine always(c),intermittent(c); itu::FrameInfo f,g;
  std::vector<float> x(240),r(240),o(240),p(240),a(240),b(240),i(240),j(240),u(240),v(240);
  for(unsigned k=0;k<250;++k) {
    for(unsigned n=0;n<240;++n) x[n]=.1f*std::sin((k*240+n)*.14f);
    const bool request=k%7==0;
    require(always.process(x.data(),240,r.data(),240,o.data(),240,f,a.data(),u.data(),nullptr,i.data())==itu::Status::ok,"continuous taps");
    require(intermittent.process(x.data(),240,r.data(),240,p.data(),240,g,request?b.data():nullptr,
            request?v.data():nullptr,nullptr,request?j.data():nullptr)==itu::Status::ok,"intermittent taps");
    require(o==p,"requesting taps changed audio");
    if(request) require(a==b && i==j && u==v,"optional tap histories skipped frames");
  }
}
void protected_ns_noise_control() {
  for(unsigned rate:{16000,24000,48000}) {
    auto c=itu::vehicle_stream_config(); c.sample_rate=rate; c.frame_size=rate/100;
    c.aec=c.automatic_gain=c.limiter=false;
    itu::AudioEngine e(c); itu::FrameInfo f;
    std::vector<float> x(c.frame_size),o(c.frame_size);
    std::uint32_t seed=123; double input=0,output=0;
    for(unsigned k=0;k<600;++k) {
      for(float& value:x) {seed=seed*1664525u+1013904223u; value=.05f*(int(seed>>16)-32768)/32768.f;}
      require(e.process(x.data(),x.size(),nullptr,0,o.data(),o.size(),f)==itu::Status::ok,"protected NS noise process");
      if(k>=400) for(unsigned i=0;i<c.frame_size;++i) {input+=double(x[i])*x[i]; output+=double(o[i])*o[i];}
    }
    const double attenuation=10*std::log10(input/std::max(output,1e-30));
    std::cout<<"protected NS rate="<<rate<<" pure_noise_attenuation="<<attenuation<<std::endl;
    require(attenuation>=6,"speech protection bypassed noise suppression");
  }
}
void clocks() {
  for(unsigned rate:{16000,24000,48000}) for(double ppm:{-300.,0.,300.}) {
    itu::ReferenceClock c(rate); const unsigned h=rate/100;
    std::vector<float> render(h),out(h); std::uint64_t written=0;
    // Actual event timestamps and PCM sample count are generated from independent clocks.
    const double render_rate=rate*(1+ppm*1e-6);
    for(unsigned k=0;k<10000;++k) {
      const double t=k*.01;
      if(k%50==0) {
        const auto ns=std::int64_t(1000000000LL+std::llround(t*1e9));
        require(c.observe_capture({std::uint64_t(std::llround(t*rate)),ns}),"capture timestamp rejected");
        require(c.observe_render({std::uint64_t(std::llround(t*render_rate)),ns}),"render timestamp rejected");
      }
      const auto available=std::uint64_t(std::floor((t+.08)*render_rate));
      while(written+h<=available) {
        for(unsigned i=0;i<h;++i) render[i]=.2f*std::sin(2*3.141592653589793*997*(written+i)/render_rate);
        require(c.push(render.data(),h),"clock FIFO overflow"); written+=h;
      }
      require(c.pull(out.data(),h),"clock FIFO underrun");
    }
    const auto info=c.info();
    std::cout<<"clock rate="<<rate<<" injected_ppm="<<ppm<<" fitted_ppm="<<info.drift_ppm<<" queued="<<info.queued_samples<<std::endl;
    require(info.locked && std::abs(info.drift_ppm-ppm)<4,"clock fit wrong");
    require(!info.underruns && !info.overruns && info.queued_samples<rate*.2,"clock buffer unstable");
    require(!c.observe_capture({0,1000000000LL}),"backward timestamp accepted");
    c.reset(); require(!c.info().locked && !c.info().queued_samples,"clock epoch not reset");
    require(!c.pull(out.data(),h) && c.info().underruns==1,"empty negative control");
    render.back()=NAN; require(!c.push(render.data(),h),"invalid render accepted");
  }
}
void clock_waveform() {
  for(unsigned rate:{16000,24000,48000}) for(double ppm:{-300.,300.}) {
    itu::ReferenceClock corrected(rate),negative(rate);
    const unsigned h=rate/100; const double rr=rate*(1+ppm*1e-6);
    std::vector<float> input(h),a(h),b(h);
    std::uint64_t written=0;
    double aa=0,as=0,ac=0,bb=0,bs=0,bc=0; unsigned measured=0;
    for(unsigned k=0;k<10000;++k) {
      const double t=k*.01;
      if(k%50==0) {
        const auto ns=std::int64_t(1000000000LL+std::llround(t*1e9));
        corrected.observe_capture({std::uint64_t(std::llround(t*rate)),ns});
        corrected.observe_render({std::uint64_t(std::llround(t*rr)),ns});
      }
      const auto available=std::uint64_t(std::floor((t+.08)*rr));
      while(written+h<=available) {
        for(unsigned i=0;i<h;++i) input[i]=.2f*std::sin(2*3.141592653589793*997*(written+i)/rr);
        require(corrected.push(input.data(),h) && negative.push(input.data(),h),"waveform push"); written+=h;
      }
      require(corrected.pull(a.data(),h) && negative.pull(b.data(),h),"waveform pull");
      if(k>=8000) for(unsigned i=0;i<h;++i) {
        const double phase=2*3.141592653589793*997*(k*h+i)/rate;
        const double sn=std::sin(phase),cs=std::cos(phase);
        aa+=double(a[i])*a[i]; as+=a[i]*sn; ac+=a[i]*cs;
        bb+=double(b[i])*b[i]; bs+=b[i]*sn; bc+=b[i]*cs; ++measured;
      }
    }
    // A fixed AEC path can absorb phase/offset, but cannot absorb clock drift.
    const double retained_a=2*(as*as+ac*ac)/(measured*aa);
    const double retained_b=2*(bs*bs+bc*bc)/(measured*bb);
    std::cout<<"clock waveform rate="<<rate<<" ppm="<<ppm<<" corrected="<<retained_a<<" uncorrected="<<retained_b<<std::endl;
    require(retained_a>.98 && retained_b<.05,"ASRC ineffective versus negative control");
  }
}
}
int main() {
  try {rates(); latency(); module_contracts(); beam_delays(); highband(); reference_protection(); intermittent_taps(); protected_ns_noise_control(); clocks(); clock_waveform(); std::cout<<"PASS multirate and clock groups\n";}
  catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n'; return 1;}
}
