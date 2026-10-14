#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include "spatial_frontend.hpp"
#include "noise_transition.hpp"
#include "linear_phase_lowcut.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
void require(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
void steering_anchor() {
  constexpr unsigned h=160,m=4;
  itu::SpatialFrontend learned(16000,h,m,true,false),fixed(16000,h,m,true,false,true),continuous(16000,h,m,true,false,true,true);
  std::mt19937 random(907);std::normal_distribution<float> draw(0,1);
  const float interferer_gain[]={2.f,.1f,.1f,.1f};
  std::vector<float> input(h*m),a(h),b(h),c(h),target,observed,anchored,adapted;
  for(unsigned frame=0;frame<700;++frame) {
    for(unsigned i=0;i<h;++i) {
      const float near=frame<200 ? 0.f : .03f*draw(random);
      const float interference=frame<200 ? 0.f : .18f*draw(random);
      target.push_back(near);
      for(unsigned ch=0;ch<m;++ch) input[i*m+ch]=near+interference*interferer_gain[ch]+.03f*draw(random);
    }
    // Both frontends see identical PCM and speech/noise decisions. A strong
    // off-axis talker must not redefine the calibrated driver's direction.
    learned.process(input.data(),nullptr,nullptr,a.data(),nullptr,nullptr,frame<200);
    fixed.process(input.data(),nullptr,nullptr,b.data(),nullptr,nullptr,frame<200);
    continuous.process(input.data(),nullptr,nullptr,c.data(),nullptr,nullptr,frame<200);
    observed.insert(observed.end(),a.begin(),a.end());anchored.insert(anchored.end(),b.begin(),b.end());
    adapted.insert(adapted.end(),c.begin(),c.end());
  }
  double p=0,old_cross=0,new_cross=0,adapted_cross=0,frozen_error=0,adapted_error=0;
  for(unsigned i=350*h;i+fixed.delay()<anchored.size();++i) {
    p+=target[i]*target[i];old_cross+=target[i]*observed[i+learned.delay()];new_cross+=target[i]*anchored[i+fixed.delay()];
    adapted_cross+=target[i]*adapted[i+continuous.delay()];
    frozen_error+=std::pow(anchored[i+fixed.delay()]-target[i],2);
    adapted_error+=std::pow(adapted[i+continuous.delay()]-target[i],2);
  }
  const double old_gain=old_cross/p,new_gain=new_cross/p;
  std::cout<<"MVDR off-axis covariance driver_projection learned="<<old_gain<<" fixed="<<new_gain<<'\n';
  require(old_gain<.8,"off-axis steering negative control did not expose target loss");
  require(new_gain>.85 && new_gain<1.15,"fixed steering failed to preserve calibrated target");
  const double adapted_gain=adapted_cross/p,improvement=10*std::log10(frozen_error/adapted_error);
  std::cout<<"MPDR speech-time training driver_projection="<<adapted_gain<<" improvement_vs_frozen_db="<<improvement<<'\n';
  require(adapted_gain>.85 && adapted_gain<1.15,"mixture covariance cancelled calibrated target");
  require(improvement>8,"frozen noise VAD negative control did not expose off-axis leakage");
  require(fixed.updates()==200 && continuous.updates()==700 && continuous.noise_updates()==200,"mixture/noise training counts");
}
int main() {
  try {
    steering_anchor();
    for(unsigned rate:{16000,24000,48000}) {
      itu::LinearPhaseLowCut scalar(2,60,rate),block(2,60,rate);
      const unsigned hop=rate/100;
      std::mt19937 random(515); std::normal_distribution<float> draw(0,.05f);
      std::vector<float> input(hop),expected(hop),output(hop);
      double power=0,error=0;
      for(unsigned frame=0;frame<80;++frame) for(unsigned ch=0;ch<2;++ch) {
        for(unsigned i=0;i<hop;++i) {input[i]=draw(random);expected[i]=scalar.process(input[i],ch);}
        block.process_frame(input.data(),output.data(),hop,ch);
        for(unsigned i=0;i<hop;++i) {power+=expected[i]*expected[i];error+=std::pow(expected[i]-output[i],2);}
      }
      const double snr=10*std::log10(power/error);
      std::cout<<"FIR overlap-save rate="<<rate<<" scalar_equivalence_snr="<<snr<<'\n';
      require(snr>110,"block FIR phase/gain/history matches scalar");
    }
    for(unsigned rate:{16000,24000,48000}) for(unsigned m:{2,4}) {
      unsigned h=rate/100; itu::SpatialFrontend bf(rate,h,m,true,true,true,true);
      std::vector<float> in(h*m),out(h),linear(h),low(h),source,result;
      std::mt19937 random(4); std::normal_distribution<float> draw(0,.03);
      for(unsigned frame=0;frame<100;++frame) {
        for(unsigned i=0;i<h;++i) {float v=draw(random); source.push_back(v); for(unsigned ch=0;ch<m;++ch) in[i*m+ch]=v;}
        bf.process(in.data(),in.data(),in.data(),out.data(),linear.data(),low.data(),false);
        for(unsigned i=0;i<h;++i) require(out[i]==linear[i] && out[i]==low[i],"shared spatial branch weights/delay");
        result.insert(result.end(),out.begin(),out.end());
      }
      double power=0,error=0,bad=0;
      for(unsigned i=h;i+bf.delay()<result.size();++i) {power+=source[i]*source[i]; error+=std::pow(result[i+bf.delay()]-source[i],2); bad+=std::pow(result[i]-source[i],2);}
      double snr=10*std::log10(power/error); std::cout<<"WOLA rate="<<rate<<" m="<<m<<" delay="<<bf.delay()<<" identity_snr="<<snr<<'\n';
      require(snr>80 && bad>power,"WOLA reconstruction/delay negative control");
      require(bf.updates()==100 && bf.noise_updates()==0,"MPDR trained despite speech-only VAD");
    }
    itu::NoiseTransition state;
    require(state.observe(.1f,.9f,false,false)>0,"cold speech protected");
    for(unsigned i=0;i<60;++i) state.observe(.01f,0,false,false);
    require(state.state()==1,"noise learned");
    for(unsigned i=0;i<8;++i) state.observe(.1f,0,false,false);
    require(state.events()>0 && state.state()==2,"noise jump reacquired");
    require(state.observe(.1f,.9f,true,false)==0,"no startup dry leak on far-only");
    {
      constexpr unsigned h=160,m=4;
      itu::SpatialFrontend bf(16000,h,m,true,true);
      std::mt19937 random(40); std::normal_distribution<float> draw(0,.02);
      std::vector<float> input(h*m),output(h),target,das,actual;
      for(unsigned frame=0;frame<400;++frame) {
        for(unsigned i=0;i<h;++i) {
          float clean=frame>=150 ? .04f*std::sin(2*3.141592653589793*997*(frame*h+i)/16000) : 0.f;
          target.push_back(clean); float sum=0;
          for(unsigned ch=0;ch<m;++ch) {input[i*m+ch]=clean+draw(random)*(ch==0?6.f:1.f); sum+=input[i*m+ch];}
          das.push_back(sum/m);
        }
        bf.process(input.data(),nullptr,nullptr,output.data(),nullptr,nullptr,frame<150);
        actual.insert(actual.end(),output.begin(),output.end());
      }
      double before=0,after=0;
      for(unsigned i=200*h;i+bf.delay()<actual.size();++i) {
        before+=std::pow(das[i]-target[i],2); after+=std::pow(actual[i+bf.delay()]-target[i],2);
      }
      const double improvement=10*std::log10(before/after);
      std::cout<<"MVDR sensor-imbalance improvement_vs_DAS_db="<<improvement<<'\n';
      require(improvement>4,"MVDR disabled/ineffective negative control");
    }
    for(unsigned rate:{16000,24000,48000}) {
      itu::Config c; c.sample_rate=rate; c.frame_size=rate/100; c.reference_channels=2;
      c.noise_suppression=c.automatic_gain=false;
      itu::AudioEngine e(c); itu::FrameInfo f;
      std::vector<float> mic(c.frame_size),ref(c.frame_size*2),out(c.frame_size);
      ref[1]=.1f;
      require(e.process(mic.data(),mic.size(),ref.data(),c.frame_size,out.data(),out.size(),f)==itu::Status::invalid_length,"mono-length rejected in stereo engine");
      require(e.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),f)==itu::Status::ok && f.frame_index==0,"multi-render validation/reset");
      require(!e.config().reference_guard && !e.config().coherence_dtd,"mono-reference guard not misapplied");
      e.reset(); require(e.process(mic.data(),mic.size(),ref.data(),ref.size(),out.data(),out.size(),f)==itu::Status::ok && f.frame_index==0,"stereo reset");
    }
    itu_audio_config c; itu_audio_default_config(&c); itu_audio_extensions x; itu_audio_default_extensions(&x);
    char error[200]; x.reference_channels=2;
    auto* e=itu_audio_create_extended(&c,2,0,nullptr,&x,error,sizeof(error)); require(e,"C extension create"); itu_audio_destroy(e);
    x.version=2; require(!itu_audio_create_extended(&c,2,0,nullptr,&x,error,sizeof(error)),"bad extension version");
    std::cout<<"vehicle frontend contracts passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
