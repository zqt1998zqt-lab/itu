#pragma once
#include "fft.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace itu {
struct AecFusionFrame {
  float shadow_weight=0,primary_error_power=0,shadow_error_power=0;
  float reference_confidence=0;
  bool double_talk=false;
  unsigned updates=0,delay_samples=0,path_changes=0;
  bool delay_valid=false;
};

// A short adaptive tail follows a causal GCC-PHAT bulk-delay observation.
// The MDF clock/state and its exported linear tap remain independent. All
// storage, including FFT workspace and mirrored render history, is preallocated.
class AecFusion {
 public:
  AecFusion(unsigned rate,unsigned hop,float learning_rate,unsigned tail_ms)
      : rate_(rate),hop_(hop),stride_(rate/8000),
        taps_(std::max(16u,rate*tail_ms/1000+1)),
        ring_size_(detail::next_power_two(rate/2+taps_+hop)),
        learning_rate_(learning_rate),coefficients_(taps_),
        history_(2*ring_size_),shadow_(hop),fft_(kFft) {}

  void process(const float* capture,const float* render,bool adapt,
               bool protect_near=false) noexcept {
    double render_power=0;
    for(unsigned i=0;i<hop_;++i) {
      head_=(head_+ring_size_-1)&(ring_size_-1);
      history_[head_]=history_[head_+ring_size_]=render[i];
      render_power+=double(render[i])*render[i];
      decimated_capture_+=capture[i];decimated_render_+=render[i];
      if(++decimation_phase_==stride_) {
        capture_history_[detector_head_]=decimated_capture_/stride_;
        render_history_[detector_head_]=decimated_render_/stride_;
        detector_head_=(detector_head_+1)&(kFft-1);
        decimation_phase_=0;decimated_capture_=decimated_render_=0;
        ++observations_;
      }
    }
    reference_power_=float(render_power/hop_);
    ++frames_;
    const bool excited=reference_power_>1e-7f;
    if(excited && observations_>=kWindow && frames_%4==0) observe_delay();
    if(frames_-last_evidence_frame_>24) delay_valid_=false;

    // Observe this frame with the frozen predictor before deciding whether to
    // update it. A near-end onset cannot train itself out of the DTD evidence.
    double capture_power=0,prediction_power=0,cross=0,innovation_power=0;
    for(unsigned i=0;i<hop_;++i) {
      const float* x=history_.data()+head_+hop_-1-i+anchor_;
      double prediction=0;
      for(unsigned t=0;t<taps_;++t) prediction+=double(coefficients_[t])*x[t];
      const float error=capture[i]-float(prediction);
      shadow_[i]=error;
      capture_power+=double(capture[i])*capture[i];
      prediction_power+=prediction*prediction;
      cross+=capture[i]*prediction;
      innovation_power+=double(error)*error;
    }
    const double predictor_coherence=cross*cross/std::max(1e-20,capture_power*prediction_power);
    double_talk_=protect_near || (model_ready_ &&
        innovation_power>.05*capture_power && predictor_coherence<.90);
    updates_=0;
    const bool learn=adapt && excited && delay_valid_ && !double_talk_;
    if(learn) {
      const float mu=frames_-last_path_frame_<20 ? std::max(.35f,learning_rate_) : learning_rate_;
      for(unsigned i=0;i<hop_;++i) {
        const float* x=history_.data()+head_+hop_-1-i+anchor_;
        double prediction=0,norm=1e-7;
        for(unsigned t=0;t<taps_;++t) {
          prediction+=double(coefficients_[t])*x[t];norm+=double(x[t])*x[t];
        }
        // Pre-update (causal) error; no fitted block output is used to choose w.
        const float error=capture[i]-float(prediction);
        shadow_[i]=error;
        if(norm>1e-6) {
          const float step=mu*error/float(norm);
          for(unsigned t=0;t<taps_;++t)
            coefficients_[t]=std::clamp(coefficients_[t]+step*x[t],-2.f,2.f);
          ++updates_;
        }
      }
    }
    if(learn && innovation_power<.05*capture_power && prediction_power>1e-8)
      model_ready_=true;
  }

  void blend(float* primary,bool far_active,unsigned reference_state,
             unsigned near_hold) noexcept {
    double p=0,s=0,cross=0;
    for(unsigned i=0;i<hop_;++i) {
      p+=double(primary[i])*primary[i];s+=double(shadow_[i])*shadow_[i];
      cross+=double(primary[i])*shadow_[i];
    }
    primary_power_=float(p/hop_);shadow_power_=float(s/hop_);
    double_talk_=double_talk_ || near_hold>0;
    // A stale or worse short-tail branch must be able to leave completely.
    // Local delay evidence is faster than the one-second upstream monitor;
    // that monitor may still report the previous acoustic path during recovery.
    const bool trusted=far_active && delay_valid_ && reference_power_>1e-7f;
    const bool better=s<.98*p && p>1e-12;
    if(!trusted || double_talk_ || !better || (reference_state==2 && confidence_<.5f)) {
      shadow_weight_=0;
      return;
    }
    const float target=std::clamp(float((p-cross)/std::max(1e-20,p+s-2*cross)),0.f,.95f);
    shadow_weight_+=std::clamp(.35f*(target-shadow_weight_),-.20f,.15f);
    for(unsigned i=0;i<hop_;++i)
      primary[i]+=(shadow_[i]-primary[i])*shadow_weight_;
  }

  AecFusionFrame frame() const noexcept {
    return {shadow_weight_,primary_power_,shadow_power_,confidence_,double_talk_,
            updates_,delay_samples_,path_changes_,delay_valid_};
  }
  std::size_t memory_bytes() const noexcept {
    return sizeof(*this)+(coefficients_.capacity()+history_.capacity()+shadow_.capacity())*sizeof(float)+fft_.memory();
  }
 private:
  static constexpr unsigned kFft=4096,kWindow=512,kMaxLag=2000;
  void observe_delay() noexcept {
    for(unsigned i=0;i<kFft;++i) {
      const unsigned index=(detector_head_+i)&(kFft-1);
      capture_spectrum_[i]={i>=kFft-kWindow ? capture_history_[index] : 0.f,0.f};
      render_spectrum_[i]={render_history_[index],0.f};
    }
    fft_.run(capture_spectrum_.data());fft_.run(render_spectrum_.data());
    for(unsigned i=0;i<kFft;++i) {
      const auto product=capture_spectrum_[i]*std::conj(render_spectrum_[i]);
      capture_spectrum_[i]=product/std::max(1e-12f,std::abs(product));
    }
    capture_spectrum_[0]={};fft_.run(capture_spectrum_.data(),true);
    unsigned lag=0;float peak=0;double corr_power=0;
    for(unsigned i=0;i<=kMaxLag;++i) {
      const float value=std::abs(capture_spectrum_[i].real());
      corr_power+=double(value)*value;
      if(value>peak) {peak=value;lag=i;}
    }
    double cp=0,rp=0,cr=0;
    for(unsigned i=0;i<kWindow;++i) {
      const unsigned ci=(detector_head_+kFft-kWindow+i)&(kFft-1);
      const unsigned ri=(ci+kFft-lag)&(kFft-1);
      const double c=capture_history_[ci],r=render_history_[ri];
      cp+=c*c;rp+=r*r;cr+=c*r;
    }
    const float peak_ratio=peak/float(std::sqrt(corr_power/(kMaxLag+1)+1e-20));
    const double coherence=cr*cr/std::max(1e-20,cp*rp);
    if(peak_ratio<8.f || coherence<.12 || rp/kWindow<1e-8) return;
    if(candidate_count_ && std::abs(int(lag)-int(candidate_lag_))<=2) ++candidate_count_;
    else candidate_count_=1;
    candidate_lag_=lag;
    if(candidate_count_<2) return;
    confidence_=std::clamp((peak_ratio-6.f)/14.f,0.f,1.f);
    last_evidence_frame_=frames_;
    const unsigned delay=lag*stride_;
    const unsigned margin=rate_/250; // 4 ms before the bulk arrival.
    const unsigned anchor=delay>margin ? delay-margin : 0;
    if(!delay_valid_ || std::abs(int(anchor)-int(anchor_))>int(rate_/500)) {
      if(ever_locked_) ++path_changes_;
      anchor_=anchor;std::fill(coefficients_.begin(),coefficients_.end(),0.f);
      model_ready_=false;last_path_frame_=frames_;shadow_weight_=0;
    }
    ever_locked_=delay_valid_=true;delay_samples_=delay;
  }
  unsigned rate_,hop_,stride_,taps_,ring_size_;
  float learning_rate_;
  std::vector<float> coefficients_,history_,shadow_;
  detail::Fft fft_;
  std::array<float,kFft> capture_history_{},render_history_{};
  std::array<std::complex<float>,kFft> capture_spectrum_{},render_spectrum_{};
  unsigned head_=0,detector_head_=0,decimation_phase_=0,observations_=0;
  unsigned frames_=0,last_evidence_frame_=0,last_path_frame_=0;
  unsigned candidate_count_=0,candidate_lag_=0,anchor_=0,delay_samples_=0,path_changes_=0,updates_=0;
  float decimated_capture_=0,decimated_render_=0,reference_power_=0;
  float shadow_weight_=0,primary_power_=0,shadow_power_=0,confidence_=0;
  bool delay_valid_=false,ever_locked_=false,model_ready_=false,double_talk_=false;
};
} // namespace itu
