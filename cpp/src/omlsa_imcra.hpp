#pragma once
#include "fft.hpp"
#include "super_gaussian.hpp"
#include "ns_spectral_frame.hpp"
#include "complex_spectral_control.hpp"
#include "itu/omlsa_config.hpp"
#include "omlsa_adaptive_control.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace itu::detail {
// Independent implementation of Cohen's causal IMCRA (2003) and OM-LSA.
// Equations: Cohen/Gannot, Spectral Enhancement Methods (2008), 44.29--41,
// 44.67--89 and Table 44.5. No third-party MATLAB implementation is included.
// The 20 ms sine WOLA / 10 ms hop is an engineering operating point, rather
// than the paper's 32/8 ms example. All workspace is allocated at construction.
// The normalized [.25,.5,.25] frequency kernel is a deliberate engineering
// variant (not MATLAB's symmetric hanning(3)). B_min was measured with this
// window/kernel at each rate by tools/calibrate_imcra_minima.py, seeds 718--720.
// Local/global gain confidence uses current causal DD evidence for fast attack.
inline double omlsa_e1(double x) noexcept {
  x=std::max(x,1e-12);
  if(x<=1.) {
    double term=1.,sum=0.;
    for(unsigned k=1;k<=40;++k) {
      term*=-x/k;
      const double add=term/k;sum+=add;
      if(std::abs(add)<1e-15) break;
    }
    return -0.5772156649015328606-std::log(x)-sum;
  }
  if(x>80.) return 0.;
  double b=x+1.,c=1e30,d=1./b,h=d;
  for(unsigned k=1;k<=100;++k) {
    const double a=-double(k)*k;b+=2.;
    d=1./(a*d+b);c=b+a/c;
    const double change=c*d;h*=change;
    if(std::abs(change-1.)<1e-12) break;
  }
  return std::exp(-x)*h;
}
inline float omlsa_posterior(float prior,float xi,float gamma) noexcept {
  if(prior<=0) return 0;
  if(prior>=1) return 1;
  const double odds=(1.-prior)/prior*(1.+xi)*std::exp(-double(gamma)*xi/(1.+xi));
  return float(1./(1.+odds));
}
inline float omlsa_h1_gain(float xi,float gamma) noexcept {
  const double w=double(xi)/(1.+xi),v=std::max(1e-12,w*gamma);
  // A passive enhancer: prohibit noise amplification and numerical overflow.
  return float(std::min(1.,w*std::exp(.5*omlsa_e1(v))));
}

class ImcraNoiseEstimator {
 public:
  static constexpr unsigned subwindows=8,subwindow_frames=12;
  static constexpr float alpha_s=.9f,alpha_d=.85f,beta=1.47f;
  explicit ImcraNoiseEstimator(unsigned bins,float b_min=1.66f,
                               float minima_rebuild_slew_db=0.f,
                               unsigned unconditional_history_depth=8u,
                               unsigned conditional_history_depth=8u)
      : bins_(bins),b_min_(b_min),smooth_(bins),conditional_(bins),minimum_(bins),conditional_min_(bins),
        sub_min_(bins),conditional_sub_min_(bins),noise_(bins),indicator_(bins),
        history_(bins*subwindows),conditional_history_(bins*subwindows),
        minimum_rebuild_jump_db_(bins),conditional_rebuild_jump_db_(bins),
        minima_rebuild_slew_db_(minima_rebuild_slew_db),
        unconditional_history_depth_(unconditional_history_depth),
        conditional_history_depth_(conditional_history_depth) {
    if(unconditional_history_depth_<1 || unconditional_history_depth_>subwindows ||
       conditional_history_depth_<1 || conditional_history_depth_>subwindows)
      throw std::invalid_argument("IMCRA history depths must be 1..8");
  }
  void initialize(const float* power,float seed=1.f) noexcept {
    for(unsigned k=0;k<bins_;++k) {
      smooth_[k]=conditional_[k]=frequency(power,k);
      minimum_[k]=sub_min_[k]=smooth_[k]*seed;
      conditional_min_[k]=conditional_sub_min_[k]=conditional_[k]*seed;
      // Store the uncorrected noise average. beta is applied exactly once.
      noise_[k]=std::max(1e-16f,power[k]*seed/beta);
      for(unsigned u=0;u<subwindows;++u) {
        history_[u*bins_+k]=minimum_[k];
        conditional_history_[u*bins_+k]=conditional_min_[k];
      }
    }
    initialized_=true;
  }
  float noise(unsigned k) const noexcept {return std::max(1e-16f,beta*noise_[k]);}
  bool initialized() const noexcept {return initialized_;}
  float indicator(unsigned k) const noexcept {return indicator_[k];}
  float smooth(unsigned k) const noexcept {return smooth_[k];}
  float conditional(unsigned k) const noexcept {return conditional_[k];}
  float minimum(unsigned k) const noexcept {return minimum_[k];}
  float conditional_minimum(unsigned k) const noexcept {return conditional_min_[k];}
  float history_minimum(unsigned k) const noexcept {
    float value=std::numeric_limits<float>::max();
    for(unsigned u=0;u<unconditional_history_depth_;++u)
      value=std::min(value,history_[recent_history_index(u)*bins_+k]);
    return value;
  }
  float conditional_history_minimum(unsigned k) const noexcept {
    float value=std::numeric_limits<float>::max();
    for(unsigned u=0;u<conditional_history_depth_;++u)
      value=std::min(value,conditional_history_[recent_history_index(u)*bins_+k]);
    return value;
  }
  float minimum_rebuild_jump_db(unsigned k) const noexcept {
    return minimum_rebuild_jump_db_[k];
  }
  float conditional_rebuild_jump_db(unsigned k) const noexcept {
    return conditional_rebuild_jump_db_[k];
  }
  std::size_t excluded_bins() const noexcept {return excluded_bins_;}
  std::size_t noise_updates() const noexcept {return noise_updates_;}
  unsigned rotations() const noexcept {return rotations_;}
  unsigned unconditional_history_depth() const noexcept {return unconditional_history_depth_;}
  unsigned conditional_history_depth() const noexcept {return conditional_history_depth_;}
  std::size_t step_updates() const noexcept {return step_updates_;}
  void update(const float* power,const float* xi,const float* gamma,float step_recovery=0,
              const float* speech_hold=nullptr,float update_scale=1.f,
              float pause_confidence=0.f,const float* independent_noise=nullptr) noexcept {
    // First iteration: unconditional time/frequency smoothing and minima.
    for(unsigned k=0;k<bins_;++k) {
      smooth_[k]=alpha_s*smooth_[k]+(1-alpha_s)*frequency(power,k);
      minimum_[k]=std::min(minimum_[k],smooth_[k]);
      sub_min_[k]=std::min(sub_min_[k],smooth_[k]);
      const float denominator=std::max(1e-16f,b_min_*minimum_[k]);
      indicator_[k]=power[k]<4.6f*denominator && smooth_[k]<1.67f*denominator ? 1.f:0.f;
      excluded_bins_+=indicator_[k]==0.f;
    }
    // Second iteration: frequency smoothing EXCLUDES bins identified as speech.
    // With no eligible neighbour, retain the previous conditional spectrum.
    for(unsigned k=0;k<bins_;++k) {
      float numerator=0,denominator=0;
      for(int offset=-1;offset<=1;++offset) {
        const unsigned j=unsigned(std::clamp(int(k)+offset,0,int(bins_)-1));
        const float weight=(offset==0?.5f:.25f)*indicator_[j];
        numerator+=weight*power[j];denominator+=weight;
      }
      if(denominator>0)
        conditional_[k]=alpha_s*conditional_[k]+(1-alpha_s)*numerator/denominator;
      conditional_min_[k]=std::min(conditional_min_[k],conditional_[k]);
      conditional_sub_min_[k]=std::min(conditional_sub_min_[k],conditional_[k]);
      const float d=std::max(1e-16f,b_min_*conditional_min_[k]);
      const float g=power[k]/d,z=smooth_[k]/d;
      const float prior=z<1.67f ? std::clamp((g-1.f)/2.f,0.f,1.f):1.f;
      const float p=omlsa_posterior(prior,xi[k],gamma[k]);
      float a=alpha_d+(1-alpha_d)*p;
      const float previous=noise_[k];
      float next=a*previous+(1-a)*power[k];
      // An independent persistent broadband-step observer may reopen a frozen
      // upward recursion. It never resets minima or consumes reference labels.
      if(step_recovery>0 && gamma[k]>2.f) {
        const float bounded=std::min(a,1.f-.05f*step_recovery);
        // A false event must not learn an arbitrarily loud harmonic in one
        // frame. Bound only the extra recursion; preserve the original PSD
        // update exactly when recovery is disabled or the estimator is open.
        next+=(a-bounded)*(std::min(power[k],4.f*previous)-previous);
        step_updates_+=bounded<a;a=bounded;
      }
      // Only slow upward noise learning on independently supported speech.
      // Downward adaptation and the original conditional-minimum posterior
      // remain intact. A nonzero update scale prevents a frozen estimator.
      if(speech_hold && update_scale<1.f && next>previous && step_recovery<=0.f) {
        const float weight=1.f-(1.f-update_scale)*speech_hold[k];
        next=previous+weight*(next-previous);
      }
      if(pause_confidence>0.f && independent_noise) {
        // Reopen a frozen posterior only after independent capture evidence
        // confirms a pause. Bound both the PSD rise and the extra update.
        const float target=std::min(power[k],
            std::min(4.f*previous,2.f*independent_noise[k]/beta));
        if(target>previous)
          next=std::max(next,previous+.06f*pause_confidence*(target-previous));
      }
      noise_[k]=std::max(1e-16f,next);
      noise_updates_+=a<.99999f;
    }
    if(++frames_in_subwindow_==subwindow_frames) {
      const float max_upward_ratio=minima_rebuild_slew_db_>0.f ?
          std::pow(10.f,minima_rebuild_slew_db_/10.f) :
          std::numeric_limits<float>::infinity();
      for(unsigned k=0;k<bins_;++k) {
        history_[head_*bins_+k]=sub_min_[k];
        conditional_history_[head_*bins_+k]=conditional_sub_min_[k];
        const float previous_minimum=minimum_[k];
        const float previous_conditional=conditional_min_[k];
        float rebuilt_minimum=std::numeric_limits<float>::max();
        float rebuilt_conditional=std::numeric_limits<float>::max();
        for(unsigned u=0;u<unconditional_history_depth_;++u)
          rebuilt_minimum=std::min(
              rebuilt_minimum,history_[rebuild_history_index(u)*bins_+k]);
        for(unsigned u=0;u<conditional_history_depth_;++u)
          rebuilt_conditional=std::min(
              rebuilt_conditional,
              conditional_history_[rebuild_history_index(u)*bins_+k]);
        if(minima_rebuild_slew_db_>0.f) {
          rebuilt_minimum=std::min(
              rebuilt_minimum,std::max(1e-16f,previous_minimum)*max_upward_ratio);
          rebuilt_conditional=std::min(
              rebuilt_conditional,std::max(1e-16f,previous_conditional)*max_upward_ratio);
        }
        minimum_[k]=rebuilt_minimum;
        conditional_min_[k]=rebuilt_conditional;
        minimum_rebuild_jump_db_[k]=10.f*std::log10(
            std::max(1e-16f,rebuilt_minimum)/
            std::max(1e-16f,previous_minimum));
        conditional_rebuild_jump_db_[k]=10.f*std::log10(
            std::max(1e-16f,rebuilt_conditional)/
            std::max(1e-16f,previous_conditional));
        sub_min_[k]=smooth_[k];conditional_sub_min_[k]=conditional_[k];
      }
      head_=(head_+1)%subwindows;frames_in_subwindow_=0;++rotations_;
    }
  }
 private:
  unsigned recent_history_index(unsigned offset) const noexcept {
    return (head_ + subwindows - 1u - offset) % subwindows;
  }
  unsigned rebuild_history_index(unsigned offset) const noexcept {
    return (head_ + subwindows - offset) % subwindows;
  }
  float frequency(const float* p,unsigned k) const noexcept {
    return .25f*p[k?k-1:0]+.5f*p[k]+.25f*p[std::min(k+1,bins_-1)];
  }
  unsigned bins_,head_=0,frames_in_subwindow_=0,rotations_=0;
  float b_min_;
  bool initialized_=false;
  std::size_t excluded_bins_=0,noise_updates_=0;
  std::size_t step_updates_=0;
  std::vector<float> smooth_,conditional_,minimum_,conditional_min_,sub_min_,conditional_sub_min_,
      noise_,indicator_,history_,conditional_history_,
      minimum_rebuild_jump_db_,conditional_rebuild_jump_db_;
  float minima_rebuild_slew_db_=0.f;
  unsigned unconditional_history_depth_=subwindows,conditional_history_depth_=subwindows;
};

class OmlsaImcra {
 public:
  explicit OmlsaImcra(unsigned rate,float floor_db=-18.f,bool protect_start=true,
                      SpectralGainRule rule=SpectralGainRule::omlsa,
                      itu::OmlsaTuningConfig tuning={})
      : rate_(check_rate(rate)),hop_(rate_/100),n_(next_power_two(2*hop_)),bins_(n_/2+1),
        fft_(n_),noise_(bins_,rate_==16000?1.519743f:rate_==24000?1.450226f:1.448289f),window_(2*hop_),past_(hop_),overlap_(hop_),power_(bins_),
        gamma_(bins_),xi_(bins_),previous_h1_snr_(bins_,1.f),zeta_(bins_),gain_(bins_,1.f),noise_used_(bins_),probability_(bins_),
      local_(kernel(1)),global_(kernel(std::max(1u,unsigned(std::lround(470.*n_/rate_))))),
        spectrum_(n_),floor_gain_(std::pow(10.f,floor_db/20)),protect_start_(protect_start),rule_(rule),tuning_(tuning),
        speech_floor_gain_(std::pow(10.f,tuning.speech_floor_db/20)) {
    if(!std::isfinite(floor_db) || floor_db>0 || floor_db< -40) throw std::invalid_argument("OM-LSA floor must be -40..0 dB");
    if(!std::isfinite(tuning.noise_update_scale) || tuning.noise_update_scale<.1f || tuning.noise_update_scale>1.f ||
       !std::isfinite(tuning.speech_recovery) || tuning.speech_recovery<0.f || tuning.speech_recovery>1.f ||
       !std::isfinite(tuning.release_ms) || tuning.release_ms<0.f || tuning.release_ms>200.f ||
       !std::isfinite(tuning.speech_floor_db) || tuning.speech_floor_db< -40.f || tuning.speech_floor_db>0.f)
      throw std::invalid_argument("OM-LSA tuning requires noise scale .1..1, speech recovery 0..1, release 0..200 ms, speech floor -40..0 dB");
    adaptive_enabled_=tuning.adaptive_noise_tracking || tuning.high_snr_transparency ||
                      tuning.low_band_protection;
    tuning_enabled_=tuning.noise_update_scale<1.f || tuning.speech_recovery>0.f ||
                    tuning.speech_floor_db> -40.f || adaptive_enabled_;
    if(tuning_enabled_ && rule_!=SpectralGainRule::omlsa)
      throw std::invalid_argument("OM-LSA tuning is unavailable for other spectral gain rules");
    if(tuning_enabled_) {
      speech_hold_.resize(bins_);ungated_probability_.resize(bins_);
      previous_power_.resize(bins_);stable_frames_.resize(bins_);
      release_=tuning.release_ms>0.f ? std::exp(-10.f/tuning.release_ms):0.f;
    }
    if(adaptive_enabled_) {
      adaptive_=std::make_unique<OmlsaAdaptiveControl>(rate_,n_,bins_);
      independent_noise_.resize(bins_);previous_gain_.assign(bins_,1.f);
    }
    for(unsigned i=0;i<2*hop_;++i) window_[i]=std::sin(3.14159265358979323846*(i+.5)/(2*hop_));
  }
  unsigned delay() const noexcept {return hop_;}
  unsigned fft_size() const noexcept {return n_;}
  float mean_probability() const noexcept {return mean_probability_;}
  float mean_gain() const noexcept {return mean_gain_;}
  float noise_power() const noexcept {return noise_power_;}
  float bootstrap_seed() const noexcept {return bootstrap_seed_;}
  const ImcraNoiseEstimator& estimator() const noexcept {return noise_;}
  // Engine-provided evidence is computed from current/past capture PCM by
  // SpectralExcess. It is not an external VAD or a clean-reference label.
  void set_capture_evidence(float excess_fraction) noexcept {
    capture_evidence_=std::isfinite(excess_fraction) && excess_fraction>=0.f ?
        std::clamp(excess_fraction,0.f,1.f):-1.f;
  }
  NsSpectralFrame spectral_frame() noexcept {
    return {rate_,n_,bins_,power_.data(),noise_used_.data(),xi_.data(),gamma_.data(),probability_.data(),gain_.data()};
  }
  void process(float* audio,NsSpectralControl* control=nullptr,float step_recovery=0,
               ComplexSpectralControl* complex_control=nullptr) noexcept {
    for(unsigned i=0;i<hop_;++i) {
      spectrum_[i]={past_[i]*window_[i],0};
      spectrum_[i+hop_]={audio[i]*window_[i+hop_],0};past_[i]=audio[i];
    }
    std::fill(spectrum_.begin()+2*hop_,spectrum_.end(),std::complex<float>{});
    fft_.run(spectrum_.data());
    for(unsigned k=0;k<bins_;++k) power_[k]=std::max(1e-16f,std::norm(spectrum_[k]));
    if(adaptive_) {
      adaptive_->observe(power_.data());
      for(unsigned k=0;k<bins_;++k) independent_noise_[k]=adaptive_->anchor(k);
      const bool supplied=capture_evidence_>=0.f;
      const float evidence=supplied ? std::clamp((capture_evidence_-.25f)/.50f,0.f,1.f):
          adaptive_->current_speech();
      capture_speech_=std::max(evidence,capture_speech_*std::exp(-10.f/90.f));
      capture_quiet_frames_=evidence<.12f ? std::min(1000u,capture_quiet_frames_+1):0;
      capture_pause_=supplied && adaptive_frames_>=30 && capture_quiet_frames_>=10 &&
          capture_speech_<.25f ? std::clamp((float(capture_quiet_frames_)-9.f)/10.f,0.f,1.f):
          (supplied ? 0.f:adaptive_->pause());
      ++adaptive_frames_;
    }
    if(!noise_.initialized()) {
      // A bounded cold-start prior, using only this capture frame. Tonal/speech
      // starts must not immediately become the noise floor. It expires as the
      // finite minima history rotates; no waveform or oracle is held/read.
      double logsum=0,sum=0;unsigned count=0;
      for(unsigned k=1;k<bins_;++k) if(k*double(rate_)/n_>=150 && k*double(rate_)/n_<=6000) {
        logsum+=std::log(power_[k]);sum+=power_[k];++count;
      }
      const double flat=count?std::exp(logsum/count)/std::max(1e-16,sum/count):1.;
      bootstrap_seed_=protect_start_ && flat<.15 ? .05f:1.f;
      noise_.initialize(power_.data(),bootstrap_seed_);
    }
    float frame_zeta=0;
    for(unsigned k=0;k<bins_;++k) {
      gamma_[k]=std::min(1e8f,power_[k]/noise_.noise(k));
      // H1-conditional decision direction (44.41), not the uncertain output.
      xi_[k]=std::clamp(.92f*previous_h1_snr_[k]+.08f*std::max(gamma_[k]-1.f,0.f),1e-3f,1e8f);
      zeta_[k]=.7f*zeta_[k]+.3f*xi_[k];frame_zeta+=zeta_[k];
    }
    frame_zeta/=bins_;
    float frame_probability=0;
    if(frame_zeta>.1f) {
      if(frame_zeta>previous_frame_zeta_) {
        frame_probability=1;peak_zeta_=std::clamp(frame_zeta,1.f,10.f);
      } else frame_probability=probability(frame_zeta/peak_zeta_);
    }
    previous_frame_zeta_=frame_zeta;
    if(tuning_enabled_) {
      for(unsigned k=0;k<bins_;++k) {
        const float prior=std::max(.005f,probability(average(zeta_,k,local_))*
                                               probability(average(zeta_,k,global_)));
        ungated_probability_[k]=omlsa_posterior(prior,xi_[k],gamma_[k]);
      }
      for(unsigned k=0;k<bins_;++k) {
        unsigned neighbours=0;
        for(int offset=-1;offset<=1;++offset) {
          const unsigned j=unsigned(std::clamp(int(k)+offset,0,int(bins_)-1));
          neighbours+=ungated_probability_[j]>.75f && gamma_[j]>1.5f && xi_[j]>.15f;
        }
        const float change=std::abs(power_[k]-previous_power_[k])/
            std::max(std::max(power_[k],previous_power_[k]),1e-16f);
        stable_frames_[k]=change<.08f ? std::min(1000u,stable_frames_[k]+1):0;
        const float hz=float(k)*rate_/n_;
        const float evidence=hz>=150.f && hz<=7000.f && neighbours>=2 && stable_frames_[k]<8 ?
            std::clamp((ungated_probability_[k]-.60f)/.35f,0.f,1.f):0.f;
        float current_evidence=evidence;
        if(adaptive_ && tuning_.adaptive_noise_tracking) {
          // Statistical speech evidence on steady car noise cannot keep the
          // synthesis hold alive across an independently confirmed pause.
          current_evidence*=std::max(.15f,capture_speech_);
        }
        speech_hold_[k]=std::max(current_evidence,speech_hold_[k]*release_);
        if(adaptive_ && tuning_.adaptive_noise_tracking && capture_pause_>0.f)
          speech_hold_[k]*=1.f-.25f*capture_pause_;
        previous_power_[k]=power_[k];
      }
    }
    mean_probability_=mean_gain_=noise_power_=0;
    for(unsigned k=0;k<bins_;++k) {
      const float prior=std::max(.005f,probability(average(zeta_,k,local_))*probability(average(zeta_,k,global_))*frame_probability);
      const float p=omlsa_posterior(prior,xi_[k],gamma_[k]);
      probability_[k]=p;noise_used_[k]=noise_.noise(k);
      // The optional super-Gaussian experiment changes H1 gain only. IMCRA,
      // Gaussian evidence for soft speech presence, floor and timebase stay
      // identical. This hybrid is not a super-Gaussian marginal likelihood.
      float bin_floor=floor_gain_;
      if(adaptive_ && tuning_.adaptive_noise_tracking)
        bin_floor=floor_gain_*std::pow(10.f,-9.f*capture_pause_/20.f);
      const float h1=std::max(floor_gain_,rule_==SpectralGainRule::omlsa ?
          omlsa_h1_gain(xi_[k],gamma_[k]):super_gaussian_h1_gain(xi_[k],gamma_[k]));
      float synthesis_p=p;
      if(adaptive_ && tuning_.adaptive_noise_tracking)
        synthesis_p*=1.f-.85f*capture_pause_;
      gain_[k]=std::exp(synthesis_p*std::log(h1)+(1-synthesis_p)*std::log(bin_floor));
      if(tuning_enabled_ && tuning_.speech_recovery>0.f) {
        // Preserve weak continuations only up to their conditional H1 gain;
        // no speech hold can lift a noise bin above its H1 estimate. Keep
        // this synthesis probability out of the DD/noise evidence.
        const float protected_probability=std::max(synthesis_p,
            tuning_.speech_recovery*speech_hold_[k]);
        gain_[k]=std::exp(protected_probability*std::log(h1)+
                         (1.f-protected_probability)*std::log(bin_floor));
      }
      if(adaptive_) {
        const float hz=float(k)*rate_/n_;
        const float local_excess=adaptive_->local_excess(k,power_.data());
        if(tuning_.high_snr_transparency && capture_speech_>.35f && hz>=80.f) {
          // Only restore bins with current local SNR support; no broadband
          // dry bypass that would carry low-frequency rumble into pauses.
          const float clean=std::clamp((local_excess-.90f)/.08f,0.f,1.f);
          const float frame=std::clamp((adaptive_->mid_snr_db()-8.f)/10.f,0.f,1.f);
          gain_[k]+=clean*frame*(1.f-gain_[k]);
        }
        if(tuning_.low_band_protection && hz>=80.f && hz<400.f &&
           capture_speech_>.4f && adaptive_->mid_snr_db()>3.f) {
          const float edge=std::clamp((hz-80.f)/50.f,0.f,1.f)*
                           std::clamp((400.f-hz)/100.f,0.f,1.f);
          const float evidence=std::clamp((local_excess-.75f)/.20f,0.f,1.f)*
                               capture_speech_*edge;
          const float clean_gain=std::sqrt(std::clamp(local_excess,0.f,1.f));
          gain_[k]=std::max(gain_[k],gain_[k]+.6f*evidence*(clean_gain-gain_[k]));
        }
        if(capture_evidence_<0.f && adaptive_->cold_tonal_unconfirmed() && gamma_[k]>2.f)
          gain_[k]=std::max(gain_[k],std::sqrt(std::clamp(1.f-1.f/gamma_[k],0.f,1.f)));
        gain_[k]=std::max(gain_[k],adaptive_->narrow_tone_protection(k));
        // Continuous attack/release on synthesis only; PSD/DD remain separate.
        const float min_gain=tuning_.adaptive_noise_tracking ? floor_gain_*.35481339f:floor_gain_;
        const float desired=std::clamp(gain_[k],min_gain,1.f);
        const float alpha=desired>previous_gain_[k] ? .15f:.65f;
        gain_[k]=alpha*previous_gain_[k]+(1.f-alpha)*desired;
        previous_gain_[k]=gain_[k];
      }
      if(tuning_enabled_ && tuning_.speech_floor_db> -40.f) {
        const float hz=float(k)*rate_/n_;
        if(hz>=150.f && hz<=7000.f) {
          // Lift only bins already supported by the causal speech-hold evidence.
          // This synthesis-only floor cannot feed DD, H1 state or noise estimation.
          const float evidence=std::clamp(speech_hold_[k],0.f,1.f);
          const float bounded_floor=floor_gain_+
              evidence*(std::max(floor_gain_,speech_floor_gain_)-floor_gain_);
          gain_[k]=std::min(1.f,std::max(gain_[k],bounded_floor));
        }
      }
      previous_h1_snr_[k]=h1*h1*gamma_[k];
      mean_probability_+=p;mean_gain_+=gain_[k];noise_power_+=noise_.noise(k);
    }
    mean_probability_/=bins_;mean_gain_/=bins_;
    noise_power_/=bins_*float(hop_); // spectral power normalized to PCM units
    if(control) {
      control->observe(spectral_frame());
      mean_gain_=0;for(float g:gain_) mean_gain_+=g;mean_gain_/=bins_;
    }
    // Estimate t+1 only after this frame's gain uses the t noise estimate.
    noise_.update(power_.data(),xi_.data(),gamma_.data(),step_recovery,
                  tuning_enabled_ ? speech_hold_.data():nullptr,tuning_.noise_update_scale,
                  adaptive_ && tuning_.adaptive_noise_tracking ? capture_pause_:0.f,
                  adaptive_ ? independent_noise_.data():nullptr);
    for(unsigned k=0;k<bins_;++k) spectrum_[k]*=gain_[k];
    if(complex_control) complex_control->observe(
        {rate_,n_,bins_,spectrum_.data(),gain_.data()});
    for(unsigned k=1;k<n_/2;++k) spectrum_[n_-k]=std::conj(spectrum_[k]);
    fft_.run(spectrum_.data(),true);
    for(unsigned i=0;i<hop_;++i) {
      audio[i]=overlap_[i]+spectrum_[i].real()*window_[i];
      overlap_[i]=spectrum_[i+hop_].real()*window_[i+hop_];
    }
  }
 private:
  static unsigned check_rate(unsigned r) {
    if(r!=16000 && r!=24000 && r!=48000) throw std::invalid_argument("OM-LSA requires native 16/24/48 kHz");
    return r;
  }
  static std::vector<float> kernel(unsigned radius) {
    std::vector<float> result(2*radius+1);float sum=0;
    for(unsigned i=0;i<result.size();++i) {
      result[i]=.5f-.5f*std::cos(6.283185307179586*(i+1)/(result.size()+1));sum+=result[i];
    }
    for(float& x:result) x/=sum;
    return result;
  }
  float average(const std::vector<float>& values,unsigned k,const std::vector<float>& weights) const noexcept {
    float result=0;const int radius=int(weights.size()/2);
    for(unsigned i=0;i<weights.size();++i) result+=weights[i]*values[unsigned(std::clamp(int(k)+int(i)-radius,0,int(bins_)-1))];
    return result;
  }
  static float probability(float snr) noexcept {
    if(snr<=.1f) return 0;
    if(snr>=.316227766f) return 1;
    return std::log(snr/.1f)/1.1512925465f;
  }
  unsigned rate_,hop_,n_,bins_;
  Fft fft_;
  ImcraNoiseEstimator noise_;
  std::vector<float> window_,past_,overlap_,power_,gamma_,xi_,previous_h1_snr_,zeta_,gain_,noise_used_,probability_,local_,global_;
  std::vector<std::complex<float>> spectrum_;
  float floor_gain_,previous_frame_zeta_=0,peak_zeta_=1;
  float mean_probability_=0,mean_gain_=1,noise_power_=0,bootstrap_seed_=1;
  bool protect_start_;
  SpectralGainRule rule_;
  itu::OmlsaTuningConfig tuning_;
  float speech_floor_gain_;
  bool tuning_enabled_=false;
  bool adaptive_enabled_=false;
  std::unique_ptr<OmlsaAdaptiveControl> adaptive_;
  std::vector<float> independent_noise_,previous_gain_;
  float capture_evidence_=-1.f,capture_speech_=0.f,capture_pause_=0.f;
  unsigned capture_quiet_frames_=0,adaptive_frames_=0;
  float release_=0.f;
  std::vector<float> speech_hold_,ungated_probability_,previous_power_;
  std::vector<unsigned> stable_frames_;
};
}
