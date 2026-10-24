#pragma once
#include "fft.hpp"
#include "super_gaussian.hpp"
#include "ns_spectral_frame.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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
  explicit ImcraNoiseEstimator(unsigned bins,float b_min=1.66f)
      : bins_(bins),b_min_(b_min),smooth_(bins),conditional_(bins),minimum_(bins),conditional_min_(bins),
        sub_min_(bins),conditional_sub_min_(bins),noise_(bins),indicator_(bins),
        history_(bins*subwindows),conditional_history_(bins*subwindows) {}
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
  std::size_t excluded_bins() const noexcept {return excluded_bins_;}
  std::size_t noise_updates() const noexcept {return noise_updates_;}
  unsigned rotations() const noexcept {return rotations_;}
  std::size_t step_updates() const noexcept {return step_updates_;}
  void update(const float* power,const float* xi,const float* gamma,float step_recovery=0) noexcept {
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
      noise_[k]=std::max(1e-16f,next);
      noise_updates_+=a<.99999f;
    }
    if(++frames_in_subwindow_==subwindow_frames) {
      for(unsigned k=0;k<bins_;++k) {
        history_[head_*bins_+k]=sub_min_[k];
        conditional_history_[head_*bins_+k]=conditional_sub_min_[k];
        minimum_[k]=conditional_min_[k]=std::numeric_limits<float>::max();
        for(unsigned u=0;u<subwindows;++u) {
          minimum_[k]=std::min(minimum_[k],history_[u*bins_+k]);
          conditional_min_[k]=std::min(conditional_min_[k],conditional_history_[u*bins_+k]);
        }
        sub_min_[k]=smooth_[k];conditional_sub_min_[k]=conditional_[k];
      }
      head_=(head_+1)%subwindows;frames_in_subwindow_=0;++rotations_;
    }
  }
 private:
  float frequency(const float* p,unsigned k) const noexcept {
    return .25f*p[k?k-1:0]+.5f*p[k]+.25f*p[std::min(k+1,bins_-1)];
  }
  unsigned bins_,head_=0,frames_in_subwindow_=0,rotations_=0;
  float b_min_;
  bool initialized_=false;
  std::size_t excluded_bins_=0,noise_updates_=0;
  std::size_t step_updates_=0;
  std::vector<float> smooth_,conditional_,minimum_,conditional_min_,sub_min_,conditional_sub_min_,
      noise_,indicator_,history_,conditional_history_;
};

class OmlsaImcra {
 public:
  explicit OmlsaImcra(unsigned rate,float floor_db=-18.f,bool protect_start=true,
                      SpectralGainRule rule=SpectralGainRule::omlsa)
      : rate_(check_rate(rate)),hop_(rate_/100),n_(next_power_two(2*hop_)),bins_(n_/2+1),
        fft_(n_),noise_(bins_,rate_==16000?1.519743f:rate_==24000?1.450226f:1.448289f),window_(2*hop_),past_(hop_),overlap_(hop_),power_(bins_),
        gamma_(bins_),xi_(bins_),previous_h1_snr_(bins_,1.f),zeta_(bins_),gain_(bins_,1.f),noise_used_(bins_),probability_(bins_),
      local_(kernel(1)),global_(kernel(std::max(1u,unsigned(std::lround(470.*n_/rate_))))),
        spectrum_(n_),floor_gain_(std::pow(10.f,floor_db/20)),protect_start_(protect_start),rule_(rule) {
    if(!std::isfinite(floor_db) || floor_db>0 || floor_db< -40) throw std::invalid_argument("OM-LSA floor must be -40..0 dB");
    for(unsigned i=0;i<2*hop_;++i) window_[i]=std::sin(3.14159265358979323846*(i+.5)/(2*hop_));
  }
  unsigned delay() const noexcept {return hop_;}
  unsigned fft_size() const noexcept {return n_;}
  float mean_probability() const noexcept {return mean_probability_;}
  float mean_gain() const noexcept {return mean_gain_;}
  float noise_power() const noexcept {return noise_power_;}
  float bootstrap_seed() const noexcept {return bootstrap_seed_;}
  const ImcraNoiseEstimator& estimator() const noexcept {return noise_;}
  NsSpectralFrame spectral_frame() noexcept {
    return {rate_,n_,bins_,power_.data(),noise_used_.data(),xi_.data(),gamma_.data(),probability_.data(),gain_.data()};
  }
  void process(float* audio,NsSpectralControl* control=nullptr,float step_recovery=0) noexcept {
    for(unsigned i=0;i<hop_;++i) {
      spectrum_[i]={past_[i]*window_[i],0};
      spectrum_[i+hop_]={audio[i]*window_[i+hop_],0};past_[i]=audio[i];
    }
    std::fill(spectrum_.begin()+2*hop_,spectrum_.end(),std::complex<float>{});
    fft_.run(spectrum_.data());
    for(unsigned k=0;k<bins_;++k) power_[k]=std::max(1e-16f,std::norm(spectrum_[k]));
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
    mean_probability_=mean_gain_=noise_power_=0;
    for(unsigned k=0;k<bins_;++k) {
      const float prior=std::max(.005f,probability(average(zeta_,k,local_))*probability(average(zeta_,k,global_))*frame_probability);
      const float p=omlsa_posterior(prior,xi_[k],gamma_[k]);
      probability_[k]=p;noise_used_[k]=noise_.noise(k);
      // The optional super-Gaussian experiment changes H1 gain only. IMCRA,
      // Gaussian evidence for soft speech presence, floor and timebase stay
      // identical. This hybrid is not a super-Gaussian marginal likelihood.
      const float h1=std::max(floor_gain_,rule_==SpectralGainRule::omlsa ?
          omlsa_h1_gain(xi_[k],gamma_[k]):super_gaussian_h1_gain(xi_[k],gamma_[k]));
      gain_[k]=std::exp(p*std::log(h1)+(1-p)*std::log(floor_gain_));
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
    noise_.update(power_.data(),xi_.data(),gamma_.data(),step_recovery);
    for(unsigned k=0;k<bins_;++k) spectrum_[k]*=gain_[k];
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
};
}
