#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace itu::detail {
// Capture-only evidence. The observer never consumes a clean target, a file
// boundary, a scene label, a future frame, or the enhancer's output spectrum.
// Band minima are independent of the recursive IMCRA/DD speech posterior.
class OmlsaAdaptiveControl {
 public:
  OmlsaAdaptiveControl(unsigned rate,unsigned fft,unsigned bins)
      : rate_(rate),fft_(fft),bins_(bins),smooth_(bins),anchor_(bins),
        minimum_(bins),history_(bins*10),band_of_(bins) {
    for(unsigned k=0;k<bins_;++k) {
      const float hz=float(k)*rate_/fft_;
      band_of_[k]=std::min(23u,unsigned(std::max(0.f,hz-80.f)*24.f/7920.f));
    }

  }
  void observe(const float* power) noexcept {
    std::array<double,24> bands{};
    double total=0,excess=0,mid=0,mid_excess=0,noise=0,mid_noise=0;
    double spectrum_sum=0,spectrum_logs=0;
    unsigned spectrum_count=0;
    unsigned elevated=0,mid_elevated=0;
    const bool first=frames_==0;
    for(unsigned k=0;k<bins_;++k) {
      const float p=power[k];
      if(first) {
        smooth_[k]=anchor_[k]=minimum_[k]=p;
        for(unsigned h=0;h<10;++h) history_[h*bins_+k]=p;
      } else smooth_[k]=.8f*smooth_[k]+.2f*p;
      minimum_[k]=std::min(minimum_[k],smooth_[k]);
      bands[band_of_[k]]+=p;
      const float hz=float(k)*rate_/fft_;
      if(hz>=150.f && hz<=6000.f) {
        spectrum_sum+=p;spectrum_logs+=std::log(std::max(p,1e-16f));++spectrum_count;
      }
    }
    // A spectrally isolated low/mid tone at stream start is ambiguous. It
    // must not become a pause merely because the excess observer saw that
    // same tone while seeding its noise floor. This narrow-bin safeguard
    // does not grant a fullband dry bypass to coloured engine/road noise.
    unsigned peak=0;
    float peak_power=0;
    double speech_band_power=0;
    for(unsigned k=1;k<bins_;++k) {
      const float hz=float(k)*rate_/fft_;
      if(hz>=150.f && hz<=6000.f) speech_band_power+=power[k];
      if(hz>=250.f && hz<=1200.f && power[k]>peak_power) {
        peak=k;peak_power=power[k];
      }
    }
    double narrow_power=0;
    for(int d=-2;d<=2;++d) {
      const unsigned k=unsigned(std::clamp(int(peak)+d,0,int(bins_)-1));
      narrow_power+=power[k];
    }
    const float concentration=float(narrow_power/std::max(speech_band_power,1e-16));
    if(frames_<5 && concentration>.92f && peak_power>1e-10f)
      cold_tone_bin_=peak;
    const bool continuing=cold_tone_bin_>0 && concentration>.85f &&
        std::abs(int(peak)-int(cold_tone_bin_))<=1;
    if(cold_tone_bin_>0 && !continuing) cold_tone_bin_=0;
    tone_protected_=continuing;
    std::array<double,24> floors{};
    for(unsigned k=0;k<bins_;++k) floors[band_of_[k]]+=anchor_[k];
    for(unsigned b=0;b<24;++b) {
      // This is a current capture/anchor excess, not an OM-LSA posterior.
      const double e=std::max(0.,bands[b]-2.5*floors[b]);
      total+=bands[b];noise+=floors[b];excess+=e;
      elevated+=bands[b]>3.*std::max(floors[b],1e-16);
      // Exclude the rumble-dominated lowest band from the speech anchor.
      if(b>=1 && b<11) {
        mid+=bands[b];mid_noise+=floors[b];mid_excess+=e;
        mid_elevated+=bands[b]>3.*std::max(floors[b],1e-16);
      }
    }
    const float broad=std::clamp((float(elevated)-1.f)/4.f,0.f,1.f);
    const float mid_broad=std::clamp(float(mid_elevated)/3.f,0.f,1.f);
    const float all_score=std::clamp((float(excess/std::max(total,1e-16))-.15f)/.55f,0.f,1.f)*broad;
    const float mid_score=std::clamp((float(mid_excess/std::max(mid,1e-16))-.15f)/.55f,0.f,1.f)*mid_broad;
    current_speech_=std::max(mid_score,all_score);
    const float flatness=spectrum_count ? float(std::exp(spectrum_logs/spectrum_count)/
        std::max(1e-16,spectrum_sum/spectrum_count)):1.f;
    if(first) tonal_cold_start_=flatness<.02f;
    // Immediate onset protection, short release, no binary output gate.
    speech_=std::max(current_speech_,speech_*std::exp(-10.f/80.f));
    quiet_frames_=current_speech_<.12f ? std::min(1000u,quiet_frames_+1):0;
    pause_=frames_>=20 && quiet_frames_>=12 && speech_<.20f &&
        (!tonal_cold_start_ || speech_seen_) ?
        std::clamp((float(quiet_frames_)-11.f)/12.f,0.f,1.f):0.f;
    speech_seen_=speech_seen_ || current_speech_>.4f;
    frame_snr_db_=10.f*std::log10(std::max(1e-6,(total-noise)/std::max(noise,1e-16)));
    mid_snr_db_=10.f*std::log10(std::max(1e-6,(mid-mid_noise)/std::max(mid_noise,1e-16)));
    // Causal independent minima, with bounded upward anchor tracking. A pause
    // cannot be declared from a high recursive speech probability alone.
    for(unsigned k=0;k<bins_;++k) {
      const float target=std::max(1e-16f,smooth_[k]);
      const float rate=target<anchor_[k] ? .12f : (pause_>0.f ? .04f:.001f);
      const float bounded=std::min(target,anchor_[k]*1.25f);
      anchor_[k]+=rate*((target<anchor_[k] ? target:bounded)-anchor_[k]);
    }
    if(++subframe_==20) {
      for(unsigned k=0;k<bins_;++k) {
        history_[head_*bins_+k]=minimum_[k];
        float m=history_[k];
        for(unsigned h=1;h<10;++h) m=std::min(m,history_[h*bins_+k]);
        // Downward minima corrections only. An old minimum never forces a
        // sudden upward reset at the history rotation boundary.
        anchor_[k]=std::min(anchor_[k],std::max(1e-16f,2.f*m));
        minimum_[k]=smooth_[k];
      }
      head_=(head_+1)%10;subframe_=0;
    }
    ++frames_;
  }
  float speech() const noexcept {return speech_;}
  float current_speech() const noexcept {return current_speech_;}
  float pause() const noexcept {return pause_;}
  float frame_snr_db() const noexcept {return frame_snr_db_;}
  float mid_snr_db() const noexcept {return mid_snr_db_;}
  float anchor(unsigned k) const noexcept {return std::max(1e-16f,anchor_[k]);}
  bool cold_tonal_unconfirmed() const noexcept {return tonal_cold_start_ && !speech_seen_;}
  float narrow_tone_protection(unsigned k) const noexcept {
    if(!tone_protected_) return 0.f;
    const unsigned distance=unsigned(std::abs(int(k)-int(cold_tone_bin_)));
    return distance<=2 ? .98f:(distance==3 ? .70f:0.f);
  }
  float local_excess(unsigned k,const float* power) const noexcept {
    double p=0,n=0;
    for(int d=-1;d<=1;++d) {
      const unsigned j=unsigned(std::clamp(int(k)+d,0,int(bins_)-1));
      p+=power[j];n+=anchor_[j];
    }
    return float(std::max(0.,1.-n/std::max(p,1e-16)));
  }
 private:
  unsigned rate_,fft_,bins_,frames_=0,quiet_frames_=0,subframe_=0,head_=0;
  std::vector<float> smooth_,anchor_,minimum_,history_;
  std::vector<unsigned> band_of_;
  float current_speech_=0,speech_=0,pause_=0,frame_snr_db_=-60,mid_snr_db_=-60;
  bool tonal_cold_start_=false,speech_seen_=false;
  unsigned cold_tone_bin_=0;
  bool tone_protected_=false;
};
}
