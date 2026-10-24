#pragma once
#include "itu/audio_engine.hpp"
#include "spatial_frontend.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace itu {
// Capture geometry belongs before cancellation. The downstream engine receives
// one physical beam and owns AEC/NR/EQ/AGC, regardless of its backend or I/O rate.
class PreAecPipeline {
 public:
  explicit PreAecPipeline(const Config& c)
      : c_(c), core_(core_config(c),4096,true), aligned_(c.frame_size*c.microphones),
        beam_(c.frame_size), aec_(c.frame_size), input_(c.frame_size) {
    float maximum=0;
    for(unsigned ch=0;ch<c.microphones;++ch) maximum=std::max(maximum,c.beam_delays_samples[ch]);
    delay_=static_cast<unsigned>(std::ceil(maximum));
    size_=delay_+2;
    history_.resize(size_*c.microphones);
    if(c.spatial_mvdr || c.wind_noise_control)
      spatial_=std::make_unique<SpatialFrontend>(c.sample_rate,c.frame_size,c.microphones,
          c.spatial_mvdr,c.wind_noise_control,c.spatial_fixed_steering,c.spatial_continuous_covariance);
  }

  Status process(const float* mic,const float* ref,float* out,FrameInfo& info,
                 float* at,float* bt,float* lt,float* it) noexcept {
    const unsigned m=c_.microphones,h=c_.frame_size;
    double raw_power=0;
    for(unsigned i=0;i<h;++i) {
      for(unsigned ch=0;ch<m;++ch) {
        const float value=mic[i*m+ch];
        history_[head_*m+ch]=value;
        raw_power+=double(value)*value;
      }
      float sum=0;
      for(unsigned ch=0;ch<m;++ch) {
        const float delay=c_.beam_delays_samples[ch];
        const unsigned d=static_cast<unsigned>(delay);
        const float fraction=delay-d;
        const unsigned p=(head_+size_-d)%size_,q=(p+size_-1)%size_;
        const float value=(1-fraction)*history_[p*m+ch]+fraction*history_[q*m+ch];
        aligned_[i*m+ch]=value;
        sum+=value;
      }
      beam_[i]=sum/m;
      head_=(head_+1)%size_;
    }
    if(spatial_) {
      double render_power=0;
      if(ref) for(unsigned i=0;i<h*c_.reference_channels;++i) render_power+=double(ref[i])*ref[i];
      const bool far=ref && render_power/(h*c_.reference_channels)>c_.activity_floor;
      // Only past causal speech evidence is available before this frame's AEC.
      spatial_->process(aligned_.data(),nullptr,nullptr,beam_.data(),nullptr,nullptr,
          !far && previous_speech_<.2f);
    }
    // Always advance every diagnostic history, including when no tap is asked for.
    const auto status=core_.process(beam_.data(),h,ref,ref?h*c_.reference_channels:0,
        out,h,info,aec_.data(),nullptr,linear_.data(),input_.data());
    if(status!=Status::ok) return status;
    previous_speech_=std::max({info.speech_probability,info.linear_speech_probability,
                              info.adaptive_near_confidence});
    if(bt) std::copy(beam_.begin(),beam_.end(),bt);
    // Preserve the public buffer extents. These are explicitly documented
    // replicas of ONE post-BF AEC channel, not independent microphone residuals.
    for(unsigned i=0;i<h;++i) for(unsigned ch=0;ch<m;++ch) {
      if(at) at[i*m+ch]=aec_[i];
      if(it) it[i*m+ch]=input_[i];
    }
    if(lt) for(unsigned ch=0;ch<m;++ch) std::copy(linear_.begin(),linear_.end(),lt+ch*160);
    const unsigned spatial_delay=spatial_ ? spatial_->delay() : 0;
    const unsigned front_delay=delay_+spatial_delay;
    info.bf_delay_samples=front_delay;
    info.latency_samples+=front_delay;
    info.aec_input_delay_samples+=front_delay;
    info.aec_output_delay_samples+=front_delay;
    if(c_.aec) info.linear_aec_delay_samples+=static_cast<unsigned>(std::lround(double(front_delay)*16000/c_.sample_rate));
    info.microphone_rms=static_cast<float>(std::sqrt(raw_power/(h*m)));
    info.aec_capture_channels=1;
    info.beamforming_before_aec=true;
    if(spatial_) {
      info.spatial_delay_samples=spatial_delay;
      info.spatial_noise_updates=spatial_->noise_updates();
      info.spatial_covariance_updates=spatial_->updates();
      info.spatial_fallback_bins=spatial_->fallbacks();
      info.wind_score=spatial_->wind_score();
    }
    if(c_.diagnostic_stages) {
      auto& taps=*c_.diagnostic_stages;
      taps.core_delay_samples+=static_cast<unsigned>(std::ceil(double(front_delay)*taps.core_sample_rate/c_.sample_rate));
      taps.eq_delay_samples+=static_cast<unsigned>(std::ceil(double(front_delay)*taps.eq_sample_rate/c_.sample_rate));
    }
    return Status::ok;
  }

  std::size_t memory_bytes() const noexcept {
    return sizeof(*this)+(history_.capacity()+aligned_.capacity()+beam_.capacity()+aec_.capacity()+input_.capacity())*sizeof(float)
        +core_.working_memory_bytes()+(spatial_ ? spatial_->memory_bytes() : 0);
  }
 private:
  static Config core_config(Config c) {
    c.microphones=1;c.beamforming=false;c.beam_delays_samples={};
    c.spatial_mvdr=c.wind_noise_control=c.spatial_fixed_steering=c.spatial_continuous_covariance=false;
    c.rx_processing=c.rx_config.enabled=false;
    return c;
  }
  Config c_;
  AudioEngine core_;
  std::vector<float> history_,aligned_,beam_,aec_,input_;
  std::array<float,160> linear_{};
  std::unique_ptr<SpatialFrontend> spatial_;
  unsigned head_=0,size_=2,delay_=0;
  float previous_speech_=0;
};
}
