#pragma once
#include "itu/audio_engine.hpp"
#include "eq_processor.hpp"
#include "drc.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace itu {
// Native-hop PCM never enters a 160/256 audio FIFO. Only an independent
// 10ms control observer retains 160 past samples for the existing Guarded5
// policy. It emits gain decisions, not audio. Consequently this route retains
// the gain policy but uses explicit causal gain ramps rather than APM's
// fixed-10ms gain/limiter execution; it requires whole-chain quality validation.
class Native16PostPipeline {
 public:
  explicit Native16PostPipeline(const Config& c):c_(c),observer_(observer_config(c)),drc_(c.sample_rate,c.tx_drc) {
    if(c.sample_rate!=16000 || c.frame_size!=256 || c.microphones!=1 || c.aec || c.beamforming || c.rx_processing || c.rx_config.enabled)
      throw std::invalid_argument("native16 post requires mono 16kHz/256 TX without AEC/BF/RX");
    if(c.adaptive_gain || c.unified_eq_agc || c.output_eq.phase!=EqPhase::minimum)
      throw std::invalid_argument("native16 post supports Guarded5 and minimum-phase EQ only");
    if(c.noise_suppression && !c.ns_stream_process && !c.post_ns_pcm)
      throw std::invalid_argument("native16 NS requires actual stateful callback");
    delay_=c.ns_stream_process ? c.ns_stream_delay_samples : c.post_ns_pcm ? c.post_ns_delay_samples : 0;
    if(delay_!=0 && delay_!=256) throw std::invalid_argument("native16 NS delay must be 256");
    if(ParametricEqEnabled(c.output_eq)) eq_=std::make_unique<EqProcessor>(c.output_eq,c.sample_rate);
  }

  Status process(const float* mic,float* out,FrameInfo& info,const PostAecContext* context=nullptr) noexcept {
    info={};
    if(!mic || !out) return Status::invalid_pointer;
    if(failed_) {info.backend_error=-1016;return Status::backend_error;}
    try {
      for(unsigned i=0;i<256;++i) if(!std::isfinite(mic[i])) throw std::runtime_error("nonfinite input");
      info.pre_ns_rms=rms(mic,256);
      if(c_.ns_stream_process) {
        const auto mark=std::chrono::steady_clock::now();
        if(!c_.ns_stream_process(mic,ns_.data(),256)) throw std::runtime_error("native16 NS callback failed");
        info.ns_stream_roundtrip_ms=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-mark).count();
        ++applied_;
        info.ns_external_stream_enabled=true;
      } else if(c_.post_ns_pcm) {
        const auto start=frames_*256;
        if(start+256>c_.post_ns_pcm->size()) throw std::runtime_error("native16 PCM exhausted");
        std::copy_n(c_.post_ns_pcm->data()+start,256,ns_.data());
        ++applied_;info.ns_external_pcm_enabled=true;
      } else std::copy_n(mic,256,ns_.data());
      for(float sample:ns_) if(!std::isfinite(sample)) throw std::runtime_error("nonfinite NS output");
      info.post_ns_rms=rms(ns_.data(),256);
      if(c_.diagnostic_stages) {
        auto& t=*c_.diagnostic_stages;
        t.core_sample_rate=t.eq_sample_rate=16000;t.core_frame_size=t.eq_frame_size=256;
        t.core_delay_samples=t.agc_delay_samples=t.pre_eq_delay_samples=t.post_eq_delay_samples=delay_;
        t.eq_delay_samples=delay_;t.drc_delay_samples=delay_+drc_.delay_samples();
        t.experiment_ns_input_delay_samples=0;t.nr_selected_available=true;
        std::copy_n(mic,256,t.experiment_ns_input.data());
        std::copy_n(ns_.data(),256,t.nr_selected.data());std::copy_n(ns_.data(),256,t.nr_protected.data());
        std::copy_n(ns_.data(),256,t.pre_eq.data());
      }
      if(eq_) eq_->process(ns_.data(),256);
      if(c_.diagnostic_stages) std::copy_n(ns_.data(),256,c_.diagnostic_stages->post_eq.data());
      const bool far=context && context->far_active;
      for(unsigned i=0;i<256;++i) {
        float physical=mic[i];bool delayed_far=far;
        if(delay_) {
          physical=physical_delay_[i];physical_delay_[i]=mic[i];
          delayed_far=far_delay_[i];far_delay_[i]=far;
        }
        control_[control_count_]=std::clamp(physical,-1.f,1.f);
        control_ref_[control_count_]=delayed_far?.05f:0.f;
        if(++control_count_==160) {
          FrameInfo observed;
          if(observer_.process(control_.data(),160,control_ref_.data(),160,control_output_.data(),160,observed)!=Status::ok)
            throw std::runtime_error("Guarded5 control observer failed");
          control_info_=observed;control_count_=0;
          target_gain_=c_.automatic_gain ? std::pow(10.f,observed.applied_gain_db/20.f) : 1.f;
          gain_step_=(target_gain_-gain_)/160.f;gain_remaining_=160;
        }
        if(gain_remaining_) {gain_+=gain_step_;if(--gain_remaining_==0)gain_=target_gain_;}
        out[i]=ns_[i]*gain_;
      }
      if(c_.diagnostic_stages) std::copy_n(out,256,c_.diagnostic_stages->post_agc.data());
      drc_.process(out,256);
      if(c_.diagnostic_stages) std::copy_n(out,256,c_.diagnostic_stages->post_drc.data());
      for(unsigned i=0;i<256;++i) {
        if(c_.limiter && std::abs(out[i])>c_.output_ceiling) {out[i]=std::copysign(c_.output_ceiling,out[i]);++info.limited_samples;}
        if(!std::isfinite(out[i])) throw std::runtime_error("nonfinite post output");
      }
      info.frame_index=frames_++;info.microphone_rms=info.pre_ns_rms;info.output_rms=rms(out,256);
      info.speech_probability=control_info_.speech_probability;info.applied_gain_db=control_info_.applied_gain_db;
      info.gain_speech_qualified=control_info_.gain_speech_qualified;info.gain_control_state=control_info_.gain_control_state;
      info.gain_control_noise_power=control_info_.gain_control_noise_power;
      info.ns_external_applied_frames=applied_;info.ns_external_delay_samples=delay_;info.ns_delay_samples=delay_;
      info.tx_drc_delay_samples=drc_.delay_samples();info.latency_samples=delay_+drc_.delay_samples();
      info.far_active=far;info.post_aec_context_applied=context!=nullptr;
      return Status::ok;
    } catch(...) {failed_=true;info.backend_error=-1016;return Status::backend_error;}
  }

  std::size_t memory_bytes() const noexcept {return sizeof(*this)+observer_.working_memory_bytes()+drc_.memory_bytes()+(eq_?eq_->memory_bytes():0);}
 private:
  static bool ParametricEqEnabled(const EqConfig& c) {if(c.preamp_db!=0)return true;for(unsigned i=0;i<c.bands;++i)if(c.band[i].gain_db!=0)return true;return false;}
  static float rms(const float* p,unsigned n) {double sum=0;for(unsigned i=0;i<n;++i)sum+=double(p[i])*p[i];return std::sqrt(sum/n);}
  static Config observer_config(Config c) {
    c.frame_size=160;c.microphones=c.reference_channels=1;c.aec=c.beamforming=false;
    c.aec_backend=AecBackend::webrtc;c.noise_suppression=false;c.ns_stream_process={};c.ns_stream_delay_samples=0;
    c.post_ns_pcm.reset();c.post_ns_delay_samples=0;c.output_eq={};c.tx_drc.enabled=false;c.limiter=false;
    c.diagnostic_stages=nullptr;c.rx_processing=c.rx_config.enabled=false;
    c.spatial_mvdr=c.wind_noise_control=c.spatial_fixed_steering=c.spatial_continuous_covariance=false;
    c.reference_guard=c.correlation_delay=c.coherence_dtd=c.analyze_linear_noise=c.linear_speech_protection=false;
    c.speex_residual_suppression=c.speex_coherence_residual_suppression=c.speex_post_context=c.speex_multirender_context=false;
    c.post_aec_evidence_window=false;c.complex_pf_cpu_pcm.reset();c.complex_pf_residual_capture=c.complex_pf_reference_capture=nullptr;
    c.unified_eq_agc=false;return c;
  }
  Config c_;AudioEngine observer_;DynamicRangeCompressor drc_;std::unique_ptr<EqProcessor> eq_;
  std::array<float,256> ns_{},physical_delay_{};std::array<bool,256> far_delay_{};
  std::array<float,160> control_{},control_ref_{},control_output_{};FrameInfo control_info_{};
  unsigned delay_=0,control_count_=0,gain_remaining_=0;std::uint64_t frames_=0,applied_=0;
  float gain_=1,target_gain_=1,gain_step_=0;bool failed_=false;
};
}  // namespace itu
