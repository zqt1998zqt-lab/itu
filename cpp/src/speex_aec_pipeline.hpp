#pragma once
#include "itu/audio_engine.hpp"
#include "speex/speex_echo.h"
#include "speex/speex_preprocess.h"
#include "linear_phase_lowcut.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace itu {
// Independent multi-render MDF candidate. Synchronous cancellation adds no
// playback queue; the existing streaming post chain owns BF/NS/AGC and delays.
class SpeexAecPipeline {
 public:
  explicit SpeexAecPipeline(const Config& c):c_(c),
      capture_(c.frame_size*c.microphones),render_(c.frame_size*c.reference_channels),
      residual_(capture_.size()),audio_(capture_.size()),conditioned_(capture_.size()),
      filtered_render_(render_.size()),scratch_(c.frame_size),
      dry_history_((conditioning_delay(c)+(c.speex_residual_suppression ? c.frame_size : 0))*c.microphones),
      channel_(c.frame_size),channel_output_(c.frame_size),post_(post_config(c)) {
    if(c.linear_phase_low_cut)
      lowcut_=std::make_unique<LinearPhaseLowCut>(c.microphones+c.reference_channels,c.low_cut_hz,c.sample_rate);
    if(c.speex_residual_suppression) {
      // Upstream get_residual() is mono-capture only (mdf.c MC FIXME). Each
      // microphone therefore owns a mono-capture MDF with all render channels.
      for(unsigned ch=0;ch<c.microphones;++ch) {
        Echo echo(speex_echo_state_init_mc(c.frame_size,c.sample_rate*c.speex_echo_tail_ms/1000,1,c.reference_channels));
        Preprocess res(speex_preprocess_state_init(c.frame_size,c.sample_rate));
        if(!echo || !res) throw std::runtime_error("Speex MDF/RES initialization failed");
        int rate=c.sample_rate,enabled=1,disabled=0,noise=0,echo_floor=-40,active=c.speex_active_echo_suppression_db;
        if(speex_echo_ctl(echo.get(),SPEEX_ECHO_SET_SAMPLING_RATE,&rate)) throw std::runtime_error("MDF rate rejected");
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_DENOISE,&enabled);
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_NOISE_SUPPRESS,&noise);
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_AGC,&disabled);
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_VAD,&disabled);
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_ECHO_SUPPRESS,&echo_floor);
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE,&active);
        speex_preprocess_ctl(res.get(),SPEEX_PREPROCESS_SET_ECHO_STATE,echo.get());
        echoes_.push_back(std::move(echo));res_.push_back(std::move(res));
      }
      return;
    }
    state_=speex_echo_state_init_mc(c.frame_size,c.sample_rate*c.speex_echo_tail_ms/1000,
                                  c.microphones,c.reference_channels);
    if(!state_) throw std::runtime_error("Speex MDF initialization failed");
    int rate=c.sample_rate;
    if(speex_echo_ctl(state_,SPEEX_ECHO_SET_SAMPLING_RATE,&rate)) {
      speex_echo_state_destroy(state_); state_=nullptr;
      throw std::runtime_error("Speex MDF sample rate rejected");
    }
  }
  ~SpeexAecPipeline() {if(state_) speex_echo_state_destroy(state_);}
  Status process(const float* mic,const float* ref,float* out,FrameInfo& info,
                 float* at,float* bt,float* lt,float* it) noexcept {
    // Identical capture/render conditioning before MDF. The input tap is this
    // float path before quantization, rather than an unrelated post-chain tap.
    auto condition=[&](const float* input,std::vector<float>& output,unsigned channels,unsigned offset) {
      for(unsigned ch=0;ch<channels;++ch) {
        for(unsigned i=0;i<c_.frame_size;++i) scratch_[i]=input[i*channels+ch];
        if(lowcut_) lowcut_->process_frame(scratch_.data(),scratch_.data(),c_.frame_size,offset+ch);
        for(unsigned i=0;i<c_.frame_size;++i) output[i*channels+ch]=scratch_[i];
      }
    };
    condition(mic,conditioned_,c_.microphones,0);
    condition(ref,filtered_render_,c_.reference_channels,c_.microphones);
    for(std::size_t i=0;i<capture_.size();++i) capture_[i]=pcm(conditioned_[i]);
    for(std::size_t i=0;i<render_.size();++i) render_[i]=pcm(filtered_render_[i]);
    if(res_.empty()) speex_echo_cancellation(state_,capture_.data(),render_.data(),residual_.data());
    else for(unsigned ch=0;ch<res_.size();++ch) {
      for(unsigned i=0;i<channel_.size();++i) channel_[i]=capture_[i*res_.size()+ch];
      speex_echo_cancellation(echoes_[ch].get(),channel_.data(),render_.data(),channel_output_.data());
      speex_preprocess_run(res_[ch].get(),channel_output_.data());
      for(unsigned i=0;i<channel_.size();++i) residual_[i*res_.size()+ch]=channel_output_[i];
    }
    double render_power=0;for(float v:filtered_render_) render_power+=double(v)*v;
    // Keep MDF adapting, but preserve fullband capture after the observed
    // playback echo tail drains. MDF's internal DC notch otherwise modifies
    // unreferenced near speech. No scene/SNR/clean label enters this decision.
    if(render_power/render_.size()>c_.activity_floor)
      render_hold_=(c_.speex_echo_tail_ms+c_.stream_delay_ms+9)/10;
    else if(render_hold_) --render_hold_;
    const float target=c_.speech_preserving_low_cut && !render_hold_ ? 1.f : 0.f;
    dry_mix_+=std::clamp(target-dry_mix_,-.2f,.05f);
    for(std::size_t i=0;i<audio_.size();++i) {
      const float dry=dry_history_.empty() ? mic[i] : dry_history_[dry_head_];
      if(!dry_history_.empty()) {
        dry_history_[dry_head_]=mic[i];dry_head_=(dry_head_+1)%dry_history_.size();
      }
      audio_[i]=(1-dry_mix_)*(residual_[i]/32768.f)+dry_mix_*dry;
    }
    const auto status=post_.process(audio_.data(),audio_.size(),ref,render_.size(),
                                   out,post_.config().frame_size,info,at,bt,nullptr,it);
    // The protected MDF/RES residual is the pre-post input, possibly blended
    // with aligned capture after playback drains. It is not a WebRTC linear
    // tap; declare unavailable rather than populate a misleading ERLE metric.
    info.linear_aec_available=false;
    info.nearend_mix=dry_mix_;
    info.aec_input_delay_samples=conditioning_delay(c_);
    info.low_cut_delay_samples=conditioning_delay(c_);
    info.latency_samples+=conditioning_delay(c_);
    info.aec_output_delay_samples+=conditioning_delay(c_);
    if(!res_.empty()) {
      info.latency_samples+=channel_.size();
      info.aec_output_delay_samples+=channel_.size();
    }
    info.microphone_rms=0;
    double power=0; for(std::size_t i=0;i<audio_.size();++i) power+=double(mic[i])*mic[i];
    info.microphone_rms=float(std::sqrt(power/audio_.size()));
    if(it) std::copy(conditioned_.begin(),conditioned_.end(),it);
    if(lt) std::fill_n(lt,160*post_.config().microphones,0.f);
    return status;
  }
  std::size_t memory_bytes() const noexcept {
    // Speex opaque state memory is not observable from its public API.
    return sizeof(*this)+(capture_.capacity()+render_.capacity()+residual_.capacity()+channel_.capacity()+channel_output_.capacity())*sizeof(spx_int16_t)
        +(audio_.capacity()+conditioned_.capacity()+filtered_render_.capacity()+scratch_.capacity()+dry_history_.capacity())*sizeof(float)
        +echoes_.capacity()*sizeof(Echo)+res_.capacity()*sizeof(Preprocess)+(lowcut_ ? lowcut_->memory_bytes() : 0)
        +post_.working_memory_bytes();
  }
 private:
  static unsigned conditioning_delay(const Config& c) {return c.linear_phase_low_cut ? c.sample_rate*16/1000 : 0;}
  static Config post_config(Config c) {
    // Standalone NS retains its own conditioned/fullband speech branches.
    // This second low-cut is after the AEC/BF taps, and its delay is owned by
    // post_. The MDF input low-cut above is independently accounted for.
    c.aec=false;c.aec_backend=AecBackend::webrtc;return c;
  }
  static spx_int16_t pcm(float v) noexcept {
    return static_cast<spx_int16_t>(std::clamp(std::lround(v*32768.f),-32768l,32767l));
  }
  Config c_;
  SpeexEchoState* state_=nullptr;
  std::unique_ptr<LinearPhaseLowCut> lowcut_;
  unsigned render_hold_=0;
  std::size_t dry_head_=0;
  float dry_mix_=0;
  struct EchoDelete {void operator()(SpeexEchoState* p) const noexcept {speex_echo_state_destroy(p);}};
  struct ResDelete {void operator()(SpeexPreprocessState* p) const noexcept {speex_preprocess_state_destroy(p);}};
  using Echo=std::unique_ptr<SpeexEchoState,EchoDelete>;
  using Preprocess=std::unique_ptr<SpeexPreprocessState,ResDelete>;
  std::vector<Echo> echoes_;
  std::vector<Preprocess> res_;
  std::vector<spx_int16_t> capture_,render_,residual_;
  std::vector<float> audio_,conditioned_,filtered_render_,scratch_,dry_history_;
  std::vector<spx_int16_t> channel_,channel_output_;
  AudioEngine post_;
};
}
