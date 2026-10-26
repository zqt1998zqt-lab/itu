#pragma once
#include "itu/audio_engine.hpp"
#include "speex/speex_echo.h"
#include "speex/speex_preprocess.h"
#include "linear_phase_lowcut.hpp"
#include "rate_converter.hpp"
#include "reference_monitor.hpp"
#include "echo_coherence.hpp"
#include "speex_multirender_context.hpp"
#include "ns_evidence.hpp"
#include "aec_fusion.hpp"
#include <algorithm>
#include <array>
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
      channel_(c.frame_size),channel_output_(c.frame_size),linear_native_(capture_.size()),
      linear_channel_(c.frame_size),linear_output_(160),post_(post_config(c),4096,true) {
    for(unsigned ch=0;ch<c.microphones;++ch)
      linear_down_.emplace_back(c.frame_size==160 ? nullptr : std::make_unique<RateConverter>(c.frame_size,160));
    // A mono render has an identifiable GCC observation. Do not classify a
    // multi-render mixture from one arbitrarily selected playback channel.
    if(c.reference_guard && c.reference_channels==1) {
      reference_monitor_=std::make_unique<ReferenceMonitor>(c.speex_post_context);
      if(c.speex_post_context && c.coherence_dtd) coherence_=std::make_unique<EchoCoherence>();
      if(c.frame_size!=160) {
        monitor_capture_down_=std::make_unique<RateConverter>(c.frame_size,160);
        monitor_render_down_=std::make_unique<RateConverter>(c.frame_size,160);
      }
    }
    // This observer is independent of the mono reference_guard/coherence_dtd
    // switches, which checked() disables for multi-render streams.
    if(c.speex_post_context && c.speex_multirender_context && c.reference_channels>1) {
      multirender_reference_=std::make_unique<SpeexMultiRenderReference>(c.reference_channels);
      multirender_native_.resize(c.frame_size);
      multirender_render_.resize(160*c.reference_channels);
      if(c.frame_size!=160) multirender_capture_down_=std::make_unique<RateConverter>(c.frame_size,160);
      for(unsigned ch=0;ch<c.reference_channels;++ch)
        multirender_down_.emplace_back(c.frame_size==160 ? nullptr : std::make_unique<RateConverter>(c.frame_size,160));
    }
    if(c.speex_post_context) {
      linear_evidence_=std::make_unique<NsEvidence>();
      post_evidence_=std::make_unique<NsEvidence>();
      if(c.frame_size!=160) post_down_=std::make_unique<RateConverter>(c.frame_size,160);
      cancellation_band_=std::make_unique<LinearPhaseLowCut>(2*c.microphones,300.f,c.sample_rate);
    }
    if(c.linear_phase_low_cut && !c.speex_native_conditioning)
      lowcut_=std::make_unique<LinearPhaseLowCut>(c.microphones+c.reference_channels,c.low_cut_hz,c.sample_rate);
    if(c.aec_backend==AecBackend::speex_fused)
      fused_aec_=std::make_unique<AecFusion>(c.sample_rate,c.frame_size,
          c.speex_fused_learning_rate,c.speex_fused_shadow_taps_ms);
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
    if(reference_monitor_) {
      for(unsigned i=0;i<c_.frame_size;++i) scratch_[i]=conditioned_[i*c_.microphones];
      if(monitor_capture_down_) {
        if(monitor_capture_down_->Resample(scratch_.data(),c_.frame_size,monitor_capture_.data(),160) ||
           monitor_render_down_->Resample(filtered_render_.data(),c_.frame_size,monitor_render_.data(),160)) return Status::backend_error;
      } else {
        std::copy_n(scratch_.data(),160,monitor_capture_.data());
        std::copy_n(filtered_render_.data(),160,monitor_render_.data());
      }
      reference_monitor_->push(monitor_capture_.data(),monitor_render_.data(),1);
      if(coherence_) coherence_->push(monitor_capture_.data(),monitor_render_.data(),1,
          reference_monitor_->delay_available() ? reference_monitor_->delay_samples() : 0,true);
    }
    SpeexMultiRenderReference::Evidence multirender_evidence;
    if(multirender_reference_) {
      for(unsigned i=0;i<c_.frame_size;++i) {
        float sum=0;
        for(unsigned ch=0;ch<c_.microphones;++ch) sum+=conditioned_[i*c_.microphones+ch];
        multirender_native_[i]=sum/c_.microphones;
      }
      if(multirender_capture_down_) {
        if(multirender_capture_down_->Resample(multirender_native_.data(),c_.frame_size,
              multirender_capture_.data(),160)) return Status::backend_error;
      } else std::copy_n(multirender_native_.data(),160,multirender_capture_.data());
      for(unsigned ch=0;ch<c_.reference_channels;++ch) {
        for(unsigned i=0;i<c_.frame_size;++i) multirender_native_[i]=filtered_render_[i*c_.reference_channels+ch];
        if(multirender_down_[ch]) {
          if(multirender_down_[ch]->Resample(multirender_native_.data(),c_.frame_size,
                multirender_channel_.data(),160)) return Status::backend_error;
        } else std::copy_n(multirender_native_.data(),160,multirender_channel_.data());
        for(unsigned i=0;i<160;++i) multirender_render_[i*c_.reference_channels+ch]=multirender_channel_[i];
      }
      double excitation=0;
      for(float sample:multirender_render_) excitation+=double(sample)*sample;
      multirender_evidence=multirender_reference_->observe(multirender_capture_.data(),
          multirender_render_.data(),excitation/multirender_render_.size()>c_.activity_floor);
    }
    for(std::size_t i=0;i<capture_.size();++i) capture_[i]=pcm(conditioned_[i]);
    for(std::size_t i=0;i<render_.size();++i) render_[i]=pcm(filtered_render_[i]);
    if(res_.empty()) {
      speex_echo_cancellation(state_,capture_.data(),render_.data(),residual_.data());
      for(std::size_t i=0;i<residual_.size();++i) linear_native_[i]=residual_[i]/32768.f;
    }
    else for(unsigned ch=0;ch<res_.size();++ch) {
      for(unsigned i=0;i<channel_.size();++i) channel_[i]=capture_[i*res_.size()+ch];
      speex_echo_cancellation(echoes_[ch].get(),channel_.data(),render_.data(),channel_output_.data());
      for(unsigned i=0;i<channel_.size();++i) linear_native_[i*res_.size()+ch]=channel_output_[i]/32768.f;
      speex_preprocess_run(res_[ch].get(),channel_output_.data());
      for(unsigned i=0;i<channel_.size();++i) residual_[i*res_.size()+ch]=channel_output_[i];
    }
    // Export the actual MDF error before RES, dry restoration, NS and AGC.
    // Keep every channel's converter advancing even when the caller omits lt.
    context_linear_.fill(0.f);
    for(unsigned ch=0;ch<c_.microphones;++ch) {
      for(unsigned i=0;i<c_.frame_size;++i) linear_channel_[i]=linear_native_[i*c_.microphones+ch];
      if(linear_down_[ch]) {
        if(linear_down_[ch]->Resample(linear_channel_.data(),c_.frame_size,linear_output_.data(),160)) return Status::backend_error;
      } else std::copy_n(linear_channel_.data(),160,linear_output_.data());
      if(lt) std::copy_n(linear_output_.data(),160,lt+ch*160);
      if(c_.speex_post_context) for(unsigned i=0;i<160;++i)
        context_linear_[i]+=linear_output_[i]/c_.microphones;
    }
    double render_power=0;for(float v:filtered_render_) render_power+=double(v)*v;
    // Keep MDF adapting, but preserve fullband capture after the observed
    // playback echo tail drains. MDF's internal DC notch otherwise modifies
    // unreferenced near speech. No scene/SNR/clean label enters this decision.
    if(render_power/render_.size()>c_.activity_floor)
      render_hold_=(c_.speex_echo_tail_ms+c_.stream_delay_ms+9)/10;
    else if(render_hold_) --render_hold_;
    double capture_power=0,error_power=0;
    for(std::size_t i=0;i<linear_native_.size();++i) {
      capture_power+=double(conditioned_[i])*conditioned_[i];
      error_power+=double(linear_native_[i])*linear_native_[i];
    }
    capture_power_= .95*capture_power_+.05*capture_power/linear_native_.size();
    error_power_= .95*error_power_+.05*error_power/linear_native_.size();
    const float residual_ratio=error_power_/std::max(capture_power_,1e-20);
    double band_capture=0,band_error=0;
    if(cancellation_band_) for(unsigned ch=0;ch<c_.microphones;++ch) {
      for(unsigned i=0;i<c_.frame_size;++i) scratch_[i]=conditioned_[i*c_.microphones+ch];
      cancellation_band_->process_frame(scratch_.data(),scratch_.data(),c_.frame_size,ch);
      for(float v:scratch_) band_capture+=double(v)*v;
      for(unsigned i=0;i<c_.frame_size;++i) scratch_[i]=linear_native_[i*c_.microphones+ch];
      cancellation_band_->process_frame(scratch_.data(),scratch_.data(),c_.frame_size,c_.microphones+ch);
      for(float v:scratch_) band_error+=double(v)*v;
    }
    band_capture_=.95*band_capture_+.05*band_capture/std::max(std::size_t(1),linear_native_.size());
    band_error_=.95*band_error_+.05*band_error/std::max(std::size_t(1),linear_native_.size());
    const float cancellation_ratio=cancellation_band_ ? band_error_/std::max(band_capture_,1e-20) : residual_ratio;
    // GCC can be weak on a long diffuse tail. Measured linear cancellation is
    // independent evidence against declaring that reference unrelated.
    const bool cancelling=capture_power_>1e-6 && cancellation_ratio<.72f;
    const bool unrelated=((reference_monitor_ && reference_monitor_->unreliable()) ||
        (multirender_reference_ && multirender_evidence.unrelated)) && !cancelling;
    PostAecContext context;
    context.far_active=render_hold_>0;
    context.reference_state=!render_hold_ ? 0 : unrelated ? 2 :
        cancelling || (reference_monitor_ && reference_monitor_->reliable()) ||
        (multirender_reference_ && multirender_evidence.valid) ? 1 : 0;
    context.reference_residual_ratio=residual_ratio;
    context.reference_confidence=reference_monitor_ ? reference_monitor_->confidence() : 0.f;
    float control_cancellation_ratio=cancellation_ratio;
    bool dry_unrelated=unrelated;
    bool multirender_path_changed=multirender_evidence.path_changed;
    if(c_.speex_post_context) {
      context.linear_speech_probability=linear_evidence_->observe(context_linear_.data());
      context.incoherent_ratio=coherence_ ? coherence_->ratio() : 1.f;
      if(multirender_reference_) {
        context.multirender_context=true;
        context.incoherent_ratio=multirender_evidence.incoherent_ratio;
        context.reference_evidence_valid=multirender_evidence.valid;
        context.multirender_frames=multirender_evidence.frames;
        context.reference_confidence=multirender_evidence.confidence;
      }
      if(!res_.empty()) {
        // Speex RES emits the previous input block. Delay observations, not
        // PCM, so DTD combines that block's linear/reference evidence with its
        // actual post-RES observation. Preserve the existing output clock.
        const auto current=context;
        context=previous_context_; previous_context_=current;
        control_cancellation_ratio=previous_cancellation_ratio_;
        previous_cancellation_ratio_=cancellation_ratio;
        dry_unrelated=previous_unrelated_; previous_unrelated_=unrelated;
        multirender_path_changed=previous_multirender_path_changed_;
        previous_multirender_path_changed_=multirender_evidence.path_changed;
      }
    }
    if(fused_aec_) {
      // The linear evidence above belongs to this frame. The nonlinear speech
      // observation is produced below and cannot be consumed here yet.
      const bool protect=near_hold_>0;
      fused_aec_->process(conditioned_.data(),filtered_render_.data(),
          render_power/render_.size()>c_.activity_floor,protect);
      fused_aec_->blend(linear_native_.data(),context.far_active,context.reference_state,near_hold_);
    }
    const bool dry_idle=c_.speex_post_context ? !context.far_active : !render_hold_;
    // An idle render has reference_state=unknown, while the monitor retains
    // its unrelated-reference verdict across silence. Keep that independent
    // verdict, delayed with RES, rather than erasing dry protection at idle.
    const float target=(c_.speech_preserving_low_cut && dry_idle) || dry_unrelated ? 1.f : 0.f;
    dry_mix_+=std::clamp(target-dry_mix_,-.2f,.05f);
    float residual_gain = 1.f;
    if(c_.speex_coherence_residual_suppression && c_.reference_channels==1) {
      // A causal, frame-rate residual echo postfilter. It only acts after a
      // measured mono-reference cancellation state is available. Correlation
      // with the current render estimates coherent late echo; incoherent
      // residual and confirmed near speech keep the raw MDF error intact.
      double residual_power = 0., render_power_now = 0., render_nonlinear_power = 0.;
      double cross = 0., cross_nonlinear = 0.;
      for(unsigned i=0;i<c_.frame_size;++i) {
        const float render = filtered_render_[i];
        const float nonlinear_render = render*std::abs(render);
        render_power_now += double(render) * render;
        render_nonlinear_power += double(nonlinear_render) * nonlinear_render;
        for(unsigned ch=0; ch<c_.microphones; ++ch) {
          const float residual = linear_native_[i*c_.microphones+ch];
          residual_power += double(residual) * residual;
          cross += double(residual) * render;
          cross_nonlinear += double(residual) * nonlinear_render;
        }
      }
      const double channels = std::max(1u, c_.microphones);
      residual_power /= channels;
      cross /= channels;
      cross_nonlinear /= channels;
      const double linear_denom = std::sqrt(std::max(1e-20, residual_power * render_power_now));
      const double nonlinear_denom = std::sqrt(std::max(1e-20, residual_power * render_nonlinear_power));
      const float linear_correlation = static_cast<float>(std::abs(cross) / linear_denom);
      const float nonlinear_correlation = static_cast<float>(std::abs(cross_nonlinear) / nonlinear_denom);
      const float correlation = std::max(linear_correlation, nonlinear_correlation);
      const float render_fraction = static_cast<float>(render_power_now /
          std::max(1e-20, render_power_now + residual_power));
      const float coherent_echo = std::clamp((correlation - .12f) / .35f, 0.f, 1.f) *
          std::clamp(render_fraction * 2.f, 0.f, 1.f);
      const bool protect_near = near_hold_ > 0 || (fused_aec_ && fused_aec_->frame().double_talk) || (c_.speex_post_context &&
          (context.incoherent_ratio > .45f || context.linear_speech_probability > .75f));
      const bool trusted = context.far_active && context.reference_state == 1 &&
          !dry_unrelated && !protect_near;
      const float target_gain = trusted ? std::clamp(1.f - .72f * coherent_echo, .28f, 1.f) : 1.f;
      residual_gain_ += std::clamp(target_gain - residual_gain_,
          target_gain < residual_gain_ ? -.22f : .08f, .08f);
      residual_gain = residual_gain_;
    }
    for(std::size_t i=0;i<audio_.size();++i) {
      const float dry=dry_history_.empty() ? mic[i] : dry_history_[dry_head_];
      if(!dry_history_.empty()) {
        dry_history_[dry_head_]=mic[i];dry_head_=(dry_head_+1)%dry_history_.size();
      }
      const float wet=fused_aec_ ? linear_native_[i] : residual_[i]/32768.f;
      audio_[i]=(1-dry_mix_)*(residual_gain*wet)+dry_mix_*dry;
    }
    if(c_.speex_post_context) {
      for(unsigned i=0;i<c_.frame_size;++i) {
        float sum=0;for(unsigned ch=0;ch<c_.microphones;++ch) sum+=audio_[i*c_.microphones+ch];
        scratch_[i]=sum/c_.microphones;
      }
      if(post_down_) {
        if(post_down_->Resample(scratch_.data(),c_.frame_size,context_post_.data(),160)) return Status::backend_error;
      } else std::copy_n(scratch_.data(),160,context_post_.data());
      context.nonlinear_speech_probability=post_evidence_->observe(context_post_.data());
      if(multirender_reference_) {
        const auto decision=multirender_near_.update(context.far_active,
            context.reference_evidence_valid,context.reference_state==1,multirender_path_changed,
            control_cancellation_ratio,context.incoherent_ratio,
            context.linear_speech_probability,context.nonlinear_speech_probability);
        near_hold_=decision.hold;
        near_clear_frames_=decision.clear_frames;
        echo_floor_=decision.echo_floor;
        coherence_floor_=decision.coherence_floor;
        context.near_state=decision.state;
      } else {
      const bool trusted=context.far_active && context.reference_state==1 && bool(coherence_);
      // Relative minima alone turn ordinary far-end envelope fluctuations into
      // double talk. Bound the learned baseline and require absolute residual
      // and incoherence support. A syllable valley is not a confirmed return
      // to echo-only: require 200 ms of coherent cancellation before clearing.
      const bool innovation=trusted && near_observed_ && control_cancellation_ratio>std::max(.15f,2.f*echo_floor_) &&
          context.incoherent_ratio>std::max(.48f,1.6f*coherence_floor_) &&
          context.nonlinear_speech_probability>.35f;
      // Once independently entered, sustain through far-dominant syllables
      // using contrast against the learned echo-only baseline. Capture may
      // remain mostly coherent with render even while audible near speech is
      // present in the linear error. Do not reuse the stricter entry bound.
      const bool sustained_near=trusted && near_hold_ &&
          control_cancellation_ratio>std::max(.12f,1.5f*echo_floor_) &&
          context.incoherent_ratio>std::max(.12f,1.2f*coherence_floor_) &&
          context.nonlinear_speech_probability>.35f;
      near_confirmation_=innovation ? std::min(3u,near_confirmation_+1) : 0;
      const bool coherent_echo=trusted && control_cancellation_ratio<.08f && context.incoherent_ratio<.25f &&
          context.nonlinear_speech_probability<.2f;
      near_clear_frames_=coherent_echo ? std::min(20u,near_clear_frames_+1) : 0;
      if(!trusted) near_hold_=0;
      else if(near_confirmation_>=3 || sustained_near) near_hold_=60;
      else if(near_clear_frames_>=20) near_hold_=0;
      else if(near_hold_) --near_hold_;
      if(trusted && !near_hold_ && (!near_observed_ || context.incoherent_ratio<.4f)) {
        echo_floor_+=std::clamp(control_cancellation_ratio-echo_floor_,-.02f,.0002f);
        echo_floor_=std::clamp(echo_floor_,.04f,.6f);
        coherence_floor_+=std::clamp(context.incoherent_ratio-coherence_floor_,-.02f,.0002f);
        coherence_floor_=std::clamp(coherence_floor_,.1f,.8f);
      }
      }
      context.near_hold_frames=near_hold_;
      if(fused_aec_ && fused_aec_->frame().double_talk)
        context.near_hold_frames=std::max(context.near_hold_frames,3u);
    }
    const auto status=post_.process_impl(audio_.data(),audio_.size(),ref,render_.size(),
        out,post_.config().frame_size,info,at,bt,nullptr,it,c_.speex_post_context ? &context : nullptr);
    if(c_.speex_post_context && !context.far_active) {
      const bool observed=info.control_evidence_probability>.55f &&
          context.nonlinear_speech_probability>.5f && context.linear_speech_probability>.5f;
      if(c_.post_aec_evidence_window) {
        // At 24 kHz the control evidence flickers frame to frame, so a strictly
        // consecutive run of 20 never completes and the near latch stays unset
        // for 61% of double-talk scenes. Hold that evidence for three frames and
        // require 20 qualifying frames inside a 30-frame window instead. The
        // per-frame conjunction and every threshold are unchanged: only the
        // counter mechanics differ. The spectral and nonlinear terms stay
        // strictly current-frame evidence.
        evidence_history_[2]=evidence_history_[1];
        evidence_history_[1]=evidence_history_[0];
        evidence_history_[0]=info.control_evidence_probability;
        const float peak=std::max(evidence_history_[0],
            std::max(evidence_history_[1],evidence_history_[2]));
        const bool qualified=peak>.55f && context.nonlinear_speech_probability>.5f &&
            context.linear_speech_probability>.5f;
        const bool previous=evidence_window_[evidence_head_]!=0;
        evidence_window_[evidence_head_]=qualified ? 1 : 0;
        evidence_head_=(evidence_head_+1)%kEvidenceWindow;
        evidence_count_+=int(qualified)-int(previous);
        if(evidence_count_>=20) near_observed_=true;
      } else {
        near_observations_=observed ? std::min(20u,near_observations_+1) : 0;
        if(near_observations_>=20) near_observed_=true;
      }
    }
    info.linear_aec_available=true;
    info.linear_aec_delay_samples=unsigned(std::lround(conditioning_delay(c_)*16000./c_.sample_rate+
        (linear_down_[0] ? linear_down_[0]->output_delay() : 0)));
    info.nearend_mix=dry_mix_;
    info.reference_state=context.reference_state;
    info.reference_residual_ratio=context.reference_residual_ratio;
    if(c_.speex_post_context) {
      info.post_aec_cancellation_ratio=control_cancellation_ratio;
      info.post_aec_echo_floor=echo_floor_;
      info.post_aec_coherence_floor=coherence_floor_;
      info.post_aec_near_observed=near_observed_;
      info.post_aec_echo_clear_frames=near_clear_frames_;
    }
    if(reference_monitor_ || multirender_reference_) info.reference_confidence=context.reference_confidence;
    if(fused_aec_) {
      const auto fused=fused_aec_->frame();
      info.aec_fused_shadow_weight=fused.shadow_weight;
      info.aec_fused_primary_error_power=fused.primary_error_power;
      info.aec_fused_shadow_error_power=fused.shadow_error_power;
      info.aec_fused_reference_confidence=fused.reference_confidence;
      info.aec_fused_double_talk=fused.double_talk;
      info.aec_fused_updates=fused.updates;
      info.aec_fused_delay_samples=fused.delay_samples;
      info.aec_fused_path_changes=fused.path_changes;
      info.aec_fused_delay_valid=fused.delay_valid;
    }
    info.post_aec_reference_evidence_valid=context.reference_evidence_valid;
    info.post_aec_near_state=context.near_state;
    info.post_aec_multirender_frames=context.multirender_frames;
    info.aec_input_delay_samples=conditioning_delay(c_);
    info.low_cut_delay_samples=conditioning_delay(c_);
    info.latency_samples+=conditioning_delay(c_);
    info.aec_output_delay_samples+=conditioning_delay(c_);
    if(!res_.empty()) {
      info.latency_samples+=channel_.size();
      info.aec_output_delay_samples+=channel_.size();
    }
    if(c_.diagnostic_stages) {
      auto& taps=*c_.diagnostic_stages;
      const unsigned extra=conditioning_delay(c_)+(res_.empty() ? 0 : c_.frame_size);
      taps.core_delay_samples+=unsigned(std::lround(double(extra)*taps.core_sample_rate/c_.sample_rate));
      taps.eq_delay_samples+=unsigned(std::lround(double(extra)*taps.eq_sample_rate/c_.sample_rate));
    }
    info.microphone_rms=0;
    double power=0; for(std::size_t i=0;i<audio_.size();++i) power+=double(mic[i])*mic[i];
    info.microphone_rms=float(std::sqrt(power/audio_.size()));
    if(it) std::copy(conditioned_.begin(),conditioned_.end(),it);
    return status;
  }
  std::size_t memory_bytes() const noexcept {
    // Speex opaque state memory is not observable from its public API.
    return sizeof(*this)+(capture_.capacity()+render_.capacity()+residual_.capacity()+channel_.capacity()+channel_output_.capacity())*sizeof(spx_int16_t)
        +(audio_.capacity()+conditioned_.capacity()+filtered_render_.capacity()+scratch_.capacity()+dry_history_.capacity()+
          linear_native_.capacity()+linear_channel_.capacity()+linear_output_.capacity())*sizeof(float)
        +linear_down_.capacity()*sizeof(std::unique_ptr<RateConverter>)
        +(reference_monitor_ ? reference_monitor_->memory_bytes() : 0)
        +(monitor_capture_down_ ? sizeof(RateConverter)*2 : 0)
        +(coherence_ ? coherence_->memory_bytes() : 0)
        +(multirender_reference_ ? multirender_reference_->memory_bytes() : 0)
        +(multirender_native_.capacity()+multirender_render_.capacity())*sizeof(float)
        +multirender_down_.capacity()*sizeof(std::unique_ptr<RateConverter>)
        +(multirender_capture_down_ ? sizeof(RateConverter) : 0)
        +std::count_if(multirender_down_.begin(),multirender_down_.end(),
            [](const auto& converter){return bool(converter);})*sizeof(RateConverter)
        +(linear_evidence_ ? sizeof(NsEvidence)*2 : 0)+(post_down_ ? sizeof(RateConverter) : 0)
        +(cancellation_band_ ? cancellation_band_->memory_bytes() : 0)
        +echoes_.capacity()*sizeof(Echo)+res_.capacity()*sizeof(Preprocess)+(lowcut_ ? lowcut_->memory_bytes() : 0)
        +(fused_aec_ ? fused_aec_->memory_bytes() : 0)
        +post_.working_memory_bytes();
  }
 private:
  static unsigned conditioning_delay(const Config&) {return 0;}
  static Config post_config(Config c) {
    // Standalone NS retains its own conditioned/fullband speech branches.
    // This second low-cut is after the AEC/BF taps, and its delay is owned by
    // post_. The MDF input low-cut above is independently accounted for.
    // This post-AEC low-cut is an NS noise branch. Its fullband complement is
    // restored for qualified near speech; it is never applied to MDF input.
    if(c.speex_native_conditioning && !c.speex_post_context) c.linear_phase_low_cut=false;
    c.aec=false;
    c.aec_backend=AecBackend::webrtc;
    // The post-AEC engine is a WebRTC/NS/AGC tail. Speex-only residual
    // controls must not leak into this converted configuration.
    c.speex_coherence_residual_suppression=false;
    return c;
  }
  static spx_int16_t pcm(float v) noexcept {
    return static_cast<spx_int16_t>(std::clamp(std::lround(v*32768.f),-32768l,32767l));
  }
  Config c_;
  SpeexEchoState* state_=nullptr;
  std::unique_ptr<LinearPhaseLowCut> lowcut_;
  std::unique_ptr<AecFusion> fused_aec_;
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
  std::vector<float> linear_native_,linear_channel_,linear_output_;
  std::vector<std::unique_ptr<RateConverter>> linear_down_;
  std::unique_ptr<ReferenceMonitor> reference_monitor_;
  std::unique_ptr<RateConverter> monitor_capture_down_,monitor_render_down_;
  std::array<float,160> monitor_capture_{},monitor_render_{};
  std::unique_ptr<EchoCoherence> coherence_;
  std::unique_ptr<SpeexMultiRenderReference> multirender_reference_;
  SpeexMultiRenderNearState multirender_near_;
  std::unique_ptr<RateConverter> multirender_capture_down_;
  std::vector<std::unique_ptr<RateConverter>> multirender_down_;
  std::vector<float> multirender_native_,multirender_render_;
  std::array<float,160> multirender_capture_{},multirender_channel_{};
  bool previous_multirender_path_changed_=false;
  std::unique_ptr<NsEvidence> linear_evidence_,post_evidence_;
  std::unique_ptr<RateConverter> post_down_;
  std::array<float,160> context_linear_{},context_post_{};
  std::unique_ptr<LinearPhaseLowCut> cancellation_band_;
  PostAecContext previous_context_;
  float previous_cancellation_ratio_=1.f;
  bool previous_unrelated_=false;
  unsigned near_hold_=0,near_confirmation_=0,near_clear_frames_=0;
  unsigned near_observations_=0;
  static constexpr unsigned kEvidenceWindow=30;
  std::array<unsigned char,kEvidenceWindow> evidence_window_{};
  std::array<float,3> evidence_history_{};
  unsigned evidence_head_=0;
  int evidence_count_=0;
  bool near_observed_=false;
  float residual_gain_=1.f;
  float echo_floor_=.08f,coherence_floor_=.2f;
  double band_capture_=0,band_error_=0;
  double capture_power_=0,error_power_=0;
  AudioEngine post_;
};
}
