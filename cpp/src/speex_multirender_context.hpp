#pragma once
#include "echo_subspace.hpp"
#include "reference_monitor.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace itu {
// Native 16 kHz observation only. Each render gets an independent GCC path;
// dependent/anti-phase stereo remains a joint subspace, never a render sum.
// All storage, FFT work and monitors are allocated at construction.
class SpeexMultiRenderReference {
 public:
  struct Evidence {
    bool valid=false, unrelated=false, path_changed=false;
    float incoherent_ratio=1.f, confidence=0.f;
    unsigned frames=0;
  };
  explicit SpeexMultiRenderReference(unsigned channels):channels_(channels),
      joint_(channels),history_(channels*8192),aligned_(channels*160) {
    if(channels<2 || channels>8) throw std::invalid_argument("multi-render channels must be 2..8");
    for(unsigned ch=0;ch<channels;++ch)
      monitors_.emplace_back(std::make_unique<ReferenceMonitor>(true));
  }
  Evidence observe(const float* capture,const float* render,bool excited) noexcept {
    Evidence out;
    bool measured_path=false,all_unrelated=true,changed=false;
    float confidence=0;
    for(unsigned ch=0;ch<channels_;++ch) {
      for(unsigned i=0;i<160;++i) channel_[i]=render[i*channels_+ch];
      monitors_[ch]->push(capture,channel_.data(),1);
      confidence=std::max(confidence,monitors_[ch]->confidence());
      measured_path|=monitors_[ch]->delay_available();
      all_unrelated&=monitors_[ch]->unreliable();
      // ReferenceMonitor already requires two consistent, strongly correlated
      // GCC windows. A confirmed near latch must not block a newly measured
      // path: otherwise stale alignment can sustain that latch forever.
      if(monitors_[ch]->delay_available()) {
        const unsigned delay=monitors_[ch]->delay_samples();
        if(std::abs(int(delay)-int(delays_[ch]))>64) {
          delays_[ch]=delay;changed=true;
        }
      }
    }
    if(changed) joint_.reset();
    for(unsigned i=0;i<160;++i) {
      for(unsigned ch=0;ch<channels_;++ch) {
        history_[head_*channels_+ch]=render[i*channels_+ch];
        aligned_[i*channels_+ch]=history_[((head_+8192-delays_[ch])%8192)*channels_+ch];
      }
      head_=(head_+1)%8192;
    }
    joint_.push(capture,aligned_.data(),1,excited,0);
    out.valid=excited && measured_path && joint_.frames()>=80;
    out.unrelated=excited && all_unrelated;
    out.path_changed=changed;
    out.incoherent_ratio=joint_.ratio();
    out.confidence=confidence;
    out.frames=joint_.frames();
    return out;
  }
  std::size_t memory_bytes() const noexcept {
    std::size_t bytes=sizeof(*this)+joint_.memory_bytes()+
      (history_.capacity()+aligned_.capacity())*sizeof(float)+
      monitors_.capacity()*sizeof(std::unique_ptr<ReferenceMonitor>);
    for(const auto& monitor:monitors_) bytes+=monitor->memory_bytes();
    return bytes;
  }
 private:
  unsigned channels_,head_=0;
  EchoSubspace joint_;
  std::vector<float> history_,aligned_;
  std::vector<std::unique_ptr<ReferenceMonitor>> monitors_;
  std::array<float,160> channel_{};
  std::array<unsigned,8> delays_{};
};

// Temporal model separates "unobserved" from measured echo-only and near.
// Ratios come from live MDF/subspace observations, not clean PCM, labels,
// schedules or a hard-coded scene SNR. Baselines freeze during near evidence.
class SpeexMultiRenderNearState {
 public:
  struct Decision {
    unsigned state=0,hold=0,clear_frames=0;
    float echo_floor=0.f,coherence_floor=0.f;
  };
  Decision update(bool far,bool valid,bool reference_trusted,bool path_changed,
                  float cancellation,float incoherence,float linear_speech,
                  float post_speech) noexcept {
    if(path_changed) reset();
    if(!far) {if(!ready_) learning_=0;hold_=confirmation_=clear_=0;return decision(0);}
    if(!valid || !reference_trusted || !std::isfinite(cancellation) ||
       !std::isfinite(incoherence)) {
      if(!ready_) learning_=0;
      confirmation_=clear_=0;
      if(hold_) --hold_;
      return decision(hold_ ? 2u : 0u);
    }
    cancellation=std::clamp(cancellation,0.f,2.f);
    incoherence=std::clamp(incoherence,0.f,1.f);
    if(!ready_) {
      // Require sustained reference-explained capture and cancellation before
      // learning echo-only. Cold double talk with no such evidence stays
      // unknown and therefore does not authorize extra fullband attenuation.
      const bool cancelled_echo=cancellation<.25f && incoherence<.35f;
      // A measured, stable playback subspace can identify echo even while
      // MDF is slowly relearning a changed path. This is strictly stronger
      // correlation evidence, not a relaxed cancellation threshold.
      const bool subspace_echo=cancellation<1.25f && incoherence<.02f;
      const bool echo_support=cancelled_echo || subspace_echo;
      if(echo_support) {
        if(!learning_) {echo_=cancellation;coherence_=incoherence;}
        else {
          echo_+=.05f*(cancellation-echo_);
          coherence_+=.05f*(incoherence-coherence_);
        }
        if(++learning_>=40) {
          ready_=true;
          joint_only_baseline_=echo_>=.25f;
        }
      } else learning_=0;
      return decision(ready_ ? 1u : 0u);
    }
    const bool speech=linear_speech>.35f && post_speech>.35f;
    const bool ratio_innovation=cancellation>std::max(.12f,std::max(echo_+.045f,1.35f*echo_));
    const bool joint_innovation=incoherence>std::max(.15f,std::max(coherence_+.055f,1.2f*coherence_));
    // When MDF has not converged, adding near speech barely changes its
    // already-large error/capture ratio. Innovation outside a previously
    // 98%-explained joint subspace is independent evidence in that state.
    const bool innovation=speech && joint_innovation &&
        (ratio_innovation || (joint_only_baseline_ && cancellation>.08f));
    confirmation_=innovation ? std::min(3u,confirmation_+1) : 0;
    const bool sustained=speech && incoherence>coherence_+.015f &&
        (cancellation>std::max(.10f,echo_+.02f) ||
         (joint_only_baseline_ && cancellation>.08f));
    const bool echo_return=cancellation<echo_*1.15f+.015f &&
        incoherence<coherence_+.04f;
    clear_=echo_return ? std::min(20u,clear_+1) : 0;
    if(confirmation_>=3 || (hold_ && sustained)) hold_=60;
    else if(clear_>=20) hold_=0;
    else if(hold_) --hold_;
    if(!hold_ && !innovation) {
      // Downward correction follows a recovered path quickly; upward drift is
      // deliberately slow, so a near transition cannot train itself away.
      echo_+=std::clamp(cancellation-echo_,-.015f,.0003f);
      coherence_+=std::clamp(incoherence-coherence_,-.015f,.0003f);
      echo_=std::clamp(echo_,.01f,1.25f);
      if(echo_<.25f && coherence_<.1f) joint_only_baseline_=false;
      coherence_=std::clamp(coherence_,.005f,.72f);
    }
    return decision(hold_ ? 2u : 1u);
  }
  void reset() noexcept {ready_=joint_only_baseline_=false;learning_=hold_=confirmation_=clear_=0;echo_=coherence_=0;}
 private:
  Decision decision(unsigned state) const noexcept {
    return {state,hold_,clear_,echo_,coherence_};
  }
  bool ready_=false,joint_only_baseline_=false;
  unsigned learning_=0,hold_=0,confirmation_=0,clear_=0;
  float echo_=0,coherence_=0;
};
} // namespace itu
