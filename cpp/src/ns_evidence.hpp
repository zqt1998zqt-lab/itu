/*
 * NS statistics adapted from the pinned WebRTC noise_suppressor.cc.
 * Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 * BSD license in the pinned WebRTC LICENSE; PATENTS grant also applies.
 */
#pragma once
#include "modules/audio_processing/ns/ns_fft.h"
#include "modules/audio_processing/ns/noise_estimator.h"
#include "modules/audio_processing/ns/speech_probability_estimator.h"
#include "modules/audio_processing/ns/wiener_filter.h"
#include <algorithm>
#include <array>
#include <cmath>
namespace itu {
// NS noise/SNR posterior observation only. No RNN, external model, output PCM,
// activity gate or lookahead. This is still statistical speech-presence evidence.
class NsEvidence {
 public:
  NsEvidence():params_(webrtc::NsConfig::SuppressionLevel::k6dB),noise_(params_),filter_(params_) {
    for(unsigned i=0;i<256;++i)window_[i]=i<96 ? std::sin(3.141592653589793*i/192) : i<=160 ? 1.f : std::sin(3.141592653589793*(256-i)/192);
  }
  float observe(const float* input) noexcept {
    noise_.PrepareAnalysis();
    std::move(history_.begin()+160,history_.end(),history_.begin());
    for(unsigned i=0;i<160;++i)history_[i+96]=input[i]*32768.f;
    double power=0;for(float x:history_)power+=double(x)*x;
    if(power<1e-16)return 0;
    ++frames_;
    for(unsigned i=0;i<256;++i)time_[i]=history_[i]*window_[i];
    fft_.Fft(time_,real_,imag_);
    float energy=0,sum=0;
    for(unsigned i=0;i<129;++i) {
      signal_[i]=(i==0 || i==128 ? std::abs(real_[i]) : std::sqrt(real_[i]*real_[i]+imag_[i]*imag_[i]))+1.f;
      sum+=signal_[i];energy+=real_[i]*real_[i]+imag_[i]*imag_[i];
    }
    noise_.PreUpdate(frames_,signal_,sum);
    const auto np=noise_.get_noise_spectrum(),prev=noise_.get_prev_noise_spectrum(),gain=filter_.get_filter();
    for(unsigned i=0;i<129;++i) {
      post_[i]=std::max(0.f,signal_[i]/(np[i]+.0001f)-1.f);
      prior_[i]=.98f*previous_[i]/(prev[i]+.0001f)*gain[i]+.02f*post_[i];
    }
    presence_.Update(frames_,prior_,post_,noise_.get_conservative_noise_spectrum(),signal_,sum,energy/129);
    noise_.PostUpdate(presence_.get_probability(),signal_);
    filter_.Update(frames_,noise_.get_noise_spectrum(),noise_.get_prev_noise_spectrum(),noise_.get_parametric_noise_spectrum(),signal_);
    previous_=signal_;
    const auto probability=presence_.get_probability();
    double weighted=0,total=0;for(unsigned i=2;i<112;++i){weighted+=probability[i]*signal_[i];total+=signal_[i];}
    return total>0 ? std::clamp(float(weighted/total),0.f,1.f) : 0;
  }
 private:
  webrtc::SuppressionParams params_;webrtc::NoiseEstimator noise_;webrtc::WienerFilter filter_;
  webrtc::SpeechProbabilityEstimator presence_;webrtc::NrFft fft_;
  std::array<float,256>history_{},time_{},window_{},real_{},imag_{};
  std::array<float,129>signal_{},previous_{},prior_{},post_{};
  int frames_=-1;
};
}
