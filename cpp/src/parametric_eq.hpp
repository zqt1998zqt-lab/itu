#pragma once
#include "itu/audio_engine.hpp"
#include "modules/audio_processing/utility/cascaded_biquad_filter.h"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace itu {
class ParametricEq {
 public:
  static void validate(const EqConfig& c,unsigned rate) {
    if(c.bands>c.band.size() || !std::isfinite(c.preamp_db) || c.preamp_db>0 || c.preamp_db<-24)
      throw std::invalid_argument("EQ requires 0..8 bands and preamp -24..0 dB");
    float boosts=0;
    for(unsigned i=0;i<c.bands;++i) {
      const auto& b=c.band[i];
      if((b.kind!=EqKind::peak && b.kind!=EqKind::low_shelf && b.kind!=EqKind::high_shelf) ||
          !std::isfinite(b.frequency_hz) || b.frequency_hz<20 || b.frequency_hz>rate*.45f ||
          !std::isfinite(b.gain_db) || std::abs(b.gain_db)>12 ||
          !std::isfinite(b.q) || b.q<.2f || b.q>10)
        throw std::invalid_argument("EQ frequency 20..0.45*sample_rate, gain +/-12 dB, Q 0.2..10");
      boosts+=std::max(0.f,b.gain_db);
    }
    if(boosts+c.preamp_db>1e-5f) throw std::invalid_argument("EQ boosts require explicit negative preamp headroom");
  }
  static bool enabled(const EqConfig& c) {
    if(c.preamp_db!=0) return true;
    for(unsigned i=0;i<c.bands;++i) if(c.band[i].gain_db!=0) return true;
    return false;
  }
  ParametricEq(const EqConfig& c,unsigned rate):gain_(std::pow(10.f,c.preamp_db/20)),
      filter_(coefficients(c,rate)) {}
  void process(float* pcm,unsigned count) noexcept {
    for(unsigned i=0;i<count;++i) pcm[i]*=gain_;
    filter_.Process(std::span<float>(pcm,count));
  }
  unsigned delay() const noexcept {return 0;}
  std::size_t memory_bytes() const noexcept {return sizeof(*this);}
 private:
  using Coefficients=webrtc::CascadedBiQuadFilter::BiQuadCoefficients;
  static std::vector<Coefficients> coefficients(const EqConfig& c,unsigned rate) {
    validate(c,rate);
    std::vector<Coefficients> result;
    for(unsigned i=0;i<c.bands;++i) {
      const auto& b=c.band[i];
      if(b.gain_db==0) continue;
      // W3C Audio EQ Cookbook; normalization matches WebRTC's direct-form I.
      const double a=std::pow(10.,b.gain_db/40.), w=2*3.141592653589793*b.frequency_hz/rate;
      const double cs=std::cos(w),sn=std::sin(w),alpha=sn/(2*b.q),shelf=sn/std::sqrt(2.);
      double b0,b1,b2,a0,a1,a2;
      if(b.kind==EqKind::peak) {
        b0=1+alpha*a; b1=-2*cs; b2=1-alpha*a;
        a0=1+alpha/a; a1=-2*cs; a2=1-alpha/a;
      } else if(b.kind==EqKind::low_shelf) {
        const double t=2*std::sqrt(a)*shelf;
        b0=a*((a+1)-(a-1)*cs+t); b1=2*a*((a-1)-(a+1)*cs); b2=a*((a+1)-(a-1)*cs-t);
        a0=(a+1)+(a-1)*cs+t; a1=-2*((a-1)+(a+1)*cs); a2=(a+1)+(a-1)*cs-t;
      } else {
        const double t=2*std::sqrt(a)*shelf;
        b0=a*((a+1)+(a-1)*cs+t); b1=-2*a*((a-1)+(a+1)*cs); b2=a*((a+1)+(a-1)*cs-t);
        a0=(a+1)-(a-1)*cs+t; a1=2*((a-1)-(a+1)*cs); a2=(a+1)-(a-1)*cs-t;
      }
      result.push_back({{float(b0/a0),float(b1/a0),float(b2/a0)}, {float(a1/a0),float(a2/a0)}});
    }
    return result;
  }
  float gain_;
  webrtc::CascadedBiQuadFilter filter_;
};
}
