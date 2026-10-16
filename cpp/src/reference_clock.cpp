#include "itu/reference_clock.hpp"
#include "common_audio/resampler/sinc_resampler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace itu {
namespace {
struct ClockFit {
  std::array<AudioTimestamp,32> points{};
  unsigned count=0,head=0;
  double rate=0;
  bool valid=false;
  bool observe(AudioTimestamp t,unsigned nominal) noexcept {
    if(t.monotonic_ns<=0) return false;
    if(count) {
      const auto last=points[(head+31)%32];
      if(t.monotonic_ns<=last.monotonic_ns || t.frame_position<=last.frame_position) return false;
      const double elapsed=double(t.monotonic_ns-last.monotonic_ns)*1e-9;
      const double hz=double(t.frame_position-last.frame_position)/elapsed;
      if(std::abs(hz/nominal-1)>.01) return false;
      if(elapsed<.2) return true; // Timestamp quantization needs a long baseline.
    }
    points[head]=t; head=(head+1)%32; count=std::min(32u,count+1);
    const auto origin=points[(head+32-count)%32];
    if(count<5 || t.monotonic_ns-origin.monotonic_ns<2000000000LL) return true;
    double sx=0,sy=0,sxx=0,sxy=0;
    for(unsigned i=0;i<count;++i) {
      const auto p=points[(head+32-count+i)%32];
      const double x=double(p.monotonic_ns-origin.monotonic_ns)*1e-9;
      const double y=double(p.frame_position-origin.frame_position);
      sx+=x; sy+=y; sxx+=x*x; sxy+=x*y;
    }
    const double denominator=count*sxx-sx*sx;
    if(denominator<=0) return false;
    const double candidate=(count*sxy-sx*sy)/denominator;
    if(std::abs(candidate/nominal-1)>.002) return false;
    rate=candidate; valid=true; return true;
  }
};
}
struct ReferenceClock::State : webrtc::SincResamplerCallback {
  unsigned rate,h;
  ClockFit capture,render;
  std::vector<float> fifo,scratch;
  std::size_t head=0,size=0;
  std::uint64_t rejected=0,underruns=0,overruns=0;
  double ratio=1;
  std::int64_t ratio_time=0;
  bool priming=true;
  webrtc::SincResampler resampler;
  explicit State(unsigned r):rate(r),h(r/100),fifo(r*2),scratch(h*2),resampler(1,h,this) {
    // Prime with zeros, exactly as the upstream push wrapper, to avoid future input.
    resampler.Resample(resampler.ChunkSize(),scratch.data()); priming=false;
  }
  void Run(std::size_t n,float* out) override {
    if(priming) {std::fill_n(out,n,0); return;}
    for(std::size_t i=0;i<n;++i) {out[i]=fifo[head]; head=(head+1)%fifo.size();}
    size-=n;
  }
  void update_ratio(AudioTimestamp t) noexcept {
    if(!capture.valid || !render.valid || t.monotonic_ns-ratio_time<1000000000LL) return;
    const double desired=render.rate/capture.rate;
    if(std::abs(desired-1)>.0015) return;
    if(std::abs(desired-ratio)>1e-6) {resampler.SetRatio(desired); ratio=desired;}
    ratio_time=t.monotonic_ns;
  }
};
ReferenceClock::ReferenceClock(unsigned rate) {
  if(rate!=16000 && rate!=24000 && rate!=48000) throw std::invalid_argument("clock rate must be 16/24/48 kHz");
  state_=std::make_unique<State>(rate);
}
ReferenceClock::~ReferenceClock()=default;
void ReferenceClock::reset() {auto fresh=std::make_unique<State>(state_->rate); state_=std::move(fresh);}
bool ReferenceClock::observe_capture(AudioTimestamp t) noexcept {
  auto& s=*state_; if(!s.capture.observe(t,s.rate)) {++s.rejected; return false;} s.update_ratio(t); return true;
}
bool ReferenceClock::observe_render(AudioTimestamp t) noexcept {
  auto& s=*state_; if(!s.render.observe(t,s.rate)) {++s.rejected; return false;} s.update_ratio(t); return true;
}
bool ReferenceClock::push(const float* pcm,std::size_t n) noexcept {
  auto& s=*state_; if(!pcm || !n) return false;
  for(std::size_t i=0;i<n;++i) if(!std::isfinite(pcm[i]) || std::abs(pcm[i])>1) return false;
  if(n>s.fifo.size()-s.size) {++s.overruns; return false;}
  for(std::size_t i=0;i<n;++i) s.fifo[(s.head+s.size+i)%s.fifo.size()]=pcm[i];
  s.size+=n; return true;
}
bool ReferenceClock::pull(float* pcm,std::size_t n) noexcept {
  auto& s=*state_; if(!pcm || n!=s.h) return false;
  // A bounded-rate 10 ms request can trigger at most two upstream callbacks.
  if(s.size<2*s.h) {++s.underruns; return false;}
  s.resampler.Resample(n,pcm); return true;
}
ClockInfo ReferenceClock::info() const noexcept {
  const auto& s=*state_;
  const bool both=s.capture.valid && s.render.valid;
  ClockInfo out;
  out.locked=both;
  out.drift_ppm=(s.ratio-1)*1e6;
  out.capture_rate_hz=s.capture.valid ? s.capture.rate : 0;
  out.render_rate_hz=s.render.valid ? s.render.rate : 0;
  out.desired_ppm=(both && s.capture.rate>0) ? (s.render.rate/s.capture.rate-1)*1e6 : 0;
  out.capture_valid=s.capture.valid;
  out.render_valid=s.render.valid;
  out.queued_samples=s.size;
  out.rejected_timestamps=s.rejected;
  out.underruns=s.underruns;
  out.overruns=s.overruns;
  return out;
}
}
