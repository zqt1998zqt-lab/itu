#include "native_pipeline.hpp"
#include "rate_converter.hpp"
#include "eq_processor.hpp"
#include "linear_phase_eq.hpp"
#include "parametric_eq.hpp"
#include "denormal_scope.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>

namespace itu {
struct NativePipeline::State {
  Config c;
  unsigned h;
  AudioEngine core;
  std::unique_ptr<EqProcessor> output_eq;
  std::vector<float> capture,channel,aec,conditioned;
  std::array<float,480> output{},beam{},scratch{};
  std::vector<float> render;
  std::vector<std::unique_ptr<RateConverter>> up,aec_down,input_down,render_up;
  RateConverter output_down,beam_down;
  static Config core_config(Config c) {
    const unsigned io_rate=c.sample_rate;
    // Freeze the FIR geometry before the internal 48 kHz rate is selected.
    // The taps and their physical duration belong to the configured I/O EQ,
    // not to a second interpretation at the bridge rate.
    if(c.output_eq.phase==EqPhase::linear && c.output_eq.linear_phase_taps==0)
      c.output_eq.linear_phase_taps=
          LinearPhaseEq::taps_for(c.output_eq,io_rate);
    if(!c.eq_before_agc) {
      if(ParametricEq::enabled(c.output_eq)) c.limiter=false;
      c.output_eq={}; // Legacy post-AGC EQ runs exactly once at the I/O rate.
    }
    // RX belongs to the outer engine at its playback rate, never the TX core.
    c.rx_processing=c.rx_config.enabled=false;
    c.policy_sample_rate=io_rate;
    const float ratio=48000.f/c.sample_rate;
    for(float& delay:c.beam_delays_samples) delay*=ratio;
    c.sample_rate=48000; c.frame_size=480; c.native_fullband=false;
    return c;
  }
  explicit State(Config config):c(config),h(c.frame_size),core(core_config(c),12288,true),
      output_eq(!c.eq_before_agc && ParametricEq::enabled(c.output_eq) ? std::make_unique<EqProcessor>(c.output_eq,c.sample_rate) : nullptr),
      capture(480*c.microphones),channel(h),aec(capture.size()),conditioned(capture.size()),
      render(480*c.reference_channels),output_down(480,h),beam_down(480,h) {
    for(unsigned ch=0;ch<c.microphones;++ch) {
      up.emplace_back(std::make_unique<RateConverter>(h,480));
      aec_down.emplace_back(std::make_unique<RateConverter>(480,h));
      input_down.emplace_back(std::make_unique<RateConverter>(480,h));
    }
    for(unsigned ch=0;ch<c.reference_channels;++ch) render_up.emplace_back(std::make_unique<RateConverter>(h,480));
  }
};
NativePipeline::NativePipeline(const Config& c):state_(std::make_unique<State>(c)) {}
NativePipeline::~NativePipeline()=default;
std::size_t NativePipeline::memory_bytes() const noexcept {
  const auto& s=*state_;
  return sizeof(State)+(s.capture.capacity()+s.channel.capacity()+s.aec.capacity()+s.conditioned.capacity()+s.render.capacity())*sizeof(float)+
      s.core.working_memory_bytes()+(s.output_eq ? s.output_eq->memory_bytes() : 0);
}
Status NativePipeline::process(const float* mic,const float* ref,float* out,FrameInfo& f,
    float* at,float* bt,float* lt,float* it,const PostAecContext* context) noexcept {
  auto& s=*state_;
  using Clock=std::chrono::steady_clock;
  auto mark=s.c.profile_stages ? Clock::now() : Clock::time_point{};
  auto stage=[&](float& measurement) {
    if(s.c.profile_stages) {
      auto now=Clock::now();
      measurement=std::chrono::duration<float,std::milli>(now-mark).count(); mark=now;
    }
  };
  float capture_ms=0,render_ms=0;
  for(unsigned ch=0;ch<s.c.microphones;++ch) {
    for(unsigned i=0;i<s.h;++i) {
      const float value=mic[i*s.c.microphones+ch];
      // Match the NS component's native-rate PCM16 entry conversion once,
      // before SRC. The inner module retains the resampled float precision.
      s.channel[i]=s.c.vehicle_ns_control ?
          std::lround(std::clamp(value*32768.f,-32768.f,32767.f))/32768.f : value;
    }
    if(s.up[ch]->Resample(s.channel.data(),s.h,s.output.data(),480)) return Status::backend_error;
    for(unsigned i=0;i<480;++i) s.capture[i*s.c.microphones+ch]=s.output[i];
  }
  stage(capture_ms);
  if(ref) for(unsigned ch=0;ch<s.c.reference_channels;++ch) {
    for(unsigned i=0;i<s.h;++i) s.channel[i]=ref[i*s.c.reference_channels+ch];
    if(s.render_up[ch]->Resample(s.channel.data(),s.h,s.scratch.data(),480)) return Status::backend_error;
    for(unsigned i=0;i<480;++i) s.render[i*s.c.reference_channels+ch]=s.scratch[i];
  }
  stage(render_ms);
  auto status=s.core.process_impl(s.capture.data(),s.capture.size(),ref?s.render.data():nullptr,ref?s.render.size():0,
      s.output.data(),480,f,s.aec.data(),s.beam.data(),lt,s.conditioned.data(),context);
  if(status!=Status::ok) return status;
  f.native_capture_ms=capture_ms; f.native_render_ms=render_ms;
  stage(f.native_core_ms);
  if(s.output_down.Resample(s.output.data(),480,out,s.h)) return Status::backend_error;
  if(s.c.diagnostic_stages && !s.c.eq_before_agc) std::copy_n(out,s.h,s.c.diagnostic_stages->pre_eq.data());
  if(s.output_eq) {DenormalScope denormals; s.output_eq->process(out,s.h);}
  if(s.c.diagnostic_stages && !s.c.eq_before_agc) std::copy_n(out,s.h,s.c.diagnostic_stages->post_eq.data());
  for(unsigned i=0;i<s.h;++i) if(s.c.limiter && std::abs(out[i])>s.c.output_ceiling) {
    out[i]=std::copysign(s.c.output_ceiling,out[i]); ++f.limited_samples;
  }
  double render_power=0;
  if(ref) for(unsigned i=0;i<s.h*s.c.reference_channels;++i)
    render_power+=double(ref[i])*ref[i];
  const bool render_quiet=!ref || render_power/std::max(1u,s.h*s.c.reference_channels)<=s.c.activity_floor;
  const bool quiet_boundary=f.pre_ns_rms<1.5e-4f && render_quiet &&
      f.near_protection_hold_frames==0 && f.input_excess_score<.05f &&
      f.speech_probability<.05f && f.linear_speech_probability<.05f &&
      f.nonlinear_speech_probability<.05f;
  if(quiet_boundary && !s.c.vehicle_ns_control) {
    float quiet_peak=0;
    for(unsigned i=0;i<s.h;++i) quiet_peak=std::max(quiet_peak,std::abs(out[i]));
    if(quiet_peak<5e-4f) std::fill_n(out,s.h,0.f);
    else for(unsigned i=0;i<s.h;++i) if(std::abs(out[i])<1.5e-4f) out[i]=0.f;
  }
  stage(f.native_output_ms);
  for(unsigned ch=0;ch<s.c.microphones;++ch) {
    for(auto pair:{std::pair<float*,std::vector<float>*>(at,&s.aec),{it,&s.conditioned}}) {
      for(unsigned i=0;i<480;++i) s.scratch[i]=(*pair.second)[i*s.c.microphones+ch];
      auto& converter=pair.second==&s.aec ? s.aec_down[ch] : s.input_down[ch];
      if(converter->Resample(s.scratch.data(),480,s.channel.data(),s.h)) return Status::backend_error;
      if(pair.first) for(unsigned i=0;i<s.h;++i) pair.first[i*s.c.microphones+ch]=s.channel[i];
    }
  }
  if(s.beam_down.Resample(s.beam.data(),480,s.channel.data(),s.h)) return Status::backend_error;
  if(bt) std::copy(s.channel.begin(),s.channel.end(),bt);
  stage(f.native_taps_ms);
  const double ratio=double(s.c.sample_rate)/48000;
  const unsigned conversion=unsigned(std::lround(s.up[0]->output_delay()*ratio+s.output_down.output_delay()));
  const unsigned legacy_eq_delay=s.output_eq ? s.output_eq->delay() : 0;
  f.latency_samples=conversion+legacy_eq_delay+
      unsigned(std::ceil(f.latency_samples*ratio));
  f.aec_output_delay_samples=conversion+unsigned(std::lround(f.aec_output_delay_samples*ratio));
  f.aec_input_delay_samples=conversion+unsigned(std::lround(f.aec_input_delay_samples*ratio));
  if(s.c.aec) f.linear_aec_delay_samples+=unsigned(std::lround(s.up[0]->output_delay()/3));
  f.low_cut_delay_samples=unsigned(std::lround(f.low_cut_delay_samples*ratio));
  f.post_low_cut_delay_samples=unsigned(std::lround(f.post_low_cut_delay_samples*ratio));
  f.ns_delay_samples=unsigned(std::lround(f.ns_delay_samples*ratio));
  f.spatial_delay_samples=unsigned(std::ceil(f.spatial_delay_samples*ratio));
  f.bf_delay_samples=conversion+unsigned(std::ceil(f.bf_delay_samples*ratio));
  if(s.c.diagnostic_stages) {
    auto& taps=*s.c.diagnostic_stages;
    // The core taps also include the causal input-upsample delay.
    taps.core_delay_samples+=s.up[0]->output_delay();
    if(s.c.eq_before_agc) taps.eq_delay_samples+=s.up[0]->output_delay();
    else {
      taps.eq_sample_rate=s.c.sample_rate; taps.eq_frame_size=s.h;
      taps.eq_delay_samples=f.latency_samples;
    }
  }
  double p=0; for(unsigned i=0;i<s.h;++i) p+=double(out[i])*out[i];
  f.output_rms=float(std::sqrt(p/s.h));
  return Status::ok;
}
}
