#include "itu/vehicle_ns.hpp"
#include "vehicle_ns_controller.hpp"
#include "vehicle_rumble_filter.hpp"
#include "ns_repair_alignment.hpp"
#include "speex_ns.hpp"
#include "rate_converter.hpp"
#include "omlsa_imcra.hpp"
#include "unified_ns.hpp"
#include "hybrid_ns_control.hpp"
#include "residual_noise_suppressor.hpp"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/ns/noise_suppressor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace itu {
namespace {
class TransientSpectralGuard final : public detail::NsSpectralControl {
 public:
  TransientSpectralGuard(const VehicleNsController* controller, float max_db)
      : controller_(controller), max_db_(max_db) {}
  void observe(detail::NsSpectralFrame frame) noexcept override {
    double probability_sum = 0;
    for (unsigned k = 1; k < frame.bins; ++k)
      probability_sum += frame.probability[k];
    const float mean_probability = frame.bins > 1 ?
        float(probability_sum / (frame.bins - 1)) : 0.f;
    const bool indicator =
        controller_->transient_periodicity_500ms() > .55f;
    const float low_posterior = indicator ? std::clamp(
        (.30f - mean_probability) / .20f, 0.f, 1.f) : 1.f;
    const float evidence = controller_->transient_noise_evidence() *
                           low_posterior;
    float& hold = indicator ? indicator_hold_ : wind_hold_;
    hold = std::max(evidence, hold * std::exp(-10.f / 240.f));
    // The release envelope keeps repeated pulses connected, but it must not
    // continue attenuating a speech onset after the current transient evidence
    // has disappeared. Require live evidence before applying the held depth.
    const float current_evidence = std::clamp(evidence / .10f, 0.f, 1.f);
    const float strength = max_db_ * hold * current_evidence;
    const float speech = std::clamp(
        (mean_probability - .45f) / .25f, 0.f, 1.f);
    last_applied_ = 1.f;
    if (strength <= 0) return;
    for (unsigned k = 1; k < frame.bins; ++k) {
      const float frequency = float(k) * frame.rate / frame.fft_size;
      const bool selected = indicator ?
          frequency >= 1000.f && frequency < 8000.f : frequency < 300.f;
      if (!selected) continue;
      // Protect whole frames when the independent statistical posterior
      // supports speech, but do not let a high posterior on the transient's
      // own high-energy bins shield those bins inside an otherwise
      // noise-dominant frame. The transient evidence already carries the
      // class-specific periodicity/shape and low-confidence checks.
      const float attenuation_db = strength * (1.f - speech);
      const float factor = std::pow(10.f, -attenuation_db / 20.f);
      frame.gain[k] *= factor;
      last_applied_ = std::min(last_applied_, factor);
    }
  }
  float last_applied() const noexcept { return last_applied_; }
 private:
  const VehicleNsController* controller_;
  float max_db_, wind_hold_ = 0, indicator_hold_ = 0;
  float last_applied_ = 1;
};
}
struct VehicleNsProcessor::Impl {
  unsigned rate,hop,core_rate,core_hop,repair_mode,startup_ms,core_delay,delay;
  int strength,mild_strength;
  float dry_max,snr_threshold,noise_gain,slew_ms,release_ms,quiet_floor_db;
  unsigned floor_guard;
  std::string backend,policy;
  bool direct_fullband,quiet_enabled,policy_enabled=false,dual=false,adaptive_rumble;
  bool dynamic_rumble_corner=false;
  bool quantize_input;
  bool residual_enabled=false;
  bool unified_raw_budget=false;
  float unified_high_snr_bypass_db=0,unified_high_snr_probability=.80f;
  bool unified_high_snr_bypass=false;
  float raw_mix_target=0,raw_mix_applied_min=0,raw_mix_applied_max=0;
  float residual_max_db=0,residual_speech_guard=.75f;
  float residual_gate_snr_db=-60,residual_gate_width_db=3;
  double src_delay;
  std::unique_ptr<VehicleNsController> controller;
  std::unique_ptr<RateConverter> up,down;
  VehicleRumbleFilter rumble,mild_rumble;
  LinearRumbleFilter linear_filter;
  std::unique_ptr<SpeexNs> speex,mild_speex,steady_speex;
  std::unique_ptr<detail::OmlsaImcra> omlsa;
  std::unique_ptr<detail::UnifiedNs> unified;
  std::unique_ptr<detail::HybridNsControl> student;
  std::unique_ptr<detail::SpeechBandProtection> speech_band;
  std::unique_ptr<TransientSpectralGuard> transient_spectral;
  std::unique_ptr<detail::NsSpectralChain> spectral_chain;
  std::unique_ptr<ResidualNoiseSuppressor> residual;
  std::unique_ptr<webrtc::NoiseSuppressor> ns,mild_ns;
  std::unique_ptr<webrtc::AudioBuffer> buffer,mild_buffer,neutral_buffer,startup_buffer;
  std::vector<float> neutral,dry_history,startup_raw,startup_history,input,core,output,rumble_amount;
  unsigned dry_head=0,startup_head=0,strength_changes=0;
  float previous_dry=0,previous_gain=1,dynamic_db,startup_target=0;
  float rumble_speech_hold=0,rumble_target=0,rumble_applied=0;
  float rumble_recovery=0,rumble_credible_hold=0,rumble_adaptive_hz=0,rumble_corner_hz=0;
  float rumble_hz_base=0;
  float noise_step_recovery=0;
  float steady_speech_hold=0,steady_mix_target=0;
  float spectral_dry_guard=0,spectral_hold=0,spectral_guard_applied=1;
  float transient_dry_guard=0,transient_guard_applied=1;
  NsRepairAlignment aligned,startup_control,rumble_control;
  std::size_t frame=0;
  static const char* name(VehicleNsBackend b) {
    switch(b) {case VehicleNsBackend::webrtc:return "webrtc"; case VehicleNsBackend::speex:return "speex";
      case VehicleNsBackend::bypass:return "bypass";case VehicleNsBackend::filterbank:return "filterbank";
      case VehicleNsBackend::omlsa_imcra:return "omlsa_imcra";
      case VehicleNsBackend::super_gaussian:return "super_gaussian";
      case VehicleNsBackend::unified:return "unified";}
    throw std::invalid_argument("invalid vehicle NS backend");
  }
  static unsigned checked_rate(unsigned r,const VehicleNsConfig& c) {
    const bool dual_policy=(c.backend==VehicleNsBackend::webrtc ||
        c.backend==VehicleNsBackend::speex) &&
        (c.policy=="parallel" || c.policy=="parallel_start");
    if(c.unified_raw_budget &&
       (dual_policy || c.policy=="dynamic" || c.policy=="steady_dynamic"))
      throw std::invalid_argument(
          "unified raw budget requires a raw donor; use aligned/fixedblend/legacy "
          "(statistical backends also support parallel_start)");
    if(!c.student_model_path.empty() && c.backend!=VehicleNsBackend::omlsa_imcra)
      throw std::invalid_argument("hybrid student requires the native OM-LSA backend");
    if((r!=16000 && r!=24000 && r!=48000) ||
       (c.native_24 && (r!=24000 || (c.backend!=VehicleNsBackend::speex && c.backend!=VehicleNsBackend::omlsa_imcra && c.backend!=VehicleNsBackend::super_gaussian && c.backend!=VehicleNsBackend::unified))) ||
       c.repair_mode>4 || c.floor_guard>2 || c.startup_ms>500 ||
       !std::isfinite(c.dry_max+c.snr_threshold_db+c.noise_gain+c.slew_ms+c.release_ms+c.quiet_floor_db+c.spectral_dry_guard+c.unified_learning_rate+c.unified_minima_rebuild_slew_db) ||
       c.spectral_dry_guard<0 || c.spectral_dry_guard>1 ||
       (c.spectral_dry_guard>0 && c.backend!=VehicleNsBackend::omlsa_imcra && c.backend!=VehicleNsBackend::super_gaussian) ||
       c.dry_max<0 || c.dry_max>1 || c.snr_threshold_db<0 || c.snr_threshold_db>30 ||
       c.noise_gain<.2f || c.noise_gain>1 || c.slew_ms<0 || c.slew_ms>500 ||
       c.release_ms<0 || c.release_ms>500 || c.quiet_floor_db< -300 || c.quiet_floor_db> -50 ||
       !std::isfinite(c.residual_max_db+c.residual_speech_guard) ||
       c.residual_max_db<0 || c.residual_max_db>24 ||
       c.residual_speech_guard<0 || c.residual_speech_guard>1 ||
       !std::isfinite(c.residual_gate_snr_db+c.residual_gate_width_db) ||
       c.residual_gate_snr_db< -60 || c.residual_gate_snr_db>30 ||
       c.residual_gate_width_db<=0 || c.residual_gate_width_db>20 ||
       !std::isfinite(c.rumble_recovery) || c.rumble_recovery<0 || c.rumble_recovery>1 ||
       (c.rumble_recovery>0 && !c.adaptive_rumble) ||
       !std::isfinite(c.rumble_adaptive_hz) || c.rumble_adaptive_hz<0 || c.rumble_adaptive_hz>120 ||
       (c.rumble_adaptive_hz>0 && (!c.adaptive_rumble || c.rumble_adaptive_hz<=c.rumble_hz)) ||
       !std::isfinite(c.weak_speech_recovery) || c.weak_speech_recovery<0 || c.weak_speech_recovery>1 ||
       !std::isfinite(c.speech_band_recovery) || c.speech_band_recovery<0 || c.speech_band_recovery>1 ||
       (c.speech_band_recovery>0 && c.backend!=VehicleNsBackend::omlsa_imcra) ||
       !std::isfinite(c.noise_step_recovery) || c.noise_step_recovery<0 || c.noise_step_recovery>1 ||
       !std::isfinite(c.transient_dry_guard) || c.transient_dry_guard<0 || c.transient_dry_guard>1 ||
       (c.transient_dry_guard>0 && c.backend!=VehicleNsBackend::omlsa_imcra) ||
       !std::isfinite(c.transient_spectral_guard_db) ||
       c.transient_spectral_guard_db<0 || c.transient_spectral_guard_db>24 ||
       (c.transient_spectral_guard_db>0 &&
        c.backend!=VehicleNsBackend::omlsa_imcra) ||
       (c.backend==VehicleNsBackend::webrtc && (c.strength<0 || c.strength>3 || c.mild_strength<0 || c.mild_strength>3)) ||
       ((c.backend==VehicleNsBackend::omlsa_imcra || c.backend==VehicleNsBackend::super_gaussian || c.backend==VehicleNsBackend::unified) && (c.strength<0 || c.strength>3)) ||
       (c.backend==VehicleNsBackend::speex && (c.strength< -40 || c.strength>0 || c.mild_strength< -40 || c.mild_strength>0)) ||
       (c.policy!="legacy" && c.policy!="aligned" && c.policy!="parallel" && c.policy!="dynamic" && c.policy!="steady_dynamic" && c.policy!="fixedblend" && c.policy!="parallel_start") ||
       ((c.policy=="dynamic" || c.policy=="steady_dynamic") && c.backend!=VehicleNsBackend::speex) ||
       c.steady_suppression_db < -40 || c.steady_suppression_db>0 ||
       c.unified_learning_rate<0 || c.unified_learning_rate>1 ||
       c.unified_minima_rebuild_slew_db<0 || c.unified_minima_rebuild_slew_db>24 ||
       c.unified_minima_history_depth<1 || c.unified_minima_history_depth>8 ||
       c.unified_conditional_minima_history_depth<1 || c.unified_conditional_minima_history_depth>8 ||
       !std::isfinite(c.unified_high_snr_bypass_db+c.unified_high_snr_probability) ||
       c.unified_high_snr_bypass_db<0 || c.unified_high_snr_bypass_db>60 ||
       c.unified_high_snr_probability<0 || c.unified_high_snr_probability>1 ||
       (c.unified_high_snr_bypass_db>0 && !c.unified_raw_budget) ||
       ((c.unified_minima_rebuild_slew_db>0 || c.unified_minima_history_depth!=8 ||
         c.unified_conditional_minima_history_depth!=8) && c.backend!=VehicleNsBackend::unified) ||
       (c.adaptive_rumble && (!c.linear_rumble || !(c.rumble_hz>0))))
      throw std::invalid_argument("invalid vehicle NS operating envelope");
    return r;
  }
  Impl(unsigned r,const VehicleNsConfig& c,bool quantize)
    : rate(checked_rate(r,c)),hop(rate/100),core_rate(rate==24000 && !c.native_24 && c.backend!=VehicleNsBackend::omlsa_imcra && c.backend!=VehicleNsBackend::super_gaussian && c.backend!=VehicleNsBackend::unified ? 48000:rate),core_hop(core_rate/100),
      repair_mode(c.repair_mode),startup_ms(c.startup_ms),
      core_delay(c.backend==VehicleNsBackend::speex || c.backend==VehicleNsBackend::omlsa_imcra || c.backend==VehicleNsBackend::super_gaussian || c.backend==VehicleNsBackend::unified ? core_hop:c.backend==VehicleNsBackend::webrtc ? (core_rate==48000?336:96):c.backend==VehicleNsBackend::filterbank && core_rate==48000?48:0),
      delay(0),strength(c.strength),mild_strength(c.mild_strength),dry_max(c.dry_max),snr_threshold(c.snr_threshold_db),noise_gain(c.noise_gain),
      slew_ms(c.slew_ms),release_ms(c.release_ms),quiet_floor_db(c.quiet_floor_db),floor_guard(c.floor_guard),backend(name(c.backend)),policy(c.policy),
      direct_fullband(c.unified_raw_budget || repair_mode>=2),quiet_enabled(quiet_floor_db> -200 || repair_mode>=3),adaptive_rumble(c.adaptive_rumble),quantize_input(quantize),src_delay(0),
      rumble(core_rate,c.linear_rumble?0:c.rumble_hz),mild_rumble(core_rate,c.linear_rumble?0:c.rumble_hz),linear_filter(core_rate,c.linear_rumble?c.rumble_hz:0),
      neutral(core_hop),dry_history(std::max(1u,core_delay-(backend=="webrtc" && core_rate==48000 ? 48u:0u))),
      startup_raw(core_hop),startup_history(direct_fullband?std::max(1u,core_delay):dry_history.size()),input(hop),core(core_hop),output(hop),rumble_amount(core_hop),
      dynamic_db(float(strength)),aligned(core_delay,core_hop,slew_ms,c.unified_raw_budget ? false : release_ms>0),startup_control(core_delay,core_hop,0,release_ms>0 || quiet_enabled),
      rumble_control(core_delay+linear_filter.delay(),core_hop,30,false) {
    unified_raw_budget=c.unified_raw_budget;
    unified_high_snr_bypass_db=c.unified_high_snr_bypass_db;
    unified_high_snr_probability=c.unified_high_snr_probability;
    spectral_dry_guard=c.spectral_dry_guard;
    rumble_recovery=c.rumble_recovery;
    rumble_adaptive_hz=c.rumble_adaptive_hz;
    rumble_hz_base=c.rumble_hz;
    dynamic_rumble_corner=rumble_adaptive_hz>c.rumble_hz;
    noise_step_recovery=c.noise_step_recovery;
    transient_dry_guard=c.transient_dry_guard;
    residual_enabled=c.residual_max_db>0;
    residual_max_db=c.residual_max_db;
    residual_speech_guard=c.residual_speech_guard;
    residual_gate_snr_db=c.residual_gate_snr_db;
    residual_gate_width_db=c.residual_gate_width_db;
    policy_enabled=dry_max>0 || noise_gain<1 || c.weak_speech_recovery>0 ||
        noise_step_recovery>0 || transient_dry_guard>0 ||
        c.transient_spectral_guard_db>0;
    dual=(backend=="speex" || backend=="webrtc") && (policy=="parallel" || policy=="parallel_start");
    if(policy_enabled || adaptive_rumble)
      controller=std::make_unique<itu::VehicleNsController>(
          core_rate,dry_max,snr_threshold,noise_gain,startup_ms,release_ms,
          quiet_floor_db,floor_guard,repair_mode,c.weak_speech_recovery,
          noise_step_recovery,transient_dry_guard);
    if(core_rate!=rate) {
      up=std::make_unique<itu::RateConverter>(hop,core_hop);
      down=std::make_unique<itu::RateConverter>(core_hop,hop);
    }
    if(backend=="speex") speex=std::make_unique<itu::SpeexNs>(core_rate,strength);
    if(backend=="omlsa_imcra" || backend=="super_gaussian") {
      constexpr float floors[]={-9.f,-12.f,-18.f,-24.f};
      omlsa=std::make_unique<detail::OmlsaImcra>(core_rate,floors[strength],true,
          backend=="super_gaussian" ? detail::SpectralGainRule::super_gaussian_map:detail::SpectralGainRule::omlsa);
      if(!c.student_model_path.empty()) student=std::make_unique<detail::HybridNsControl>(
          core_rate,omlsa->fft_size(),c.student_model_path,floors[strength],
          c.student_spp_mix,c.student_snr_mix,c.student_speech_floor_db);
      if(c.speech_band_recovery>0) {
        speech_band=std::make_unique<detail::SpeechBandProtection>(
            core_rate,omlsa->fft_size(),c.speech_band_recovery);
      }
      if(c.transient_spectral_guard_db>0)
        transient_spectral=std::make_unique<TransientSpectralGuard>(
            controller.get(),c.transient_spectral_guard_db);
      if(spectral_chain || student || speech_band || transient_spectral)
        spectral_chain=std::make_unique<detail::NsSpectralChain>(
            student.get(),speech_band.get(),transient_spectral.get());
    }
    if(policy=="steady_dynamic") steady_speex=std::make_unique<itu::SpeexNs>(core_rate,c.steady_suppression_db);
    if(backend=="unified") {
      constexpr float floors[]={-9.f,-12.f,-18.f,-24.f};
      unified=std::make_unique<detail::UnifiedNs>(
          core_rate,floors[strength],c.unified_learning_rate,
          c.unified_minima_rebuild_slew_db,c.unified_minima_history_depth,
          c.unified_conditional_minima_history_depth);
    }
    if(residual_enabled) {
      residual=std::make_unique<ResidualNoiseSuppressor>(
          core_rate,residual_max_db,residual_speech_guard,
          residual_gate_snr_db,residual_gate_width_db);
    }
    if(backend=="webrtc") {
      webrtc::NsConfig config;
      config.target_level=static_cast<webrtc::NsConfig::SuppressionLevel>(strength);
      ns=std::make_unique<webrtc::NoiseSuppressor>(config,core_rate,1);
    }
    if(dual && policy_enabled) {
      if(backend=="speex") mild_speex=std::make_unique<itu::SpeexNs>(core_rate,mild_strength);
      else if(backend=="webrtc") {
        webrtc::NsConfig c;c.target_level=static_cast<webrtc::NsConfig::SuppressionLevel>(mild_strength);
        mild_ns=std::make_unique<webrtc::NoiseSuppressor>(c,core_rate,1);
        mild_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
      }
    }
    if(backend=="webrtc" || backend=="filterbank")
      buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    if(policy_enabled && backend=="webrtc" && !dual) neutral_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    // A bounded, explicitly configured cold-start crossfade. Both NS states
    // keep learning throughout; raw PCM never enters either noise model twice.
    if((policy=="parallel_start" || quiet_enabled) && backend=="webrtc" && !direct_fullband)
      startup_buffer=std::make_unique<webrtc::AudioBuffer>(core_rate,1,core_rate,1,core_rate,1);
    src_delay=up ? up->output_delay()*double(rate)/core_rate+down->output_delay():0.;
    delay=unsigned(std::llround((core_delay+linear_filter.delay())*double(rate)/core_rate+src_delay));
  }
  bool process(float* samples,unsigned count) noexcept {
    if(!samples || count!=hop) return false;
    for(unsigned i=0;i<count;++i) if(!std::isfinite(samples[i])) return false;
    raw_mix_target=raw_mix_applied_min=raw_mix_applied_max=0;
    unified_high_snr_bypass=false;
    for(unsigned i=0;i<hop;++i) input[i]=quantize_input ? std::lround(std::clamp(samples[i]*32768.f,-32768.f,32767.f))/32768.f : samples[i];
      if(up) {
        if(up->Resample(input.data(),hop,core.data(),core_hop)) return false;
      } else std::copy(input.begin(),input.end(),core.begin());
      if(controller) {
        controller->observe(core.data());
        if(unified_raw_budget || policy=="parallel_start" || quiet_enabled) {
          std::copy(core.begin(),core.end(),startup_raw.begin());
          if(startup_buffer) {
            const float* in[]={startup_raw.data()};float* out[]={startup_raw.data()};
            startup_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
            if(core_rate==48000) startup_buffer->SplitIntoFrequencyBands();
            if(core_rate==48000) startup_buffer->MergeFrequencyBands();
            startup_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
          }
          // max(1,delay) is storage capacity, not a physical one-sample
          // delay for zero-latency backends. Preserve historical behavior
          // unless the explicit unified mode selected the true raw donor.
          if(!unified_raw_budget || core_delay) for(unsigned i=0;i<core_hop;++i) {
            const float delayed=startup_history[startup_head];startup_history[startup_head]=startup_raw[i];startup_raw[i]=delayed;
            startup_head=(startup_head+1)%startup_history.size();
          }
          // Immediate protection attack; smooth release after the fixed bound.
          startup_target=policy=="parallel_start" && frame*10 < unsigned(startup_ms) ? 1.f : startup_target*std::exp(-10.f/std::max(10.f,slew_ms));
          startup_control.begin(std::max(startup_target,controller->quiet()),1.f);
        }
        std::copy(core.begin(),core.end(),neutral.begin());
        if(neutral_buffer && policy!="dynamic") {
          const float* in[]={neutral.data()};float* out[]={neutral.data()};
          neutral_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
          if(core_rate==48000) neutral_buffer->SplitIntoFrequencyBands();
          if(core_rate==48000) neutral_buffer->MergeFrequencyBands();
          neutral_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
        }
        if(core_delay && !dual && policy!="dynamic" && policy!="steady_dynamic") for(unsigned i=0;i<core_hop;++i) {
          const float delayed=dry_history[dry_head];dry_history[dry_head]=neutral[i];neutral[i]=delayed;
          dry_head=(dry_head+1)%dry_history.size();
        }
      }
      if(controller && (policy=="dynamic" || policy=="steady_dynamic")) {
        const float target=float(strength)+(mild_strength-strength)*controller->dry();
        dynamic_db+=(1.f-std::exp(-10.f/std::max(10.f,slew_ms)))*(target-dynamic_db);
        const int db=int(std::lround(dynamic_db));
        speex->suppression(db);++strength_changes;
      }
      if(mild_speex) mild_speex->process(neutral.data());
      if(steady_speex) steady_speex->process(neutral.data());
      if(mild_buffer) {
        const float* in[]={neutral.data()};float* out[]={neutral.data()};
        mild_buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
        if(core_rate==48000) mild_buffer->SplitIntoFrequencyBands();
        mild_ns->Analyze(*mild_buffer);mild_ns->Process(mild_buffer.get());
        if(core_rate==48000) mild_buffer->MergeFrequencyBands();
        mild_buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
      }
      if(speex) speex->process(core.data());
      detail::NsSpectralControl* spectral_control =
          spectral_chain ? static_cast<detail::NsSpectralControl*>(spectral_chain.get()) :
          student ? static_cast<detail::NsSpectralControl*>(student.get()) : nullptr;
      if(omlsa) omlsa->process(core.data(),spectral_control,noise_step_recovery*(controller?controller->noise_step_probability():0.f));
      if(unified) {
        unified->process(core.data());
        unified_high_snr_bypass =
            unified_high_snr_bypass_db > 0.f &&
            unified->frame_snr_db() >= unified_high_snr_bypass_db &&
            unified->mean_probability() >= unified_high_snr_probability;
      }
      if(buffer) {
        const float* in[]={core.data()};float* out[]={core.data()};
        buffer->CopyFrom(in,webrtc::StreamConfig(core_rate,1));
        if(core_rate==48000) buffer->SplitIntoFrequencyBands();
        if(ns) {ns->Analyze(*buffer); ns->Process(buffer.get());}
        if(core_rate==48000) buffer->MergeFrequencyBands();
        buffer->CopyTo(webrtc::StreamConfig(core_rate,1),out);
      }
      rumble.process(core.data(),core_hop);
      if(dual || steady_speex) mild_rumble.process(neutral.data(),core_hop);
      if(controller) {
        float next_dry=policy=="fixedblend" ? dry_max : controller->dry();
        const float next_gain=controller->gain();
        if(omlsa && spectral_dry_guard>0) {
          // Statistical and broadband observers must both support a strong
          // raw blend. A short confidence release protects syllable valleys;
          // this changes only mix control, never waveform scheduling.
          const float evidence=std::clamp((omlsa->mean_probability()-.05f)/.75f,0.f,1.f);
          spectral_hold=std::max(evidence,spectral_hold*std::exp(-10.f/80.f));
          spectral_guard_applied=1-spectral_dry_guard*(1-spectral_hold);
          next_dry*=spectral_guard_applied;
        }
        if(omlsa && transient_dry_guard>0) {
          // Indicator pulse trains can look speech-like to the broadband
          // observer. Require the independent statistical posterior to be
          // low before reducing only the raw/dry blend. The OM-LSA gain and
          // IMCRA state remain untouched.
          const float low_posterior=std::clamp(
              (.30f-omlsa->mean_probability())/.20f,0.f,1.f);
          const float evidence=controller->transient_noise_evidence()*
              (controller->transient_periodicity_500ms()>.55f ?
               low_posterior : 1.f);
          transient_guard_applied=1-transient_dry_guard*evidence;
          next_dry*=transient_guard_applied;
        }
        float blend=policy=="dynamic" ? 0.f : next_dry;
        if(steady_speex) {
          // Keep a separate, continuously learned strong-noise state. A speech
          // confidence hold protects valleys between syllables without holding
          // PCM or changing the primary state. No reference or labels are used.
          steady_speech_hold=std::max(controller->confidence(),steady_speech_hold*std::exp(-10.f/400.f));
          // A quiet learned floor cannot justify stronger suppression. This
          // continuous margin also protects low-level speech after a pause.
          const float background=std::clamp((controller->noise_floor_db()-(quiet_floor_db+10.f))/12.f,0.f,1.f);
          blend=background*std::clamp((.2f-steady_speech_hold)/.2f,0.f,1.f);
          steady_mix_target=blend;
        }
        if(unified_raw_budget) {
          // Independent protection requests are alternative evidence for the
          // SAME capture samples, not extra bypasses. Quiet/startup can spend
          // the remaining budget but cannot raise the total above the active
          // raw budget. High-SNR protection expands the synthesis mix only;
          // the estimator keeps its PSD, DD and minima state continuous.
          const float raw_budget=unified_high_snr_bypass ? 1.f : dry_max;
          const float protection=std::max(
              quiet_enabled ? controller->quiet() : 0.f,
              policy=="parallel_start" ? startup_target : 0.f);
          const float guarded_protection=raw_budget*protection*
              spectral_guard_applied*transient_guard_applied;
          blend=std::clamp(std::max(next_dry,guarded_protection),0.f,raw_budget);
        }
        raw_mix_target=unified_raw_budget ? blend :
            std::clamp(blend+(1.f-blend)*
                ((policy=="parallel_start" || quiet_enabled) ?
                 std::max(startup_target,controller->quiet()) : 0.f),0.f,1.f);
        raw_mix_applied_min=1.f;
        aligned.begin(blend,next_gain);
        for(unsigned i=0;i<core_hop;++i) {
          const float f=float(i+1)/core_hop;
          float mix=previous_dry+f*(next_dry-previous_dry),gain=previous_gain+f*(next_gain-previous_gain);
          if(unified_raw_budget || policy!="legacy") aligned.sample(i,mix,gain);
          float total_raw=mix;
          if(unified_raw_budget) {
            // One fullband donor with the backend's declared fixed delay.
            // The observer, startup, guards and noise gain never add PCM
            // bypasses around this convex mix; no process-time allocation.
            mix=std::clamp(mix,0.f,unified_high_snr_bypass ? 1.f : dry_max);
            total_raw=mix;
            core[i]=gain*((1-mix)*core[i]+mix*startup_raw[i]);
          } else {
            // Keep the historical arithmetic/order unchanged for replay.
            core[i]=gain*((1-mix)*core[i]+mix*neutral[i]);
            if(policy=="parallel_start" || quiet_enabled) {
              float start_mix,unused_gain;startup_control.sample(i,start_mix,unused_gain);
              core[i]=(1-start_mix)*core[i]+start_mix*startup_raw[i];
              total_raw=start_mix+(1-start_mix)*mix;
            }
          }
          raw_mix_applied_min=std::min(raw_mix_applied_min,total_raw);
          raw_mix_applied_max=std::max(raw_mix_applied_max,total_raw);
        }
        previous_dry=next_dry;previous_gain=next_gain;
      }
      if(adaptive_rumble) {
        const float fraction=controller->rumble_fraction();
        const float dominant=std::clamp((fraction-.6f)/.3f,0.f,1.f);
        // Extend credible mid-band speech evidence over short low-frequency
        // breaths. Continuous rumble cannot refresh this 250 ms confidence
        // release merely by being loud below 80 Hz. Audio is never held.
        const float support=std::clamp((.95f-fraction)/.3f,0.f,1.f);
        const float speech=controller->confidence()*support;
        rumble_speech_hold=std::max(speech,rumble_speech_hold*std::exp(-10.f/250.f));
        rumble_target=dominant*(1-rumble_speech_hold);
        if(rumble_recovery>0) {
          // A weak broadband fluctuation must not veto persistent bass removal.
          // The added path remains frequency selective and shares the FIR's
          // original timebase. Strong mid-band speech can still release it.
          const float credible=std::clamp((controller->confidence()-.35f)/.45f,0.f,1.f)*support;
          rumble_credible_hold=std::max(credible,rumble_credible_hold*std::exp(-10.f/250.f));
          const float bass=std::clamp((fraction-.05f)/.25f,0.f,1.f);
          const float recovered=std::max(rumble_target,bass*(1-rumble_credible_hold));
          rumble_target+=rumble_recovery*(recovered-rumble_target);
        }
        // Select a stricter corner only while the causal capture spectrum has
        // excess energy below 80 Hz relative to 80-300 Hz. Hysteresis avoids
        // rapid switching, and both FIRs share the same 20 ms delay.
        if(dynamic_rumble_corner && controller) {
          const float ratio=controller->rumble_low_ratio_db();
          if(ratio>=0.f) rumble_corner_hz=rumble_adaptive_hz;
          else if(ratio< -3.f) rumble_corner_hz=0;
        }
        // Delay the evidence to the centre of the same FIR output sample.
        // Both dry and conditioned samples retain the common declared delay.
        rumble_control.begin(rumble_target,1.f);
        for(unsigned i=0;i<core_hop;++i) {float unused;rumble_control.sample(i,rumble_amount[i],unused);}
        rumble_applied=rumble_amount.back();
      }
      if(dynamic_rumble_corner) linear_filter.retune(rumble_corner_hz>0 ? rumble_adaptive_hz : rumble_hz_base);
      linear_filter.process(core.data(),core_hop,adaptive_rumble?rumble_amount.data():nullptr);
      if(residual) residual->process(core.data(),core_hop,
          controller ? controller->snr_db() : 0.f);
      if(down) {
        if(down->Resample(core.data(),core_hop,output.data(),hop)) return false;
      } else std::copy(core.begin(),core.end(),output.begin());
    std::copy(output.begin(),output.end(),samples);
    ++frame;
    return true;
  }
};
VehicleNsProcessor::VehicleNsProcessor(unsigned r,const VehicleNsConfig& c,bool quantize):impl_(std::make_unique<Impl>(r,c,quantize)){}
VehicleNsProcessor::~VehicleNsProcessor()=default;
bool VehicleNsProcessor::process(float* samples,unsigned count) noexcept {return impl_->process(samples,count);}
unsigned VehicleNsProcessor::delay_samples() const noexcept {return impl_->delay;}
unsigned VehicleNsProcessor::core_rate() const noexcept {return impl_->core_rate;}
unsigned VehicleNsProcessor::core_delay_samples() const noexcept {return impl_->core_delay;}
unsigned VehicleNsProcessor::rumble_delay_samples() const noexcept {return impl_->linear_filter.delay();}
double VehicleNsProcessor::src_delay_samples() const noexcept {return impl_->src_delay;}
VehicleNsStatistics VehicleNsProcessor::statistics() const noexcept {
  const auto& s=*impl_; VehicleNsStatistics x;
  x.unified_raw_budget=s.unified_raw_budget;
  x.raw_mix_budget=s.unified_raw_budget ? (s.unified_high_snr_bypass ? 1.f : s.dry_max) : 1.f;
  x.raw_mix_target=s.raw_mix_target;
  x.raw_mix_applied_min=s.raw_mix_applied_min;
  x.raw_mix_applied_max=s.raw_mix_applied_max;
  if(s.controller) {x.confidence=s.controller->confidence();x.snr_db=s.controller->snr_db();
    x.quiet_mix_target=s.controller->quiet();x.noise_floor_db=s.controller->noise_floor_db();x.rumble_fraction=s.controller->rumble_fraction();}
  if(s.controller) x.noise_step_probability=s.controller->noise_step_probability();
  if(s.controller) {
    x.transient_noise_evidence=s.controller->transient_noise_evidence();
    x.transient_periodicity_100ms=s.controller->transient_periodicity_100ms();
    x.transient_periodicity_500ms=s.controller->transient_periodicity_500ms();
  }
  x.transient_guard_applied=s.transient_guard_applied;
  if(s.transient_spectral)
    x.transient_spectral_guard_applied=
        s.transient_spectral->last_applied();
  x.dry_mix_target=s.previous_dry;x.gain_target=s.previous_gain;x.suppression_db=s.dynamic_db;x.suppression_updates=s.strength_changes;
  x.rumble_mix_target=s.rumble_target;x.rumble_mix_applied=s.rumble_applied;
  x.rumble_corner_hz=s.rumble_corner_hz;
  if(s.controller) x.rumble_low_ratio_db=s.controller->rumble_low_ratio_db();
  x.steady_mix_target=s.steady_mix_target;
  x.spectral_guard_applied=s.spectral_guard_applied;
  if(s.omlsa) x.noise_step_updates=s.omlsa->estimator().step_updates();
  if(s.residual) {
    x.residual_speech_probability=s.residual->speech_probability();
    x.residual_attenuation_db=s.residual->extra_attenuation_db();
  }
  if(s.omlsa) {
    x.omlsa_probability=s.omlsa->mean_probability();x.omlsa_gain=s.omlsa->mean_gain();
    x.omlsa_noise_power=s.omlsa->noise_power();x.omlsa_bootstrap_seed=s.omlsa->bootstrap_seed();
    x.imcra_excluded_bins=s.omlsa->estimator().excluded_bins();
    x.imcra_noise_updates=s.omlsa->estimator().noise_updates();
    x.imcra_minima_rotations=s.omlsa->estimator().rotations();
  }
  if(s.student) {
    x.student_state=unsigned(s.student->corrector.state());x.student_applied=s.student->applied;
    x.student_inferences=s.student->inferences;x.student_fallback_frames=s.student->fallback_frames;
  }
  if(s.unified) {
    x.unified_weights=s.unified->weights();
    x.unified_target_weights=s.unified->target_weights();
    x.unified_weight_entropy=s.unified->weight_entropy();
    x.unified_frame_snr_db=s.unified->frame_snr_db();
    x.unified_flatness=s.unified->flatness();x.unified_tonality=s.unified->tonality();
    x.unified_minima_confidence=s.unified->minima_confidence();
    x.unified_transient_evidence=s.unified->transient_evidence();
    x.unified_noise_step_evidence=s.unified->noise_step_evidence();
    x.unified_probability=s.unified->mean_probability();x.unified_gain=s.unified->mean_gain();
    x.unified_noise_power=s.unified->noise_power();x.unified_bootstrap_seed=s.unified->bootstrap_seed();
    x.unified_noise_updates=s.unified->noise_updates();x.unified_minima_rotations=s.unified->minima_rotations();
    x.unified_minima_band_db=s.unified->minima_band_db();
    x.unified_conditional_minima_band_db=s.unified->conditional_minima_band_db();
    x.unified_history_minima_band_db=s.unified->history_minima_band_db();
    x.unified_noise_psd_band_db=s.unified->noise_psd_band_db();
    x.unified_smooth_band_db=s.unified->smooth_band_db();
    x.unified_conditional_band_db=s.unified->conditional_band_db();
    x.unified_dd_prior_band_db=s.unified->dd_prior_band_db();
    x.unified_xi_band_db=s.unified->xi_band_db();
    x.unified_gamma_band_db=s.unified->gamma_band_db();
    x.unified_gain_band_db=s.unified->gain_band_db();
    x.unified_minima_rebuild_jump_db=s.unified->minima_rebuild_jump_db();
    x.unified_conditional_minima_rebuild_jump_db=s.unified->conditional_minima_rebuild_jump_db();
    x.unified_high_snr_bypass=s.unified_high_snr_bypass;
  }
  if(s.speech_band) {
    x.speech_band_frames=s.speech_band->frames();
    x.speech_band_floor=s.speech_band->last_floor();
    x.speech_band_gain_delta=s.speech_band->last_delta();
    x.speech_band_evidence=s.speech_band->last_evidence();
    x.speech_band_mid_probability=s.speech_band->last_mid_probability();
    x.speech_band_anchor_probability=s.speech_band->last_anchor_probability();
    x.speech_band_anchor_gain=s.speech_band->last_anchor_gain();
  }
  return x;
}
}
