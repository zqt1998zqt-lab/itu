#include "itu/audio_engine.hpp"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/environment/environment_factory.h"
#include "rtc_base/logging.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/ns/noise_suppressor.h"
#include "modules/audio_processing/agc2/vad_wrapper.h"
#include "linear_phase_lowcut.hpp"
#include "reference_monitor.hpp"
#include "noise_profile.hpp"
#include "echo_coherence.hpp"
#include "echo_subspace.hpp"
#include "denormal_scope.hpp"
#include "native_pipeline.hpp"
#include "speex_ns.hpp"
#include "speex_aec_pipeline.hpp"
#include "rate_converter.hpp"
#include "speech_gain.hpp"
#include "utterance_gain.hpp"
#include "legacy_utterance_gain.hpp"
#include "min_stat_noise.hpp"
#include "eq_processor.hpp"
#include "linear_phase_eq.hpp"
#include "parametric_eq.hpp"
#include "unified_eq_agc.hpp"
#include "spatial_frontend.hpp"
#include "noise_transition.hpp"
#include "residual_noise_gate.hpp"
#include "repeated_tone.hpp"
#include "spectral_excess.hpp"
#include "ns_evidence.hpp"
#include "neural_ns.hpp"
#include "adaptive_controller.hpp"
#include "drc.hpp"
#include "downlink_pipeline.hpp"
#include "pre_aec_pipeline.hpp"
#include "near_protection_gap.hpp"
#ifdef ITU_HAS_RNNOISE
#include "rnnoise.h"
#endif
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace itu {
namespace {
Config checked(Config c,unsigned maximum_beam_samples=4096) {
  ParametricEq::validate(c.output_eq,c.sample_rate);
  if(c.output_eq.phase==EqPhase::linear)
    static_cast<void>(LinearPhaseEq::taps_for(c.output_eq,c.sample_rate));
  if ((c.sample_rate != 16000 && c.sample_rate != 24000 && c.sample_rate != 48000) ||
      c.frame_size != c.sample_rate/100 || !c.microphones ||
      c.microphones > kMaxMicrophones || c.noise_level > 3 || c.stream_delay_ms > 500 ||
      c.aec_filter_blocks<13 || c.aec_filter_blocks>128)
    throw std::invalid_argument("requires 16/24/48kHz, exactly 10ms, 1..8 microphones, NS level 0..3, delay 0..500ms");
  if(static_cast<unsigned>(c.mode)>static_cast<unsigned>(ProcessingMode::vehicle_stream))
    throw std::invalid_argument("unknown processing mode");
  if(!c.reference_channels || c.reference_channels>8)
    throw std::invalid_argument("render channels must be 1..8");
  if(c.aec_backend!=AecBackend::webrtc && c.aec_backend!=AecBackend::speex &&
     c.aec_backend!=AecBackend::speex_fused)
    throw std::invalid_argument("unknown AEC backend");
  if(c.aec && c.aec_backend==AecBackend::speex_fused &&
      ((!c.beamforming && c.microphones!=1) || c.speex_residual_suppression))
    throw std::invalid_argument("fused Speex AEC requires a mono MDF input (or BF before AEC) and no MDF RES");
  if(c.speex_echo_tail_ms<100 || c.speex_echo_tail_ms>500)
    throw std::invalid_argument("Speex MDF tail must be 100..500 ms");
  if(c.speex_multirender_context &&
      (!c.aec || c.aec_backend!=AecBackend::speex || !c.speex_post_context)) {
    if(c.mode==ProcessingMode::vehicle_stream) c.speex_multirender_context=false;
    else throw std::invalid_argument("multi-reference context requires enabled Speex MDF and post-AEC context");
  }
  if(c.speex_active_echo_suppression_db>0 || c.speex_active_echo_suppression_db<-30)
    throw std::invalid_argument("Speex active echo suppression must be -30..0 dB");
  if(c.speex_coherence_residual_suppression &&
     c.aec_backend!=AecBackend::speex && c.aec_backend!=AecBackend::speex_fused) {
    if(c.mode==ProcessingMode::vehicle_stream) c.speex_coherence_residual_suppression=false;
    else throw std::invalid_argument("coherence residual suppression requires Speex MDF");
  }
  if(c.speex_coherence_residual_suppression && !c.aec) {
    if(c.mode==ProcessingMode::vehicle_stream) c.speex_coherence_residual_suppression=false;
    else throw std::invalid_argument("coherence residual suppression requires enabled AEC");
  }
  if(c.speex_coherence_residual_suppression && c.reference_channels!=1) {
    // The VCP profile remains usable with stereo/multichannel playback
    // references. Coherence RES is a mono-reference gate, so the profile
    // falls back to the un-gated MDF path for those inputs and reports the
    // effective value through AudioEngine::config(). Standalone Config users
    // still get a hard error instead of silently changing their request.
    if(c.mode==ProcessingMode::vehicle_stream) c.speex_coherence_residual_suppression=false;
    else throw std::invalid_argument("coherence residual suppression requires mono-reference Speex MDF");
  }
  if((c.spatial_mvdr || c.wind_noise_control) && (!c.beamforming || c.microphones<2))
    throw std::invalid_argument("spatial MVDR/wind requires calibrated multi-microphone BF");
  if(c.spatial_fixed_steering && !c.spatial_mvdr)
    throw std::invalid_argument("fixed spatial steering requires MVDR");
  if(c.spatial_continuous_covariance && !c.spatial_fixed_steering)
    throw std::invalid_argument("continuous spatial covariance requires fixed calibrated steering");
  // Mono-reference projection/coherence guards do not model a multichannel echo
  // subspace. Use AEC3's native multi-render DTD/suppression for this new path.
  if(c.reference_channels>1) {
    c.reference_guard=false; c.coherence_dtd=false; c.linear_speech_protection=false;
    c.analyze_linear_noise=false; c.correlation_delay=false;
    if(c.multirender_speech_protection) c.linear_speech_protection=true;
  }
  if((c.aec_backend==AecBackend::speex || c.aec_backend==AecBackend::speex_fused) &&
      (c.multirender_nearend_policy || c.multirender_speech_protection || c.linear_output_diagnostic))
    throw std::invalid_argument("AEC3 diagnostic protection options require the WebRTC AEC backend");
  if(c.noise_backend!=NoiseBackend::webrtc && c.noise_backend!=NoiseBackend::rnnoise && c.noise_backend!=NoiseBackend::speex && c.noise_backend!=NoiseBackend::ulunas && c.noise_backend!=NoiseBackend::omlsa_imcra && c.noise_backend!=NoiseBackend::super_gaussian && c.noise_backend!=NoiseBackend::unified)
    throw std::invalid_argument("unknown noise backend");
  if(!c.vehicle_ns.student_model_path.empty()) {
    if(!c.noise_suppression || c.noise_backend!=NoiseBackend::omlsa_imcra)
      throw std::invalid_argument("hybrid NS model requires enabled OM-LSA NS");
    if(c.sample_rate==24000 && c.aec && c.aec_backend==AecBackend::webrtc)
      throw std::invalid_argument("native24 hybrid NS cannot use AEC3's 48 kHz bridge; use native Speex AEC or standalone NS");
  }
  if(c.speex_suppression_db>0 || c.speex_suppression_db<-40) throw std::invalid_argument("Speex suppression must be -40..0 dB");
  if(!std::isfinite(c.speex_fused_learning_rate) || c.speex_fused_learning_rate<.005f || c.speex_fused_learning_rate>.5f ||
     c.speex_fused_shadow_taps_ms<1 || c.speex_fused_shadow_taps_ms>64)
    throw std::invalid_argument("fused AEC learning rate/taps out of range");
  if(c.hybrid_speech_suppression_db>0 || c.hybrid_speech_suppression_db<-40)
    throw std::invalid_argument("hybrid speech suppression must be -40..0 dB");
  if(!std::isfinite(c.high_snr_dry_mix) || c.high_snr_dry_mix<0 || c.high_snr_dry_mix>1)
    throw std::invalid_argument("high SNR dry mix must be 0..1");
  if(!std::isfinite(c.multirender_innovation_threshold) || c.multirender_innovation_threshold<.05f || c.multirender_innovation_threshold>.9f)
    throw std::invalid_argument("multiple-coherence innovation threshold must be .05.. .9");
  // The selected v2 bundle is deliberately separate from diagnostic factors.
  if(c.ecnr_v3_g1_control && !c.ecnr_v2_control)
    throw std::invalid_argument("G1 requires explicitly selected ecnr v2 control");
  if(c.ecnr_v2_control) {c.ecnr_coordinated_control=true; c.ecnr_v2_double_evidence=true;}
  auto range=[](float v,float lo,float hi){return std::isfinite(v) && v>=lo && v<=hi;};
  auto valid_drc=[](const DrcConfig& d,auto range_fn){
    return range_fn(d.threshold_dbfs,-40.f,0.f) &&
           range_fn(d.ratio,1.f,20.f) &&
           range_fn(d.knee_width_db,0.f,12.f) &&
           range_fn(d.attack_ms,.5f,50.f) &&
           range_fn(d.release_ms,10.f,500.f) &&
           range_fn(d.makeup_gain_db,-12.f,18.f) &&
           range_fn(d.ceiling_dbfs,-6.f,0.f) &&
           range_fn(d.lookahead_ms,0.f,10.f);
  };
  if(!valid_drc(c.tx_drc,range) || !valid_drc(c.rx_config.drc,range))
    throw std::invalid_argument("DRC threshold/ratio/knee/attack/release/makeup/ceiling/lookahead out of range");
  if(!range(c.ecnr_v2_nonlinear_vad_min,0.f,1.f) || !range(c.ecnr_v2_coherence_margin,1.f,4.f))
    throw std::invalid_argument("invalid v2 evidence thresholds");
  if (!range(c.activity_floor,1e-12f,1.f) || !range(c.agc_headroom_db,0.f,30.f) ||
      !range(c.low_cut_hz,1.f,120.f) ||
      !range(c.fullband_min_highband_fraction,0.f,.2f) ||
      !range(c.max_gain_db,0.f,50.f) || !range(c.initial_gain_db,0.f,c.max_gain_db) ||
      !range(c.max_gain_change_db_per_second,.01f,100.f) ||
      !range(c.max_output_noise_level_dbfs,-90.f,0.f) || !range(c.fixed_gain_db,0.f,50.f) ||
      !range(c.output_ceiling,.01f,1.f)) throw std::invalid_argument("invalid WebRTC gain or level configuration");
  for(float d:c.beam_delays_samples) if(!range(d,0.f,float(maximum_beam_samples)))
    throw std::invalid_argument("BF delays must be causal, finite, 0..4096 samples");
  return c;
}
float rms(const float* x,std::size_t n) noexcept {
  double sum=0; for(std::size_t i=0;i<n;++i) sum+=double(x[i])*x[i];
  return static_cast<float>(std::sqrt(sum/n));
}
webrtc::ProcessingConfig streams(unsigned microphones,unsigned rate,unsigned render=1) {
  webrtc::ProcessingConfig p;
  p.input_stream()=p.output_stream()=webrtc::StreamConfig(rate,microphones);
  p.reverse_input_stream()=p.reverse_output_stream()=webrtc::StreamConfig(rate,render);
  return p;
}
}
Config referenced_radio_config() {
  Config c;
  c.mode=ProcessingMode::referenced_radio;
  c.correlation_delay=false;
  c.stable_echo_coherence=true;
  c.raw_nearend_protection=true;
  c.speech_preserving_low_cut=true;
  c.fast_aec_adaptation=false;
  return c;
}
Config vehicle_stream_config() {
  auto c=referenced_radio_config();
  c.mode=ProcessingMode::vehicle_stream;
  c.continuous_vehicle_policy=true;
  c.adaptive_noise_policy=false;
  c.adaptive_controller=true;
  c.ecnr_v2_control=true;
  c.speech_gain_policy=false;
  c.high_snr_dry_mix=1.f;
  c.very_transparent_nearend=true;
  return c;
}
Config barrot_vcp_config() {
  auto c=vehicle_stream_config();
  c.eq_before_agc=true;
  c.speech_gain_policy=true;
  c.ecnr_v2_fullband_gain_calibration=true;
  c.repair_stage=1;
  c.rx_processing=true;
  c.tx_drc.enabled=true;
  c.rx_config.enabled=true;
  c.rx_config.enable_nr=true;
  c.rx_config.drc.enabled=true;
  // VCP uses one speech path: Speex MDF plus the live post-AEC context and
  // the coherence-gated residual suppressor for mono-reference playback.
  // The coherence stage is disabled automatically by checked() for the
  // multichannel-reference cases that it cannot model.
  c.aec=true;
  c.aec_backend=AecBackend::speex;
  c.speex_echo_tail_ms=500;
  c.speex_native_conditioning=true;
  c.speex_post_context=true;
  c.speex_coherence_residual_suppression=true;
  c.post_aec_evidence_window=true;
  // Low-level projection floor stays off by default: on 4,412 it traded
  // quiet-utterance gains for SI-SDR losses on clean -32 dBFS scenes.
  c.low_level_projection_gain=false;
  c.min_stat_ns=false;
  return c;
}
void apply_vcp_joint_control(Config& c) {
  c.speex_multirender_context=true;
  c.vehicle_ns_control=true;
  c.noise_backend=NoiseBackend::super_gaussian;
  c.vehicle_ns.unified_raw_budget=true;
  c.vehicle_ns.dry_max=.25f;
  c.vehicle_ns.snr_threshold_db=24.f;
  c.vehicle_ns.rumble_hz=0.f;
  c.tx_drc.enabled=true;
  c.tx_drc.threshold_dbfs=-6.f;
  c.tx_drc.ratio=2.f;
  c.tx_drc.knee_width_db=4.f;
}
Config barrot_vcp_joint_config() {
  auto c=barrot_vcp_config();
  apply_vcp_joint_control(c);
  return c;
}
Config barrot_vcp_fused_aec_config() {
  auto c=barrot_vcp_config();
  c.aec_backend=AecBackend::speex_fused;
  c.speex_fused_learning_rate=.12f;
  c.speex_fused_shadow_taps_ms=32;
  c.speex_coherence_residual_suppression=true;
  return c;
}
Config barrot_vcp_unified_config() {
  auto c=barrot_vcp_fused_aec_config();
  c.vehicle_ns_control=true;
  c.noise_backend=NoiseBackend::unified;
  c.vehicle_ns.unified_raw_budget=false;
  c.vehicle_ns.dry_max=0.f;
  c.vehicle_ns.noise_gain=1.f;
  c.vehicle_ns.unified_learning_rate=.08f;
  c.unified_eq_agc=true;
  return c;
}
struct AudioEngine::Impl {
  Config c;
  unsigned aec_delay,linear_delay,ns_delay,lowcut_delay;
  std::vector<std::unique_ptr<RateConverter>> monitor_down,nonlinear_down,linear_up;
  std::unique_ptr<RateConverter> monitor_ref_down,profile_down;
  std::vector<float> monitor_capture,linear_eight_history,nonlinear_history;
  std::array<float,160> monitor_reference{},profile_audio{},delta{};
  std::array<float,480> delta_up{};
  std::vector<float> reconstructed_linear;
  unsigned eight_head=0,nonlinear_head=0;
#ifdef ITU_HAS_RNNOISE
  DenoiseState* rn=nullptr;
  ~Impl() {if(rn) rnnoise_destroy(rn);}
#endif
  std::unique_ptr<NativePipeline> native;
  std::unique_ptr<SpeexAecPipeline> speex_aec;
  std::unique_ptr<EqProcessor> output_eq;
  std::unique_ptr<UnifiedEqAgc> unified_eq_agc;
  std::unique_ptr<SpatialFrontend> spatial;
  std::unique_ptr<NeuralNs> neural;
  NoiseTransition noise_transition;
  std::unique_ptr<NoiseProfile> ns_residual_profile;
  ResidualNoiseGate ns_residual_gate;
  std::unique_ptr<RepeatedTone> repeated_tone;
  std::unique_ptr<RateConverter> tone_down;
  std::array<float,160> tone_audio{};
  std::unique_ptr<SpectralExcess> spectral_excess;
  std::unique_ptr<RateConverter> excess_down;
  std::array<float,160> excess_audio{};
  std::vector<float> spatial_capture,spatial_linear,spatial_low,transition_dry,transition_history;
  unsigned transition_head=0;
  float spatial_vad=1;
  float input_excess_control=0;
  std::unique_ptr<NsEvidence> ns_evidence,linear_evidence,nonlinear_evidence;
  std::unique_ptr<AdaptiveController> adaptive_controller;
  std::vector<std::unique_ptr<DownlinkPipeline>> downlink_pipeline;
  std::vector<float> downlink_input,downlink_output;
  std::unique_ptr<PreAecPipeline> pre_aec;
  std::unique_ptr<DynamicRangeCompressor> tx_drc;
  std::unique_ptr<RateConverter> linear_evidence_down,nonlinear_evidence_down;
  std::array<float,160> linear_evidence_audio{},nonlinear_evidence_audio{};
  std::array<float,480> evidence_native{};
  std::unique_ptr<SpeexNs> speex;
  std::unique_ptr<VehicleNsProcessor> vehicle_ns;
  std::unique_ptr<SpeexNs> speech_speex;
  webrtc::scoped_refptr<webrtc::AudioProcessing> aec, post;
  std::unique_ptr<webrtc::NoiseSuppressor> ns;
  std::unique_ptr<webrtc::NoiseSuppressor> moderate_ns;
  std::unique_ptr<webrtc::NoiseSuppressor> fullband_ns;
  std::unique_ptr<webrtc::AudioBuffer> fullband_ns_audio;
  std::vector<float> fullband_ns_output,fullband_dry_history;
  std::vector<float> ns_padding_history;
  unsigned ns_padding_head=0;
  unsigned fullband_dry_head=0,fullband_noise_frames=0,fullband_quiet_frames=0;
  double fullband_noise_power=0;
  double speech_branch_power=0,voiced_reference_power=0;
  unsigned fullband_speech_hold=0,fullband_speech_confirmation=0;
  float fullband_ns_mix=0;
  std::unique_ptr<webrtc::AudioBuffer> ns_audio;
  std::unique_ptr<webrtc::AudioBuffer> ns_analysis_audio;
  std::unique_ptr<webrtc::AudioBuffer> moderate_audio;
  std::unique_ptr<NoiseProfile> noise_profile;
  std::vector<float> moderate_output,ns_dry_history,ns_dry_output;
  unsigned ns_dry_head=0;
  unsigned ns_protection_hold=0;
  float ns_protection_mix=0;
  std::unique_ptr<webrtc::VoiceActivityDetectorWrapper> vad;
  std::unique_ptr<webrtc::VoiceActivityDetectorWrapper> aec_vad;
  std::unique_ptr<webrtc::VoiceActivityDetectorWrapper> nonlinear_vad;
  std::vector<float> vad_audio;
  std::unique_ptr<LinearPhaseLowCut> low_cut;
  std::unique_ptr<ReferenceMonitor> reference_monitor;
  std::unique_ptr<EchoCoherence> echo_coherence;
  std::vector<std::unique_ptr<EchoSubspace>> echo_subspaces;
  std::unique_ptr<ReferenceMonitor> subspace_delay;
  std::vector<std::unique_ptr<RateConverter>> subspace_down;
  std::vector<float> subspace_reference;
  std::vector<float> filtered_reference;
  std::vector<float> conditioned_capture;
  unsigned speech_calibration_frames=0;
  std::array<float,30> speech_calibration_levels{};
  bool gain_calibrated=false;
  float calibrated_gain_db=0;
  SpeechGain speech_gain;
  UtteranceGain utterance_gain;
  LegacyUtteranceGain legacy_utterance_gain;
  MinStatNoise min_stat_ns_gate;
  float applied_gain_db=0;
  float previous_commanded_gain_db=0;
  std::vector<float> raw_history,protected_capture;
  std::vector<float> fullband_history,fullband_capture,lowband_bf_history;
  std::vector<float> lowband_output,lowband_ns_history;
  unsigned fullband_head=0,lowband_ns_head=0,lowband_noise_frames=0,lowband_speech_hold=0,lowband_speech_candidate_frames=0;
  double lowband_noise_power=0,lowband_echo_power=0;
  unsigned lowband_echo_frames=0;
  float lowband_restore_mix=0;
  unsigned raw_head=0,reference_observation_frames=0,reference_window_frames=0,reference_state=0;
  double reference_input_power=0,reference_linear_power=0;
  float reference_residual_ratio=1,fallback_mix=0;
  std::vector<float> aligned_linear,linear_history;
  std::vector<float> linear_bf_history,linear_mono;
  unsigned linear_head=0,near_hangover=0,render_hangover=0;
  unsigned near_candidate_frames=0,reference_near_candidate_frames=0;
  unsigned gain_noise_evidence=0;float gain_noise_compensation=1;
  // Single-microphone low-band wind evidence. `wind_tilt_db` is the smoothed
  // 60-300 Hz / sub-60 Hz energy ratio in dB; speech is positive, low-frequency
  // noise is negative. `wind_lowband_gain` is the soft gain applied to the low
  // band (1 = untouched, 0 = removed).
  struct WindLowPass {double z1=0,z2=0;};
  WindLowPass wind_lowpass;
  double wind_lp_b0=1,wind_lp_b1=0,wind_lp_b2=0,wind_lp_a1=0,wind_lp_a2=0;
  float wind_tilt_db=0,wind_envelope=0,wind_envelope_deviation=0,wind_lowband_gain=1;
  double residual_smoothed=0,echo_smoothed=0,echo_floor_ratio=1;
  float nearend_mix=0;
  unsigned startup_echo_confirmation=0,v2_near_confirmation=0;
  NearProtectionGap g1_gap;
  unsigned v2_delay_frames=0;
  float v2_observed_delay_ms=-1;
  double predicted_power=0,predicted_cross=0;
  float subtraction_gain=1,noise_speech_mix=0,floor_gain_smoothed=.42f;
  float observed_peak=0,noise_profile_mix=0;
  unsigned noise_speech_hold=0,confirmed_near_frames=0;
  bool near_observed=false;
  unsigned coherence_learning_frames=0,coherent_echo_frames=0;
  unsigned coherence_delay_samples=~0u,coherence_settle_frames=0;
  float coherence_baseline=1,coherence_smoothed=1;
  std::vector<float> planar, taps, bf_history, mono;
  std::vector<std::array<float,160>> linear;
  std::array<const float*,kMaxMicrophones> inputs{};
  std::array<float*,kMaxMicrophones> outputs{};
  unsigned beam_size=1, beam_head=0;
  std::uint64_t frames=0;
  int failed_backend_error=0;
  bool internal_pcm=false;
  unsigned maximum_beam_samples=4096;
  explicit Impl(Config config,unsigned maximum_beam=4096,bool allow_internal_pcm=false):c(checked(config,maximum_beam)),
      aec_delay(c.sample_rate==48000 ? 912 : 128),linear_delay(c.sample_rate==48000 ? 216 : 64),
      ns_delay(c.noise_backend==NoiseBackend::rnnoise ? 960 :
          c.hybrid_speech_ns && c.speech_preserving_low_cut && c.linear_phase_low_cut &&
          c.noise_backend==NoiseBackend::webrtc ? c.frame_size :
          c.sample_rate==48000 ? 336 : 96),lowcut_delay(0),
      fullband_ns_output(c.frame_size),fullband_dry_history(ns_delay),
      moderate_output(c.frame_size),ns_dry_history(ns_delay),ns_dry_output(c.frame_size),
      vad_audio(c.frame_size),filtered_reference(c.frame_size*c.reference_channels),lowband_output(c.frame_size),
      lowband_ns_history(ns_delay),planar(c.frame_size*c.microphones),
      taps(planar.size()),mono(c.frame_size),linear(c.microphones),
      internal_pcm(allow_internal_pcm),maximum_beam_samples(maximum_beam) {
    webrtc::LogMessage::LogToDebug(webrtc::LS_NONE);
    webrtc::LogMessage::SetLogToStderr(false);
    if(c.rx_processing || c.rx_config.enabled) {
      for(unsigned ch=0;ch<c.reference_channels;++ch)
        downlink_pipeline.emplace_back(std::make_unique<DownlinkPipeline>(c.sample_rate,c.frame_size,c.rx_config));
      downlink_input.resize(c.frame_size);downlink_output.resize(c.frame_size);
    }
    if(c.beamforming) {
      pre_aec=std::make_unique<PreAecPipeline>(c);
      return;
    }
    if(c.tx_drc.enabled) tx_drc=std::make_unique<DynamicRangeCompressor>(c.sample_rate,c.tx_drc);
    if(c.adaptive_controller) {
      const unsigned policy_rate=c.policy_sample_rate ? c.policy_sample_rate : c.sample_rate;
      adaptive_controller=std::make_unique<AdaptiveController>(AdaptiveController::Context{
          c.mode,policy_rate,c.frame_size,c.microphones,c.reference_channels});
    }
    if(c.aec && (c.aec_backend==AecBackend::speex || c.aec_backend==AecBackend::speex_fused)) {
      speex_aec=std::make_unique<SpeexAecPipeline>(c); return;
    }
    // The standalone OM-LSA path and Speex MDF's native post chain retain
    // native 24 kHz. AEC3 still owns its explicit legacy 48 kHz bridge.
    if((c.sample_rate==24000 && ((c.noise_backend!=NoiseBackend::omlsa_imcra && c.noise_backend!=NoiseBackend::super_gaussian && c.noise_backend!=NoiseBackend::unified) || c.aec)) || (c.sample_rate==16000 && c.noise_backend==NoiseBackend::rnnoise)) {
      if(c.vehicle_ns_control && c.vehicle_ns.native_24)
        throw std::invalid_argument("AudioEngine uses the 48 kHz core at 24 kHz; native24 is available through VehicleNsProcessor");
      native=std::make_unique<NativePipeline>(c);
      return;
    }
    if(c.unified_eq_agc)
      unified_eq_agc=std::make_unique<UnifiedEqAgc>(c.output_eq,c.sample_rate,c.frame_size);
    else if(ParametricEq::enabled(c.output_eq))
      output_eq=std::make_unique<EqProcessor>(c.output_eq,c.sample_rate);
    if((c.vehicle_ns_control || c.noise_backend==NoiseBackend::omlsa_imcra || c.noise_backend==NoiseBackend::super_gaussian || c.noise_backend==NoiseBackend::unified) && c.noise_suppression) {
      if(c.noise_backend!=NoiseBackend::webrtc && c.noise_backend!=NoiseBackend::speex && c.noise_backend!=NoiseBackend::omlsa_imcra && c.noise_backend!=NoiseBackend::super_gaussian && c.noise_backend!=NoiseBackend::unified)
        throw std::invalid_argument("vehicle NS supports WebRTC, Speex, OM-LSA/IMCRA, Super-Gaussian or unified fusion");
      auto options=c.vehicle_ns;
      options.backend=c.noise_backend==NoiseBackend::unified ? VehicleNsBackend::unified : c.noise_backend==NoiseBackend::super_gaussian ? VehicleNsBackend::super_gaussian : c.noise_backend==NoiseBackend::omlsa_imcra ? VehicleNsBackend::omlsa_imcra : c.noise_backend==NoiseBackend::speex ? VehicleNsBackend::speex : VehicleNsBackend::webrtc;
      options.strength=c.noise_backend==NoiseBackend::speex ? c.speex_suppression_db : int(c.noise_level);
      if(!c.vehicle_ns_control) {
        // Raw backend selection must not silently enable the optional common
        // protection controller. OM-LSA's own soft probabilities stay active.
        options.dry_max=0;options.noise_gain=1;options.rumble_hz=0;options.adaptive_rumble=false;
      }
      vehicle_ns=std::make_unique<VehicleNsProcessor>(c.sample_rate,options,
          !(internal_pcm && c.policy_sample_rate==24000));
      ns_delay=vehicle_ns->delay_samples();
    }
    if(c.noise_suppression && c.noise_backend==NoiseBackend::ulunas) {
      neural=std::make_unique<NeuralNs>(c.sample_rate,c.neural_model_path);
      ns_delay=neural->delay(); lowband_ns_history.resize(ns_delay);
    }
    if(c.spatial_mvdr || c.wind_noise_control) {
      spatial=std::make_unique<SpatialFrontend>(c.sample_rate,c.frame_size,c.microphones,c.spatial_mvdr,c.wind_noise_control,c.spatial_fixed_steering,c.spatial_continuous_covariance);
      spatial_capture.resize(planar.size()); spatial_linear.resize(planar.size()); spatial_low.resize(planar.size());
    }
    if(c.noise_transition_control && c.noise_suppression && !vehicle_ns) {
      transition_dry.resize(c.frame_size);
      transition_history.resize(c.noise_backend==NoiseBackend::speex ? c.frame_size : ns_delay);
    }
    if(c.sample_rate==48000) {
      monitor_capture.resize(160*c.microphones);
      reconstructed_linear.resize(planar.size());
      linear_eight_history.resize(152*c.microphones);
      nonlinear_history.resize(480*c.microphones);
      for(unsigned ch=0;ch<c.microphones;++ch) {
        monitor_down.emplace_back(std::make_unique<RateConverter>(480,160));
        nonlinear_down.emplace_back(std::make_unique<RateConverter>(480,160));
        linear_up.emplace_back(std::make_unique<RateConverter>(160,480));
      }
      monitor_ref_down=std::make_unique<RateConverter>(480,160);
      profile_down=std::make_unique<RateConverter>(480,160);
    }
    const unsigned kAecOutputDelaySamples=aec_delay,kLinearAecDelaySamples=linear_delay,
        kLowCutDelaySamples=lowcut_delay;
    float maximum=0;
    for(unsigned ch=0;ch<c.microphones;++ch) maximum=std::max(maximum,c.beam_delays_samples[ch]);
    beam_size=static_cast<unsigned>(std::ceil(maximum))+2;
    bf_history.resize(beam_size*c.microphones);
    if(c.aec && (c.reference_guard || c.speech_preserving_low_cut || c.linear_speech_protection || c.analyze_linear_noise)) {
      raw_history.resize(kAecOutputDelaySamples*c.microphones);
      protected_capture.resize(c.frame_size*c.microphones);
    }
    if(c.aec && (c.linear_speech_protection || c.linear_output_diagnostic || c.analyze_linear_noise)) {
      aligned_linear.resize(c.frame_size*c.microphones);
      linear_history.resize((kAecOutputDelaySamples-kLinearAecDelaySamples)*c.microphones);
      linear_bf_history.resize(beam_size*c.microphones);
      linear_mono.resize(c.frame_size);
      if(c.external_vad) aec_vad=std::make_unique<webrtc::VoiceActivityDetectorWrapper>(webrtc::GetAvailableCpuFeatures(),c.sample_rate);
      if(c.external_vad) nonlinear_vad=std::make_unique<webrtc::VoiceActivityDetectorWrapper>(webrtc::GetAvailableCpuFeatures(),c.sample_rate);
    }
    if(c.aec && (c.reference_guard || c.correlation_delay || c.coherence_dtd)) reference_monitor=std::make_unique<ReferenceMonitor>();
    if(c.aec && c.coherence_dtd) echo_coherence=std::make_unique<EchoCoherence>();
    if(c.aec && c.reference_channels>1 && c.multirender_speech_protection) {
      for(unsigned ch=0;ch<c.microphones;++ch)
        echo_subspaces.emplace_back(std::make_unique<EchoSubspace>(c.reference_channels));
      subspace_delay=std::make_unique<ReferenceMonitor>();
      subspace_reference.resize(160*c.reference_channels);
      if(c.sample_rate==48000) for(unsigned ch=0;ch<c.reference_channels;++ch)
        subspace_down.emplace_back(std::make_unique<RateConverter>(480,160));
    }
    {
      constexpr double pi=3.14159265358979323846;
      constexpr double q=0.7071067811865475244;
      const double omega=2.*pi*300./static_cast<double>(c.sample_rate);
      const double cosine=std::cos(omega),sine=std::sin(omega);
      const double alpha=sine/(2.*q),a0=1.+alpha;
      wind_lp_b0=(1.-cosine)/(2.*a0);
      wind_lp_b1=(1.-cosine)/a0;
      wind_lp_b2=(1.-cosine)/(2.*a0);
      wind_lp_a1=-2.*cosine/a0;
      wind_lp_a2=(1.-alpha)/a0;
    }
    if(c.linear_phase_low_cut && (c.aec || c.noise_suppression))
      low_cut=std::make_unique<LinearPhaseLowCut>(c.aec ? c.microphones+c.reference_channels : 1,c.low_cut_hz,c.sample_rate);
    if(low_cut && ((c.aec && c.raw_nearend_protection && c.reference_guard) || c.speech_preserving_low_cut)) {
      fullband_history.resize((kLowCutDelaySamples+(c.aec ? kAecOutputDelaySamples : 0))*c.microphones);
      fullband_capture.resize(planar.size());
      lowband_bf_history.resize(beam_size*c.microphones);
    }
    conditioned_capture.resize(planar.size());
    if(!c.external_vad) {
      spectral_excess=std::make_unique<SpectralExcess>();
      if(c.ns_statistical_control)ns_evidence=std::make_unique<NsEvidence>();
      // Backend statistical evidence is independent of optional RNN VAD.
      if(c.aec && c.linear_speech_protection) {
        linear_evidence=std::make_unique<NsEvidence>();nonlinear_evidence=std::make_unique<NsEvidence>();
        if(c.sample_rate!=16000) {
          linear_evidence_down=std::make_unique<RateConverter>(c.frame_size,160);
          nonlinear_evidence_down=std::make_unique<RateConverter>(c.frame_size,160);
        }
      }
      if(c.sample_rate!=16000) excess_down=std::make_unique<RateConverter>(c.frame_size,160);
    }
    if(c.external_vad && (c.noise_suppression || c.automatic_gain || !fullband_history.empty()))
      vad=std::make_unique<webrtc::VoiceActivityDetectorWrapper>(
          webrtc::GetAvailableCpuFeatures(),c.sample_rate);
    for(unsigned ch=0;ch<c.microphones;++ch) inputs[ch]=outputs[ch]=planar.data()+ch*c.frame_size;
    if(c.aec) {
      webrtc::AudioProcessing::Config a;
      a.pipeline.maximum_internal_processing_rate=48000;
      a.pipeline.multi_channel_capture=true;
      a.pipeline.multi_channel_render=c.reference_channels>1;
      a.high_pass_filter.enabled=false;
      a.echo_canceller.enabled=true;
      a.echo_canceller.export_linear_aec_output=true;
      a.echo_canceller.enforce_high_pass_filtering=false;
      webrtc::EchoCanceller3Config mono_config;
      auto multi_config=webrtc::EchoCanceller3Config::CreateDefaultMultichannelConfig();
      mono_config.filter.export_linear_aec_output=true;
      multi_config.filter.export_linear_aec_output=true;
      for(auto* cfg:{&mono_config,&multi_config}) {
        cfg->delay.use_external_delay_estimator=c.external_delay_estimator || c.correlation_delay;
        cfg->delay.delay_headroom_samples=c.aec_delay_headroom_samples;
        cfg->filter.refined.length_blocks=c.aec_filter_blocks;
        cfg->filter.coarse.length_blocks=c.aec_filter_blocks;
        if(c.fast_aec_adaptation) {
          cfg->filter.refined.noise_gate=cfg->filter.refined_initial.noise_gate=100000.f;
          cfg->filter.coarse.noise_gate=cfg->filter.coarse_initial.noise_gate=100000.f;
          cfg->filter.coarse.rate=cfg->filter.coarse_initial.rate=1;
          cfg->filter.refined.error_ceil=cfg->filter.refined_initial.error_ceil=10;
          cfg->filter.refined.leakage_converged=.001f;
          cfg->filter.refined.leakage_diverged=.1f;
        }
      }
      if(c.transparent_nearend) {
        for(auto* cfg:{&mono_config,&multi_config}) {
          cfg->suppressor.dominant_nearend_detection.enr_threshold=.75f;
          cfg->suppressor.dominant_nearend_detection.snr_threshold=10.f;
          cfg->suppressor.dominant_nearend_detection.trigger_threshold=4;
          cfg->suppressor.nearend_tuning.mask_hf={1.09f,1.1f,.3f};
        }
      }
      if(c.very_transparent_nearend) for(auto* cfg:{&mono_config,&multi_config}) {
        // The entry threshold must stay below enr_exit_threshold (10.f).
        // Setting them equal removes the detector's hysteresis; a loud voiced
        // far-end burst then raises the linear residual so that
        // echo_sum/ne_sum (~0.6-1.2 on a strong onset) clears a 10.f entry
        // threshold with no near-end present, latching transparent mode for the
        // whole 50-block hold and letting the residual echo through.
        // 1.0f was chosen by sweeping this knob against the 16/24/48 kHz streams.
        cfg->suppressor.dominant_nearend_detection.enr_threshold=1.0f;
        cfg->suppressor.dominant_nearend_detection.snr_threshold=1.f;
        cfg->suppressor.dominant_nearend_detection.trigger_threshold=2;
        cfg->suppressor.nearend_tuning.mask_lf={10.f,11.f,10.f};
        cfg->suppressor.nearend_tuning.mask_hf={10.f,11.f,10.f};
      }
      if(c.reference_channels>1 && c.multirender_nearend_policy) {
        multi_config.suppressor.nearend_tuning.mask_lf={2.f,2.5f,.3f};
        multi_config.suppressor.nearend_tuning.mask_hf={2.f,2.5f,.3f};
      }
      if(!webrtc::EchoCanceller3Config::Validate(&mono_config) ||
         !webrtc::EchoCanceller3Config::Validate(&multi_config))
        throw std::invalid_argument("invalid AEC3 configuration");
      aec=webrtc::BuiltinAudioProcessingBuilder(a)
          .SetEchoCancellerConfig(mono_config,multi_config).Build(webrtc::CreateEnvironment());
      if(!aec || aec->Initialize(streams(c.microphones,c.sample_rate,c.reference_channels))!=0)
        throw std::runtime_error("WebRTC AEC3 initialization failed");
    }
    if(c.noise_suppression && c.noise_backend==NoiseBackend::speex && !vehicle_ns)
      speex=std::make_unique<SpeexNs>(c.sample_rate,c.speex_suppression_db);
    if(c.noise_suppression && c.noise_backend==NoiseBackend::rnnoise) {
#ifdef ITU_HAS_RNNOISE
      rn=rnnoise_create(nullptr);
      if(!rn || rnnoise_get_frame_size()!=480) throw std::runtime_error("RNNoise initialization failed");
#else
      throw std::invalid_argument("RNNoise not built");
#endif
    }
    if(c.noise_suppression && c.noise_backend==NoiseBackend::webrtc && !vehicle_ns) {
      webrtc::NsConfig ns_config;
      ns_config.target_level=static_cast<webrtc::NsConfig::SuppressionLevel>(c.noise_level);
      ns=std::make_unique<webrtc::NoiseSuppressor>(ns_config,c.sample_rate,1);
      if(c.repair_stage>=2 && c.repair_stage<4) ns_residual_profile=std::make_unique<NoiseProfile>(true);
      if(c.repeated_tone_control) {
        repeated_tone=std::make_unique<RepeatedTone>(c.sample_rate);
        if(c.sample_rate!=16000) tone_down=std::make_unique<RateConverter>(c.frame_size,160);
      }
      ns_audio=std::make_unique<webrtc::AudioBuffer>(c.sample_rate,1,c.sample_rate,1,c.sample_rate,1);
      ns_analysis_audio=std::make_unique<webrtc::AudioBuffer>(c.sample_rate,1,c.sample_rate,1,c.sample_rate,1);
      if(c.speech_preserving_low_cut && (low_cut || c.speex_post_context)) {
        if(c.hybrid_speech_ns) {
          speech_speex=std::make_unique<SpeexNs>(c.sample_rate,c.hybrid_speech_suppression_db);
          ns_padding_history.resize(c.frame_size-(c.sample_rate==48000 ? 336 : 96));
        } else {
          fullband_ns=std::make_unique<webrtc::NoiseSuppressor>(ns_config,c.sample_rate,1);
          fullband_ns_audio=std::make_unique<webrtc::AudioBuffer>(c.sample_rate,1,c.sample_rate,1,c.sample_rate,1);
        }
      }
      if(c.adaptive_noise_policy && c.noise_level==0) {
        ns_config.target_level=webrtc::NsConfig::SuppressionLevel::k12dB;
        moderate_ns=std::make_unique<webrtc::NoiseSuppressor>(ns_config,c.sample_rate,1);
        moderate_audio=std::make_unique<webrtc::AudioBuffer>(c.sample_rate,1,c.sample_rate,1,c.sample_rate,1);
        noise_profile=std::make_unique<NoiseProfile>(c.ecnr_coordinated_control);
      }
    }
    if(c.automatic_gain) {
      webrtc::AudioProcessing::Config a;
      a.pipeline.maximum_internal_processing_rate=48000;
      a.high_pass_filter.enabled=false;
      a.gain_controller1.enabled=false;
      auto& g=a.gain_controller2;
      g.enabled=c.automatic_gain;
      g.input_volume_controller.enabled=false;
      g.adaptive_digital.enabled=c.adaptive_gain;
      g.adaptive_digital.headroom_db=c.agc_headroom_db;
      g.adaptive_digital.max_gain_db=c.max_gain_db;
      g.adaptive_digital.initial_gain_db=c.initial_gain_db;
      g.adaptive_digital.max_gain_change_db_per_second=c.max_gain_change_db_per_second;
      g.adaptive_digital.max_output_noise_level_dbfs=c.max_output_noise_level_dbfs;
      g.fixed_digital.gain_db=c.fixed_gain_db;
      post=webrtc::BuiltinAudioProcessingBuilder(a).Build(webrtc::CreateEnvironment());
      if(!post || post->Initialize(streams(1,c.sample_rate))!=0)
        throw std::runtime_error("WebRTC NS/AGC2 initialization failed");
    }
  }
};
AudioEngine::AudioEngine(const Config& c):impl_(std::make_unique<Impl>(c)) {}
AudioEngine::AudioEngine(const Config& c,unsigned maximum_beam_samples,bool internal_pcm):impl_(std::make_unique<Impl>(c,maximum_beam_samples,internal_pcm)) {}
AudioEngine::~AudioEngine()=default;
AudioEngine::AudioEngine(AudioEngine&&) noexcept=default;
AudioEngine& AudioEngine::operator=(AudioEngine&&) noexcept=default;
const Config& AudioEngine::config() const noexcept {return impl_->c;}
void AudioEngine::reset() {if(impl_) {auto fresh=std::make_unique<Impl>(impl_->c,impl_->maximum_beam_samples,impl_->internal_pcm); impl_=std::move(fresh);}}
std::size_t AudioEngine::working_memory_bytes() const noexcept {
  if(!impl_) return 0;
  const auto& s=*impl_;
  // This is the wrapper-owned footprint. WebRTC APM allocations remain
  // intentionally excluded and are reported separately as unavailable.
  std::size_t bytes=sizeof(Impl);
  if(s.pre_aec) bytes+=s.pre_aec->memory_bytes();
  if(s.native) bytes+=s.native->memory_bytes();
  bytes+=(s.downlink_input.capacity()+s.downlink_output.capacity())*sizeof(float);
  bytes+=s.downlink_pipeline.capacity()*sizeof(std::unique_ptr<DownlinkPipeline>);
  for(const auto& pipeline:s.downlink_pipeline) bytes+=pipeline->wrapper_memory_bytes();
  if(s.tx_drc) bytes+=s.tx_drc->memory_bytes();
  if(s.adaptive_controller) bytes+=sizeof(AdaptiveController);
  if(s.output_eq) bytes+=s.output_eq->memory_bytes();
  if(s.unified_eq_agc) bytes+=s.unified_eq_agc->memory_bytes();
  bytes+=(s.planar.capacity()+s.taps.capacity()+s.bf_history.capacity()+
      s.mono.capacity()+s.conditioned_capture.capacity()+s.raw_history.capacity()+
      s.protected_capture.capacity()+s.aligned_linear.capacity()+
      s.fullband_history.capacity()+s.fullband_capture.capacity()+s.lowband_bf_history.capacity()+
      s.linear_history.capacity()+s.linear_bf_history.capacity()+
      s.linear_mono.capacity())*sizeof(float);
  bytes+=s.linear.capacity()*sizeof(std::array<float,160>);
  bytes+=(s.fullband_ns_output.capacity()+s.fullband_dry_history.capacity()+s.moderate_output.capacity()+
      s.ns_dry_history.capacity()+s.ns_dry_output.capacity()+s.vad_audio.capacity()+s.filtered_reference.capacity()+
      s.lowband_output.capacity()+s.lowband_ns_history.capacity()+s.monitor_capture.capacity()+
      s.linear_eight_history.capacity()+s.nonlinear_history.capacity()+s.reconstructed_linear.capacity())*sizeof(float);
  bytes+=s.ns_padding_history.capacity()*sizeof(float);
  bytes+=(s.spatial_capture.capacity()+s.spatial_linear.capacity()+s.spatial_low.capacity()+
      s.transition_dry.capacity()+s.transition_history.capacity())*sizeof(float);
  if(s.spatial) bytes+=s.spatial->memory_bytes();
  if(s.neural) bytes+=s.neural->memory_bytes();
  if(s.speex) bytes+=s.speex->wrapper_memory_bytes();
  if(s.speech_speex) bytes+=s.speech_speex->wrapper_memory_bytes();
#ifdef ITU_HAS_RNNOISE
  if(s.rn) bytes+=rnnoise_get_size();
#endif
  if(s.low_cut) bytes+=s.low_cut->memory_bytes();
  if(s.reference_monitor) bytes+=s.reference_monitor->memory_bytes();
  if(s.echo_coherence) bytes+=s.echo_coherence->memory_bytes();
  for(const auto& subspace:s.echo_subspaces) bytes+=subspace->memory_bytes();
  bytes+=s.echo_subspaces.capacity()*sizeof(std::unique_ptr<EchoSubspace>)+s.subspace_reference.capacity()*sizeof(float);
  if(s.subspace_delay) bytes+=s.subspace_delay->memory_bytes();
  if(s.noise_profile) bytes+=s.noise_profile->memory_bytes();
  if(s.speex_aec) bytes+=s.speex_aec->memory_bytes();
  if(s.ns_residual_profile) bytes+=s.ns_residual_profile->memory_bytes();
  if(s.repeated_tone) bytes+=s.repeated_tone->memory_bytes();
  return bytes;
}
Status AudioEngine::process(const float* mic,std::size_t mic_n,const float* ref,std::size_t ref_n,
    float* out,std::size_t out_n,FrameInfo& info,float* aec_tap,float* bf_tap,float* linear_tap,float* aec_input_tap) noexcept {
  return process_impl(mic,mic_n,ref,ref_n,out,out_n,info,aec_tap,bf_tap,linear_tap,aec_input_tap,nullptr);
}
Status AudioEngine::process_impl(const float* mic,std::size_t mic_n,const float* ref,std::size_t ref_n,
    float* out,std::size_t out_n,FrameInfo& info,float* aec_tap,float* bf_tap,float* linear_tap,float* aec_input_tap,
    const PostAecContext* context) noexcept {
  info={};
  if(!impl_ || !mic || !out || (impl_->c.aec && !ref)) return Status::invalid_pointer;
  auto& s=*impl_; const auto& c=s.c; unsigned h=c.frame_size,m=c.microphones;
  if(context && c.aec) return Status::invalid_length;
  const unsigned kAecOutputDelaySamples=s.aec_delay,kLinearAecDelaySamples=s.linear_delay,
      kNsDelaySamples=s.ns_delay,kLowCutDelaySamples=s.lowcut_delay;
  if(mic_n!=h*m || out_n!=h || (ref && ref_n!=h*c.reference_channels) || (!ref && ref_n)) return Status::invalid_length;
  // Public PCM stays normalized. Internal BF/SRC may legitimately overshoot;
  // preserve its finite float headroom rather than clipping one AEC input.
  for(std::size_t i=0;i<mic_n;++i) if(!std::isfinite(mic[i]) || (!s.internal_pcm && std::abs(mic[i])>1.f)) return Status::non_finite;
  for(std::size_t i=0;ref && i<ref_n;++i) if(!std::isfinite(ref[i]) || (!s.internal_pcm && std::abs(ref[i])>1.f)) return Status::non_finite;
  if(s.failed_backend_error) {info.backend_error=s.failed_backend_error; return Status::backend_error;}
  if(s.pre_aec) {
    DenormalScope denormals;
    const auto status=s.pre_aec->process(mic,ref,out,info,aec_tap,bf_tap,linear_tap,aec_input_tap);
    if(status==Status::backend_error) {
      s.failed_backend_error=info.backend_error ? info.backend_error : -1004;
      info.backend_error=s.failed_backend_error;
    }
    return status;
  }
  if(s.speex_aec) {
    DenormalScope denormals;
    const auto status=s.speex_aec->process(mic,ref,out,info,aec_tap,bf_tap,linear_tap,aec_input_tap);
    if(status==Status::backend_error) {
      s.failed_backend_error=info.backend_error ? info.backend_error : -1003;
      info.backend_error=s.failed_backend_error;
    }
    return status;
  }
  if(s.native) {
    const auto status=s.native->process(mic,ref,out,info,aec_tap,bf_tap,linear_tap,aec_input_tap,context);
    if(status==Status::backend_error) {
      s.failed_backend_error=info.backend_error ? info.backend_error : -1002;
      info.backend_error=s.failed_backend_error;
    }
    return status;
  }
  DenormalScope denormals;
  info.frame_index=s.frames; info.microphone_rms=rms(mic,mic_n);
  info.aec_capture_channels=m;
  const bool render_active=ref && std::pow(rms(ref,ref_n),2)>c.activity_floor;
  if(render_active) s.render_hangover=100;
  else if(s.render_hangover) --s.render_hangover;
  info.far_active=s.render_hangover>0;
  if(context) {
    info.post_aec_context_applied=true;
    info.post_aec_reference_evidence_valid=context->reference_evidence_valid;
    info.post_aec_near_state=context->near_state;
    info.post_aec_multirender_frames=context->multirender_frames;
    info.far_active=context->far_active;
    s.near_hangover=context->near_hold_frames;
    s.reference_state=info.reference_state=context->reference_state;
    info.reference_residual_ratio=context->reference_residual_ratio;
    info.linear_residual_ratio=context->reference_residual_ratio;
    info.reference_confidence=context->reference_confidence;
    info.linear_speech_probability=context->linear_speech_probability;
    info.nonlinear_speech_probability=context->nonlinear_speech_probability;
    info.incoherent_ratio=context->incoherent_ratio;
    info.near_protection_hold_frames=s.near_hangover;
  }
  const bool has_aec=c.aec || context;
  // An unrelated reference is still physically active, but cannot classify
  // local speech as echo or freeze the post-chain noise observation.
  const bool post_far_active=info.far_active && (!context || context->reference_state!=2);
  const bool joint_multirender=context && context->multirender_context && c.reference_channels>1;
  const bool joint_reference_unknown=joint_multirender && context->near_state==0;
  // Render activity alone cannot prove echo-only during startup or path changes.
  // Keep noise learning frozen, but reserve the extra far attenuation for
  // the joint-reference detector's confirmed echo-only state.
  const bool joint_far_suppression_allowed=!joint_multirender || context->near_state==1;
  if(!s.fullband_history.empty()) for(unsigned i=0;i<h;++i) {
    for(unsigned ch=0;ch<m;++ch) {
      const unsigned p=s.fullband_head*m+ch;
      s.fullband_capture[i*m+ch]=s.fullband_history[p];
      s.fullband_history[p]=mic[i*m+ch];
    }
    s.fullband_head=(s.fullband_head+1)%(kLowCutDelaySamples+(c.aec ? kAecOutputDelaySamples : 0));
  } else if(!s.fullband_capture.empty()) {
    // With the causal IIR low-cut there is no conditioning delay to align.
    // Keep the raw frame available for the low-band restore path instead of
    // reading the zero-length legacy history as an all-zero capture.
    std::copy_n(mic,mic_n,s.fullband_capture.data());
  }
  for(unsigned ch=0;ch<m;++ch) for(unsigned i=0;i<h;++i) {
    const float value=mic[i*m+ch];
    s.planar[ch*h+i]=value;
  }
  if(c.aec && s.low_cut) for(unsigned ch=0;ch<m;++ch)
    s.low_cut->process_frame(s.planar.data()+ch*h,s.planar.data()+ch*h,h,ch);
  for(unsigned ch=0;ch<m;++ch) for(unsigned i=0;i<h;++i)
    s.conditioned_capture[i*m+ch]=s.planar[ch*h+i];
  if(aec_input_tap) std::copy(s.conditioned_capture.begin(),s.conditioned_capture.end(),aec_input_tap);
  if(s.aec) {
    std::array<const float*,8> reverse{};
    for(unsigned ch=0;ch<c.reference_channels;++ch) {
      for(unsigned i=0;i<h;++i) s.filtered_reference[ch*h+i]=ref[i*c.reference_channels+ch];
      if(s.low_cut) s.low_cut->process_frame(s.filtered_reference.data()+ch*h,s.filtered_reference.data()+ch*h,h,m+ch);
      reverse[ch]=s.filtered_reference.data()+ch*h;
    }
    if(c.sample_rate==48000) {
      for(unsigned ch=0;ch<m;++ch) {
        if(s.monitor_down[ch]->Resample(s.planar.data()+ch*h,h,s.delta.data(),160)) return Status::backend_error;
        for(unsigned i=0;i<160;++i) s.monitor_capture[i*m+ch]=s.delta[i];
      }
      if(s.monitor_ref_down->Resample(reverse[0],h,s.monitor_reference.data(),160)) return Status::backend_error;
    }
    const float* monitor_capture=c.sample_rate==48000 ? s.monitor_capture.data() : s.conditioned_capture.data();
    const float* monitor_reference=c.sample_rate==48000 ? s.monitor_reference.data() : reverse[0];
    if(!s.echo_subspaces.empty()) {
      for(unsigned ch=0;ch<c.reference_channels;++ch) {
        const float* channel=reverse[ch];
        if(c.sample_rate==48000) {
          if(s.subspace_down[ch]->Resample(channel,h,s.delta.data(),160)) return Status::backend_error;
          channel=s.delta.data();
        }
        for(unsigned i=0;i<160;++i) s.subspace_reference[i*c.reference_channels+ch]=channel[i];
      }
      s.subspace_delay->push(monitor_capture,monitor_reference,m);
      const unsigned delay=s.subspace_delay->delay_available() ? s.subspace_delay->delay_samples() : 0;
      for(unsigned ch=0;ch<m;++ch) {
        s.echo_subspaces[ch]->push(monitor_capture+ch,s.subspace_reference.data(),m,render_active,delay);
        info.incoherent_ratio=std::max(info.incoherent_ratio,s.echo_subspaces[ch]->ratio());
      }
      info.coherence_learning_frames=s.echo_subspaces[0]->frames();
      info.reference_confidence=s.subspace_delay->confidence();
    }
    int error=s.aec->AnalyzeReverseStream(reverse.data(),webrtc::StreamConfig(c.sample_rate,c.reference_channels));
    if(s.reference_monitor) {
      s.reference_monitor->push(monitor_capture,monitor_reference,m);
      info.reference_confidence=s.reference_monitor->confidence();
    }
    const bool delay_available=s.reference_monitor && (c.continuous_vehicle_policy ?
        s.reference_monitor->delay_available() : s.reference_monitor->reliable());
    const unsigned observed_delay=c.correlation_delay && delay_available ?
        (s.reference_monitor->delay_samples()+8)/16 : c.stream_delay_ms;
    if(s.echo_coherence) {
      // Acoustic alignment is independent of the AEC3 buffer-delay policy.
      const unsigned acoustic_delay=c.stable_echo_coherence && delay_available ?
          s.reference_monitor->delay_samples() : observed_delay*16;
      const bool delay_changed=s.coherence_delay_samples==~0u ||
          std::abs(int(acoustic_delay)-int(s.coherence_delay_samples))>(c.continuous_vehicle_policy ? 16 : 0);
      if(c.stable_echo_coherence && delay_changed) {
        s.coherence_delay_samples=acoustic_delay;
        s.coherence_settle_frames=50;
        s.coherence_learning_frames=s.coherent_echo_frames=0;
        s.coherence_baseline=s.coherence_smoothed=1;
      } else if(s.coherence_settle_frames) --s.coherence_settle_frames;
      s.echo_coherence->push(monitor_capture,monitor_reference,m,
                            c.stable_echo_coherence ? s.coherence_delay_samples : acoustic_delay,c.stable_echo_coherence);
      info.incoherent_ratio=s.echo_coherence->ratio();
      const bool eligible=c.stable_echo_coherence ? s.reference_monitor && s.reference_monitor->reliable() && !s.coherence_settle_frames : s.frames>200;
      const bool learning_input=c.continuous_vehicle_policy ? info.incoherent_ratio<.55f : !s.near_observed;
      if(render_active && learning_input && eligible && (c.continuous_vehicle_policy || s.coherence_learning_frames<200)) {
        s.coherence_baseline=++s.coherence_learning_frames==1 ? info.incoherent_ratio : .98f*s.coherence_baseline+.02f*info.incoherent_ratio;
      }
      info.echo_coherence_baseline=s.coherence_baseline;
    }
    if(!error) error=s.aec->set_stream_delay_ms(observed_delay);
    if(!error) error=s.aec->ProcessStream(s.inputs.data(),webrtc::StreamConfig(c.sample_rate,m),webrtc::StreamConfig(c.sample_rate,m),s.outputs.data());
    if(error) {s.failed_backend_error=info.backend_error=error; return Status::backend_error;}
    for(float value:s.planar) if(!std::isfinite(value)) {
      s.failed_backend_error=info.backend_error=-1000; return Status::backend_error;
    }
    info.linear_aec_available=s.aec->GetLinearAecOutput(s.linear);
    if(linear_tap) for(unsigned ch=0;ch<m;++ch)
      std::copy(s.linear[ch].begin(),s.linear[ch].end(),linear_tap+ch*160);
    auto stats=s.aec->GetStatistics();
    info.internal_delay_mix=c.correlation_delay || c.external_delay_estimator ? 0.f : 1.f;
    if(stats.echo_return_loss) {info.statistics_valid|=erl_valid; info.echo_return_loss_db=*stats.echo_return_loss;}
    if(stats.echo_return_loss_enhancement) {info.statistics_valid|=erle_valid; info.echo_return_loss_enhancement_db=*stats.echo_return_loss_enhancement;}
    if(stats.delay_ms) {info.statistics_valid|=delay_valid; info.estimated_delay_ms=*stats.delay_ms;}
  } else if(linear_tap) std::fill_n(linear_tap,m*160,0.f);
  if(c.aec && c.sample_rate==48000) {
    // Replace only the nonlinear low band. 432 + down(240) + up(240)
    // is 912 samples; preserve the nonlinear high band at that same delay.
    for(unsigned ch=0;ch<m;++ch) {
      if(s.nonlinear_down[ch]->Resample(s.planar.data()+ch*h,h,s.delta.data(),160)) return Status::backend_error;
      for(unsigned i=0;i<160;++i) {
        const unsigned p=((s.eight_head+i)%152)*m+ch;
        const float linear=s.linear_eight_history[p];
        s.linear_eight_history[p]=s.linear[ch][i];
        s.delta[i]=linear-s.delta[i];
      }
      if(s.linear_up[ch]->Resample(s.delta.data(),160,s.delta_up.data(),h)) return Status::backend_error;
      for(unsigned i=0;i<h;++i) {
        const unsigned p=((s.nonlinear_head+i)%480)*m+ch;
        const float delayed=s.nonlinear_history[p];
        s.nonlinear_history[p]=s.planar[ch*h+i];
        s.planar[ch*h+i]=delayed;
        s.reconstructed_linear[i*m+ch]=delayed+s.delta_up[i];
      }
    }
    s.eight_head=(s.eight_head+160)%152;
    s.nonlinear_head=(s.nonlinear_head+h)%480;
  }
  if(c.aec && (c.reference_guard || c.speech_preserving_low_cut)) {
    for(unsigned i=0;i<h;++i) {
      for(unsigned ch=0;ch<m;++ch) {
        const unsigned oldest=s.raw_head*m+ch;
        s.protected_capture[i*m+ch]=s.raw_history[oldest];
        const float aligned=s.raw_history[((s.raw_head+kAecOutputDelaySamples-kLinearAecDelaySamples)%kAecOutputDelaySamples)*m+ch];
        if(info.far_active && s.frames>10 && info.linear_aec_available) {
          s.reference_input_power+=double(aligned)*aligned;
          s.reference_linear_power+=double(s.linear[ch][i*160/h])*s.linear[ch][i*160/h];
        }
        s.raw_history[oldest]=s.conditioned_capture[i*m+ch];
      }
      s.raw_head=(s.raw_head+1)%kAecOutputDelaySamples;
    }
    if(info.far_active && s.frames>10 && info.linear_aec_available) {
      ++s.reference_observation_frames;
      if(++s.reference_window_frames>=50) {
        s.reference_residual_ratio=static_cast<float>(s.reference_linear_power/std::max(s.reference_input_power,1e-20));
        if(s.reference_input_power>1e-6 && s.reference_residual_ratio<.72f) s.reference_state=1;
        else if(s.reference_state!=1 && s.reference_observation_frames>=400) s.reference_state=2;
        s.reference_window_frames=0;
        s.reference_input_power=s.reference_linear_power=0;
      }
    }
    info.reference_state=s.reference_state;
    info.reference_residual_ratio=s.reference_residual_ratio;
    if(s.reference_monitor && c.reference_guard) {
      s.reference_state=!info.far_active ? 0 : s.reference_monitor->unreliable() ? 2 : s.reference_monitor->reliable() ? 1 : 0;
      info.reference_state=s.reference_state;
    }
    const float target=c.reference_guard && s.reference_state==2 ? 1.f : 0.f;
    s.fallback_mix+=std::clamp(target-s.fallback_mix,-.05f,.05f);
  }
  if(!s.aligned_linear.empty() && info.linear_aec_available) {
    const unsigned delay=kAecOutputDelaySamples-kLinearAecDelaySamples;
    double residual=0,echo_power=0,cross=0;
    for(unsigned i=0;i<h;++i) {
      float average=0;
      for(unsigned ch=0;ch<m;++ch) {
        const unsigned p=s.linear_head*m+ch;
        const float value=c.sample_rate==48000 ? s.reconstructed_linear[i*m+ch] : s.linear_history[p];
        s.aligned_linear[i*m+ch]=value;
        if(c.sample_rate==16000) s.linear_history[p]=s.linear[ch][i];
        average+=value/m;
        if(!s.protected_capture.empty()) {
          const float predicted=s.protected_capture[i*m+ch]-value;
          echo_power+=double(predicted)*predicted;
          cross+=double(s.protected_capture[i*m+ch])*predicted;
        }
      }
      s.vad_audio[i]=average*32768.f;
      residual+=double(average)*average;
      s.linear_head=(s.linear_head+1)%delay;
    }
    if(c.external_vad) {
    const float* channels[]={s.vad_audio.data()};
    info.linear_speech_probability=s.aec_vad->Analyze(webrtc::DeinterleavedView<const float>(channels,h,1));
    for(unsigned i=0;i<h;++i) {
      float value=0;
      for(unsigned ch=0;ch<m;++ch) value+=s.planar[ch*h+i]/m;
      s.vad_audio[i]=value*32768.f;
    }
    info.nonlinear_speech_probability=s.nonlinear_vad->Analyze(webrtc::DeinterleavedView<const float>(channels,h,1));
    info.external_vad_calls+=2;
    }
    if(s.linear_evidence) {
      for(unsigned i=0;i<h;++i) {float v=0;for(unsigned ch=0;ch<m;++ch)v+=s.aligned_linear[i*m+ch]/m;s.evidence_native[i]=v;}
      if(s.linear_evidence_down && s.linear_evidence_down->Resample(s.evidence_native.data(),h,s.linear_evidence_audio.data(),160))return Status::backend_error;
      info.linear_speech_probability=s.linear_evidence->observe(s.linear_evidence_down ? s.linear_evidence_audio.data() : s.evidence_native.data());
      for(unsigned i=0;i<h;++i) {float v=0;for(unsigned ch=0;ch<m;++ch)v+=s.planar[ch*h+i]/m;s.evidence_native[i]=v;}
      if(s.nonlinear_evidence_down && s.nonlinear_evidence_down->Resample(s.evidence_native.data(),h,s.nonlinear_evidence_audio.data(),160))return Status::backend_error;
      info.nonlinear_speech_probability=s.nonlinear_evidence->observe(s.nonlinear_evidence_down ? s.nonlinear_evidence_audio.data() : s.evidence_native.data());
    }
    s.predicted_power=.99*s.predicted_power+.01*echo_power;
    s.predicted_cross=.99*s.predicted_cross+.01*cross;
    s.subtraction_gain=c.projection_echo_safeguard ? std::clamp(s.predicted_cross/std::max(s.predicted_power,1e-12),0.,1.) : 1;
    info.echo_subtraction_gain=s.subtraction_gain;
    s.residual_smoothed=.98*s.residual_smoothed+.02*residual/h;
    s.echo_smoothed=.98*s.echo_smoothed+.02*echo_power/(h*m);
    const double ratio=s.residual_smoothed/std::max(s.echo_smoothed,1e-12);
    info.linear_residual_ratio=ratio;
    if(info.far_active && !s.near_hangover && s.frames>300 && ratio<1)
      s.echo_floor_ratio=.98*s.echo_floor_ratio+.02*ratio;
    if(c.external_vad || s.linear_evidence) {
    const bool speech=s.frames>30 && info.linear_speech_probability>.35f;
    if(s.frames>30 && !info.far_active && info.nonlinear_speech_probability>.5f && ++s.confirmed_near_frames>=20)
      s.near_observed=true;
    const bool innovation=ratio>std::max(.02,(s.near_hangover ? 1.25 : 2.)*s.echo_floor_ratio);
    const bool support=info.nonlinear_speech_probability>(s.near_observed ? .1f : .3f) || info.linear_speech_probability > .70f;
    // A single spectral/VAD spike in a no-far frame is common in wind noise.
    // Keep the existing near hold duration, but require four consecutive
    // independent observations before entering it.
    if(!speech || info.far_active) s.near_candidate_frames=0;
    const bool coherent_echo=info.far_active && s.echo_coherence && s.coherence_learning_frames>=100 &&
        info.incoherent_ratio<std::min(.98f,s.coherence_baseline*(s.near_observed ? 1.015f : 1.3f));
    s.coherence_smoothed=.98f*s.coherence_smoothed+.02f*info.incoherent_ratio;
    const float echo_threshold=s.coherence_baseline<.15f ?
        .5f*s.coherence_baseline : std::clamp(.9f*s.coherence_baseline,.25f,.30f);
    const double echo_residual_limit=s.coherence_baseline<.15f ? .008 : 1.5;
    const bool tracked_echo=info.far_active && s.reference_monitor && s.reference_monitor->reliable() &&
        s.echo_coherence && s.coherence_learning_frames>=100 &&
        s.coherence_smoothed<std::min(.7f,1.3f*s.coherence_baseline) && ratio<.6 &&
        info.nonlinear_speech_probability<.2f;
    const bool confirmed_echo=tracked_echo || (info.far_active && s.echo_coherence && s.coherence_learning_frames>=100 &&
        (!s.near_observed ? coherent_echo :
        (s.coherence_smoothed<echo_threshold && ratio<echo_residual_limit)));
    if(confirmed_echo) s.coherent_echo_frames=std::min(20u,s.coherent_echo_frames+1);
    else if(s.coherent_echo_frames) --s.coherent_echo_frames;
    // A long-term baseline needs many excited frames and is often absent after
    // intermittent playback. Do not let stale near-speech hangover expose the
    // linear residual when the live microphone is almost fully render-coherent
    // and AEC3's nonlinear output has no speech evidence. Three confirmations
    // are required; genuine double talk raises the unexplained coherence power.
    const bool startup_echo=c.continuous_vehicle_policy && render_active && s.echo_coherence &&
        s.frames>30 && info.incoherent_ratio<.03f && info.nonlinear_speech_probability<.2f;
    s.startup_echo_confirmation=startup_echo ? std::min(3u,s.startup_echo_confirmation+1) : 0;
    info.startup_echo_confirmation=s.startup_echo_confirmation;
    // Coherence is measured before beamforming. Apply this new decision only
    // to the validated delay-sum array path; single-mic and adaptive spatial
    // output keep their established speech protection. A high unexplained
    // baseline is also insufficient reference evidence for this extra guard.
    // The new guard has only been validated for short echo paths. This is an
    // algorithm applicability range, not a reference calibration verdict:
    // require measured AEC3 delay plus independent acoustic alignment within
    // 100 ms, and twenty consistent delay frames. Long/unknown paths retain v1.
    const bool supported_delay=(c.ecnr_v2_control || c.ecnr_v2_double_evidence) && (info.statistics_valid & delay_valid) &&
        std::isfinite(info.estimated_delay_ms) && info.estimated_delay_ms>=0 &&
        info.estimated_delay_ms<=100 && s.coherence_delay_samples<=1600;
    if(supported_delay) s.v2_delay_frames=std::abs(info.estimated_delay_ms-s.v2_observed_delay_ms)<=4 ?
        std::min(20u,s.v2_delay_frames+1) : 1;
    else s.v2_delay_frames=0;
    s.v2_observed_delay_ms=supported_delay ? info.estimated_delay_ms : -1;
    const bool v2_evidence_eligible=(c.ecnr_v2_control || c.ecnr_v2_double_evidence) && m>1 && c.beamforming && !c.spatial_mvdr &&
        s.v2_delay_frames>=20 && c.reference_channels==1 && render_active && s.reference_state==1 &&
        s.reference_monitor && s.reference_monitor->reliable() &&
        s.echo_coherence && s.coherence_learning_frames>=100 && s.coherence_baseline<.30f;
    info.v2_double_evidence_eligible=v2_evidence_eligible;
    // Lost reference/applicability invalidates the new latch immediately.
    // The original fallback branch below remains unchanged.
    if(c.ecnr_v3_g1_control && !v2_evidence_eligible) s.g1_gap.reset();
    if(c.reference_channels>1 && c.multirender_speech_protection) {
      // AEC3's linear residual uses all render channels. Require independent
      // speech evidence after residual suppression as well; residual VAD alone
      // classifies uncancelled radio speech as near-end speech.
      const bool innovation=!s.echo_subspaces.empty() && s.subspace_delay->delay_available() &&
          s.echo_subspaces[0]->frames()>100 && info.incoherent_ratio>c.multirender_innovation_threshold;
      const bool double_talk=speech && info.nonlinear_speech_probability>.15f && ratio>.08 && innovation;
      if(!info.far_active && speech) s.near_hangover=50;
      else if(double_talk) s.near_hangover=50;
      else if(!innovation) s.near_hangover=0;
      else if(s.near_hangover) --s.near_hangover;
    } else if(v2_evidence_eligible) {
      // Residual radio speech can pass the linear VAD. Require independent
      // nonlinear speech evidence and unexplained render-coherence energy
      // before renewing linear/NS restoration. A high unexplained baseline
      // indicates weak reference tracking; preserve the established protection
      // there as well as for unverified references and the
      // learning period retain the original fallback and near-end policy.
      const bool dual=speech && info.nonlinear_speech_probability>c.ecnr_v2_nonlinear_vad_min &&
          info.incoherent_ratio>std::max(.48f,s.coherence_baseline*1.6f);
      s.v2_near_confirmation=dual ? std::min(3u,s.v2_near_confirmation+1) : 0;
      bool startup_clear=s.startup_echo_confirmation>=3;
      bool coherent_clear=s.coherent_echo_frames>=(s.near_observed ? 20u : 1u);
      if(c.ecnr_v3_g1_control) {
        // The original first-entry evidence and clear priority stay intact.
        // Only a strong ORIGINAL confirmation renews this short-lived latch.
        // While valid, both existing clears require continuous echo evidence;
        // the union counter cannot invent a new clear or renew its own lifetime.
        const bool original_strict=s.v2_near_confirmation>=3;
        s.g1_gap.observe(true,original_strict,info.nonlinear_speech_probability,
                         startup_echo,confirmed_echo);
        const bool defer=s.g1_gap.defer_clear(s.near_hangover);
        info.g1_strict_near_age=s.g1_gap.strict_age();
        info.g1_startup_echo_frames=s.g1_gap.startup_frames();
        info.g1_startup_clear_threshold=3u; // Original startup predicate; union threshold is separate.
        info.g1_startup_clear_deferred=startup_clear && defer;
        info.g1_echo_evidence_frames=s.g1_gap.echo_frames();
        info.g1_echo_clear_threshold=s.g1_gap.threshold(s.near_hangover);
        info.g1_echo_clear_deferred=(startup_clear || coherent_clear) && defer;
        info.g1_strong_near_confirmation=original_strict &&
            std::isfinite(info.nonlinear_speech_probability) &&
            info.nonlinear_speech_probability>=NearProtectionGap::kStrongNonlinearVad;
        info.g1_near_latch_active=s.g1_gap.protects(s.near_hangover);
        info.g1_original_confirmed_echo=confirmed_echo;
        info.g1_original_coherent_clear=coherent_clear;
        startup_clear=startup_clear && !defer;
        coherent_clear=coherent_clear && !defer;
      }
      if(startup_clear) s.near_hangover=0;
      else if(s.v2_near_confirmation>=3) s.near_hangover=200;
      else if(coherent_clear) s.near_hangover=0;
      else if(s.near_hangover) --s.near_hangover;
    } else if(s.startup_echo_confirmation>=3) s.near_hangover=0;
    else if(speech && !info.far_active) {
      const float near_vad = c.external_vad ? s.spatial_vad : s.input_excess_control;
      if(near_vad > .2f) {
        s.near_candidate_frames=std::min(4u,s.near_candidate_frames+1);
        if(s.near_candidate_frames>=4) s.near_hangover=50;
      } else {
        s.near_candidate_frames=0;
        if(s.near_hangover) --s.near_hangover;
      }
    }
    else if(speech && info.far_active && (!tracked_echo && !coherent_echo && support && (s.near_observed || innovation) && (!c.ecnr_v2_control || info.incoherent_ratio > std::max(.48f, s.coherence_baseline * 1.6f)))) s.near_hangover=200;
    else if(s.coherent_echo_frames>=(s.near_observed ? 20u : 1u)) s.near_hangover=0;
    else if(s.near_hangover) --s.near_hangover;
    if(!v2_evidence_eligible) s.v2_near_confirmation=0;
    info.v2_near_confirmation=s.v2_near_confirmation;
    info.near_protection_hold_frames=s.near_hangover;
    const float target=c.linear_output_diagnostic || (c.linear_speech_protection && s.near_hangover) ? 1.f : 0.f;
    s.nearend_mix+=std::clamp(target-s.nearend_mix,
                            c.multirender_speech_protection ? -.2f : -.05f,.05f);
    info.nearend_mix=s.nearend_mix;
    }
  }
  for(unsigned i=0;i<h;++i) {
    for(unsigned ch=0;ch<m;++ch) {
      s.taps[i*m+ch]=s.planar[ch*h+i];
      const float safe_linear=!(s.aec_vad || s.linear_evidence) ? s.taps[i*m+ch] : s.protected_capture.empty() ? s.aligned_linear[i*m+ch] :
          (1-s.subtraction_gain)*s.protected_capture[i*m+ch]+s.subtraction_gain*s.aligned_linear[i*m+ch];
      const float cancelled=c.speech_preserving_low_cut && c.aec && !info.far_active && !s.protected_capture.empty() ?
          s.protected_capture[i*m+ch] : (s.aec_vad || s.linear_evidence) ? (1-s.nearend_mix)*s.taps[i*m+ch]+s.nearend_mix*safe_linear : s.taps[i*m+ch];
      s.bf_history[s.beam_head*m+ch]=c.aec && c.reference_guard ?
          (1-s.fallback_mix)*cancelled+s.fallback_mix*(s.protected_capture.empty()?cancelled:s.protected_capture[i*m+ch]) : cancelled;
      if(!s.linear_bf_history.empty()) s.linear_bf_history[s.beam_head*m+ch]=
          (1-s.fallback_mix)*s.aligned_linear[i*m+ch]+s.fallback_mix*(s.protected_capture.empty() ? s.aligned_linear[i*m+ch] : s.protected_capture[i*m+ch]);
      if(!s.lowband_bf_history.empty()) s.lowband_bf_history[s.beam_head*m+ch]=
          s.fullband_capture.empty() ? 0.f : s.fullband_capture[i*m+ch]-(c.aec && !s.protected_capture.empty() ? s.protected_capture[i*m+ch] : 0.f);
    }
    float sum=0,linear_sum=0,lowband_sum=0;
    if(c.beamforming) for(unsigned ch=0;ch<m;++ch) {
      float delay=c.beam_delays_samples[ch]; unsigned d=static_cast<unsigned>(delay); float fraction=delay-d;
      unsigned p=(s.beam_head+s.beam_size-d)%s.beam_size;
      unsigned q=(p+s.beam_size-1)%s.beam_size;
      const float beam=(1-fraction)*s.bf_history[p*m+ch]+fraction*s.bf_history[q*m+ch];
      const float lin=s.linear_bf_history.empty() ? 0.f : (1-fraction)*s.linear_bf_history[p*m+ch]+fraction*s.linear_bf_history[q*m+ch];
      const float low=s.lowband_bf_history.empty() ? 0.f : (1-fraction)*s.lowband_bf_history[p*m+ch]+fraction*s.lowband_bf_history[q*m+ch];
      sum+=beam; linear_sum+=lin; lowband_sum+=low;
      if(s.spatial) {s.spatial_capture[i*m+ch]=beam; s.spatial_linear[i*m+ch]=lin; s.spatial_low[i*m+ch]=low;}
    }
    s.mono[i]=c.beamforming ? sum/m : s.bf_history[s.beam_head*m];
    if(!s.linear_mono.empty()) s.linear_mono[i]=c.beamforming ? linear_sum/m : s.linear_bf_history[s.beam_head*m];
    if(!s.lowband_bf_history.empty()) s.lowband_output[i]=c.beamforming ? lowband_sum/m : s.lowband_bf_history[s.beam_head*m];
    s.beam_head=(s.beam_head+1)%s.beam_size;
  }
  if(s.spatial) {
    s.spatial->process(s.spatial_capture.data(),s.spatial_linear.data(),s.spatial_low.data(),
        s.mono.data(),s.linear_mono.empty()?nullptr:s.linear_mono.data(),s.lowband_output.data(),
        (c.external_vad ? s.spatial_vad : s.input_excess_control)<.2f &&
        !info.far_active && !s.near_hangover);
    info.spatial_delay_samples=s.spatial->delay(); info.spatial_noise_updates=s.spatial->noise_updates();
    info.spatial_covariance_updates=s.spatial->updates();
    info.spatial_fallback_bins=s.spatial->fallbacks(); info.wind_score=s.spatial->wind_score();
  }
  info.aec_rms=rms(s.taps.data(),s.taps.size());
  if(aec_tap) std::copy(s.taps.begin(),s.taps.end(),aec_tap);
  if(bf_tap) std::copy(s.mono.begin(),s.mono.end(),bf_tap);
  if(s.low_cut) info.low_cut_delay_samples=kLowCutDelaySamples;
  info.post_low_cut_delay_samples=c.aec ? 0 : info.low_cut_delay_samples;
  if(s.low_cut && !c.aec) {
    s.low_cut->process_frame(s.mono.data(),s.mono.data(),h,0);
    if(!s.lowband_bf_history.empty()) for(unsigned i=0;i<h;++i) s.lowband_output[i]-=s.mono[i];
  }
  if(s.vad) {
    for(unsigned i=0;i<h;++i) s.vad_audio[i]=s.mono[i]*32768.f;
    const float* channels[]={s.vad_audio.data()};
    info.speech_probability=s.vad->Analyze(webrtc::DeinterleavedView<const float>(channels,h,1));
    s.spatial_vad=info.speech_probability;
    ++info.external_vad_calls;
  }
  if(s.repeated_tone) {
    if(s.tone_down && s.tone_down->Resample(s.mono.data(),h,s.tone_audio.data(),160)) return Status::backend_error;
    s.repeated_tone->observe(s.tone_down ? s.tone_audio.data() : s.mono.data(),!info.far_active && !s.near_hangover);
  }
  info.external_vad_active=c.external_vad;
  float control=info.speech_probability;
  if(s.spectral_excess) {
    if(s.excess_down && s.excess_down->Resample(s.mono.data(),h,s.excess_audio.data(),160)) return Status::backend_error;
    control=s.spectral_excess->observe(s.excess_down ? s.excess_audio.data() : s.mono.data(),!post_far_active);
    info.input_excess_score=control;
    info.input_excess_fraction=context && post_far_active ? 0.f : s.spectral_excess->fraction();
    if(s.ns_evidence)control=s.ns_evidence->observe(s.excess_down ? s.excess_audio.data() : s.mono.data());
    if(!has_aec && c.linear_speech_protection && !c.external_vad) {
      const float excess_prob=std::clamp((info.input_excess_fraction-.35f)/.5f,0.f,1.f);
      control=std::max(control,excess_prob);
      info.linear_speech_probability=control>.05f ? std::max(control,info.input_excess_fraction) : 0.f;
      info.nonlinear_speech_probability=control;
    }
    s.input_excess_control=control;
    // An unreliable render reference disables the normal coherence guard.
    // Re-enable near-end protection only after an independent spectral
    // fraction and both speech observers agree for several frames. The
    // fraction is measured after the current mono frame is formed, so this
    // latch intentionally takes effect on the following frame.
    const bool reference_near_candidate=!context && c.reference_guard &&
        s.reference_state==2 && post_far_active &&
        info.input_excess_fraction>.70f &&
        info.linear_speech_probability>.70f &&
        info.nonlinear_speech_probability>.70f;
    s.reference_near_candidate_frames=reference_near_candidate ?
        std::min(4u,s.reference_near_candidate_frames+1) : 0;
    if(s.reference_near_candidate_frames>=3) s.near_hangover=50;
  }
  if(context && !c.external_vad) {
    // Use the current residual observation during render activity. The
    // standalone spectral floor intentionally stops learning in those frames;
    // its stale fraction is not a current-frame speech observation.
    if(post_far_active) control=context->nonlinear_speech_probability;
    if(post_far_active && !s.near_hangover && joint_far_suppression_allowed) control=0.f;
    s.input_excess_control=control;
  }
  constexpr float kWindLowbandFloor=.25f;
  info.control_evidence_probability=control;
  const float pre_ns_level=rms(s.mono.data(),h);
  info.pre_ns_rms=pre_ns_level;
  if(c.continuous_vehicle_policy && m==1 && !s.lowband_bf_history.empty()) {
    // 300 Hz second-order Butterworth low pass over the already high-passed
    // mono signal isolates the 60-300 Hz band. The filter is causal and adds
    // no delay to the alignment contract.
    double band_power=0;
    for(unsigned i=0;i<h;++i) {
      const double y=s.wind_lp_b0*s.mono[i]+s.wind_lowpass.z1;
      s.wind_lowpass.z1=s.wind_lp_b1*s.mono[i]-s.wind_lp_a1*y+s.wind_lowpass.z2;
      s.wind_lowpass.z2=s.wind_lp_b2*s.mono[i]-s.wind_lp_a2*y;
      band_power+=y*y/h;
    }
    const double low_power=std::pow(rms(s.lowband_output.data(),h),2);
    const float tilt=10.f*std::log10(std::max(band_power,1e-20)/std::max(low_power,1e-20));
    s.wind_tilt_db=.95f*s.wind_tilt_db+.05f*tilt;
    s.wind_envelope+=.002f*(pre_ns_level-s.wind_envelope);
    const float envelope_ratio=pre_ns_level/std::max(s.wind_envelope,1e-9f);
    s.wind_envelope_deviation=.95f*s.wind_envelope_deviation+.05f*std::abs(envelope_ratio-1.f);
    // Speech is mid-band dominant with a syllabic envelope; low-frequency wind
    // and rumble are not. Only the physical contrast is used, so no speech or
    // noise decision changes.
    const bool wind_band=s.wind_tilt_db<0.f;
    const bool steady_envelope=s.wind_envelope_deviation<=.45f;
    const bool lowband_quiet=wind_band && steady_envelope && !post_far_active;
    s.wind_lowband_gain+=std::clamp((lowband_quiet?kWindLowbandFloor:1.f)-s.wind_lowband_gain,-.05f,.05f);
  }
  if(!s.lowband_bf_history.empty()) {
    const double lowband_power=std::pow(rms(s.lowband_output.data(),h),2);
    const double highband_power=double(pre_ns_level)*pre_ns_level;
    if(!post_far_active && !s.near_hangover && control<.1f && s.lowband_noise_frames<30) {
      const double rate=1./(++s.lowband_noise_frames);
      s.lowband_noise_power+=(lowband_power-s.lowband_noise_power)*rate;
    }
    if(post_far_active && !s.near_observed && !s.near_hangover && s.frames>200 &&
        s.reference_monitor && s.reference_monitor->reliable() && s.lowband_echo_frames<100) {
      s.lowband_echo_power+=(lowband_power-s.lowband_echo_power)/(++s.lowband_echo_frames);
    }
    const double interference=post_far_active ? std::max(s.lowband_noise_power,s.lowband_echo_power) : s.lowband_noise_power;
    const float clean_fraction=std::clamp(1.-interference/std::max(lowband_power,1e-12),0.,1.);
    if(post_far_active || (c.continuous_vehicle_policy && (info.wind_score>.25f || lowband_power>=6.*highband_power))) {
      s.lowband_speech_candidate_frames=0;
      s.lowband_speech_hold=0;
    } else if(control>.5f) {
      s.lowband_speech_candidate_frames=std::min(4u,s.lowband_speech_candidate_frames+1);
      if(s.lowband_speech_candidate_frames>=4) s.lowband_speech_hold=50;
    } else {
      s.lowband_speech_candidate_frames=0;
      if(s.lowband_speech_hold) --s.lowband_speech_hold;
    }
    const bool speech=s.lowband_speech_hold || (post_far_active && s.near_observed && s.near_hangover);
    const float target=speech ? clean_fraction : 0.f;
    s.lowband_restore_mix+=std::clamp(target-s.lowband_restore_mix,-.2f,.05f);
  }
  info.lowband_restore_mix=s.lowband_restore_mix;
  info.coherence_learning_frames=s.echo_subspaces.empty() ? s.coherence_learning_frames : s.echo_subspaces[0]->frames();
  const bool fullband_near=c.speech_preserving_low_cut && !post_far_active &&
      (!s.lowband_bf_history.empty() || context);
  bool high_snr_dry=false;
  const float independent_near=(context && !post_far_active) || (c.repair_stage>=4 && !has_aec) ?
      control : info.linear_speech_probability;
  double gain_qualification_power=double(pre_ns_level)*pre_ns_level;
  if(c.ecnr_coordinated_control && fullband_near) {
    gain_qualification_power=0;
    for(unsigned i=0;i<h;++i) {
      const double value=s.mono[i]+s.lowband_output[i];
      gain_qualification_power+=value*value/h;
    }
  }
  if(!s.transition_history.empty()) {
    info.startup_dry_mix=s.noise_transition.observe(pre_ns_level,control,
        post_far_active,has_aec && s.near_hangover);
    info.noise_control_state=s.noise_transition.state();
    info.noise_transition_events=s.noise_transition.events();
    for(unsigned i=0;i<h;++i) {
      s.transition_dry[i]=s.transition_history[s.transition_head];
      s.transition_history[s.transition_head]=s.mono[i]+(fullband_near || context ? s.lowband_output[i] : 0.f);
      s.transition_head=(s.transition_head+1)%s.transition_history.size();
    }
  }
  if(s.fullband_ns || s.speech_speex) {
    // Both NS histories advance continuously and share the same synthesis delay.
    for(unsigned i=0;i<h;++i) s.fullband_ns_output[i]=s.mono[i]+s.wind_lowband_gain*s.lowband_output[i];
    const float* input[]={s.fullband_ns_output.data()}; float* output[]={s.fullband_ns_output.data()};
    if(s.speech_speex) s.speech_speex->process(s.fullband_ns_output.data());
    else {
      s.fullband_ns_audio->CopyFrom(input,webrtc::StreamConfig(c.sample_rate,1));
      if(c.sample_rate==48000) s.fullband_ns_audio->SplitIntoFrequencyBands();
      s.fullband_ns->Analyze(*s.fullband_ns_audio);
      s.fullband_ns->Process(s.fullband_ns_audio.get());
      if(c.sample_rate==48000) s.fullband_ns_audio->MergeFrequencyBands();
      s.fullband_ns_audio->CopyTo(webrtc::StreamConfig(c.sample_rate,1),output);
    }
    double power=0;
    const double frame_power=std::pow(rms(s.fullband_ns_output.data(),h),2);
    const double input_power=std::pow(rms(s.mono.data(),h),2)+std::pow(rms(s.lowband_output.data(),h),2);
    s.speech_branch_power=.95*s.speech_branch_power+.05*input_power;
    const bool low_ambient_noise=s.fullband_noise_power<1.5e-3;
    const bool clean_ambient_noise=s.fullband_noise_power<6e-5;
    if(control<.1f) ++s.fullband_quiet_frames;
    else s.fullband_quiet_frames=0;
      const bool dropped_to_noise_floor=control<.1f && s.fullband_noise_frames>=5 &&
        input_power<=2.5*std::max(s.fullband_noise_power,1e-12);
    const bool in_long_silence=(s.fullband_quiet_frames>=20 && dropped_to_noise_floor) || s.fullband_quiet_frames>=40;
    if(in_long_silence) s.lowband_speech_hold=0;
    const bool protected_speech=fullband_near && s.fullband_speech_hold && clean_ambient_noise &&
        (s.fullband_noise_frames<10 || input_power>10*std::max(s.fullband_noise_power,1e-12));
    const bool confirmed_near=independent_near>.4f || s.fullband_speech_hold;
    const bool high_ratio=s.fullband_noise_frames>=10 &&
        (input_power>12.*std::max(s.fullband_noise_power,1e-12) ||
         s.speech_branch_power>12.*std::max(s.fullband_noise_power,1e-12));
    const bool cold_start_speech=c.repair_stage>=4 ?
        c.high_snr_dry_mix>0 && s.frames<50 && independent_near>.55f && high_ratio :
        c.high_snr_dry_mix>0 && s.frames>0 && s.frames<50 &&
        ((info.linear_speech_probability>.15f && s.frames>0) || input_power<2e-4);
    high_snr_dry=c.repair_stage>=4 || context ?
        fullband_near && c.high_snr_dry_mix>0 && !in_long_silence &&
        confirmed_near && high_ratio && clean_ambient_noise :
        fullband_near && c.high_snr_dry_mix>0 &&
        (cold_start_speech || (!in_long_silence && s.frames>=30 && clean_ambient_noise &&
         (s.fullband_noise_power<3.5e-5 || (s.fullband_speech_hold &&
          (s.speech_branch_power>12.*std::max(s.fullband_noise_power,1e-12) ||
           input_power>12.*std::max(s.fullband_noise_power,1e-12))))));
    for(unsigned i=0;i<h;++i) {
      const float value=s.mono[i]+s.lowband_output[i];
      power+=double(value)*value;
      const float dry=s.fullband_dry_history[s.fullband_dry_head];
      s.fullband_dry_history[s.fullband_dry_head]=value;
      s.fullband_dry_head=(s.fullband_dry_head+1)%kNsDelaySamples;
      if(fullband_near && s.fullband_noise_frames>=30 && s.fullband_noise_power<1e-12)
        s.fullband_ns_output[i]=dry;
      else if(!s.speech_speex && protected_speech && frame_power>1e-12)
        s.fullband_ns_output[i]=.1f*s.fullband_ns_output[i]+.9f*dry;
      if(high_snr_dry)
        s.fullband_ns_output[i]=(1-c.high_snr_dry_mix)*s.fullband_ns_output[i]+c.high_snr_dry_mix*dry;
    }
    const bool noise_observation=!s.speech_speex || (s.frames>=30 && !s.fullband_speech_hold);
    const bool cold_start_noise=c.continuous_vehicle_policy && s.frames>0 && s.frames<30 &&
        !s.fullband_speech_hold && independent_near<.14f && power/h>=1e-3;
    if(fullband_near && noise_observation && (control<.1f || cold_start_noise) &&
        (s.fullband_noise_frames<5 || s.fullband_quiet_frames>=3 || power/h<=2.2*std::max(s.fullband_noise_power,1e-12)) &&
        (c.continuous_vehicle_policy || s.fullband_noise_frames<30)) {
      const double rate=s.fullband_noise_frames<30 ? 1./(++s.fullband_noise_frames) : .05;
      s.fullband_noise_power+=(power/h-s.fullband_noise_power)*rate;
    }
    // The fullband near-preservation path re-injects the low band and the
    // fullband NS output. `control` is NS-derived evidence, which wind noise
    // raises while the independent linear observer stays low, so a
    // control-only confirmation latches the path on wind. Require the
    // independent linear evidence as well; the hold still needs three
    // consecutive confirmations. Gain, thresholds and denominators untouched.
    const float kFullbandIndependentSpeechMin=.20f;
    const float eff_vad = low_ambient_noise ? std::max(control, independent_near) : control;
    if(fullband_near && eff_vad>.4f && independent_near>kFullbandIndependentSpeechMin)
      s.fullband_speech_confirmation=std::min(3u,s.fullband_speech_confirmation+1);
    else s.fullband_speech_confirmation=0;
    if(!fullband_near) s.fullband_speech_hold=0;
    else if(s.fullband_speech_confirmation>=3 ||
            (c.continuous_vehicle_policy && low_ambient_noise && s.fullband_noise_power>=1e-6 &&
             eff_vad>.35f && independent_near>.25f &&
             input_power>3.2*std::max(s.fullband_noise_power,1e-12)))
      s.fullband_speech_hold=30;
    else if(s.fullband_speech_hold) --s.fullband_speech_hold;
    if(in_long_silence ||
       s.fullband_quiet_frames>=(low_ambient_noise && s.fullband_noise_power<4e-4 ? 25u : 6u))
      s.fullband_speech_hold=0;
    if(c.continuous_vehicle_policy && fullband_near && s.fullband_noise_power>=1.5e-3 &&
       s.fullband_quiet_frames>=4 && power/h<2.2*s.fullband_noise_power && s.near_hangover) {
      s.near_hangover=s.near_hangover>5 ? s.near_hangover-5 : 0;
    }
    const bool quiet_input=s.fullband_noise_frames>=30 && s.fullband_noise_power<1e-12;
    const bool initial_observation=s.speech_speex && s.frames<30;
    const bool high_snr=!s.speech_speex || s.fullband_noise_frames<10 ||
        s.speech_branch_power>3.2*std::max(s.fullband_noise_power,1e-12) ||
        input_power>3.2*std::max(s.fullband_noise_power,1e-12);
    const float target=fullband_near && high_snr && (s.fullband_speech_hold || quiet_input || initial_observation) ? 1.f : 0.f;
    if(initial_observation && fullband_near) s.fullband_ns_mix=1.f;
    else if(c.continuous_vehicle_policy && fullband_near && high_snr && (((low_ambient_noise && s.fullband_noise_power>=1e-6) ? (control>.3f && independent_near>kFullbandIndependentSpeechMin) : (control>.45f && independent_near>.45f && s.near_hangover>0)) || s.fullband_speech_hold || cold_start_speech)) s.fullband_ns_mix=1.f;
    s.fullband_ns_mix+=std::clamp(target-s.fullband_ns_mix,dropped_to_noise_floor ? -.5f : -.1f,high_snr_dry ? .25f : .1f);
    const double high_power=std::pow(rms(s.mono.data(),h),2);
    const double low_power=std::pow(rms(s.lowband_output.data(),h),2);
    info.highband_energy_fraction=high_power/std::max(high_power+low_power,1e-20);
    if(!quiet_input && info.highband_energy_fraction<c.fullband_min_highband_fraction && control<.1f && !s.fullband_speech_hold && independent_near<.3f)
      s.fullband_ns_mix=0;
    if(c.fullband_near_ns) s.fullband_ns_mix=fullband_near ? 1.f : 0.f;
    // The post-AEC fullband branch advances continuously, including the echo
    // learning period. Select that same branch during render rather than
    // changing the spectrum seen by the conditioned NS at playback edges.
    // Its input and the aligned protection tap are MDF residuals, not capture.
    if(context && post_far_active) s.fullband_ns_mix=1.f;
    info.fullband_ns_mix=s.fullband_ns_mix;
    info.fullband_noise_power=s.fullband_noise_power;
  }
  AdaptiveController::Decision adaptive_decision{};
  adaptive_decision.gain_limit_db=c.max_gain_db;
  if(s.vehicle_ns) {
    // Consume the fullband post-AEC residual once. All NS-local wet/dry paths
    // and their common delay live in this component; no later raw restoration.
    if(fullband_near || context) for(unsigned i=0;i<h;++i) s.mono[i]+=s.lowband_output[i];
    if(!s.vehicle_ns->process(s.mono.data(),h)) return Status::backend_error;
    const auto state=s.vehicle_ns->statistics();
    info.hybrid_ns_state=state.student_state;
    info.hybrid_ns_inferences=state.student_inferences;
    info.hybrid_ns_fallback_frames=state.student_fallback_frames;
    info.hybrid_ns_applied=state.student_applied;
    info.vehicle_ns_rumble_corner_hz=state.rumble_corner_hz;
    info.vehicle_ns_rumble_low_ratio_db=state.rumble_low_ratio_db;
    info.vehicle_ns_raw_mix_budget=state.raw_mix_budget;
    info.vehicle_ns_raw_mix_min=state.raw_mix_applied_min;
    info.vehicle_ns_raw_mix_max=state.raw_mix_applied_max;
    info.noise_speech_mix=state.dry_mix_target;
    info.startup_dry_mix=state.quiet_mix_target;
    info.unified_ns_weights=state.unified_weights;
    info.unified_ns_target_weights=state.unified_target_weights;
    info.unified_ns_weight_entropy=state.unified_weight_entropy;
    info.unified_ns_frame_snr_db=state.unified_frame_snr_db;
    info.unified_ns_flatness=state.unified_flatness;info.unified_ns_tonality=state.unified_tonality;
    info.unified_ns_minima_confidence=state.unified_minima_confidence;
    info.unified_ns_transient_evidence=state.unified_transient_evidence;info.unified_ns_noise_step_evidence=state.unified_noise_step_evidence;
    info.unified_ns_probability=state.unified_probability;info.unified_ns_gain=state.unified_gain;info.unified_ns_noise_power=state.unified_noise_power;
    info.unified_ns_bootstrap_seed=state.unified_bootstrap_seed;info.unified_ns_noise_updates=state.unified_noise_updates;
    if(c.diagnostic_stages) std::copy_n(s.mono.data(),h,c.diagnostic_stages->nr_selected.data());
  }
  if(s.ns) {
    for(unsigned i=0;i<h;++i) {
      s.ns_dry_output[i]=s.ns_dry_history[s.ns_dry_head];
      s.ns_dry_history[s.ns_dry_head]=s.mono[i]+(fullband_near || context ? s.lowband_output[i] : 0.f);
      s.ns_dry_head=(s.ns_dry_head+1)%kNsDelaySamples;
      if(!s.lowband_bf_history.empty()) {
        const float delayed=s.lowband_ns_history[s.lowband_ns_head];
        s.lowband_ns_history[s.lowband_ns_head]=s.lowband_output[i];
        s.lowband_output[i]=delayed;
        s.lowband_ns_head=(s.lowband_ns_head+1)%kNsDelaySamples;
      }
    }
    // Both observers use one conversion of this same physical frame.
    if((s.ns_residual_profile || s.noise_profile) && s.profile_down &&
        s.profile_down->Resample(s.mono.data(),h,s.profile_audio.data(),160)) return Status::backend_error;
    if(s.ns_residual_profile) {
      const bool cold_profile_noise=c.continuous_vehicle_policy && s.frames>0 && s.frames<=30 &&
          !post_far_active && info.linear_speech_probability<.14f && pre_ns_level>=.03f;
      s.ns_residual_profile->observe(s.profile_down ? s.profile_audio.data() : s.mono.data(),
          !post_far_active && ((s.frames>30 && control<.1f) || cold_profile_noise));
    }
    const float* input[]={s.mono.data()}; float* output[]={s.mono.data()};
    s.ns_audio->CopyFrom(input,webrtc::StreamConfig(c.sample_rate,1));
    if(c.sample_rate==48000) s.ns_audio->SplitIntoFrequencyBands();
    if(s.noise_profile) {
      s.noise_profile->observe(s.profile_down ? s.profile_audio.data() : s.mono.data(),s.frames>30 && !post_far_active && control<.3f);
      info.noise_flatness=s.noise_profile->flatness();
      info.noise_profile_observations=s.noise_profile->observations();
      s.moderate_audio->CopyFrom(input,webrtc::StreamConfig(c.sample_rate,1));
      if(c.sample_rate==48000) s.moderate_audio->SplitIntoFrequencyBands();
    }
    if(c.analyze_linear_noise && !s.linear_mono.empty()) {
      const float* analysis[]={s.linear_mono.data()};
      s.ns_analysis_audio->CopyFrom(analysis,webrtc::StreamConfig(c.sample_rate,1));
      if(c.sample_rate==48000) s.ns_analysis_audio->SplitIntoFrequencyBands();
      s.ns->Analyze(*s.ns_analysis_audio);
      if(s.moderate_ns) s.moderate_ns->Analyze(*s.ns_analysis_audio);
    } else {
      s.ns->Analyze(*s.ns_audio);
      if(s.moderate_ns) s.moderate_ns->Analyze(*s.ns_audio);
    }
    s.ns->Process(s.ns_audio.get());
    if(c.sample_rate==48000) s.ns_audio->MergeFrequencyBands();
    s.ns_audio->CopyTo(webrtc::StreamConfig(c.sample_rate,1),output);
    if(s.moderate_ns) {
      s.moderate_ns->Process(s.moderate_audio.get());
      if(c.sample_rate==48000) s.moderate_audio->MergeFrequencyBands();
      float* moderate_output[]={s.moderate_output.data()};
      s.moderate_audio->CopyTo(webrtc::StreamConfig(c.sample_rate,1),moderate_output);
      if(c.ecnr_coordinated_control) {
        const float next=s.noise_profile->white_mix();
        for(unsigned i=0;i<h;++i) {
          const float mix=s.noise_profile_mix+(next-s.noise_profile_mix)*float(i+1)/h;
          s.mono[i]=(1-mix)*s.mono[i]+mix*s.moderate_output[i];
        }
        s.noise_profile_mix=next;
      } else if(s.noise_profile->white()) std::copy(s.moderate_output.begin(),s.moderate_output.end(),s.mono.begin());
      info.noise_profile_mix=c.ecnr_coordinated_control ? s.noise_profile_mix : (s.noise_profile->white() ? 1.f : 0.f);
    }
    if(!s.ns_padding_history.empty()) for(unsigned i=0;i<h;++i) {
      const float delayed=s.ns_padding_history[s.ns_padding_head];
      s.ns_padding_history[s.ns_padding_head]=s.mono[i];
      s.ns_padding_head=(s.ns_padding_head+1)%s.ns_padding_history.size();
      s.mono[i]=delayed;
    }
    if(s.fullband_ns || s.speech_speex) for(unsigned i=0;i<h;++i)
      s.mono[i]=(1-s.fullband_ns_mix)*s.mono[i]+s.fullband_ns_mix*s.fullband_ns_output[i];
    if(c.diagnostic_stages) std::copy_n(s.mono.data(),h,c.diagnostic_stages->nr_selected.data());
    if(s.adaptive_controller) {
      adaptive_decision=s.adaptive_controller->observe({
          pre_ns_level,control,
          context && post_far_active && !s.near_hangover && joint_far_suppression_allowed ? 0.f : info.linear_speech_probability,
          context && post_far_active && !s.near_hangover && joint_far_suppression_allowed ? 0.f : info.nonlinear_speech_probability,
          info.linear_residual_ratio,info.incoherent_ratio,info.wind_score,
          post_far_active,s.near_hangover>0,info.reference_state==1});
      info.adaptive_noise_floor=adaptive_decision.noise_floor;
      info.adaptive_near_confidence=adaptive_decision.near_confidence;
      info.adaptive_far_confidence=adaptive_decision.far_confidence;
      info.adaptive_preserve_mix=adaptive_decision.preserve_mix;
      info.adaptive_gain_limit_db=adaptive_decision.gain_limit_db;
      info.adaptive_state=adaptive_decision.state;
    }
    if(!post_far_active) s.ns_protection_hold=0;
    else if(has_aec && s.near_hangover) s.ns_protection_hold=50;
    else if(s.ns_protection_hold) --s.ns_protection_hold;
    if(c.multirender_speech_protection && c.reference_channels>1 && !s.near_hangover)
      s.ns_protection_hold=0;
    const bool heavy_ambient_noise=c.continuous_vehicle_policy && adaptive_decision.noise_floor>=1.5e-3f;
    const float dry_cap=heavy_ambient_noise ? .20f : 1.f;
    const float protection=s.ns_protection_hold ? dry_cap :
        (c.continuous_vehicle_policy && fullband_near ? 0.f :
         s.adaptive_controller ? std::min(dry_cap,.65f*adaptive_decision.near_confidence) : 0.f);
    s.ns_protection_mix+=std::clamp(protection-s.ns_protection_mix,-.05f,.05f);
    for(unsigned i=0;i<h;++i) s.mono[i]=(1-s.ns_protection_mix)*s.mono[i]+s.ns_protection_mix*s.ns_dry_output[i];
    for(float value:s.mono) s.observed_peak=std::max(s.observed_peak,std::abs(value));
    if((c.adaptive_noise_policy || c.continuous_vehicle_policy) && s.adaptive_controller) {
      s.noise_speech_mix=adaptive_decision.preserve_mix;
      info.noise_speech_mix=s.noise_speech_mix;
      const bool highrate_reference_hold=c.sample_rate>=48000 && s.reference_state==2;
      const bool midrate_speech_policy=c.sample_rate==24000 || c.policy_sample_rate==24000;
      const bool protected_double_talk=post_far_active && s.near_hangover;
      const float speech_gain=joint_reference_unknown ? 1.f : c.continuous_vehicle_policy ?
          (protected_double_talk ? (highrate_reference_hold ? 1.52f : midrate_speech_policy ? 1.32f : 1.28f) : 1.f) :
          (adaptive_decision.near_confidence>.55f ? 1.258925f : 1.f);
      const bool clean_speech_hold=c.continuous_vehicle_policy && fullband_near &&
          (s.fullband_speech_hold ||
           (s.fullband_quiet_frames<20 && s.fullband_noise_power>=1e-6) ||
           (control>.20f && gain_qualification_power>2.2*std::max(s.fullband_noise_power,1e-12)));
      const float target_floor=(has_aec && post_far_active && !s.near_hangover && joint_far_suppression_allowed) ? .24f :
          joint_reference_unknown ? 1.f :
          c.repair_stage>=4 && fullband_near ? 1.f : clean_speech_hold ? 1.f : .42f;
      // A stale echo-only floor must not survive a changed/unknown path.
      if(joint_reference_unknown) s.floor_gain_smoothed=1.f;
      else s.floor_gain_smoothed+=std::clamp(target_floor-s.floor_gain_smoothed,-.04f,clean_speech_hold ? .25f : .02f);
      const float floor_gain=c.continuous_vehicle_policy ? s.floor_gain_smoothed : target_floor;
      const float eff_mix=(c.continuous_vehicle_policy && fullband_near && (high_snr_dry || clean_speech_hold)) ? 1.f : s.noise_speech_mix;
      for(unsigned i=0;i<h;++i)
        s.mono[i]=(eff_mix*speech_gain+(1-eff_mix)*floor_gain)*s.mono[i];
    } else if(c.adaptive_noise_policy) {
      const bool far_without_near=has_aec && post_far_active && !s.near_hangover && !s.ns_protection_hold && joint_far_suppression_allowed;
      const bool speech=s.frames>30 && (control>.5f ||
          (has_aec && post_far_active && s.near_hangover)) && !far_without_near;
      if(speech) s.noise_speech_hold=200;
      else if(far_without_near) s.noise_speech_hold=0;
      else if(s.noise_speech_hold) --s.noise_speech_hold;
      const float target=s.noise_speech_hold ? 1.f : 0.f;
      s.noise_speech_mix+=std::clamp(target-s.noise_speech_mix,-.05f,.05f);
      info.noise_speech_mix=s.noise_speech_mix;
      const float speech_gain=s.reference_state==2 || s.observed_peak>.65f ? 1.f : 1.258925f;
      for(unsigned i=0;i<h;++i) s.mono[i]=(s.noise_speech_mix*speech_gain+(1-s.noise_speech_mix)*.42f)*s.mono[i];
    }
  }
  if(fullband_near && !s.ns && !s.vehicle_ns) for(unsigned i=0;i<h;++i) s.mono[i]+=s.wind_lowband_gain*s.lowband_output[i];
  if(s.speex) {
    if(c.speex_speech_policy) s.speex->suppression(!post_far_active && control>.5f ? -6 : c.speex_suppression_db);
    s.speex->process(s.mono.data());
  }
#ifdef ITU_HAS_RNNOISE
  if(s.rn) {
    for(unsigned i=0;i<h;++i) s.vad_audio[i]=s.mono[i]*32768.f;
    rnnoise_process_frame(s.rn,s.mono.data(),s.vad_audio.data());
    for(float& v:s.mono) v/=32768.f;
    // The fullband restoration tap must follow the neural synthesis delay too.
    if(!s.lowband_bf_history.empty()) for(unsigned i=0;i<h;++i) {
      const float delayed=s.lowband_ns_history[s.lowband_ns_head];
      s.lowband_ns_history[s.lowband_ns_head]=s.lowband_output[i];
      s.lowband_output[i]=delayed; s.lowband_ns_head=(s.lowband_ns_head+1)%kNsDelaySamples;
    }
  }
#endif
  if(s.neural) {
    try {s.neural->process(s.mono.data());}
    catch(...) {s.failed_backend_error=info.backend_error=-1001; return Status::backend_error;}
    info.neural_inferences=s.neural->inferences();
    if(!s.lowband_bf_history.empty()) for(unsigned i=0;i<h;++i) {
      const float delayed=s.lowband_ns_history[s.lowband_ns_head];
      s.lowband_ns_history[s.lowband_ns_head]=s.lowband_output[i];
      s.lowband_output[i]=delayed; s.lowband_ns_head=(s.lowband_ns_head+1)%kNsDelaySamples;
    }
  }
  if(c.diagnostic_stages && !s.ns)
    std::copy_n(s.mono.data(),h,c.diagnostic_stages->nr_selected.data());
  if(!s.speex && !s.vehicle_ns && !s.lowband_bf_history.empty()) {
    const float eff_low_mix=(c.continuous_vehicle_policy && fullband_near && s.fullband_speech_hold) ? 1.f : s.noise_speech_mix;
    const float speech_gain=c.noise_suppression && (c.adaptive_noise_policy || c.continuous_vehicle_policy) ?
        eff_low_mix*(c.continuous_vehicle_policy || s.reference_state==2 || s.observed_peak>.65f ? 1.f : 1.258925f)+(1-eff_low_mix)*(c.continuous_vehicle_policy ? s.floor_gain_smoothed : .42f) : 1.f;
    for(unsigned i=0;i<h;++i) s.mono[i]+=s.wind_lowband_gain*(1-s.fullband_ns_mix)*s.lowband_restore_mix*speech_gain*s.lowband_output[i];
  }
  if(!s.transition_history.empty()) for(unsigned i=0;i<h;++i)
    s.mono[i]=(1-info.startup_dry_mix)*s.mono[i]+info.startup_dry_mix*s.transition_dry[i];
  if(c.repair_stage>=2 && c.repair_stage<4 && s.ns && fullband_near) {
    s.ns_residual_gate.observe(c.external_vad ? control : std::min(1.f,2.f*control),
        !post_far_active && !s.near_hangover,
        s.ns_residual_profile && s.ns_residual_profile->white());
    for(unsigned i=0;i<h;++i) s.mono[i]*=s.ns_residual_gate.sample(i,h);
    info.ns_residual_gain=s.ns_residual_gate.gain();
    info.ns_residual_quiet_frames=s.ns_residual_gate.quiet_frames();
    info.ns_residual_flatness=s.ns_residual_profile ? s.ns_residual_profile->flatness() : 0.f;
    info.ns_residual_observations=s.ns_residual_profile ? s.ns_residual_profile->observations() : 0;
  }
  if(s.repeated_tone) {
    s.repeated_tone->process(s.mono.data(),h);
    info.tone_frequency_hz=s.repeated_tone->frequency();
    info.tone_mix=s.repeated_tone->mix();
    info.tone_concentration=s.repeated_tone->concentration();
    info.tone_repeats=s.repeated_tone->repeats();
  }
  if(c.min_stat_ns) {
    double p=0; for(unsigned i=0;i<h;++i) p+=double(s.mono[i])*s.mono[i];
    const float g=s.min_stat_ns_gate.observe(float(p/std::max(h,1u)),control);
    if(g<1.f) for(unsigned i=0;i<h;++i) s.mono[i]*=g;
    info.min_stat_gain=g;
  }
  info.post_ns_rms=rms(s.mono.data(),h);
  if(c.diagnostic_stages) std::copy_n(s.mono.data(),h,c.diagnostic_stages->nr_protected.data());
  if(c.eq_before_agc) {
    if(c.diagnostic_stages) std::copy_n(s.mono.data(),h,c.diagnostic_stages->pre_eq.data());
    if(s.unified_eq_agc) {
      info.unified_eq_weight=s.unified_eq_agc->process_eq(
          s.mono.data(),h,control,info.unified_ns_probability,
          info.linear_speech_probability,info.nonlinear_speech_probability,
          post_far_active,adaptive_decision.near_confidence);
    } else if(s.output_eq) s.output_eq->process(s.mono.data(),h);
    if(c.diagnostic_stages) std::copy_n(s.mono.data(),h,c.diagnostic_stages->post_eq.data());
  }
  if(s.post) {
    if(c.speech_gain_policy && !c.adaptive_gain && c.continuous_vehicle_policy) {
      const float controller_gain_limit=s.adaptive_controller ?
          std::min(c.max_gain_db,adaptive_decision.gain_limit_db) : c.max_gain_db;
      const float calibration_level=c.ecnr_v2_fullband_gain_calibration && fullband_near ?
          std::sqrt(gain_qualification_power) : pre_ns_level;
      const float conservative=s.speech_gain.observe(calibration_level,control,post_far_active,
          c.ecnr_coordinated_control ? std::min(3.f,controller_gain_limit) : controller_gain_limit,c.max_gain_change_db_per_second,
          c.ecnr_coordinated_control && fullband_near && s.fullband_noise_frames>=10 ? s.fullband_noise_power : -1,
          c.ecnr_coordinated_control && fullband_near ? gain_qualification_power : -1);
      const float utterance_target=c.repair_stage>=4 ? s.utterance_gain.observe(info.microphone_rms,
          control,post_far_active,controller_gain_limit,
          fullband_near && s.fullband_noise_frames>=10 ? s.fullband_noise_power : -1,conservative,
          c.max_gain_change_db_per_second,c.stage4_noise_release) : c.repair_stage>=1 ?
          s.legacy_utterance_gain.observe(std::sqrt(gain_qualification_power),control,
          post_far_active,controller_gain_limit,
          fullband_near && s.fullband_noise_frames>=10 ? s.fullband_noise_power : -1,conservative,
          c.low_level_projection_gain,info.microphone_rms) : conservative;
      const bool utterance_qualified=c.repair_stage>=4 ? s.utterance_gain.qualified() :
          c.repair_stage>=1 ? s.legacy_utterance_gain.qualified() : s.speech_gain.qualified();
      float target=utterance_target;
      if(s.unified_eq_agc) {
        target=s.unified_eq_agc->fuse_gain(
            conservative,utterance_target,controller_gain_limit,control,
            info.unified_ns_probability,info.linear_speech_probability,
            info.nonlinear_speech_probability,adaptive_decision.near_confidence,
            post_far_active,s.speech_gain.qualified(),utterance_qualified);
        info.unified_agc_fused_weight=s.unified_eq_agc->agc_weight();
        info.unified_agc_gain_db=target;
      }
      info.gain_control_noise_power=c.repair_stage>=4 ? s.utterance_gain.noise() :
          c.repair_stage>=1 ? s.legacy_utterance_gain.noise() : s.speech_gain.noise_power();
      info.gain_speech_qualified=c.repair_stage>=4 ? s.utterance_gain.qualified() :
          c.repair_stage>=1 ? s.legacy_utterance_gain.qualified() : s.speech_gain.qualified();
      info.gain_control_state=c.repair_stage>=4 ? s.utterance_gain.state() :
          c.repair_stage>=1 ? s.legacy_utterance_gain.state() : 0;
      // Frame-level gain attribution: what the controller saw, what it decided.
      info.gain_gate_level=std::sqrt(static_cast<float>(gain_qualification_power));
      info.gain_limit_db=controller_gain_limit;
      info.gain_input_level=info.microphone_rms;
      if(c.repair_stage>=4) {
        info.gain_votes=s.utterance_gain.votes();
        info.gain_target_db=s.utterance_gain.target();
        info.gain_hold_db=0.f;
      } else if(c.repair_stage>=1) {
        info.gain_votes=s.legacy_utterance_gain.votes();
        info.gain_voiced_rms=s.legacy_utterance_gain.voiced_rms();
        info.gain_peak=s.legacy_utterance_gain.peak();
        info.gain_max_voiced=s.legacy_utterance_gain.max_voiced_level();
        info.gain_target_db=s.legacy_utterance_gain.target();
        info.gain_hold_db=s.legacy_utterance_gain.hold_gain();
      }
      const float step=c.max_gain_change_db_per_second/100.f;
      const float commanded=context ? s.applied_gain_db+std::clamp(target-s.applied_gain_db,-step,step) : target;
      float applied=commanded;
      if(c.gain_fault==GainFault::previous_frame) applied=s.previous_commanded_gain_db;
      else if(c.gain_fault==GainFault::inverted) applied=-commanded;
      else if(c.gain_fault==GainFault::always_boost) applied=controller_gain_limit;
      s.previous_commanded_gain_db=commanded;
      if(applied!=s.applied_gain_db) {
        s.post->SetRuntimeSetting(webrtc::AudioProcessing::RuntimeSetting::CreateCaptureFixedPostGain(applied));
        s.applied_gain_db=applied;
      }
    } else if(c.speech_gain_policy && !c.adaptive_gain) {
      if(!s.gain_calibrated && !post_far_active && control>.5f) {
        s.speech_calibration_levels[s.speech_calibration_frames]=pre_ns_level;
        if(++s.speech_calibration_frames>=30) {
          s.gain_calibrated=true;
          std::sort(s.speech_calibration_levels.begin(),s.speech_calibration_levels.end());
          const double dbfs=20*std::log10(std::max(double(s.speech_calibration_levels[15]),1e-10));
          s.calibrated_gain_db=dbfs < -40 ? std::min(8.f,c.max_gain_db) : 0.f;
        }
      }
      const float target=post_far_active ? 0.f : s.calibrated_gain_db;
      if(target!=s.applied_gain_db) {
        s.post->SetRuntimeSetting(webrtc::AudioProcessing::RuntimeSetting::CreateCaptureFixedPostGain(target));
        s.applied_gain_db=target;
      }
    }
    info.applied_gain_db=s.applied_gain_db;
    const float* input[]={s.mono.data()}; float* output[]={out};
    int error=s.post->ProcessStream(input,webrtc::StreamConfig(c.sample_rate,1),webrtc::StreamConfig(c.sample_rate,1),output);
    if(error) {s.failed_backend_error=info.backend_error=error; return Status::backend_error;}
    if(c.repair_stage>=4 && c.continuous_vehicle_policy && fullband_near && s.ns) {
      // Noise gain is controlled separately from the locked speech gain. A
      // leaky evidence counter tolerates isolated spectral spikes; strong
      // signal innovation immediately releases this guard. No PCM is buffered.
      const double physical_power=double(info.microphone_rms)*info.microphone_rms;
      const double ambient=std::max(s.fullband_noise_power,1e-14);
      const bool calibrated=s.fullband_noise_frames>=10 && ambient<6e-5 && ambient>1e-12;
      const bool quiet=calibrated && physical_power<2.5*ambient &&
          (control<.1f || physical_power<1.5*ambient);
      if(quiet) s.gain_noise_evidence=std::min(100u,s.gain_noise_evidence+1);
      else s.gain_noise_evidence=s.gain_noise_evidence>3 ? s.gain_noise_evidence-3 : 0;
      if(!calibrated || physical_power>8*ambient) s.gain_noise_evidence=0;
      // First undo the extra speech gain; only sustained ambient evidence
      // applies additional attenuation. A high confidence value from a tonal
      // noise observer cannot veto physical agreement with the ambient floor.
      const float target=s.gain_noise_evidence>=5 ?
          (s.gain_noise_evidence>=40 ? .42f : 1.f)*std::pow(10.f,-s.applied_gain_db/20.f) : 1.f;
      const float previous=s.gain_noise_compensation;
      s.gain_noise_compensation+=std::clamp(target-previous,-.1f,.5f);
      for(unsigned i=0;i<h;++i) out[i]*=previous+(s.gain_noise_compensation-previous)*float(i+1)/h;
      info.gain_noise_evidence=s.gain_noise_evidence;
      info.gain_noise_compensation=s.gain_noise_compensation;
    }
  } else std::copy(s.mono.begin(),s.mono.end(),out);
  if(c.diagnostic_stages) {
    std::copy_n(out,h,c.diagnostic_stages->post_agc.data());
    if(!c.eq_before_agc) std::copy_n(out,h,c.diagnostic_stages->pre_eq.data());
  }
  if(!c.eq_before_agc) {
    if(s.unified_eq_agc) {
      info.unified_eq_weight=s.unified_eq_agc->process_eq(
          out,h,control,info.unified_ns_probability,
          info.linear_speech_probability,info.nonlinear_speech_probability,
          post_far_active,adaptive_decision.near_confidence);
    } else if(s.output_eq) s.output_eq->process(out,h);
  }
  if(c.tx_drc.enabled && s.tx_drc && s.tx_drc->delay_samples() != 0) {
    s.tx_drc->process(out,h);
  } else if(c.tx_drc.enabled && s.tx_drc) {
    s.tx_drc->process(out,h);
  }
  if(c.diagnostic_stages && !c.eq_before_agc) std::copy_n(out,h,c.diagnostic_stages->post_eq.data());
  for(unsigned i=0;i<h;++i) if(c.limiter && std::abs(out[i])>c.output_ceiling) {
    out[i]=std::copysign(c.output_ceiling,out[i]); ++info.limited_samples;
  }
  // Remove only sub-quantization tails when the complete frame is quiet. This
  // keeps speech and ordinary noise untouched while preventing resampler
  // boundary ringing from escaping a true silence window.
  const bool quiet_boundary=pre_ns_level<1.5e-4f && !render_active &&
      s.near_hangover==0 && s.input_excess_control<.05f &&
      info.speech_probability<.05f && info.linear_speech_probability<.05f &&
      info.nonlinear_speech_probability<.05f;
  // VehicleNsProcessor already owns the delayed wet/dry timeline. Current
  // input silence cannot qualify its still-valid delayed output as a tail.
  if(quiet_boundary && !s.vehicle_ns) {
    float quiet_peak=0;
    for(unsigned i=0;i<h;++i) quiet_peak=std::max(quiet_peak,std::abs(out[i]));
    if(quiet_peak<5e-4f) std::fill_n(out,h,0.f);
    else for(unsigned i=0;i<h;++i) if(std::abs(out[i])<1.5e-4f) out[i]=0.f;
  }
  info.output_rms=rms(out,h);
  float max_delay=0;
  if(c.beamforming) for(unsigned ch=0;ch<m;++ch) max_delay=std::max(max_delay,c.beam_delays_samples[ch]);
  // AEC3: 64-sample framer + 64-sample nonlinear synthesis overlap.
  // The latency test measures both taps and the serial output in the pinned build.
  info.aec_input_delay_samples=c.aec ? info.low_cut_delay_samples : 0;
  info.aec_output_delay_samples=c.aec ? kAecOutputDelaySamples+info.aec_input_delay_samples : 0;
  info.linear_aec_delay_samples=c.aec ? (kLinearAecDelaySamples+info.aec_input_delay_samples)*16000/c.sample_rate : 0;
  // Report the backend delay on the selected output path. The causal low-cut
  // contributes zero fixed samples; hybrid/Speex NS keeps its frame contract.
  info.ns_delay_samples=c.noise_suppression ? (s.speex ? h : kNsDelaySamples) : 0;
  info.tx_drc_delay_samples=(c.tx_drc.enabled && s.tx_drc) ?
      static_cast<unsigned>(s.tx_drc->delay_samples()) : 0;
  const unsigned eq_delay=s.unified_eq_agc ? s.unified_eq_agc->delay() :
      (s.output_eq ? s.output_eq->delay() : 0);
  info.latency_samples=info.aec_output_delay_samples+info.ns_delay_samples+
      info.spatial_delay_samples+(c.aec ? 0 : info.low_cut_delay_samples)+
      info.tx_drc_delay_samples+eq_delay+
      static_cast<unsigned>(std::ceil(max_delay));
  info.bf_delay_samples=info.latency_samples-info.ns_delay_samples;
  if(c.diagnostic_stages) {
    auto& taps=*c.diagnostic_stages;
    taps.core_sample_rate=taps.eq_sample_rate=c.sample_rate;
    taps.core_frame_size=taps.eq_frame_size=h;
    taps.core_delay_samples=taps.eq_delay_samples=info.latency_samples;
    taps.nr_selected_available=c.noise_suppression;
  }
  ++s.frames;
  return Status::ok;
}

Status AudioEngine::process_downlink(const float* rx_input, std::size_t rx_samples,
                                     float* rx_speaker_output, float* aec_reference_output) noexcept {
  if(!impl_ || !rx_input || !rx_speaker_output) return Status::invalid_pointer;
  auto& s=*impl_; const auto& c=s.c;
  if(rx_samples!=c.frame_size*c.reference_channels) return Status::invalid_length;
  for(std::size_t i=0;i<rx_samples;++i)
    if(!std::isfinite(rx_input[i]) || std::abs(rx_input[i])>1.f) return Status::non_finite;
  if(s.downlink_pipeline.empty()) {
    std::copy_n(rx_input,rx_samples,rx_speaker_output);
    if(aec_reference_output) std::copy_n(rx_input,rx_samples,aec_reference_output);
    return Status::ok;
  }
  for(unsigned ch=0;ch<c.reference_channels;++ch) {
    for(unsigned i=0;i<c.frame_size;++i) s.downlink_input[i]=rx_input[i*c.reference_channels+ch];
    s.downlink_pipeline[ch]->process(s.downlink_input.data(),s.downlink_output.data(),nullptr,c.frame_size);
    for(unsigned i=0;i<c.frame_size;++i) {
      rx_speaker_output[i*c.reference_channels+ch]=s.downlink_output[i];
      if(aec_reference_output) aec_reference_output[i*c.reference_channels+ch]=s.downlink_output[i];
    }
  }
  return Status::ok;
}
}
