#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace itu {
constexpr unsigned kMaxMicrophones = 8;
inline constexpr char kWebRtcRevision[] = "52022620d7f1f97eed6bcfbfcb2744fb26acc86f";
constexpr unsigned kAecOutputDelaySamples=128, kLinearAecDelaySamples=64, kNsDelaySamples=96;
// Minimum-delay IIR low-cut is causal and does not contribute a fixed sample
// offset to stream alignment. Keep this constant for ABI/source compatibility.
constexpr unsigned kLowCutDelaySamples=0;
enum class NoiseBackend { webrtc, rnnoise, speex, ulunas };
enum class AecBackend { webrtc, speex };
// The caller selects the operating envelope. Runtime adaptation is bounded by
// this mode; it never infers a product mode from PCM alone.
enum class ProcessingMode : unsigned {
  general = 0,
  referenced_radio = 1,
  vehicle_stream = 2,
  call = 3,
  ecnr = 4,
};
enum class EqKind { peak, low_shelf, high_shelf };
struct EqBand {
  EqKind kind = EqKind::peak;
  float frequency_hz = 1000.f;
  float gain_db = 0.f;
  float q = .70710678f; // Peak Q; shelves use slope 1.
};
struct EqConfig {
  unsigned bands = 0;
  std::array<EqBand,8> band{};
  float preamp_db = 0.f; // Negative headroom must cover the sum of positive band gains.
};
struct DrcConfig {
  bool enabled = false;
  float threshold_dbfs = -12.0f; // Compression threshold in dBFS [-40, 0]
  float ratio = 4.0f;            // Compression ratio [1, 20]
  float knee_width_db = 6.0f;    // Soft knee width in dB [0, 12]
  float attack_ms = 5.0f;        // Attack time in ms [0.5, 50]
  float release_ms = 60.0f;      // Release time in ms [10, 500]
  float makeup_gain_db = 0.0f;   // Post-compression makeup gain in dB [-12, 18]
  float ceiling_dbfs = -0.5f;    // Peak limiter ceiling in dBFS [-6, 0]
};
struct DownlinkConfig {
  bool enabled = false;
  bool enable_nr = true;
  float nr_max_suppression_db = 9.0f; // Mild NR (6-12 dB) preserving far-end naturalness
  EqConfig eq{};                       // Vehicle loudspeaker acoustic response compensation
  DrcConfig drc{
      .enabled = true,
      .threshold_dbfs = -10.0f,
      .ratio = 4.0f,
      .knee_width_db = 6.0f,
      .attack_ms = 4.0f,
      .release_ms = 50.0f,
      .makeup_gain_db = 2.0f,
      .ceiling_dbfs = -0.5f
  };
};
// Caller-owned fixed buffers: the callback performs only bounded copies.
// NativePipeline preserves the core-rate NR/AGC taps and replaces EQ taps
// with the actual I/O-rate stage, after its output resampler.
struct DiagnosticStageTaps {
  std::array<float,480> nr_selected{},nr_protected{},post_agc{},pre_eq{},post_eq{};
  unsigned core_sample_rate=0,core_frame_size=0,core_delay_samples=0;
  unsigned eq_sample_rate=0,eq_frame_size=0,eq_delay_samples=0;
  bool nr_selected_available=false;
};
struct Config {
  ProcessingMode mode = ProcessingMode::general;
  unsigned sample_rate = 16000;
  unsigned frame_size = 160;
  unsigned microphones = 1;
  unsigned reference_channels = 1; // Independent synchronous render PCM, interleaved.
  bool ns_statistical_control = false; // Opt-in NS posterior control, no RNN VAD.
  bool external_vad = false; // Default off; backend-internal estimation is independent.
  bool repeated_tone_control = false; // Input-only repeated burst notch, opt-in.
  unsigned repair_stage = 0; // 1 AGC, 2 NS, 3 AEC; independent of the frozen policy.
  bool ecnr_coordinated_control = false; // Opt-in shared, matched-stage noise qualification for vehicle AGC.
  bool ecnr_v2_control = false; // Opt-in selected v2 policy; retains v1 shared noise qualification.
  bool ecnr_v3_g1_control = false; // Opt-in confidence-qualified near hold / unified echo clear; requires selected v2.
  bool ecnr_v2_double_evidence = false; // Diagnostic single factor: trusted-render DTD evidence.
  bool ecnr_v2_fullband_gain_calibration = false; // Diagnostic single factor: use qualification-domain level.
  float ecnr_v2_nonlinear_vad_min = .2f;
  float ecnr_v2_coherence_margin = 1.35f;
  DiagnosticStageTaps* diagnostic_stages = nullptr; // Optional bounded PCM copies; no file I/O.
  bool noise_transition_control = false; // Input-only cold-start / noise-step protection.
  bool spatial_mvdr = false; // Opt-in calibrated, noise-covariance MVDR; requires BF.
  bool spatial_fixed_steering = false; // Experimental MVDR anchor after geometric delay compensation.
  bool spatial_continuous_covariance = false; // Experimental fixed-steering MPDR; train from causal mixture even during speech.
  bool wind_noise_control = false; // Low-frequency (<600 Hz) spatial weighting.
  NoiseBackend noise_backend = NoiseBackend::webrtc;
  std::string neural_model_path; // Optional ULUNAS stream_simple model; loaded on control thread.
  bool native_fullband = false; // Compatibility option; all rates use the shared protection policy.
  int speex_suppression_db = -12;
  bool speex_speech_policy = false; // Diagnostic -6 dB floor on observed near speech, configured floor otherwise.
  bool aec = true;
  AecBackend aec_backend = AecBackend::webrtc;
  unsigned speex_echo_tail_ms = 200;
  bool speex_residual_suppression = false; // Opt-in MDF-linked RES, separate from downstream NS.
  int speex_active_echo_suppression_db = -3;
  bool beamforming = false;
  bool noise_suppression = true;
  bool automatic_gain = true;
  bool limiter = true;
  bool linear_phase_low_cut = true; // Compatibility flag for causal 2nd-order IIR HPF conditioning.
  float low_cut_hz = 60.f;
  bool adaptive_gain = false;
  bool speech_gain_policy = true; // Vehicle: noise-qualified utterance lock/release; general: fixed calibration.
  bool continuous_vehicle_policy = false;
  bool transparent_nearend = true;
  bool reference_guard = true;
  unsigned aec_filter_blocks = 20;
  unsigned aec_delay_headroom_samples = 32;
  bool fast_aec_adaptation = true;
  bool external_delay_estimator = false;
  bool correlation_delay = true;
  bool stable_echo_coherence = false;
  bool very_transparent_nearend = false;
  bool multirender_nearend_policy = false; // Diagnostic AEC3 near-end tuning; multi-render only.
  bool multirender_speech_protection = false; // Short evidence-qualified linear residual restoration.
  float multirender_innovation_threshold = .2f;
  bool profile_stages = false; // Opt-in host timing; no clocks read in the normal callback.
  bool linear_speech_protection = true;
  bool raw_nearend_protection = false; // Restore low-cut content during unverified-reference fallback.
  bool speech_preserving_low_cut = false; // Condition VAD/AEC separately from unreferenced speech.
  bool hybrid_speech_ns = false; // Delay-aligned Speex speech branch with WebRTC noise branch.
  int hybrid_speech_suppression_db = -6; // Hybrid branch floor, -40..0 dB.
  float high_snr_dry_mix = 0.f; // Opt-in dry blend during startup or observed high-SNR near speech.
  bool fullband_near_ns = false; // Diagnostic: bypass only fullband/highpass mixing, not NS.
  float fullband_min_highband_fraction = .002f;
  bool linear_output_diagnostic = false;
  bool analyze_linear_noise = true;
  bool adaptive_noise_policy = true;
  bool adaptive_controller = false; // Context-bounded online controller.
  bool projection_echo_safeguard = true;
  bool coherence_dtd = true;
  unsigned noise_level = 0; // WebRTC low/moderate/high/very-high: 0..3.
  unsigned stream_delay_ms = 0; // Render/capture device buffering, not an input delay line.
  float activity_floor = 1e-7f;
  std::array<float, kMaxMicrophones> beam_delays_samples{};
  float agc_headroom_db = 5.f;
  float max_gain_db = 18.f;
  float initial_gain_db = 0.f;
  float max_gain_change_db_per_second = 6.f;
  float max_output_noise_level_dbfs = -50.f;
  float fixed_gain_db = 0.f;
  float output_ceiling = 0.98f;
  EqConfig output_eq{}; // Opt-in, I/O-rate uplink EQ after AGC, before final limiter.
  DrcConfig tx_drc{};
  bool eq_before_agc = false; // Barrot VCP topology: BF -> AEC -> NR -> EQ -> AGC/DRC
  bool rx_processing = false; // Barrot VCP Downlink processing: NR -> EQ -> DRC -> Speaker & AEC Reference
  DownlinkConfig rx_config{};
  // NativePipeline keeps this outer I/O rate when its core runs at 48 kHz.
  // Zero means the configuration is already at its processing rate.
  unsigned policy_sample_rate = 0;
};
Config barrot_vcp_config();
Config referenced_radio_config();
Config vehicle_stream_config();
Config call_config();
Config ecnr_config();
enum class Status : int {
  ok = 0, invalid_pointer = -1, invalid_length = -2, non_finite = -3,
  backend_error = -4
};
enum StatisticsValidity : unsigned { erl_valid = 1, erle_valid = 2, delay_valid = 4 };
struct FrameInfo {
  std::uint64_t frame_index = 0;
  bool far_active = false;
  bool linear_aec_available = false;
  unsigned statistics_valid = 0; // Upstream field presence only, NOT estimator convergence.
  float microphone_rms = 0;
  float aec_rms = 0; // AEC3 output after residual suppression and shared speech protection.
  float output_rms = 0;
  float speech_probability = 0;
  float applied_gain_db = 0;
  float pre_ns_rms = 0, post_ns_rms = 0; // Core-rate samples, before/after NR and protection, before AGC.
  float gain_control_noise_power = 0; // Same fullband input power used for gain qualification.
  bool gain_speech_qualified = false;
  unsigned gain_control_state = 0; // repair: observe/onset/locked/quiet/conservative/far.
  float input_excess_score=0.f,input_excess_fraction=0.f;
  bool external_vad_active=false;
  unsigned external_vad_calls=0;
  float tone_frequency_hz=0.f,tone_mix=0.f,tone_concentration=0.f;
  unsigned tone_repeats=0;
  float ns_residual_gain = 1.f, ns_residual_flatness = 0;
  unsigned ns_residual_quiet_frames = 0, ns_residual_observations = 0;
  unsigned noise_profile_observations = 0;
  float noise_profile_mix = 0;
  unsigned reference_state = 0; // 0 observing/no render, 1 reduction observed, 2 unverified reference fallback.
  float reference_residual_ratio = 1;
  float reference_confidence = 0;
  float linear_speech_probability = 0;
  float nearend_mix = 0;
  bool v2_double_evidence_eligible = false;
  unsigned v2_near_confirmation = 0;
  unsigned g1_strict_near_age = 101, g1_startup_echo_frames = 0, g1_startup_clear_threshold = 3;
  bool g1_startup_clear_deferred = false;
  unsigned near_protection_hold_frames = 0;
  unsigned startup_echo_confirmation = 0; // Original consecutive confirmation, capped at three.
  unsigned g1_echo_evidence_frames = 0, g1_echo_clear_threshold = 0; // 0 threshold means original fallback.
  bool g1_echo_clear_deferred = false, g1_strong_near_confirmation = false;
  bool g1_near_latch_active = false, g1_original_confirmed_echo = false, g1_original_coherent_clear = false;
  float native_capture_ms = 0, native_render_ms = 0, native_core_ms = 0;
  float native_output_ms = 0, native_taps_ms = 0;
  float linear_residual_ratio = 1;
  float nonlinear_speech_probability = 0;
  float echo_subtraction_gain = 1;
  float noise_speech_mix = 0;
  float adaptive_noise_floor = 0;
  float adaptive_near_confidence = 0;
  float adaptive_far_confidence = 0;
  float adaptive_preserve_mix = 0;
  float adaptive_gain_limit_db = 0;
  unsigned adaptive_state = 0;
  float noise_flatness = 0;
  float incoherent_ratio = 0;
  float echo_coherence_baseline = 1;
  float internal_delay_mix = 0;
  float lowband_restore_mix = 0;
  float fullband_ns_mix = 0;
  float fullband_noise_power = 0;
  float highband_energy_fraction = 0;
  unsigned coherence_learning_frames = 0;
  float echo_return_loss_db = 0;
  float echo_return_loss_enhancement_db = 0;
  int estimated_delay_ms = 0;
  int backend_error = 0;
  unsigned limited_samples = 0;
  unsigned latency_samples = 0; // I/O-rate algorithm delay + maximum BF compensation.
  unsigned aec_output_delay_samples = 0; // I/O-rate AEC tap delay, including conversion when AEC is off.
  unsigned linear_aec_delay_samples = 0; // Always in 16 kHz samples.
  unsigned low_cut_delay_samples = 0;
  unsigned post_low_cut_delay_samples = 0; // Standalone NS low-cut after AEC/BF taps, I/O samples.
  unsigned aec_input_delay_samples = 0;
  unsigned ns_delay_samples = 0;
  unsigned linear_aec_sample_rate = 16000;
  unsigned spatial_delay_samples = 0;
  unsigned spatial_noise_updates = 0;
  unsigned spatial_covariance_updates = 0;
  unsigned spatial_fallback_bins = 0;
  float wind_score = 0;
  unsigned noise_control_state = 0; // 0 uncertain, 1 learned, 2 noise-step reacquisition.
  unsigned noise_transition_events = 0;
  float startup_dry_mix = 0;
  unsigned neural_inferences = 0;
};

// Construction/reset/destruction run on the control thread. APM uses internal
// mutexes. One instance per stream; never concurrently process/reset/destroy it.
class AudioEngine {
 public:
  explicit AudioEngine(const Config& config);
  ~AudioEngine();
  AudioEngine(AudioEngine&&) noexcept;
  AudioEngine& operator=(AudioEngine&&) noexcept;
  AudioEngine(const AudioEngine&) = delete;
  AudioEngine& operator=(const AudioEngine&) = delete;
  const Config& config() const noexcept;
  void reset();
  // Finite PCM in [-1,1]. Interleaved microphones and synchronous render reference.
  // Reference count is frame_size*reference_channels. Old create APIs remain mono.
  // AEC tap: frame_size*microphones, after selected AEC/RES and speech protection, before BF;
  // conversion-only capture when AEC is disabled.
  // Rates: 16/24/48 kHz, exactly 10 ms (160/240/480 frames per channel).
  // BF tap: frame_size. Linear tap: 160*microphones at 16 kHz, channel blocks,
  // before AEC3 nonlinear suppression (zero when AEC off; availability in FrameInfo).
  // Auxiliary taps must not alias inputs/output. Counts are float elements.
  Status process(const float* microphones, std::size_t microphone_samples,
                 const float* reference, std::size_t reference_samples,
                 float* output, std::size_t output_samples, FrameInfo& info,
                 float* aec_tap = nullptr, float* bf_tap = nullptr,
                 float* linear_aec_tap = nullptr, float* aec_input_tap = nullptr) noexcept;
  // Wrapper-owned buffers only; opaque WebRTC/Speex allocations are excluded.
  std::size_t working_memory_bytes() const noexcept;
  // Barrot VCP Downlink processing (Downlink In -> NR -> EQ -> DRC -> Speaker & AEC Reference)
  Status process_downlink(const float* rx_input, std::size_t rx_samples,
                          float* rx_speaker_output, float* aec_reference_output) noexcept;
 private:
  friend class NativePipeline;
  AudioEngine(const Config& config,unsigned maximum_beam_samples);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
