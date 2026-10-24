#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace itu {
enum class VehicleNsBackend { webrtc, speex, bypass, filterbank, omlsa_imcra, super_gaussian };
// Explicit operating envelope; all adaptation uses current/past capture only.
// No external VAD. 24 kHz WebRTC requires the explicitly selected 48 kHz bridge.
// OM-LSA and Super-Gaussian keep the I/O rate; strength 0/1/2/3 sets
// -9/-12/-18/-24 dB. Super-Gaussian is joint MAP H1 gain with shared IMCRA.
struct VehicleNsConfig {
  VehicleNsBackend backend=VehicleNsBackend::webrtc;
  int strength=1, mild_strength=0;
  float dry_max=1, snr_threshold_db=15, noise_gain=1;
  unsigned startup_ms=500;
  std::string policy="parallel_start";
  float slew_ms=50, release_ms=0, quiet_floor_db=-65;
  // Mode 4 adds a bounded weak-speech raw blend; mode 3 remains reproducible.
  unsigned floor_guard=1, repair_mode=3;
  float rumble_hz=0;
  bool linear_rumble=true, native_24=false;
  bool adaptive_rumble=false; // Continuous rumble evidence; fullband delay stays fixed.
  // Optional second Speex state; the primary state and its adaptation stay intact.
  int steady_suppression_db=-16;
  // Optional continuous cap on broadband dry mix, from statistical NS evidence.
  // Zero reproduces the original profiles; it never gates/holds output PCM.
  float spectral_dry_guard=0;
  // Opt-in correction of OM-LSA synthesis gains; the statistical estimator
  // remains independent. Native-rate model failures explicitly use OM-LSA.
  std::string student_model_path;
  float student_spp_mix=.5f,student_snr_mix=.5f,student_speech_floor_db=-3.f;
  // Causal post-backend steady residual attenuation. Zero preserves all
  // existing profiles; only noise-confident output frames are attenuated.
  float residual_max_db=0,residual_speech_guard=.75f;
  // Optional controller-evidence gate for the residual stage. Negative
  // threshold preserves the ungated experiment. When enabled, attenuation is
  // confined to a bounded low-SNR band with soft edges; no clean target,
  // reference label, future PCM or external VAD is consulted.
  float residual_gate_snr_db=-60,residual_gate_width_db=3;
  // Opt-in low-frequency recovery. It changes only the delayed FIR mix;
  // continuous mid-band speech evidence still protects clean low frequencies.
  float rumble_recovery=0;
  // Optional stricter low-frequency corner selected from causal low-band
  // evidence. Both FIR paths use the same fixed 20 ms output timebase.
  float rumble_adaptive_hz=0;
  // Bounded continuous weak-speech blend; no external VAD or PCM gating.
  float weak_speech_recovery=0;
  // Optional causal synthesis-gain floor in 300-1000 Hz. The value is the
  // maximum reduction of that floor (1=fully raw gain); statistical noise
  // estimation remains unchanged and zero preserves prior behavior bit-for-bit.
  float speech_band_recovery=0;
  // Persistent, spectrally distributed noise-step recovery; zero keeps the
  // original controller minima and IMCRA recursions bit-for-bit unchanged.
  float noise_step_recovery=0;
  // Optional causal transient-noise raw-mix guard. Repetition and spectral
  // evidence come only from capture PCM; zero preserves prior behavior.
  float transient_dry_guard=0;
  // Optional synthesis-only attenuation for repeated wind/indicator noise.
  // Wind affects 0-300 Hz; indicators affect 1-8 kHz. Per-bin speech
  // posterior gates the attenuation and the estimator state is untouched.
  float transient_spectral_guard_db=0;
  // Opt-in unified broadband protection. dry_max becomes the TOTAL raw budget:
  // dry, quiet and startup requests share one aligned raw path and smoother.
  // False preserves historical mixing and backend adaptation bit-for-bit.
  // Dynamic/steady_dynamic and WebRTC/Speex parallel/parallel_start use
  // processed donors, so they explicitly reject this raw-only mode. Use
  // aligned/fixedblend/legacy; statistical parallel_start remains supported.
  bool unified_raw_budget=false;
};
struct VehicleNsStatistics {
  bool unified_raw_budget=false;
  // Target is the final merged request; min/max are the actual per-sample
  // convex raw weights for the last frame, before the common gain<=1.
  // In historical mode the union of protection weights is reported with a
  // budget of 1 (parallel mild/steady paths may be processed rather than raw).
  float raw_mix_target=0,raw_mix_budget=1;
  float raw_mix_applied_min=0,raw_mix_applied_max=0;
  float confidence=0, snr_db=0, dry_mix_target=0, gain_target=1;
  float quiet_mix_target=0, noise_floor_db=-300, rumble_fraction=0;
  float rumble_mix_target=0, rumble_mix_applied=0;
  float rumble_corner_hz=0, rumble_low_ratio_db=-60;
  float steady_mix_target=0;
  float spectral_guard_applied=1;
  float suppression_db=0;
  unsigned suppression_updates=0;
  float omlsa_probability=0,omlsa_gain=1,omlsa_noise_power=0,omlsa_bootstrap_seed=1;
  std::size_t imcra_excluded_bins=0,imcra_noise_updates=0;
  unsigned imcra_minima_rotations=0;
  unsigned student_state=0; // TinyNsCorrector::State; 1=ready, 2..5=fallback reason.
  std::size_t student_inferences=0,student_fallback_frames=0;
  bool student_applied=false;
  float residual_speech_probability=0,residual_attenuation_db=0;
  float noise_step_probability=0;
  float transient_noise_evidence=0,transient_guard_applied=1;
  float transient_spectral_guard_applied=1;
  float transient_periodicity_100ms=0,transient_periodicity_500ms=0;
  std::size_t noise_step_updates=0;
  std::size_t speech_band_frames=0;
  float speech_band_floor=1,speech_band_gain_delta=0,speech_band_evidence=0;
  float speech_band_mid_probability=0,speech_band_anchor_probability=0,speech_band_anchor_gain=0;
};
// One instance per stream. Construct on the control thread; process() accepts
// exactly 10 ms in place and allocates no memory. Flush with zero input for
// delay_samples() beyond the source length and trim only that declared delay.
class VehicleNsProcessor {
 public:
  // Internal SRC wrappers can skip a second quantization after they have
  // applied the identical PCM16 entry conversion at the outer sample rate.
  VehicleNsProcessor(unsigned sample_rate, const VehicleNsConfig& config, bool quantize_input=true);
  ~VehicleNsProcessor();
  VehicleNsProcessor(const VehicleNsProcessor&)=delete;
  VehicleNsProcessor& operator=(const VehicleNsProcessor&)=delete;
  bool process(float* samples, unsigned count) noexcept;
  unsigned delay_samples() const noexcept;
  unsigned core_rate() const noexcept;
  unsigned core_delay_samples() const noexcept;
  unsigned rumble_delay_samples() const noexcept;
  double src_delay_samples() const noexcept;
  VehicleNsStatistics statistics() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
