#ifndef ITU_AUDIO_ENGINE_C_H
#define ITU_AUDIO_ENGINE_C_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ITU_AUDIO_ABI_VERSION 2
#define ITU_AUDIO_MAX_MICROPHONES 8
#define ITU_AUDIO_OK 0
#define ITU_AUDIO_INVALID_POINTER -1
#define ITU_AUDIO_INVALID_LENGTH -2
#define ITU_AUDIO_INVALID_SAMPLE -3
#define ITU_AUDIO_BACKEND_ERROR -4
typedef struct itu_audio_engine itu_audio_engine;
enum itu_audio_profile {
  ITU_AUDIO_PROFILE_GENERAL = 0,
  ITU_AUDIO_PROFILE_REFERENCED_RADIO = 1,
  ITU_AUDIO_PROFILE_VEHICLE_STREAM = 2,
  ITU_AUDIO_PROFILE_VCP = 3,
  ITU_AUDIO_PROFILE_VCP_UNIFIED = 4,
  ITU_AUDIO_PROFILE_VCP_FUSED_AEC = 5
};
enum itu_audio_noise_backend { ITU_AUDIO_NS_WEBRTC=0, ITU_AUDIO_NS_RNNOISE=1, ITU_AUDIO_NS_SPEEX=2, ITU_AUDIO_NS_ULUNAS=3, ITU_AUDIO_NS_OMLSA_IMCRA=4, ITU_AUDIO_NS_SUPER_GAUSSIAN=5, ITU_AUDIO_NS_UNIFIED=6 };
enum itu_audio_eq_kind { ITU_AUDIO_EQ_PEAK=0, ITU_AUDIO_EQ_LOW_SHELF=1, ITU_AUDIO_EQ_HIGH_SHELF=2 };
typedef struct itu_audio_eq_band { int kind; float frequency_hz, gain_db, q; } itu_audio_eq_band;
typedef struct itu_audio_eq_config {
  uint32_t struct_size, bands;
  float preamp_db;
  itu_audio_eq_band band[8];
} itu_audio_eq_config;
typedef struct itu_audio_config {
  uint32_t struct_size, abi_version;
  uint32_t sample_rate, frame_size, microphones;
  int aec, beamforming, noise_suppression, automatic_gain, limiter;
  uint32_t noise_level, stream_delay_ms;
  float activity_floor;
  float beam_delays_samples[ITU_AUDIO_MAX_MICROPHONES];
  float agc_headroom_db, max_gain_db, initial_gain_db, max_gain_change_db_per_second;
  float max_output_noise_level_dbfs, fixed_gain_db, output_ceiling;
} itu_audio_config;
/* Additive extension: no change to ABI v2 configuration or frame layouts.
 * Multi-render uses AEC3 native DTD; mono projection/coherence policies are off.
 * PCM reference is interleaved frame_size*reference_channels with this API. */
typedef struct itu_audio_extensions {
  uint32_t struct_size, version, reference_channels;
  int noise_transition_control, spatial_mvdr, wind_noise_control;
  const char* neural_model_path; /* Copied at creation; optional ORT backend. */
} itu_audio_extensions;
void itu_audio_default_extensions(itu_audio_extensions* extensions);
itu_audio_engine* itu_audio_create_extended(const itu_audio_config* config,
  int profile, int noise_backend, const itu_audio_eq_config* eq,
  const itu_audio_extensions* extensions, char* error, size_t error_capacity);
typedef struct itu_audio_frame_info {
  uint64_t frame_index;
  int far_active, linear_aec_available;
  uint32_t statistics_valid; /* Field presence: ERL=1, ERLE=2, delay=4, NOT convergence.
                             * Upstream delay=0 may mean no estimate; confidence is unavailable. */
  float microphone_rms, aec_rms, output_rms;
  float echo_return_loss_db, echo_return_loss_enhancement_db;
  int estimated_delay_ms, backend_error;
  uint32_t limited_samples, latency_samples;
  uint32_t aec_output_delay_samples, linear_aec_delay_samples;
} itu_audio_frame_info;
/* Sets all fields including struct_size and ABI version; NULL is a no-op. */
void itu_audio_default_config(itu_audio_config* config);
/* Returns NULL on invalid config/allocation failure; optional error is NUL-terminated if capacity>0. */
itu_audio_engine* itu_audio_create(const itu_audio_config* config, char* error, size_t error_capacity);
/* Selects internal streaming policies, then applies all config fields. ABI v2
 * layout and the general-policy create entry point remain unchanged. */
itu_audio_engine* itu_audio_create_with_profile(const itu_audio_config* config,
                                              int profile, char* error, size_t error_capacity);
/* Additive entry point; ABI v2 structs unchanged. RNNoise must be built in. */
itu_audio_engine* itu_audio_create_with_backend(const itu_audio_config* config,
                                              int profile, int noise_backend,
                                              char* error, size_t error_capacity);
/* Additive opt-in EQ, no changes to ABI v2 config/frame structs. NULL EQ disables it.
 * Applied to uplink only, after AGC and before final limiter at the I/O sample rate. */
itu_audio_engine* itu_audio_create_with_tuning(const itu_audio_config* config,
                                            int profile, int noise_backend,
                                            const itu_audio_eq_config* eq,
                                            char* error, size_t error_capacity);
/* Returns the selected preset, or -1 for NULL. */
int itu_audio_processing_profile(const itu_audio_engine* engine);
void itu_audio_destroy(itu_audio_engine* engine);
int itu_audio_reset(itu_audio_engine* engine);
/* Reset failure preserves the prior engine. A backend-failed stream stays unusable
 * until successful reset; destroy/recreate if reset cannot succeed. */
/* Native-endian float32 PCM. Counts are float elements, not bytes. Exactly one fixed frame.
 * mic: frame_size*microphones interleaved, ref: frame_size*reference_channels interleaved
 * (mono for legacy create APIs; NULL/0 only if AEC off),
 * output: frame_size mono. Enabled BF precedes AEC. AEC/input/linear taps then
 * replicate the mono beam residual across the legacy microphone-sized buffers;
 * these are not independent microphone residuals. With BF off, taps remain per mic.
 * BF tap: frame_size, before AEC when BF is enabled.
 * Linear AEC tap: 160*microphones, deinterleaved channel blocks.
 * Reference must be actual playback PCM; this TX call never processes RX again.
 * All storage must remain valid during this synchronous call; pointers are not retained.
 * Auxiliary taps must not alias input/output. Output may alias input (mono only).
 * One processing thread; stop/join before reset/destroy. APM uses internal locks.
 * Input validation rejects before state mutation. Backend errors require reset.
 * Samples must be finite in [-1,1]. Rates 16/24/48kHz use 160/240/480 sample frames.
 * Linear tap remains 160*microphones at 16kHz; linear delay is in 16kHz samples.
 */
int itu_audio_process(itu_audio_engine* engine,
                     const float* microphones, size_t microphone_samples,
                     const float* reference, size_t reference_samples,
                     float* output, size_t output_samples,
                     itu_audio_frame_info* info, float* aec_tap, float* bf_tap,
                     float* linear_aec_tap);
/* Interleaved RX PCM, exactly frame_size*reference_channels float samples.
 * Process once per playback frame. Speaker output and AEC reference are identical;
 * pass the returned reference directly to itu_audio_process. NULL reference output
 * is allowed. Invalid length/sample validation does not advance RX state. */
int itu_audio_process_downlink(itu_audio_engine* engine,
                              const float* input, size_t samples,
                              float* speaker, float* reference);
/* Number of physical AEC capture channels, before legacy tap replication. */
unsigned itu_audio_aec_capture_channels(const itu_audio_engine* engine);
/* Wrapper buffers only, excluding opaque upstream APM allocation. */
size_t itu_audio_working_memory_bytes(const itu_audio_engine* engine);
const char* itu_audio_version(void);
#ifdef __cplusplus
}
#endif
#endif
