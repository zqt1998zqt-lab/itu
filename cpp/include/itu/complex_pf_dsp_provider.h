#ifndef ITU_COMPLEX_PF_DSP_PROVIDER_H
#define ITU_COMPLEX_PF_DSP_PROVIDER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Optional HiFi adapter, statically linked by ITU_COMPLEX_PF_DSP_LIBRARY.
 * Input is the PCM16-derived raw MDF error and the actual post-RX playback.
 * The adapter owns Complex_PF STFT/GRU/OLA state and 160<->212 adaptation.
 * probe/create run on the control thread. process must not allocate, block on
 * unbounded queues, throw, or retain caller buffers after returning. A host
 * simulator MUST identify itself as host_simulator, never as hifi_dsp. */
#define ITU_COMPLEX_PF_DSP_ABI 1u
typedef struct itu_complex_pf_dsp_info_v1 {
  uint32_t struct_size, version, available;
  uint32_t sample_rate, callback_samples, delay_samples;
  uint64_t working_memory_bytes;
  char execution_provider[32], architecture[32], device_id[64], model_id[64];
  char model_sha256[65], frontend_sha256[65], provider_sha256[65];
  char reason[192];
} itu_complex_pf_dsp_info_v1;
typedef struct itu_complex_pf_dsp_frame_v1 {
  uint32_t struct_size, version;
  uint64_t frame_index, inferences, fallback_frames, cycles;
} itu_complex_pf_dsp_frame_v1;
typedef struct itu_complex_pf_dsp_provider_v1 {
  uint32_t struct_size, version;
  int (*probe)(const char* model_id, itu_complex_pf_dsp_info_v1*, char*, size_t);
  void* (*create)(const char* model_id, uint32_t rate, uint32_t frame_samples,
                 itu_complex_pf_dsp_info_v1*, char*, size_t);
  int (*process)(void*, const float* residual, const float* farend,
                 float* output, uint32_t samples, itu_complex_pf_dsp_frame_v1*);
  void (*destroy)(void*);
} itu_complex_pf_dsp_provider_v1;

/* Supplied by the vendor adapter only. The default host build does not
 * define or load an implementation and therefore rejects DSP execution. */
const itu_complex_pf_dsp_provider_v1* itu_complex_pf_dsp_get_provider_v1(void);
#ifdef __cplusplus
}
#endif
#endif
