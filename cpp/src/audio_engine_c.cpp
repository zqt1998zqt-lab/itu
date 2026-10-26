#include "itu/audio_engine_c.h"
#include "itu/audio_engine.hpp"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>

static_assert(itu::kMaxMicrophones == ITU_AUDIO_MAX_MICROPHONES);

struct itu_audio_engine {
  itu::AudioEngine core;
  int profile;
  itu_audio_engine(const itu::Config& c,int selected):core(c),profile(selected){}
};
namespace {
itu::Config convert(const itu_audio_config& c,int profile) {
  itu::Config r;
  switch(profile) {
    case ITU_AUDIO_PROFILE_GENERAL: break;
    case ITU_AUDIO_PROFILE_REFERENCED_RADIO: r=itu::referenced_radio_config(); break;
    case ITU_AUDIO_PROFILE_VEHICLE_STREAM: r=itu::vehicle_stream_config(); break;
    case ITU_AUDIO_PROFILE_VCP: r=itu::barrot_vcp_config(); break;
    case ITU_AUDIO_PROFILE_VCP_UNIFIED: r=itu::barrot_vcp_unified_config(); break;
    case ITU_AUDIO_PROFILE_VCP_FUSED_AEC: r=itu::barrot_vcp_fused_aec_config(); break;
    default: throw std::invalid_argument("unknown processing profile");
  }
  r.sample_rate=c.sample_rate; r.frame_size=c.frame_size; r.microphones=c.microphones;
  r.aec=c.aec; r.beamforming=c.beamforming;
  r.noise_suppression=c.noise_suppression; r.automatic_gain=c.automatic_gain; r.limiter=c.limiter;
  r.noise_level=c.noise_level; r.stream_delay_ms=c.stream_delay_ms; r.activity_floor=c.activity_floor;
  std::copy_n(c.beam_delays_samples,8,r.beam_delays_samples.begin());
  r.agc_headroom_db=c.agc_headroom_db; r.initial_gain_db=c.initial_gain_db;
  r.max_gain_change_db_per_second=c.max_gain_change_db_per_second;
  r.max_output_noise_level_dbfs=c.max_output_noise_level_dbfs; r.fixed_gain_db=c.fixed_gain_db;
  r.max_gain_db=c.max_gain_db; r.output_ceiling=c.output_ceiling; return r;
}
}
extern "C" {
void itu_audio_default_config(itu_audio_config* c) {
  if (!c) return;
  itu::Config d; *c={}; c->struct_size=sizeof(*c); c->abi_version=ITU_AUDIO_ABI_VERSION;
  c->sample_rate=d.sample_rate; c->frame_size=d.frame_size; c->microphones=d.microphones;
  c->aec=d.aec; c->beamforming=d.beamforming;
  c->noise_suppression=d.noise_suppression; c->automatic_gain=d.automatic_gain; c->limiter=d.limiter;
  c->noise_level=d.noise_level; c->stream_delay_ms=d.stream_delay_ms; c->activity_floor=d.activity_floor;
  c->agc_headroom_db=d.agc_headroom_db; c->initial_gain_db=d.initial_gain_db;
  c->max_gain_change_db_per_second=d.max_gain_change_db_per_second;
  c->max_output_noise_level_dbfs=d.max_output_noise_level_dbfs; c->fixed_gain_db=d.fixed_gain_db;
  c->max_gain_db=d.max_gain_db; c->output_ceiling=d.output_ceiling;
}
itu_audio_engine* itu_audio_create(const itu_audio_config* c, char* error, size_t capacity) {
  return itu_audio_create_with_profile(c,ITU_AUDIO_PROFILE_GENERAL,error,capacity);
}
itu_audio_engine* itu_audio_create_with_profile(const itu_audio_config* c,int profile,char* error,size_t capacity) {
  const int backend = profile==ITU_AUDIO_PROFILE_VCP_UNIFIED ?
      ITU_AUDIO_NS_UNIFIED : ITU_AUDIO_NS_WEBRTC;
  return itu_audio_create_with_backend(c,profile,backend,error,capacity);
}
itu_audio_engine* itu_audio_create_with_backend(const itu_audio_config* c,int profile,int backend,char* error,size_t capacity) {
  return itu_audio_create_with_tuning(c,profile,backend,nullptr,error,capacity);
}
itu_audio_engine* itu_audio_create_with_tuning(const itu_audio_config* c,int profile,int backend,
                                             const itu_audio_eq_config* eq,char* error,size_t capacity) {
  return itu_audio_create_extended(c,profile,backend,eq,nullptr,error,capacity);
}
void itu_audio_default_extensions(itu_audio_extensions* x) {
  if(x) {*x={}; x->struct_size=sizeof(*x); x->version=1; x->reference_channels=1;}
}
itu_audio_engine* itu_audio_create_extended(const itu_audio_config* c,int profile,int backend,
    const itu_audio_eq_config* eq,const itu_audio_extensions* x,char* error,size_t capacity) {
  if (error && capacity) error[0]=0;
  try {
    if (!c || c->struct_size!=sizeof(*c) || c->abi_version!=ITU_AUDIO_ABI_VERSION)
      throw std::invalid_argument("configuration ABI mismatch");
    for (int v : {c->aec,c->beamforming,c->noise_suppression,c->automatic_gain,c->limiter})
      if (v!=0 && v!=1) throw std::invalid_argument("boolean flag must be 0 or 1");
    if(backend<ITU_AUDIO_NS_WEBRTC || backend>ITU_AUDIO_NS_UNIFIED)
      throw std::invalid_argument("unknown noise backend");
    auto config=convert(*c,profile);
    if(x) {
      if(x->struct_size!=sizeof(*x) || x->version!=1) throw std::invalid_argument("extension ABI mismatch");
      for(int v:{x->noise_transition_control,x->spatial_mvdr,x->wind_noise_control})
        if(v!=0 && v!=1) throw std::invalid_argument("extension boolean must be 0 or 1");
      config.reference_channels=x->reference_channels; config.noise_transition_control=x->noise_transition_control;
      config.spatial_mvdr=x->spatial_mvdr; config.wind_noise_control=x->wind_noise_control;
      if(x->neural_model_path) config.neural_model_path=x->neural_model_path;
    }
    config.noise_backend=static_cast<itu::NoiseBackend>(backend);
    if(eq) {
      if(eq->struct_size!=sizeof(*eq) || eq->bands>8) throw std::invalid_argument("EQ configuration size or band count mismatch");
      config.output_eq.bands=eq->bands; config.output_eq.preamp_db=eq->preamp_db;
      for(unsigned i=0;i<eq->bands;++i) {
        if(eq->band[i].kind<0 || eq->band[i].kind>2) throw std::invalid_argument("unknown EQ band type");
        config.output_eq.band[i]={static_cast<itu::EqKind>(eq->band[i].kind),eq->band[i].frequency_hz,eq->band[i].gain_db,eq->band[i].q};
      }
    }
    return new itu_audio_engine(config,profile);
  } catch (const std::exception& e) { if (error && capacity) std::snprintf(error,capacity,"%s",e.what()); }
  catch (...) { if (error && capacity) std::snprintf(error,capacity,"creation failed"); }
  return nullptr;
}
int itu_audio_processing_profile(const itu_audio_engine* e) {return e ? e->profile : -1;}
void itu_audio_destroy(itu_audio_engine* e) { delete e; }
int itu_audio_reset(itu_audio_engine* e) {
  if(!e) return ITU_AUDIO_INVALID_POINTER;
  try {e->core.reset(); return ITU_AUDIO_OK;} catch(...) {return ITU_AUDIO_BACKEND_ERROR;}
}
int itu_audio_process(itu_audio_engine* e, const float* mic, size_t mn, const float* ref, size_t rn,
                      float* out, size_t on, itu_audio_frame_info* info, float* aec, float* bf, float* linear) {
  if (info) *info={};
  if (!e || !info) return ITU_AUDIO_INVALID_POINTER;
  itu::FrameInfo f; auto result=e->core.process(mic,mn,ref,rn,out,on,f,aec,bf,linear);
  info->frame_index=f.frame_index; info->far_active=f.far_active; info->linear_aec_available=f.linear_aec_available;
  info->statistics_valid=f.statistics_valid;
  info->microphone_rms=f.microphone_rms; info->aec_rms=f.aec_rms; info->output_rms=f.output_rms;
  info->echo_return_loss_db=f.echo_return_loss_db; info->echo_return_loss_enhancement_db=f.echo_return_loss_enhancement_db;
  info->estimated_delay_ms=f.estimated_delay_ms; info->backend_error=f.backend_error; info->limited_samples=f.limited_samples;
  info->latency_samples=f.latency_samples;
  info->aec_output_delay_samples=f.aec_output_delay_samples;
  info->linear_aec_delay_samples=f.linear_aec_delay_samples;
  return static_cast<int>(result);
}
size_t itu_audio_working_memory_bytes(const itu_audio_engine* e) { return e ? e->core.working_memory_bytes() : 0; }
int itu_audio_process_downlink(itu_audio_engine* e,const float* input,size_t samples,float* speaker,float* reference) {
  if(!e) return ITU_AUDIO_INVALID_POINTER;
  return static_cast<int>(e->core.process_downlink(input,samples,speaker,reference));
}
unsigned itu_audio_aec_capture_channels(const itu_audio_engine* e) {
  return e ? (e->core.config().beamforming ? 1u : e->core.config().microphones) : 0u;
}
const char* itu_audio_version(void) { return "itu-audio/0.2.0 abi/2 webrtc/52022620d7f1f97eed6bcfbfcb2744fb26acc86f"; }
}
