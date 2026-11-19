#pragma once
#include "itu/audio_engine.hpp"
#include "itu/complex_pf_dsp_provider.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace itu {
inline bool complex_pf_model_id(const std::string& id) noexcept {
  return !id.empty() && id.size()<=63 && std::all_of(id.begin(),id.end(),[](unsigned char c){
    return (c>='a' && c<='z') || (c>='A' && c<='Z') ||
        (c>='0' && c<='9') || c=='_' || c=='-';
  });
}
inline const itu_complex_pf_dsp_provider_v1* linked_complex_pf_dsp_provider() {
#ifdef ITU_HAS_COMPLEX_PF_DSP
  return itu_complex_pf_dsp_get_provider_v1();
#else
  return nullptr;
#endif
}

inline bool complex_pf_sha256(const char* s, std::size_t size) noexcept {
  if(size!=65 || s[64]!='\0') return false;
  for(unsigned i=0;i<64;++i)
    if(!((s[i]>='0' && s[i]<='9') || (s[i]>='a' && s[i]<='f'))) return false;
  return true;
}
template<std::size_t N> inline bool complex_pf_text(const char (&s)[N]) noexcept {
  return s[0] && std::memchr(s,'\0',N)!=nullptr;
}
inline std::string complex_pf_info_error(const itu_complex_pf_dsp_info_v1& i) {
  if(i.struct_size!=sizeof(i) || i.version!=ITU_COMPLEX_PF_DSP_ABI)
    return "Complex_PF provider metadata ABI mismatch";
  if(i.available!=1) return complex_pf_text(i.reason) ? i.reason : "HiFi device or Complex_PF model unavailable";
  if(!complex_pf_text(i.execution_provider) || std::strcmp(i.execution_provider,"hifi_dsp"))
    return "speex_fused_dsp requires a real hifi_dsp provider; host simulation cannot satisfy this route";
  if(!complex_pf_text(i.architecture) || std::strcmp(i.architecture,"mdf_complex_pf"))
    return "speex_fused_dsp requires the MDF + Complex_PF architecture";
  if(i.sample_rate!=16000 || i.callback_samples!=160 || i.delay_samples<212 ||
     i.delay_samples>16000 || i.delay_samples%160)
    return "Complex_PF requires 16 kHz, 160 samples and a declared whole-callback delay";
  if(!complex_pf_text(i.device_id) || !complex_pf_text(i.model_id) ||
     !complex_pf_sha256(i.model_sha256,sizeof(i.model_sha256)) ||
     !complex_pf_sha256(i.frontend_sha256,sizeof(i.frontend_sha256)) ||
     !complex_pf_sha256(i.provider_sha256,sizeof(i.provider_sha256)))
    return "Complex_PF device, registered model and SHA-256 identities are required";
  return {};
}
inline std::string complex_pf_provider_error(const itu_complex_pf_dsp_provider_v1* p) {
  if(!p) return "speex_fused_dsp unavailable: HiFi SDK/provider is not linked; register a trained Complex_PF model with the HiFi adapter";
  if(p->struct_size!=sizeof(*p) || p->version!=ITU_COMPLEX_PF_DSP_ABI ||
     !p->probe || !p->create || !p->process || !p->destroy)
    return "Complex_PF provider ABI or required callbacks are invalid";
  return {};
}
inline AecBackendCapability complex_pf_dsp_capability(
    const std::string& model_id="complex_pf_stream_v1",
    const itu_complex_pf_dsp_provider_v1* p=linked_complex_pf_dsp_provider()) {
  AecBackendCapability c;
  c.id="speex_fused_dsp"; c.architecture="mdf_complex_pf";
  if(!complex_pf_model_id(model_id)) {
    c.reason="speex_fused_dsp model ID must be a registered name, not a file path";
    return c;
  }
  c.reason=complex_pf_provider_error(p);
  if(!c.reason.empty()) return c;
  itu_complex_pf_dsp_info_v1 info{}; info.struct_size=sizeof(info); info.version=ITU_COMPLEX_PF_DSP_ABI;
  std::array<char,256> error{};
  try {
    if(p->probe(model_id.c_str(),&info,error.data(),error.size())) {
      c.reason=error.back()==0 && error[0] ? error.data() : "HiFi probe failed";
      return c;
    }
  } catch(...) {c.reason="HiFi probe threw an exception";return c;}
  c.reason=complex_pf_info_error(info);
  if(c.reason.empty() && info.model_id!=model_id)
    c.reason="HiFi probe returned a different model identity";
  if(!c.reason.empty()) return c;
  c.available=true; c.execution_provider=info.execution_provider;
  c.device_id=info.device_id; c.model_id=info.model_id;
  c.model_sha256=info.model_sha256; c.frontend_sha256=info.frontend_sha256;
  c.provider_sha256=info.provider_sha256; c.delay_samples=info.delay_samples;
  return c;
}

class ComplexPfDsp {
 public:
  explicit ComplexPfDsp(const Config& c,
      const itu_complex_pf_dsp_provider_v1* p=linked_complex_pf_dsp_provider()):provider_(p) {
    if(c.sample_rate!=16000 || c.frame_size!=160 || c.microphones!=1 || c.reference_channels!=1)
      throw std::invalid_argument("speex_fused_dsp supports only 16 kHz / 160 samples / mono capture and playback");
    if(c.speex_fused_dsp_deadline_us==0 || c.speex_fused_dsp_deadline_us>10000)
      throw std::invalid_argument("speex_fused_dsp deadline must be 1..10000 us");
    deadline_us_=c.speex_fused_dsp_deadline_us;
    const auto capability=complex_pf_dsp_capability(c.speex_fused_dsp_model_id,p);
    if(!capability.available) throw std::invalid_argument(capability.reason);
    info_.struct_size=sizeof(info_); info_.version=ITU_COMPLEX_PF_DSP_ABI;
    std::array<char,256> error{};
    handle_=p->create(c.speex_fused_dsp_model_id.c_str(),c.sample_rate,c.frame_size,
        &info_,error.data(),error.size());
    if(!handle_) throw std::runtime_error(error.back()==0 && error[0] ? error.data() : "HiFi model creation failed");
    const auto mismatch=complex_pf_info_error(info_);
    if(!mismatch.empty() || info_.model_id!=c.speex_fused_dsp_model_id ||
       capability.device_id!=info_.device_id || capability.model_sha256!=info_.model_sha256 ||
       capability.frontend_sha256!=info_.frontend_sha256 || capability.provider_sha256!=info_.provider_sha256 ||
       capability.delay_samples!=info_.delay_samples) {
      p->destroy(handle_);handle_=nullptr;
      throw std::runtime_error(mismatch.empty() ? "HiFi identity changed between probe and create" : mismatch);
    }
  }
  ~ComplexPfDsp() {if(handle_) provider_->destroy(handle_);}
  ComplexPfDsp(const ComplexPfDsp&)=delete;
  ComplexPfDsp& operator=(const ComplexPfDsp&)=delete;
  Status process(const float* residual,const float* farend,float* output) noexcept {
    if(failed_) {std::fill_n(output,160,0.f);return Status::backend_error;}
    itu_complex_pf_dsp_frame_v1 frame{};frame.struct_size=sizeof(frame);frame.version=ITU_COMPLEX_PF_DSP_ABI;
    frame.frame_index=frames_;
    const auto started=std::chrono::steady_clock::now();
    int result=-1;
    try {result=provider_->process(handle_,residual,farend,output,160,&frame);} catch(...) {}
    const auto us=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-started).count();
    if(result || frame.struct_size!=sizeof(frame) || frame.version!=ITU_COMPLEX_PF_DSP_ABI ||
       frame.frame_index!=frames_ || frame.fallback_frames || us>deadline_us_ ||
       std::any_of(output,output+160,[](float f){return !std::isfinite(f) || std::abs(f)>1.f;})) {
      std::fill_n(output,160,0.f);failed_=true;return Status::backend_error;
    }
    ++frames_;inferences_+=frame.inferences;cycles_+=frame.cycles;
    return Status::ok;
  }
  unsigned delay() const noexcept {return info_.delay_samples;}
  std::size_t memory_bytes() const noexcept {return sizeof(*this)+info_.working_memory_bytes;}
  AecRouteDiagnostics diagnostics() const {
    AecRouteDiagnostics d;
    d.execution_provider=info_.execution_provider;d.device_id=info_.device_id;d.model_id=info_.model_id;
    d.model_sha256=info_.model_sha256;d.frontend_sha256=info_.frontend_sha256;
    d.provider_sha256=info_.provider_sha256;d.processed_frames=frames_;
    d.inferences=inferences_;d.cycles=cycles_;d.neural_delay_samples=info_.delay_samples;
    return d;
  }
 private:
  const itu_complex_pf_dsp_provider_v1* provider_;
  void* handle_=nullptr;
  itu_complex_pf_dsp_info_v1 info_{};
  std::uint64_t frames_=0,inferences_=0,cycles_=0;
  unsigned deadline_us_=10000;
  bool failed_=false;
};
}
