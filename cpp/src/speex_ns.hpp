#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#ifdef ITU_HAS_SPEEXDSP
#include "speex/speex_preprocess.h"
#endif
namespace itu {
class SpeexNs {
 public:
  SpeexNs(unsigned rate,int suppression):pcm_(rate/100) {
#ifdef ITU_HAS_SPEEXDSP
    state_=speex_preprocess_state_init(pcm_.size(),rate);
    if(!state_) throw std::runtime_error("SpeexDSP initialization failed");
    int enabled=1,disabled=0;
    speex_preprocess_ctl(state_,SPEEX_PREPROCESS_SET_DENOISE,&enabled);
    speex_preprocess_ctl(state_,SPEEX_PREPROCESS_SET_AGC,&disabled);
    speex_preprocess_ctl(state_,SPEEX_PREPROCESS_SET_VAD,&disabled);
    speex_preprocess_ctl(state_,SPEEX_PREPROCESS_SET_NOISE_SUPPRESS,&suppression);
#else
    (void)suppression;
    throw std::invalid_argument("SpeexDSP not built; set ITU_SPEEXDSP_SOURCE_DIR");
#endif
  }
  ~SpeexNs() {
#ifdef ITU_HAS_SPEEXDSP
    if(state_) speex_preprocess_state_destroy(state_);
#endif
  }
  void process(float* mono) noexcept {
#ifdef ITU_HAS_SPEEXDSP
    for(unsigned i=0;i<pcm_.size();++i) pcm_[i]=short(std::lround(std::clamp(mono[i]*32768.f,-32768.f,32767.f)));
    speex_preprocess_run(state_,pcm_.data());
    for(unsigned i=0;i<pcm_.size();++i) mono[i]=pcm_[i]/32768.f;
#else
    (void)mono;
#endif
  }
  void suppression(int db) noexcept {
#ifdef ITU_HAS_SPEEXDSP
    speex_preprocess_ctl(state_,SPEEX_PREPROCESS_SET_NOISE_SUPPRESS,&db);
#else
    (void)db;
#endif
  }
  std::size_t wrapper_memory_bytes() const noexcept {
    return sizeof(*this)+pcm_.capacity()*sizeof(short);
  }
 private:
  std::vector<short> pcm_;
#ifdef ITU_HAS_SPEEXDSP
  SpeexPreprocessState* state_=nullptr;
#endif
};
}
