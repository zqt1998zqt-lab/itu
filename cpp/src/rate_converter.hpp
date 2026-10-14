#pragma once
#include <stdexcept>
#include "speex/speex_resampler.h"

namespace itu {
// Fixed 10 ms conversion; each stream owns its filter history.
class RateConverter {
 public:
  RateConverter(unsigned input,unsigned output):input_(input),output_(output) {
    int error=0;
    state_=speex_resampler_init(1,input*100,output*100,8,&error);
    if(!state_ || error) throw std::runtime_error("rate converter initialization failed");
    delay_=speex_resampler_get_output_latency(state_);
  }
  ~RateConverter() {
    if(state_) speex_resampler_destroy(state_);
  }
  RateConverter(const RateConverter&)=delete;
  RateConverter& operator=(const RateConverter&)=delete;
  int Resample(const float* source,unsigned input,float* destination,unsigned output) noexcept {
    if(input!=input_ || output!=output_) return -1;
    spx_uint32_t in=input,out=output;
    const int error=speex_resampler_process_float(state_,0,source,&in,destination,&out);
    return error || in!=input || out!=output ? -1 : 0;
  }
  double output_delay() const noexcept {return delay_;}
 private:
  unsigned input_,output_;
  double delay_=0;
  SpeexResamplerState* state_=nullptr;
};
}
