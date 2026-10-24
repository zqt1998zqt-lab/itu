#pragma once
#include "itu/audio_engine.hpp"

namespace itu {
class NativePipeline {
 public:
  explicit NativePipeline(const Config& config);
  ~NativePipeline();
  Status process(const float* mic,const float* ref,float* out,FrameInfo& info,
                 float* aec,float* bf,float* linear,float* conditioned,
                 const PostAecContext* context=nullptr) noexcept;
  std::size_t memory_bytes() const noexcept;
 private:
  struct State;
  std::unique_ptr<State> state_;
};
}
