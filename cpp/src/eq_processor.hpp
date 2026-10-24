#pragma once
#include "itu/audio_engine.hpp"
#include <memory>
#include <variant>

namespace itu {
class ParametricEq;
class LinearPhaseEq;
struct EqProcessor {
  explicit EqProcessor(const EqConfig& c, unsigned rate);
  ~EqProcessor();
  EqProcessor(EqProcessor&&) noexcept;
  EqProcessor& operator=(EqProcessor&&) noexcept;
  void process(float* pcm, unsigned count) noexcept;
  unsigned delay() const noexcept;
  std::size_t memory_bytes() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
