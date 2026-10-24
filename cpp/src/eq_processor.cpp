#include "eq_processor.hpp"
#include "linear_phase_eq.hpp"
#include "parametric_eq.hpp"

namespace itu {
struct EqProcessor::Impl {
  std::variant<std::unique_ptr<ParametricEq>,
               std::unique_ptr<LinearPhaseEq>> processor;
};
EqProcessor::EqProcessor(const EqConfig& c, unsigned rate)
    : impl_(std::make_unique<Impl>()) {
  if (c.phase == EqPhase::linear)
    impl_->processor = std::make_unique<LinearPhaseEq>(c, rate);
  else
    impl_->processor = std::make_unique<ParametricEq>(c, rate);
}
EqProcessor::~EqProcessor() = default;
EqProcessor::EqProcessor(EqProcessor&&) noexcept = default;
EqProcessor& EqProcessor::operator=(EqProcessor&&) noexcept = default;
void EqProcessor::process(float* pcm, unsigned count) noexcept {
  std::visit([&](auto& processor) { processor->process(pcm, count); },
             impl_->processor);
}
unsigned EqProcessor::delay() const noexcept {
  return std::visit([](const auto& processor) { return processor->delay(); },
                    impl_->processor);
}
std::size_t EqProcessor::memory_bytes() const noexcept {
  return sizeof(*impl_) +
      std::visit([](const auto& processor) {
        return processor->memory_bytes();
      }, impl_->processor);
}
}
