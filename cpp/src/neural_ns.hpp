#pragma once
#include <memory>
#include <string>
namespace itu {
class NeuralNs {
 public:
  NeuralNs(unsigned rate,const std::string& model);
  ~NeuralNs();
  void process(float* pcm);
  unsigned delay() const noexcept;
  unsigned inferences() const noexcept;
  unsigned nonfinite_outputs() const noexcept;
  std::size_t memory_bytes() const noexcept;
 private:
  struct State;
  std::unique_ptr<State> state_;
};
}
