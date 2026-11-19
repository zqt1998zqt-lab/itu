#include "native16_hop.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

int main() {
  itu::Native16HopAdapter stream;
  std::array<float,256> mic{}, ref{}, out{};
  std::uint64_t calls=0;
  auto model=[&](const float* input,const float* reference,float* output) {
    ++calls;
    for(unsigned i=0;i<256;++i) output[i]=input[i]-reference[i];
    return true;
  };
  for(unsigned block=0;block<751;++block) {
    for(unsigned i=0;i<256;++i) { mic[i]=float(block*256+i)/200000; ref[i]=.01f; }
    if(!stream.process(mic.data(),ref.data(),out.data(),model)) throw std::runtime_error("native hop failed");
    for(unsigned i=0;i<256;++i)
      if(out[i]!=mic[i]-ref[i]) throw std::runtime_error("sample skipped/replayed");
  }
  if(calls!=751 || stream.inferences()!=751 || stream.delay_samples()!=0)
    throw std::runtime_error("native frame contract mismatch");
  if(stream.finish_after_hops(750)) throw std::runtime_error("wrong EOS accepted");
  if(!stream.finish_after_hops(751)) throw std::runtime_error("EOS rejected");
  if(stream.process(mic.data(),ref.data(),out.data(),model)) throw std::runtime_error("post-EOS model advanced");
  if(!stream.drain(out.data()) || calls!=751) throw std::runtime_error("drain advanced model");
  for(float sample:out) if(sample!=0) throw std::runtime_error("drain not zero");
  itu::Native16HopAdapter failing;
  auto bad=[](const float*,const float*,float*) { return false; };
  if(failing.process(mic.data(),ref.data(),out.data(),bad)) throw std::runtime_error("backend error swallowed");
  if(failing.inferences()!=0 || failing.finish_after_hops(0)) throw std::runtime_error("bad state committed");
}
