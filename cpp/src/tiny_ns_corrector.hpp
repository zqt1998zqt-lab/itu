#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace itu::detail {
// Portable FP32 deployment of the exact PyTorch GRU reset-after equations.
// Export order: weight_ih, weight_hh, bias_ih, bias_hh, head.weight, head.bias.
// No model loading, allocation, locks or file I/O occurs in process().
class TinyNsCorrector {
 public:
  // Keep the original constants as the default model contract. Both widths
  // share the same input/output and feature contracts; width is file metadata.
  static constexpr unsigned bands=64,inputs=128,hidden=32,outputs=128,parameters=19776,max_hidden=64;
  static constexpr unsigned parameter_count(unsigned width) noexcept {
    return 3*width*(inputs+width)+6*width+width*outputs+outputs;
  }
  // v2 contracts: 2 = four-channel features; 3 = residual frequency Conv1d.
  static constexpr unsigned parameter_count(unsigned width,unsigned contract) noexcept {
    return 3*width*((contract==1 ? 128:256)+width)+6*width+width*outputs+outputs+(contract==3 ? 52:0);
  }
  enum class State : unsigned { disabled, ready, missing, contract_error, checksum_error, numerical_error };
  State load(const std::string& path,unsigned rate) {
    state_=State::disabled;checksum_=0;hidden_size_=0;contract_=0;weights_.clear();reset();
    if(path.empty()) return state_;
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f) return state_=State::missing;
    const auto file_bytes=f.tellg();
    if(file_bytes<std::streamoff(40)) return state_=State::contract_error;
    std::array<unsigned char,32> header{};f.seekg(0);f.read(reinterpret_cast<char*>(header.data()),header.size());
    auto read_u32=[](const unsigned char* p) {std::uint32_t v=0;for(unsigned i=0;i<4;++i) v|=std::uint32_t(p[i])<<(8*i);return v;};
    const unsigned width=read_u32(header.data()+20);
    const unsigned contract=read_u32(header.data()+24);
    const bool legacy=std::equal(header.begin(),header.begin()+8,"ITUNSGR1") && read_u32(header.data()+8)==1 && contract==1;
    const bool extended=std::equal(header.begin(),header.begin()+8,"ITUNSGR2") && read_u32(header.data()+8)==2 && (contract==2 || contract==3);
    if(!f || (!legacy && !extended) || read_u32(header.data()+12)!=rate ||
       read_u32(header.data()+16)!=bands || (width!=32 && width!=64) ||
       read_u32(header.data()+28)!=parameter_count(width,contract))
      return state_=State::contract_error;
    const unsigned count=parameter_count(width,contract);
    const std::size_t bytes=32+4*count+8;
    if(file_bytes!=std::streamoff(bytes)) return state_=State::contract_error;
    // Width and exact file size are checked before any model-sized allocation.
    std::vector<unsigned char> data(bytes);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),data.size());
    if(!f) return state_=State::contract_error;
    auto u32=[&](unsigned at) {return read_u32(data.data()+at);};
    std::uint64_t hash=14695981039346656037ull,expected=0;
    for(unsigned i=0;i<bytes-8;++i) {hash^=data[i];hash*=1099511628211ull;}
    for(unsigned i=0;i<8;++i) expected|=std::uint64_t(data[bytes-8+i])<<(8*i);
    if(hash!=expected) return state_=State::checksum_error;
    weights_.resize(count);
    for(unsigned i=0;i<count;++i) {
      const float v=std::bit_cast<float>(u32(32+4*i));
      if(!std::isfinite(v) || std::abs(v)>100.f) return state_=State::contract_error;
      weights_[i]=v;
    }
    hidden_size_=width;contract_=contract;checksum_=hash;return state_=State::ready;
  }
  State state() const noexcept {return state_;}
  std::uint64_t checksum() const noexcept {return checksum_;}
  unsigned hidden_size() const noexcept {return hidden_size_;}
  unsigned parameter_size() const noexcept {return hidden_size_ ? parameter_count(hidden_size_,contract_):0;}
  unsigned input_size() const noexcept {return contract_>1 ? 256:128;}
  unsigned contract() const noexcept {return contract_;}
  void reset() noexcept {h_.fill(0);}
  bool process(const float* features,float* spp,float* delta_log_xi) noexcept {
    if(state_!=State::ready) return false;
    for(unsigned i=0;i<input_size();++i) if(!std::isfinite(features[i]) || std::abs(features[i])>2.001f) return fail();
    if(contract_==2) return hidden_size_==64 ? process_width<64,256>(features,spp,delta_log_xi):process_width<32,256>(features,spp,delta_log_xi);
    if(contract_==3) {
      std::array<float,256> encoded{};
      for(unsigned o=0;o<4;++o) for(unsigned b=0;b<64;++b) {
        float value=weights_[48+o];
        for(unsigned i=0;i<4;++i) for(unsigned k=0;k<3;++k) {
          const int index=int(b)+int(k)-1;
          if(index>=0 && index<64) value+=weights_[(o*4+i)*3+k]*features[i*64+unsigned(index)];
        }
        encoded[o*64+b]=features[o*64+b]+std::tanh(value);
      }
      return hidden_size_==64 ? process_width<64,256,52>(encoded.data(),spp,delta_log_xi):process_width<32,256,52>(encoded.data(),spp,delta_log_xi);
    }
    return hidden_size_==64 ? process_width<64>(features,spp,delta_log_xi):process_width<32>(features,spp,delta_log_xi);
  }
 private:
  template<unsigned hidden,unsigned inputs=128,unsigned offset=0>
  bool process_width(const float* features,float* spp,float* delta_log_xi) noexcept {
    constexpr unsigned wh=3*hidden*inputs,bi=wh+3*hidden*hidden,bh=bi+3*hidden,hw=bh+3*hidden,hb=hw+outputs*hidden;
    std::array<float,3*hidden> a{},b{};std::array<float,hidden> next{};
    for(unsigned j=0;j<3*hidden;++j) {
      float x=weights_[offset+bi+j],y=weights_[offset+bh+j];
      for(unsigned i=0;i<inputs;++i) x+=weights_[offset+j*inputs+i]*features[i];
      for(unsigned i=0;i<hidden;++i) y+=weights_[offset+wh+j*hidden+i]*h_[i];
      a[j]=x;b[j]=y;
    }
    for(unsigned j=0;j<hidden;++j) {
      const float r=sigmoid(a[j]+b[j]),z=sigmoid(a[hidden+j]+b[hidden+j]);
      const float n=std::tanh(a[2*hidden+j]+r*b[2*hidden+j]);
      next[j]=(1-z)*n+z*h_[j];
      if(!std::isfinite(next[j])) return fail();
    }
    std::array<float,outputs> result{};
    for(unsigned j=0;j<outputs;++j) {
      float x=weights_[offset+hb+j];
      for(unsigned i=0;i<hidden;++i) x+=weights_[offset+hw+j*hidden+i]*next[i];
      result[j]=j<bands ? sigmoid(x):std::tanh(x)*1.3815510558f; // +/-6 dB power SNR
      if(!std::isfinite(result[j])) return fail();
    }
    std::copy_n(next.data(),hidden,h_.data());
    std::copy_n(result.data(),bands,spp);std::copy_n(result.data()+bands,bands,delta_log_xi);
    return true;
  }
  static float sigmoid(float x) noexcept {return 1.f/(1.f+std::exp(-std::clamp(x,-80.f,80.f)));}
  bool fail() noexcept {state_=State::numerical_error;return false;}
  State state_=State::disabled;
  std::uint64_t checksum_=0;
  unsigned hidden_size_=0;
  unsigned contract_=0;
  std::vector<float> weights_;
  std::array<float,max_hidden> h_{};
};
}
