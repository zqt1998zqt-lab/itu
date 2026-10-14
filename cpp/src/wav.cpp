#include "wav.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace itu {
namespace {
unsigned u16(const unsigned char* b) { return b[0]|unsigned(b[1])<<8; }
std::uint32_t u32(const unsigned char* b) { return b[0]|std::uint32_t(b[1])<<8|std::uint32_t(b[2])<<16|std::uint32_t(b[3])<<24; }
void put(std::ostream& f, std::uint32_t v, unsigned n) { for (unsigned i=0;i<n;++i) f.put(char(v>>(i*8))); }
}
Wav read_wav(const std::string& path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if (!f) throw std::runtime_error("cannot open WAV: "+path);
  auto length=f.tellg(); if (length<12 || length>512*1024*1024) throw std::runtime_error("WAV size must be 12 bytes..512 MiB");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length)); f.seekg(0); f.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
  if (!f || std::memcmp(bytes.data(),"RIFF",4) || std::memcmp(bytes.data()+8,"WAVE",4)) throw std::runtime_error("expected little-endian RIFF WAVE");
  std::size_t end=std::size_t(u32(bytes.data()+4))+8;
  if (end>bytes.size() || end<12) throw std::runtime_error("truncated RIFF");
  Wav w; unsigned format=0,bits=0,block=0; std::size_t data_at=0,data_size=0; bool has_fmt=false;
  for (std::size_t p=12;p+8<=end;) {
    std::size_t n=u32(bytes.data()+p+4), start=p+8;
    if (n>end-start) throw std::runtime_error("truncated WAV chunk");
    if (!std::memcmp(bytes.data()+p,"fmt ",4)) {
      if (n<16) throw std::runtime_error("short fmt chunk");
      format=u16(bytes.data()+start); w.channels=u16(bytes.data()+start+2); w.sample_rate=u32(bytes.data()+start+4);
      block=u16(bytes.data()+start+12); bits=u16(bytes.data()+start+14); has_fmt=true;
    } else if (!std::memcmp(bytes.data()+p,"data",4)) {
      if (data_at) throw std::runtime_error("multiple data chunks unsupported");
      data_at=start; data_size=n;
    }
    p=start+n+(n&1);
  }
  if (!has_fmt || !data_at || w.channels<1 || w.channels>8 || w.sample_rate<8000 || w.sample_rate>48000 ||
      !((format==1 && (bits==8 || bits==16 || bits==24 || bits==32)) || (format==3 && bits==32)) ||
      block!=w.channels*(bits/8) || data_size%block) throw std::runtime_error("unsupported WAV: PCM8/16/24/32 or float32, 1..8 channels, 8..48kHz required");
  w.samples.resize(data_size/(bits/8));
  for (std::size_t i=0;i<w.samples.size();++i) {
    const auto* b=bytes.data()+data_at+i*(bits/8); float v;
    if (format==3) { auto raw=u32(b); std::memcpy(&v,&raw,4); }
    else if (bits==8) v=(int(b[0])-128)/128.f;
    else if (bits==16) v=static_cast<std::int16_t>(u16(b))/32768.f;
    else if (bits==24) { int value=b[0]|int(b[1])<<8|int(b[2])<<16; if(value&0x800000) value-=0x1000000; v=value/8388608.f; }
    else v=static_cast<std::int32_t>(u32(b))/2147483648.f;
    if (!std::isfinite(v) || std::abs(v)>16) throw std::runtime_error("invalid WAV sample");
    w.samples[i]=v;
  }
  return w;
}
void write_wav(const std::string& path, unsigned rate, unsigned channels, const std::vector<float>& samples) {
  if (!channels || channels>8 || samples.size()%channels || samples.size()>(UINT32_MAX-36)/2) throw std::runtime_error("invalid WAV output shape");
  std::ofstream f(path,std::ios::binary); if(!f) throw std::runtime_error("cannot write WAV: "+path);
  auto size=static_cast<std::uint32_t>(samples.size()*2);
  f.write("RIFF",4); put(f,size+36,4); f.write("WAVEfmt ",8); put(f,16,4); put(f,1,2); put(f,channels,2);
  put(f,rate,4); put(f,rate*channels*2,4); put(f,channels*2,2); put(f,16,2); f.write("data",4); put(f,size,4);
  for (float v:samples) {
    if (!std::isfinite(v)) throw std::runtime_error("non-finite WAV output");
    int q=static_cast<int>(std::lround(std::clamp(v,-1.f,32767.f/32768.f)*32768.f));
    put(f,static_cast<std::uint16_t>(q),2);
  }
  if (!f) throw std::runtime_error("WAV output write failed");
}
}
