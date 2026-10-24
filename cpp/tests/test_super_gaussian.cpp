#include "super_gaussian.hpp"
#include "omlsa_imcra.hpp"
#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include <iostream>
#include <random>

namespace {
void require(bool x,const char* why) {if(!x) throw std::runtime_error(why);}
void reference_gain() {
  // Independent bounded SciPy maximization of eq. (36), 49-point grid.
  // Retained in build/ns-omlsa-supergaussian-20261018/sg-agent/.
  const float xi[]={.01f,.1f,1.f,10.f,100.f},gamma[]={.1f,1.f,10.f,100.f};
  const float reference[][4]={
    {.0237417277f,.00817314256f,.00359021057f,.134677827f},
    {.0809668005f,.0352655277f,.167591423f,.72574991f},
    {.306212604f,.324277848f,.733471155f,.913689494f},
    {.861382425f,.803307593f,.919848979f,.973135591f},
    {1.f,.977453232f,.978923798f,.991935134f}};
  for(unsigned i=0;i<5;++i) for(unsigned j=0;j<4;++j)
    require(std::abs(itu::detail::super_gaussian_h1_gain(xi[i],gamma[j])-reference[i][j])<2e-7f,
            "joint MAP differs from independently optimized objective");
  for(float x:{1e-12f,1e-6f,1.f,1e6f,1e12f}) for(float g:{1e-12f,1e-6f,1.f,1e6f,1e12f}) {
    const float y=itu::detail::super_gaussian_h1_gain(x,g);
    require(std::isfinite(y) && y>=0 && y<=1,"unstable/active MAP gain");
  }
  require(std::abs(itu::detail::super_gaussian_h1_gain(.1f,1.f)-
                   itu::detail::omlsa_h1_gain(.1f,1.f))>.01f,"gain selection negative control vacuous");
}
void stream() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    // Match OM-LSA's steady test: 4 s input, evaluate after 2 s adaptation.
    // Cold-start quality is independently scored on real speech fixtures.
    const unsigned hop=rate/100,n=hop*400;
    itu::detail::OmlsaImcra unity(rate,0,true,itu::detail::SpectralGainRule::super_gaussian_map);
    itu::detail::OmlsaImcra ns(rate,-12,true,itu::detail::SpectralGainRule::super_gaussian_map);
    std::mt19937 rng(rate+417);std::normal_distribution<float> random(0,.03f);
    std::vector<float> source(n+hop),frame(hop),rendered(n+hop);
    double in=0,out=0;
    for(unsigned j=0;j<n;++j) source[j]=random(rng);
    for(unsigned pos=0;pos<n+hop;pos+=hop) {
      std::copy_n(source.data()+pos,hop,frame.data());unity.process(frame.data());
      std::copy(frame.begin(),frame.end(),rendered.begin()+pos);
      std::copy_n(source.data()+pos,hop,frame.data());ns.process(frame.data());
      if(pos>=2*rate && pos<n) for(unsigned j=0;j<hop;++j) {
        in+=double(source[pos+j])*source[pos+j];out+=double(frame[j])*frame[j];
      }
      for(float x:frame) require(std::isfinite(x),"nonfinite MAP stream");
    }
    float error=0;for(unsigned j=0;j<n;++j) error=std::max(error,std::abs(source[j]-rendered[j+hop]));
    require(error<2e-6f,"joint MAP unity WOLA reconstruction/delay mismatch");
    require(10*std::log10(in/out)>6,"joint MAP steady noise suppression inactive");
    std::cout<<"Super-Gaussian native="<<rate<<" WOLA_error="<<error<<" noise_attenuation_db="<<10*std::log10(in/out)<<'\n';
  }
}
void c_api() {
  static_assert(int(itu::NoiseBackend::super_gaussian)==ITU_AUDIO_NS_SUPER_GAUSSIAN);
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu_audio_config c;itu_audio_default_config(&c);c.sample_rate=rate;c.frame_size=rate/100;
    c.aec=c.automatic_gain=c.limiter=0;c.noise_level=1;
    char error[256];auto* e=itu_audio_create_with_backend(&c,ITU_AUDIO_PROFILE_GENERAL,ITU_AUDIO_NS_SUPER_GAUSSIAN,error,sizeof(error));
    require(e,error);std::vector<float> x(c.frame_size,.02f),y(c.frame_size);itu_audio_frame_info f;
    require(itu_audio_process(e,x.data(),x.size(),nullptr,0,y.data(),y.size(),&f,nullptr,nullptr,nullptr)==ITU_AUDIO_OK,"C API MAP process");
    require(f.latency_samples==c.frame_size,"MAP C API latency changed");
    itu_audio_destroy(e);
    require(!itu_audio_create_with_backend(&c,0,6,error,sizeof(error)),"invalid backend accepted");
  }
}
}
int main() {
  try {reference_gain();stream();c_api();std::cout<<"Super-Gaussian checks passed\n";return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
