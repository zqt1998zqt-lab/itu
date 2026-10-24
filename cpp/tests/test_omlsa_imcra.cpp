#include "omlsa_imcra.hpp"
#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include <iostream>
#include <random>

namespace {
void require(bool yes,const char* message) {if(!yes) throw std::runtime_error(message);}
void scalars() {
  // Independent SciPy exp1 reference values; include both sides of the branch.
  const double x[]={1e-6,.001,.1,1.,1.001,2.,10.,50.};
  const double expected[]={13.23829589306249,6.331539364136149,1.8229239584193906,
      .2193839343955205,.21901642252746892,.048900510708061125,4.156968929685325e-6,3.783264029550459e-24};
  for(unsigned i=0;i<8;++i) require(std::abs(itu::detail::omlsa_e1(x[i])/expected[i]-1)<2e-10,"E1 independent reference mismatch");
  require(itu::detail::omlsa_posterior(0,2,3)==0 && itu::detail::omlsa_posterior(1,2,3)==1,"probability endpoints");
  const double reference=1./(1.+(1.-.2)/.2*3.*std::exp(-2.));
  require(std::abs(itu::detail::omlsa_posterior(.2f,2,3)-reference)<1e-7,"Bayes posterior mismatch");
}
void two_stage() {
  constexpr unsigned bins=17;
  itu::detail::ImcraNoiseEstimator imcra(bins);
  std::vector<float> p(bins,1),xi(bins,.1f),gamma(bins,1);
  imcra.initialize(p.data());imcra.update(p.data(),xi.data(),gamma.data());
  const float expected=.85f+.15f*1.47f;
  for(unsigned k=0;k<bins;++k) require(std::abs(imcra.noise(k)-expected)<1e-6,"IMCRA bias applied twice / wrong recursion");
  const float before=imcra.noise(8);p[7]=p[8]=p[9]=100;gamma[7]=gamma[8]=gamma[9]=100;
  imcra.update(p.data(),xi.data(),gamma.data());
  require(imcra.indicator(8)==0 && imcra.smooth(8)>5,"first IMCRA stage did not detect spectral excess");
  require(std::abs(imcra.conditional(8)-1)<1e-7,"second stage learned the excluded speech bins");
  require(std::abs(imcra.noise(8)-before)<1e-7,"strong speech poisoned noise recursion");
  require(imcra.excluded_bins()>0 && imcra.noise_updates()>0,"IMCRA trigger check vacuous");
  p.assign(bins,1);gamma.assign(bins,1);
  for(unsigned f=0;f<120;++f) imcra.update(p.data(),xi.data(),gamma.data());
  require(imcra.rotations()==10,"minima history did not advance");
  std::cout<<"IMCRA conditional smoothing active; noise updates="<<imcra.noise_updates()<<" rotations="<<imcra.rotations()<<'\n';
  itu::detail::ImcraNoiseEstimator frozen(bins),recover(bins);
  p.assign(bins,1);frozen.initialize(p.data());recover.initialize(p.data());
  p.assign(bins,16);gamma.assign(bins,16);xi.assign(bins,8);
  for(unsigned f=0;f<40;++f) {
    frozen.update(p.data(),xi.data(),gamma.data());
    recover.update(p.data(),xi.data(),gamma.data(),1.f);
  }
  require(recover.noise(8)>5*frozen.noise(8),"bounded step recovery failed to reopen frozen PSD");
  require(recover.noise(8)<24,"step recursion exceeded observed-power bias bound");
  require(recover.step_updates()>0 && frozen.step_updates()==0,"PSD recovery negative control was vacuous");
  std::cout<<"IMCRA step frozen/recovered="<<frozen.noise(8)<<'/'<<recover.noise(8)<<'\n';
  itu::detail::ImcraNoiseEstimator impulse(bins);
  p.assign(bins,1);impulse.initialize(p.data());
  const float prior_noise=impulse.noise(8);
  p.assign(bins,1e8f);gamma.assign(bins,1e8f);
  impulse.update(p.data(),xi.data(),gamma.data(),1.f);
  require(impulse.noise(8)<=1.151f*prior_noise,"step correction learned an unbounded loud frame");
}
void reconstruction() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    const unsigned h=rate/100,n=h*100;
    itu::detail::OmlsaImcra unity(rate,0);
    std::mt19937 rng(rate);std::uniform_real_distribution<float> random(-.8f,.8f);
    std::vector<float> source(n+h),frame(h),rendered(n+h);
    for(unsigned j=0;j<n;++j) source[j]=random(rng);
    source[n-1]=.7f;float maximum=0;
    for(unsigned offset=0;offset<n+h;offset+=h) {
      std::copy_n(source.data()+offset,h,frame.data());unity.process(frame.data());
      std::copy(frame.begin(),frame.end(),rendered.begin()+offset);
    }
    for(unsigned j=0;j<n;++j) maximum=std::max(maximum,std::abs(source[j]-rendered[j+h]));
    require(maximum<2e-6f,"unity WOLA reconstruction / declared delay / end drain mismatch");
    // Wrong delay is a meaningful negative control on a broadband signal.
    double wrong=0;for(unsigned j=1;j<n;++j) wrong+=std::abs(source[j]-rendered[j+h-1]);
    require(wrong/n>.1,"delay negative control vacuous");
    std::cout<<"native rate="<<rate<<" FFT="<<unity.fft_size()<<" delay="<<unity.delay()<<" WOLA maximum_error="<<maximum<<'\n';
  }
}
void attenuation_and_silence() {
  for(unsigned rate:{16000u,24000u,48000u}) {
    const unsigned h=rate/100;std::vector<float> pcm(h);
    itu::detail::OmlsaImcra ns(rate,-18);
    std::mt19937 rng(rate+71);std::normal_distribution<float> random(0,.02f);
    double before=0,after=0;
    for(unsigned f=0;f<400;++f) {
      for(float& v:pcm) {v=random(rng);if(f>=200) before+=double(v)*v;}
      ns.process(pcm.data());
      for(float v:pcm) {require(std::isfinite(v),"nonfinite noise output");if(f>=200) after+=double(v)*v;}
    }
    const double attenuation=10*std::log10(before/after);
    require(attenuation>6 && attenuation<30,"steady noise attenuation outside operating range");
    for(unsigned f=0;f<200;++f) {
      std::fill(pcm.begin(),pcm.end(),f%2?0:1e-30f);ns.process(pcm.data());
      for(float v:pcm) require(std::isfinite(v),"silence/denormal produced NaN");
    }
    std::cout<<"WGN rate="<<rate<<" attenuation_db="<<attenuation<<" mean_gain="<<ns.mean_gain()<<'\n';
  }
}
void c_api() {
  static_assert(int(itu::NoiseBackend::omlsa_imcra)==ITU_AUDIO_NS_OMLSA_IMCRA);
  for(unsigned rate:{16000u,24000u,48000u}) {
    itu_audio_config c;itu_audio_default_config(&c);c.sample_rate=rate;c.frame_size=rate/100;
    c.aec=c.automatic_gain=c.limiter=0;c.noise_level=2;
    char error[256];auto* e=itu_audio_create_with_backend(&c,ITU_AUDIO_PROFILE_GENERAL,ITU_AUDIO_NS_OMLSA_IMCRA,error,sizeof(error));
    require(e,error);std::vector<float> x(c.frame_size,.02f),y(c.frame_size);itu_audio_frame_info f;
    require(itu_audio_process(e,x.data(),x.size(),nullptr,0,y.data(),y.size(),&f,nullptr,nullptr,nullptr)==ITU_AUDIO_OK,"C API new backend process");
    require(f.latency_samples==c.frame_size,"C API native NS delay changed");
    itu_audio_destroy(e);
    require(!itu_audio_create_with_backend(&c,0,6,error,sizeof(error)),"invalid C backend accepted");
  }
}
}
int main() {
  try {scalars();two_stage();reconstruction();attenuation_and_silence();c_api();std::cout<<"OM-LSA/IMCRA checks passed\n";return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
