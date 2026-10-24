#pragma once
#include "ns_spectral_frame.hpp"
#include "tiny_ns_corrector.hpp"
#include "omlsa_imcra.hpp"
#include <array>
#include <vector>

namespace itu::detail {
// The mapping is defined in Hz for each native rate. Two nonzero weights per
// FFT bin; construction allocates, observe()/interpolate() do not.
class NsFeatureBands {
 public:
  static constexpr unsigned bands=64;
  explicit NsFeatureBands(unsigned rate,unsigned nfft)
      : left_(nfft/2+1),right_weight_(nfft/2+1) {
    const float nyquist=rate*.5f;
    for(unsigned b=0;b<bands;++b) centers_[b]=b<24 ? 1000.f*b/23.f:
        b<40 ? 1000.f+3000.f*(b-23)/16.f:4000.f+(nyquist-4000.f)*(b-39)/24.f;
    for(unsigned k=0;k<left_.size();++k) {
      const float hz=float(k)*rate/nfft;unsigned b=0;
      while(b+1<bands-1 && hz>centers_[b+1]) ++b;
      left_[k]=b;right_weight_[k]=std::clamp((hz-centers_[b])/(centers_[b+1]-centers_[b]),0.f,1.f);
      normalization_[b]+=1-right_weight_[k];normalization_[b+1]+=right_weight_[k];
    }
  }
  void project(const float* spectrum,float* out) const noexcept {
    std::fill_n(out,bands,0.f);
    for(unsigned k=0;k<left_.size();++k) {
      const unsigned b=left_[k];const float w=right_weight_[k];
      out[b]+=(1-w)*spectrum[k];out[b+1]+=w*spectrum[k];
    }
    for(unsigned b=0;b<bands;++b) out[b]/=std::max(1e-12f,normalization_[b]);
  }
  float interpolate(const float* values,unsigned k) const noexcept {
    const unsigned b=left_[k];const float w=right_weight_[k];return (1-w)*values[b]+w*values[b+1];
  }
  void observe(NsSpectralFrame f) noexcept {
    project(f.power,power.data());project(f.noise,noise.data());project(f.xi,xi.data());
    project(f.gamma,gamma.data());project(f.probability,probability.data());
    for(unsigned b=0;b<bands;++b) {
      const float lp=std::log(std::max(power[b],1e-16f));
      features[b]=std::clamp(lp-std::log(std::max(noise[b],1e-16f)),-12.f,12.f)/6.f;
      features[bands+b]=initialized_ ? std::clamp(lp-previous_log_power_[b],-12.f,12.f)/6.f:0;
      previous_log_power_[b]=lp;
      enriched_features[b]=features[b];enriched_features[bands+b]=features[bands+b];
      enriched_features[2*bands+b]=std::clamp(lp/12.f,-2.f,2.f);
      enriched_features[3*bands+b]=std::clamp(std::log(std::max(noise[b],1e-16f))/12.f,-2.f,2.f);
    }
    initialized_=true;
  }
  std::array<float,bands> power{},noise{},xi{},gamma{},probability{};
  std::array<float,2*bands> features{};
  std::array<float,4*bands> enriched_features{};
 private:
  std::array<float,bands> centers_{},normalization_{},previous_log_power_{};
  std::vector<unsigned> left_;
  std::vector<float> right_weight_;
  bool initialized_=false;
};

class HybridNsControl final : public NsSpectralControl {
 public:
  HybridNsControl(unsigned rate,unsigned nfft,const std::string& model,float floor_db,
                  float spp_mix=.5f,float snr_mix=.5f,float speech_floor_db=-3.f)
      : bands(rate,nfft),previous_gain_(nfft/2+1,1.f),floor_(std::pow(10.f,floor_db/20)),
        speech_floor_(std::max(floor_,std::pow(10.f,speech_floor_db/20))),spp_mix_(spp_mix),snr_mix_(snr_mix),
        quiet_psd_limit_(16.f*(rate/100)/(12.f*32768.f*32768.f)) {
    if(!std::isfinite(spp_mix+snr_mix+speech_floor_db) || spp_mix<0 || spp_mix>1 || snr_mix<0 || snr_mix>1 || speech_floor_db>0 || speech_floor_db< -40)
      throw std::invalid_argument("invalid hybrid NS control limits");
    corrector.load(model,rate);
  }
  void observe(NsSpectralFrame f) noexcept override {
    bands.observe(f);applied=false;
    if(corrector.state()!=TinyNsCorrector::State::ready) {++fallback_frames;return;}
    if(!corrector.process(corrector.input_size()==256 ? bands.enriched_features.data():bands.features.data(),spp.data(),delta_log_xi.data())) {++fallback_frames;return;}
    ++inferences;
    if(spp_mix_==0 && snr_mix_==0) return; // Exact disabled-gain negative control.
    for(unsigned k=0;k<f.bins;++k) {
      const float pn=bands.interpolate(spp.data(),k),delta=bands.interpolate(delta_log_xi.data(),k);
      // Uncertain model outputs stay close to the independently maintained
      // statistical branch. Low entropy is not an OOD guarantee; all guards
      // and quality gates still apply.
      const float confidence=(2*pn-1)*(2*pn-1),mix=spp_mix_*confidence;
      const float p=(1-mix)*f.probability[k]+mix*pn;
      const float xi=std::clamp(f.xi[k]*std::exp(snr_mix_*confidence*delta),1e-3f,1e8f);
      const float h1=std::max(floor_,omlsa_h1_gain(xi,f.gamma[k]));
      float g=std::exp(p*std::log(h1)+(1-p)*std::log(floor_));
      const float speech=std::clamp((pn-.85f)/.15f,0.f,1.f)*mix*std::clamp(xi/.1f,0.f,1.f);
      g=std::max(g,std::exp((1-speech)*std::log(floor_)+speech*std::log(speech_floor_)));
      // A learned correction must not override a confident high-SNR
      // statistical speech bin merely because its feature is out of training
      // range (e.g. clean PCM with a vanishing estimated noise floor).
      const float trusted=std::clamp((f.xi[k]-3.16227766f)/6.83772234f,0.f,1.f)*
          std::clamp((f.probability[k]-.9f)/.1f,0.f,1.f)*
          std::clamp(1-f.noise[k]/quiet_psd_limit_,0.f,1.f);
      g=std::max(g,(1-trusted)*g+trusted*f.gain[k]);
      // Recover speech immediately; approach deeper suppression over 30 ms.
      if(initialized_ && g<previous_gain_[k]) g=.71653131f*previous_gain_[k]+.28346869f*g;
      previous_gain_[k]=f.gain[k]=std::clamp(g,floor_,1.f);
    }
    initialized_=true;applied=true;
  }
  NsFeatureBands bands;
  TinyNsCorrector corrector;
  std::array<float,64> spp{},delta_log_xi{};
  std::size_t inferences=0,fallback_frames=0;
  bool applied=false;
 private:
  std::vector<float> previous_gain_;
  float floor_,speech_floor_,spp_mix_,snr_mix_,quiet_psd_limit_;
  bool initialized_=false;
};
}
