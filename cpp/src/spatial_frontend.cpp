#include "spatial_frontend.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace itu {
SpatialFrontend::SpatialFrontend(unsigned rate,unsigned hop,unsigned m,bool mvdr,bool wind,bool fixed_steering,bool continuous_covariance)
    :rate_(rate),hop_(hop),m_(m),mvdr_(mvdr),wind_(wind),fixed_steering_(fixed_steering),continuous_covariance_(continuous_covariance) {
  if(!m || m>8 || hop!=rate/100) throw std::invalid_argument("invalid spatial frame");
  if(continuous_covariance && (!mvdr || !fixed_steering || m<2))
    throw std::invalid_argument("continuous mixture covariance requires calibrated fixed-steering MVDR");
  while(size_<2*hop_) size_*=2;
  window_.resize(size_); history_.resize(3*m_*size_); overlap_.resize(3*size_);
  normalizer_.resize(size_); spectrum_.resize(3*m_*size_);
  covariance_.resize((size_/2+1)*m_*m_); scratch_.resize(size_); roots_.resize(size_/2);
  speech_covariance_.resize(covariance_.size());
  for(unsigned i=0;i<size_;++i) window_[i]=std::sqrt(.5f-.5f*std::cos(2*std::numbers::pi_v<float>*i/size_));
  for(unsigned i=0;i<size_/2;++i) roots_[i]=std::polar(1.f,-2*std::numbers::pi_v<float>*i/size_);
}
void SpatialFrontend::fft(C* x,bool inv) noexcept {
  for(unsigned i=1,j=0;i<size_;++i) {
    unsigned b=size_/2; for(;j&b;b/=2) j^=b; j^=b;
    if(i<j) std::swap(x[i],x[j]);
  }
  for(unsigned len=2;len<=size_;len*=2) for(unsigned i=0;i<size_;i+=len)
    for(unsigned j=0;j<len/2;++j) {
      C w=roots_[j*size_/len]; if(inv) w=std::conj(w);
      const C u=x[i+j],v=x[i+j+len/2]*w; x[i+j]=u+v; x[i+j+len/2]=u-v;
    }
  if(inv) for(unsigned i=0;i<size_;++i) x[i]/=float(size_);
}
bool SpatialFrontend::weights(unsigned bin,std::array<C,8>& w) noexcept {
  const C* r=covariance_.data()+bin*m_*m_;
  float trace=0; for(unsigned ch=0;ch<m_;++ch) trace+=std::max(0.f,r[ch*m_+ch].real());
  if(trace<1e-14f || updates_<20) return false;
  const float load=std::max(1e-10f,.03f*trace/m_);
  std::array<C,64> l{};
  // Hermitian positive-definite solve, with diagonal loading and bounded WNG.
  for(unsigned i=0;i<m_;++i) for(unsigned j=0;j<=i;++j) {
    C v=r[i*m_+j]; if(i==j) v+=load;
    for(unsigned k=0;k<j;++k) v-=l[i*8+k]*std::conj(l[j*8+k]);
    if(i==j) {
      if(!std::isfinite(v.real()) || v.real()<=load*.01f) return false;
      l[i*8+j]=std::sqrt(v.real());
    } else l[i*8+j]=v/l[j*8+j].real();
  }
  std::array<C,8> z{};
  std::array<C,8> steering{};
  C target_power=0;
  const C* speech=speech_covariance_.data()+bin*m_*m_;
  for(unsigned i=0;i<m_;++i) {
    for(unsigned j=0;j<m_;++j) steering[i]+=(speech[i*m_+j]-r[i*m_+j])/float(m_);
    target_power+=steering[i]/float(m_);
  }
  const bool target_valid=!fixed_steering_ && target_power.real()>std::max(1e-12f,trace/m_);
  bool reliable=target_valid;
  if(target_valid) for(unsigned i=0;i<m_;++i) {
    steering[i]/=target_power;
    reliable &= std::isfinite(std::abs(steering[i])) && std::abs(steering[i])<4.f;
  }
  if(!reliable) for(unsigned i=0;i<m_;++i) steering[i]=1.f;
  for(unsigned i=0;i<m_;++i) {C v=steering[i]; for(unsigned j=0;j<i;++j) v-=l[i*8+j]*z[j]; z[i]=v/l[i*8+i].real();}
  for(int i=int(m_)-1;i>=0;--i) {C v=z[i]; for(unsigned j=i+1;j<m_;++j) v-=std::conj(l[j*8+i])*w[j]; w[i]=v/l[i*8+i].real();}
  C denominator=0; for(unsigned ch=0;ch<m_;++ch) denominator+=std::conj(steering[ch])*w[ch];
  if(!std::isfinite(std::abs(denominator)) || denominator.real()<=1e-12f) return false;
  float norm=0; for(unsigned ch=0;ch<m_;++ch) {w[ch]/=denominator; norm+=std::norm(w[ch]);}
  return std::isfinite(norm) && norm<=2.f;
}
void SpatialFrontend::process(const float* capture,const float* linear,const float* low,
    float* output,float* linear_out,float* low_out,bool noise) noexcept {
  const float* inputs[]={capture,linear,low}; float* outputs[]={output,linear_out,low_out};
  for(unsigned bank=0;bank<3;++bank) for(unsigned ch=0;ch<m_;++ch) {
    float* h=history_.data()+(bank*m_+ch)*size_;
    std::move(h+hop_,h+size_,h);
    for(unsigned i=0;i<hop_;++i) h[size_-hop_+i]=inputs[bank] ? inputs[bank][i*m_+ch] : 0.f;
    C* x=spectrum_.data()+(bank*m_+ch)*size_;
    for(unsigned i=0;i<size_;++i) x[i]=h[i]*window_[i];
    fft(x,false);
  }
  if(noise) ++noise_updates_;
  const bool train=noise || continuous_covariance_;
  if(train) ++updates_;
  double incoherent=0,total=0;
  for(unsigned bin=0;bin<=size_/2;++bin) {
    C* r=covariance_.data()+bin*m_*m_;
    // With a fixed calibrated target, minimizing mixture power is MPDR: the
    // distortionless constraint preserves that target while an off-axis talker
    // can keep training the interference covariance despite a speech VAD.
    // No future frames or offline target enter this update. Loading and WNG
    // limits still bound sensitivity to imperfect calibration.
    if(train) for(unsigned i=0;i<m_;++i) for(unsigned j=0;j<m_;++j) {
      C v=spectrum_[i*size_+bin]*std::conj(spectrum_[j*size_+bin]);
      r[i*m_+j]=updates_==1 ? v : .95f*r[i*m_+j]+.05f*v;
    }
    else if(!fixed_steering_) for(unsigned i=0;i<m_;++i) for(unsigned j=0;j<m_;++j) {
      const C v=spectrum_[i*size_+bin]*std::conj(spectrum_[j*size_+bin]);
      C& q=speech_covariance_[(bin*m_+i)*m_+j]; q=.95f*q+.05f*v;
    }
    std::array<C,8> w{}; for(unsigned ch=0;ch<m_;++ch) w[ch]=1.f/m_;
    if(mvdr_ && m_>1 && updates_>=20 && !weights(bin,w)) {
      ++fallbacks_; for(unsigned ch=0;ch<m_;++ch) w[ch]=1.f/m_;
    }
    if(wind_ && m_>1 && bin*rate_/size_<=600 && updates_>=20) {
      float diagonal=0; C cross=0;
      for(unsigned ch=0;ch<m_;++ch) diagonal+=std::max(0.f,r[ch*m_+ch].real());
      for(unsigned i=0;i<m_;++i) for(unsigned j=i+1;j<m_;++j) cross+=r[i*m_+j];
      const float score=std::clamp(1.f-2.f*std::abs(cross)/std::max((m_-1)*diagonal,1e-15f),0.f,1.f);
      float invsum=0; std::array<float,8> inverse{};
      for(unsigned ch=0;ch<m_;++ch) {inverse[ch]=1.f/std::max(r[ch*m_+ch].real()+.03f*diagonal/m_,1e-10f); invsum+=inverse[ch];}
      for(unsigned ch=0;ch<m_;++ch) w[ch]=(1-score)*w[ch]+score*inverse[ch]/invsum;
      incoherent+=score*diagonal; total+=diagonal;
    }
    // Normalize against the observed reverberant target relative to DAS,
    // rather than assuming every reflection has the direct-path phase.
    for(unsigned bank=0;bank<3;++bank) {
      C value=0; for(unsigned ch=0;ch<m_;++ch) value+=std::conj(w[ch])*spectrum_[(bank*m_+ch)*size_+bin];
      if(bin==0 || bin==size_/2) value=C(value.real(),0);
      // Reuse each bank's first channel for the beam spectrum after this bin.
      spectrum_[bank*m_*size_+bin]=value;
      if(bin && bin<size_/2) spectrum_[bank*m_*size_+size_-bin]=std::conj(value);
    }
  }
  wind_score_=total ? float(incoherent/total) : 0.f;
  for(unsigned i=0;i<size_;++i) normalizer_[i]+=window_[i]*window_[i];
  for(unsigned bank=0;bank<3;++bank) {
    C* x=spectrum_.data()+bank*m_*size_; fft(x,true);
    float* o=overlap_.data()+bank*size_;
    for(unsigned i=0;i<size_;++i) o[i]+=x[i].real()*window_[i];
    if(outputs[bank]) for(unsigned i=0;i<hop_;++i) outputs[bank][i]=normalizer_[i]>1e-9f ? o[i]/normalizer_[i] : 0.f;
    std::move(o+hop_,o+size_,o); std::fill(o+size_-hop_,o+size_,0.f);
  }
  std::move(normalizer_.begin()+hop_,normalizer_.end(),normalizer_.begin());
  std::fill(normalizer_.end()-hop_,normalizer_.end(),0.f);
}
std::size_t SpatialFrontend::memory_bytes() const noexcept {
  return sizeof(*this)+(window_.capacity()+history_.capacity()+overlap_.capacity()+normalizer_.capacity())*sizeof(float)+
      (spectrum_.capacity()+covariance_.capacity()+speech_covariance_.capacity()+scratch_.capacity()+roots_.capacity())*sizeof(C);
}
}
