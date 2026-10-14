#include "neural_ns.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  try {
    for(unsigned rate:{16000,48000}) {
      itu::NeuralNs engine(rate,argv[1]); const unsigned h=rate/100;
      std::vector<float> x(h),source,after; std::mt19937 rng(4);
      // The 48k->16k candidate intentionally has an 8k bandwidth. Excite the
      // common band for an identity/latency assertion, not out-of-band energy.
      std::normal_distribution<float> draw(0,.025f); float smooth[3]{};
      for(unsigned frame=0;frame<120;++frame) {
        for(float& v:x) {smooth[0]=.85f*smooth[0]+.15f*draw(rng); smooth[1]=.85f*smooth[1]+.15f*smooth[0]; smooth[2]=.85f*smooth[2]+.15f*smooth[1]; v=smooth[2];}
        source.insert(source.end(),x.begin(),x.end()); engine.process(x.data());
        after.insert(after.end(),x.begin(),x.end());
      }
      double cross=0,p=0,q=0,wrong=0;
      for(unsigned i=rate/5;i+engine.delay()<source.size();++i) {
        cross+=source[i]*after[i+engine.delay()]; p+=source[i]*source[i]; q+=after[i+engine.delay()]*after[i+engine.delay()];
        wrong+=source[i]*after[i];
      }
      const double corr=cross/std::sqrt(p*q);
      std::cout<<"identity ORT rate="<<rate<<" delay="<<engine.delay()<<" corr="<<corr<<" calls="<<engine.inferences()<<'\n';
      if(corr<.995 || wrong/std::sqrt(p*q)>.3 || engine.inferences()<70 || engine.nonfinite_outputs())
        throw std::runtime_error("neural adapter latency/future buffer negative control");
    }
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
