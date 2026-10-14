#include "speech_gain.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
void require(bool b,const char* message) {if(!b) throw std::runtime_error(message);}
int main() {
  try {
    itu::SpeechGain baseline,candidate;
    float a=0,b=0;
    for(unsigned i=0;i<50;++i) {
      a=baseline.observe(.001f,.01f,false,18,6);
      b=candidate.observe(.001f,.01f,false,18,6,1e-6,1e-6);
    }
    require(a==0 && b==0,"noise-only startup must not gain noise");
    for(unsigned i=0;i<50;++i) {
      a=baseline.observe(.008f,.99f,false,18,6);
      b=candidate.observe(.008f,.99f,false,18,6,1e-6,64e-6);
    }
    require(a==8 && b==8,"qualified weak speech retains utterance gain");
    for(unsigned i=0;i<50;++i)
      b=candidate.observe(.008f,.99f,false,18,6,1e-6,64e-6);
    require(b==8,"stable speech gain must not pump");
    float previous=b;
    for(unsigned i=0;i<250;++i) {
      a=baseline.observe(.004f,.9f,false,18,6);
      b=candidate.observe(.004f,.9f,false,18,6,16e-6,16e-6);
      require(b<=previous && previous-b<=.06001f,"bounded monotonic noise release");
      require(!candidate.qualified(),"noise floor vetoes false-positive VAD");
      previous=b;
    }
    require(a==8 && b<.001f,"negative control exposes stale amplified noise");
    for(unsigned i=0;i<50;++i)
      b=candidate.observe(.012f,.99f,false,18,6,16e-6,144e-6);
    require(b==8 && candidate.qualified(),"qualified speech after noise must recover");
    b=candidate.observe(.012f,.99f,true,18,6,16e-6,144e-6);
    require(b==0 && !candidate.qualified(),"far-end gain policy preserved");
    std::cout<<"PASS shared noise qualification, bounded release, weak speech and far-end negative controls\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
