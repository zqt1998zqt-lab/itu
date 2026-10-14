#include "itu/audio_engine.hpp"
#include <jni.h>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
namespace {
struct Handle { itu::AudioEngine engine; explicit Handle(const itu::Config& c):engine(c){} };
Handle* handle(jlong h) {return reinterpret_cast<Handle*>(static_cast<std::uintptr_t>(h));}
void fail(JNIEnv* env,const char* message) {env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),message);}
float* buffer(JNIEnv* env,jobject b,std::size_t bytes) {
  if(!b) return nullptr;
  void* p=env->GetDirectBufferAddress(b); jlong capacity=env->GetDirectBufferCapacity(b);
  if(!p || capacity<0 || static_cast<std::uint64_t>(capacity)<bytes ||
     capacity%sizeof(float) || reinterpret_cast<std::uintptr_t>(p)%alignof(float)) return nullptr;
  return static_cast<float*>(p);
}
template<class T> void store(unsigned char* p,unsigned offset,T value) {std::memcpy(p+offset,&value,sizeof(value));}
jlong create(JNIEnv* env,jint rate,jint frame,jint channels,jint flags,jint noise_level,jint stream_delay_ms,jfloatArray delays,jint profile,jint backend=0,jint references=1,jint extensions=0,jstring model=nullptr) {
  try {
    if(flags<0 || (flags & ~91)) throw std::invalid_argument("invalid module mask; AEC3 owns DTD and residual suppression");
    itu::Config c;
    if(profile==1) c=itu::referenced_radio_config();
    else if(profile==2) c=itu::vehicle_stream_config();
    else if(profile!=0) throw std::invalid_argument("unknown processing profile");
    c.sample_rate=rate; c.frame_size=frame; c.microphones=channels;
    if(extensions<0 || (extensions & ~7)) throw std::invalid_argument("invalid extension flags");
    c.reference_channels=references; c.noise_transition_control=extensions&1;
    c.spatial_mvdr=extensions&2; c.wind_noise_control=extensions&4;
    if(backend<0 || backend>3) throw std::invalid_argument("unknown noise backend");
    c.noise_backend=static_cast<itu::NoiseBackend>(backend);
    if(model) {
      const char* path=env->GetStringUTFChars(model,nullptr); if(!path) return 0;
      c.neural_model_path=path; env->ReleaseStringUTFChars(model,path);
    }
    c.aec=flags&1; c.beamforming=flags&2; c.noise_suppression=flags&8;
    c.automatic_gain=flags&16; c.limiter=flags&64;
    c.noise_level=noise_level; c.stream_delay_ms=stream_delay_ms;
    if(delays) {jsize n=env->GetArrayLength(delays); if(n!=channels || n>8 || n<1) throw std::invalid_argument("one causal delay per microphone required"); env->GetFloatArrayRegion(delays,0,n,c.beam_delays_samples.data()); if(env->ExceptionCheck()) return 0;}
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(new Handle(c)));
  } catch(const std::exception& e) {fail(env,e.what()); return 0;}
  catch(...) {fail(env,"native create failed"); return 0;}
}
}
extern "C" {
JNIEXPORT jboolean JNICALL Java_dev_itu_audio_NativeEngine_nativeBackendAvailable(JNIEnv*,jclass,jint backend) {
  if(backend==0) return true;
#ifdef ITU_HAS_RNNOISE
  if(backend==1) return true;
#endif
#ifdef ITU_HAS_SPEEXDSP
  if(backend==2) return true;
#endif
#ifdef ITU_HAS_ORT
  if(backend==3) return true;
#endif
  return false;
}
JNIEXPORT jlong JNICALL Java_dev_itu_audio_NativeEngine_nativeCreate(JNIEnv* env,jclass,jint rate,jint frame,jint channels,jint flags,jint noise_level,jint stream_delay_ms,jfloatArray delays) {
  return create(env,rate,frame,channels,flags,noise_level,stream_delay_ms,delays,0);
}
JNIEXPORT jlong JNICALL Java_dev_itu_audio_NativeEngine_nativeCreateWithProfile(JNIEnv* env,jclass,jint rate,jint frame,jint channels,jint flags,jint noise_level,jint stream_delay_ms,jfloatArray delays,jint profile) {
  return create(env,rate,frame,channels,flags,noise_level,stream_delay_ms,delays,profile);
}
JNIEXPORT jlong JNICALL Java_dev_itu_audio_NativeEngine_nativeCreateWithBackend(JNIEnv* env,jclass,jint rate,jint frame,jint channels,jint flags,jint noise_level,jint stream_delay_ms,jfloatArray delays,jint profile,jint backend) {
  return create(env,rate,frame,channels,flags,noise_level,stream_delay_ms,delays,profile,backend);
}
JNIEXPORT jlong JNICALL Java_dev_itu_audio_NativeEngine_nativeCreateExtended(JNIEnv* env,jclass,jint rate,jint frame,jint channels,jint flags,jint noise_level,jint stream_delay_ms,jfloatArray delays,jint profile,jint backend,jint references,jint extensions) {
  return create(env,rate,frame,channels,flags,noise_level,stream_delay_ms,delays,profile,backend,references,extensions);
}
JNIEXPORT jlong JNICALL Java_dev_itu_audio_NativeEngine_nativeCreateWithModel(JNIEnv* env,jclass,jint rate,jint frame,jint channels,jint flags,jint noise_level,jint stream_delay_ms,jfloatArray delays,jint profile,jint backend,jint references,jint extensions,jstring model) {
  return create(env,rate,frame,channels,flags,noise_level,stream_delay_ms,delays,profile,backend,references,extensions,model);
}
JNIEXPORT void JNICALL Java_dev_itu_audio_NativeEngine_nativeDestroy(JNIEnv*,jclass,jlong h) {delete handle(h);}
JNIEXPORT void JNICALL Java_dev_itu_audio_NativeEngine_nativeReset(JNIEnv* env,jclass,jlong h) {
  try {if(h) handle(h)->engine.reset();} catch(const std::exception& e) {fail(env,e.what());}
  catch(...) {fail(env,"native reset failed; recreate engine");}
}
JNIEXPORT jint JNICALL Java_dev_itu_audio_NativeEngine_nativeProcess(JNIEnv* env,jclass,jlong h,jobject mic,jobject reference,jobject output,jobject statistics,jobject aec_tap,jobject bf_tap,jobject linear_tap) {
  if(!h) return -1;
  auto& engine=handle(h)->engine; const auto& c=engine.config(); std::size_t mn=std::size_t(c.frame_size)*c.microphones;
  auto* m=buffer(env,mic,mn*sizeof(float)); auto* r=buffer(env,reference,c.frame_size*c.reference_channels*sizeof(float));
  auto* o=buffer(env,output,c.frame_size*sizeof(float)); auto* stats=buffer(env,statistics,64);
  auto* at=buffer(env,aec_tap,mn*sizeof(float)); auto* bt=buffer(env,bf_tap,c.frame_size*sizeof(float));
  auto* lt=buffer(env,linear_tap,160*c.microphones*sizeof(float));
  if(!m || !o || !stats || (reference && !r) || (c.aec && !r) || (aec_tap && !at) || (bf_tap && !bt) || (linear_tap && !lt)) return -2;
  itu::FrameInfo f; auto status=engine.process(m,mn,r,r?c.frame_size*c.reference_channels:0,o,c.frame_size,f,at,bt,lt);
  if(status!=itu::Status::ok) return static_cast<int>(status);
  auto* bytes=reinterpret_cast<unsigned char*>(stats); std::memset(bytes,0,64);
  store(bytes,0,f.frame_index);
  store(bytes,8,std::uint32_t(f.far_active | (f.linear_aec_available<<1)));
  store(bytes,12,f.microphone_rms); store(bytes,16,f.aec_rms); store(bytes,20,f.output_rms);
  store(bytes,24,f.echo_return_loss_db); store(bytes,28,f.echo_return_loss_enhancement_db);
  store(bytes,32,f.limited_samples); store(bytes,36,f.latency_samples); store(bytes,40,f.statistics_valid);
  store(bytes,44,f.estimated_delay_ms); store(bytes,48,f.backend_error);
  store(bytes,52,f.aec_output_delay_samples); store(bytes,56,f.linear_aec_delay_samples);
  return 0;
}
JNIEXPORT jlong JNICALL Java_dev_itu_audio_NativeEngine_nativeMemory(JNIEnv*,jclass,jlong h) {return h ? handle(h)->engine.working_memory_bytes() : 0;}
JNIEXPORT jstring JNICALL Java_dev_itu_audio_NativeEngine_nativeRevision(JNIEnv* env,jclass) {return env->NewStringUTF(itu::kWebRtcRevision);}
}
