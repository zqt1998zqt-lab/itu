#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

static FILE* exclusive(const std::string& p){int fd=open(p.c_str(),O_CREAT|O_EXCL|O_WRONLY,0644);if(fd<0)throw std::runtime_error("exclusive output failed");FILE* f=fdopen(fd,"wb");if(!f){close(fd);throw std::runtime_error("fdopen failed");}return f;}
struct wave_data{unsigned rate=0,channels=0;std::vector<float> pcm;};
static std::uint32_t u32(const unsigned char* p){return p[0]|(p[1]<<8)|(p[2]<<16)|(std::uint32_t(p[3])<<24);}
static wave_data read_wave(const char* path){
    FILE* f=fopen(path,"rb");if(!f)throw std::runtime_error("input missing");unsigned char top[12];if(fread(top,1,12,f)!=12||memcmp(top,"RIFF",4)||memcmp(top+8,"WAVE",4))throw std::runtime_error("not WAV");
    wave_data result;unsigned bits=0,format=0;std::vector<unsigned char> data;
    for(;;){unsigned char head[8];if(fread(head,1,8,f)!=8)break;auto n=u32(head+4);std::vector<unsigned char> block(n);if(fread(block.data(),1,n,f)!=n)throw std::runtime_error("truncated WAV");if(n&1)fgetc(f);if(!memcmp(head,"fmt ",4)){if(n<16)throw std::runtime_error("bad fmt");format=block[0]|(block[1]<<8);result.channels=block[2]|(block[3]<<8);result.rate=u32(block.data()+4);bits=block[14]|(block[15]<<8);}if(!memcmp(head,"data",4))data=std::move(block);}
    fclose(f);if(format!=1||bits!=16||result.rate!=16000||result.channels!=1)throw std::runtime_error("expected mono PCM16 16k");result.pcm.resize(data.size()/2);for(std::size_t i=0;i<result.pcm.size();++i)result.pcm[i]=std::int16_t(data[2*i]|(data[2*i+1]<<8))/32768.f;return result;
}
int main(int argc,char** argv){try{
    if(argc!=8)throw std::runtime_error("usage: probe library model mono-mic.wav mono-ref.wav outprefix frames startframe");
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
    auto sym=[&](const char* s){auto p=dlsym(lib,s);if(!p)throw std::runtime_error(s);return p;};
    auto optnew=(std::uintptr_t(*)())sym("localvqe_options_new");auto optmodel=(int(*)(std::uintptr_t,const char*))sym("localvqe_options_set_model_path");auto optthreads=(int(*)(std::uintptr_t,int))sym("localvqe_options_set_threads");auto create=(std::uintptr_t(*)(std::uintptr_t))sym("localvqe_new_with_options");auto optfree=(void(*)(std::uintptr_t))sym("localvqe_options_free");auto gate=(int(*)(std::uintptr_t,int,float))sym("localvqe_set_noise_gate");auto process=(int(*)(std::uintptr_t,const float*,const float*,int,float*))sym("localvqe_process_frame_f32");auto dump=(int(*)(std::uintptr_t,FILE*,FILE*))sym("itu_lvq_probe_dump");auto release=(void(*)(std::uintptr_t))sym("localvqe_free");
    auto opts=optnew();if(!opts||optmodel(opts,argv[2])||optthreads(opts,1))throw std::runtime_error("options failed");auto ctx=create(opts);optfree(opts);if(!ctx||gate(ctx,0,-90.f))throw std::runtime_error("model setup failed");
    auto mic=read_wave(argv[3]);auto ref=read_wave(argv[4]);const int frames=std::stoi(argv[6]),start=std::stoi(argv[7]);if(frames<=0||start<0||mic.pcm.size()<std::size_t(frames)*256||ref.pcm.size()<std::size_t(frames)*256)throw std::runtime_error("input geometry");
    FILE* states=exclusive(std::string(argv[5])+".states.bin");FILE* pcm=exclusive(std::string(argv[5])+".pcm.f32");FILE* meta=exclusive(std::string(argv[5])+".meta.csv");float output[256];int size=0;
    for(int i=0;i<frames;++i){if(process(ctx,mic.pcm.data()+i*256,ref.pcm.data()+i*256,256,output))throw std::runtime_error("frame failed");for(float x:output)if(!std::isfinite(x))throw std::runtime_error("nonfinite PCM");if(fwrite(output,4,256,pcm)!=256)throw std::runtime_error("PCM write failed");if(i>=start){int n=dump(ctx,states,i==start?meta:nullptr);if(n<=0||(size&&n!=size))throw std::runtime_error("state geometry changed");size=n;}}
    fclose(states);fclose(pcm);fclose(meta);release(ctx);dlclose(lib);std::printf("probe_complete frames=%d captured=%d state_frame_bytes=%d sample_count=%d\n",frames,frames-start,size,frames*256);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"probe_failed: %s\n",e.what());return 2;}}
