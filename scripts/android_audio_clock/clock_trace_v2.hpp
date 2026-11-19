#pragma once
#include <algorithm>
#include <cstdint>
#include <cerrno>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace itu_audio_clock {
inline std::int64_t monotonic_ns() noexcept {
    timespec t{};
    if(clock_gettime(CLOCK_MONOTONIC,&t))return -1;
    return std::int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;
}
inline std::int64_t thread_cpu_ns() noexcept {
    timespec t{};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))return -1;
    return std::int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;
}
inline void write_exclusive(const std::string& path,const std::string& text) {
    int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0644);
    if(fd<0)throw std::runtime_error("refusing to overwrite evidence: "+path);
    std::size_t done=0;
    while(done<text.size()){
        const auto n=::write(fd,text.data()+done,text.size()-done);
        if(n<0 && errno==EINTR)continue;
        if(n<=0){::close(fd);throw std::runtime_error("audio-clock evidence write failed");}
        done+=std::size_t(n);
    }
    if(::close(fd))throw std::runtime_error("audio-clock evidence close failed");
}
struct TraceFrame {
    std::int64_t expected_ns=0,start_ns=0,input_ready_ns=0,completed_ns=0,collected_ns=0;
    std::int64_t process_cpu_ns=0,callback_cpu_ns=0;
    bool clocks_valid=false;
};
// Allocate once before starting the stream; callback writes one unique slot.
class Trace {
public:
    explicit Trace(std::size_t capacity):frames_(capacity){}
    TraceFrame& at(std::size_t i){if(i>=frames_.size())throw std::runtime_error("audio-clock trace capacity exceeded");return frames_[i];}
    void completed(std::size_t n)noexcept{count_=n;}
    void write(const std::string& prefix,const std::string& clock_json,const std::string& hint_json,bool mock,bool hardware_executed=false)const{
        std::ostringstream csv;
        csv<<"frame,expected_ns,callback_start_ns,input_ready_ns,completed_ns,collected_ns,process_wall_ns,process_thread_cpu_ns,callback_thread_cpu_ns,input_copy_ns,collection_ns,release_lateness_ns,completion_late,clocks_valid\n";
        std::size_t clock_errors=0,late=0;
        for(std::size_t i=0;i<count_;++i){const auto& r=frames_[i];
            const bool miss=r.completed_ns>r.expected_ns+16000000LL;late+=miss;clock_errors+=!r.clocks_valid;
            csv<<i<<','<<r.expected_ns<<','<<r.start_ns<<','<<r.input_ready_ns<<','<<r.completed_ns<<','<<r.collected_ns<<','
               <<r.completed_ns-r.input_ready_ns<<','<<r.process_cpu_ns<<','<<r.callback_cpu_ns<<','<<r.input_ready_ns-r.start_ns<<','
               <<r.collected_ns-r.completed_ns<<','<<std::max<std::int64_t>(0,r.start_ns-r.expected_ns)<<','<<miss<<','<<r.clocks_valid<<'\n';
        }
        write_exclusive(prefix+".audio-clock.frames.csv",csv.str());
        std::ostringstream report;
        report<<"{\n  \"schema\":\"itu-android-audio-clock-v1\",\n"
              <<"  \"scope\":\"frozen PCM processed by unchanged VCP; optional silent AAudio output supplies callback clock\",\n"
              <<"  \"mock_clock\":"<<(mock?"true":"false")<<",\n"
              <<"  \"hardware_audio_clock_executed\":"<<(hardware_executed?"true":"false")<<",\n"
              <<"  \"microphone_capture_verified\":false,\n  \"processed_audio_playback_verified\":false,\n"
              <<"  \"end_to_end_acoustic_latency_verified\":false,\n  \"formal_itu_compliance_verified\":false,\n"
              <<"  \"callback_deadline_scope\":\"16ms relative to first callback start; not DAC presentation or hardware xrun proof\",\n"
              <<"  \"callbacks\":"<<count_<<",\n  \"clock_errors\":"<<clock_errors<<",\n"
              <<"  \"completion_late_including_drain\":"<<late<<",\n"
              <<"  \"audio_stream\":"<<clock_json<<",\n  \"performance_hint\":"<<hint_json<<"\n}\n";
        write_exclusive(prefix+".audio-clock.json",report.str());
    }
private:
    std::vector<TraceFrame> frames_;
    std::size_t count_=0;
};
}
