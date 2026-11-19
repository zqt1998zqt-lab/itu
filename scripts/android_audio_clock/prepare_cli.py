"""Extract the frozen callback body; add an opt-in Android audio clock.

All DSP statements remain in their original order. No historical source is
modified. The mock callback exists only in host validation builds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/zhangqingtian/project/itu')
SOURCE=ROOT/'build/android-native16-20261117/root/overlay/itu_audio_cli.cpp'
SOURCE_SHA='c7ace82d55932e4f700fc599231608e592c2797cfea7941bfa4584fccc46ab29'


def transform()->tuple[str,dict]:
    original=SOURCE.read_bytes()
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA:raise RuntimeError('Frozen CLI source changed')
    text=original.decode()
    start_text='    const auto stream_start=std::chrono::steady_clock::now();\n    for(std::size_t k=0;k<total_frames;++k) {\n'
    end_text='    }\n    flush_rx_diagnostic_tail();'
    if text.count(start_text)!=1 or text.count(end_text)!=1:raise RuntimeError('Callback extraction boundary changed')
    start=text.index(start_text);end=text.index(end_text,start)
    body=text[start+len(start_text):end]
    prelude='''      const auto release=stream_start+std::chrono::milliseconds(k*16);
      if(paced) std::this_thread::sleep_until(release);
      const auto actual_release=std::chrono::steady_clock::now();
'''
    if body.count(prelude)!=1:raise RuntimeError('Pacing prelude differs')
    core=body.replace(prelude,'',1)
    input_mark='      auto t=std::chrono::steady_clock::now();\n'
    completed_mark='      const auto completed=std::chrono::steady_clock::now();\n'
    if core.count(input_mark)!=1 or core.count(completed_mark)!=1:raise RuntimeError('Timing marks ambiguous')
    core=core.replace(input_mark,input_mark+'''      const auto clock_input_ready=itu_audio_clock::monotonic_ns();
      const auto clock_process_cpu_start=itu_audio_clock::thread_cpu_ns();
''',1)
    core=core.replace(completed_mark,completed_mark+'''      const auto clock_process_cpu_end=itu_audio_clock::thread_cpu_ns();
      const auto clock_completed=itu_audio_clock::monotonic_ns();
''',1)
    replacement='''    if(audio_clock_mode && (mock_audio_clock || paced)) throw std::runtime_error("AAudio, test clock and timer pacing are mutually exclusive");
    if(hint_requested && !audio_clock_mode) throw std::runtime_error("performance hints require a real AAudio callback");
    if((audio_clock_mode || mock_audio_clock) && (c.sample_rate!=16000 || c.frame_size!=256 || !rx_in_path.empty() || simulate_clock || diagnostic_stages)) throw std::runtime_error("audio clock requires native256 frozen TX input; RX was processed upstream");
#ifndef ITU_AUDIO_CLOCK_TESTING
    if(mock_audio_clock) throw std::runtime_error("test audio clock is not present in this build");
#endif
    const bool timer_paced=paced;
    if(audio_clock_mode || mock_audio_clock)paced=true;
    itu_audio_clock::Trace clock_trace(frames+drain_capacity);
    itu_audio_clock::PerformanceHint performance_hint(hint_requested);
    performance_hint.set_target_duration_ns(16000000);
    auto process_one=[&](std::size_t k,std::int64_t expected_ns,std::int64_t callback_start_ns)->bool {
      const auto release=std::chrono::steady_clock::time_point(std::chrono::nanoseconds(expected_ns));
      const auto actual_release=std::chrono::steady_clock::time_point(std::chrono::nanoseconds(callback_start_ns));
      const auto clock_callback_cpu_start=itu_audio_clock::thread_cpu_ns();
'''+core+'''      const auto clock_collected=itu_audio_clock::monotonic_ns();
      const auto clock_callback_cpu_end=itu_audio_clock::thread_cpu_ns();
      auto& clock_row=clock_trace.at(k);
      clock_row={expected_ns,callback_start_ns,clock_input_ready,clock_completed,clock_collected,
          clock_process_cpu_end-clock_process_cpu_start,clock_callback_cpu_end-clock_callback_cpu_start,
          clock_input_ready>=0 && clock_completed>=clock_input_ready && clock_collected>=clock_completed && clock_process_cpu_start>=0 && clock_process_cpu_end>=clock_process_cpu_start && clock_callback_cpu_start>=0 && clock_callback_cpu_end>=clock_callback_cpu_start};
      clock_trace.completed(k+1);
      if(hint_requested)performance_hint.submit_callback_duration(clock_collected-callback_start_ns);
      return k+1<total_frames;
    };
    std::string audio_stream_json="null";
    if(audio_clock_mode) {
      itu_audio_clock::Config audio_config;
      audio_config.exclusive=!audio_clock_shared;
      audio_config.on_main_poll=[&](){performance_hint.drain();};
      audio_config.on_callback_thread=[&](std::int32_t tid){performance_hint.register_callback_tid(tid);};
      itu_audio_clock::Runner runner(audio_config,process_one);
      const auto audio_result=runner.run(double(n)/c.sample_rate+5.0);
      audio_stream_json=audio_result.to_json();
    } else {
      const auto base_ns=itu_audio_clock::monotonic_ns();
      for(std::size_t k=0;k<total_frames;++k) {
        const auto expected_ns=base_ns+std::int64_t(k)*16000000;
        if(timer_paced)std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(expected_ns)));
        if(!process_one(k,expected_ns,itu_audio_clock::monotonic_ns()))break;
      }
    }
    performance_hint.drain();performance_hint.close();
    clock_trace.write(prefix,audio_stream_json,performance_hint.snapshot().to_json(),mock_audio_clock);
    flush_rx_diagnostic_tail();'''
    text=text[:start]+replacement+text[end+len(end_text):]
    anchor='#include "native16_cofi.hpp"\n'
    if text.count(anchor)!=1:raise RuntimeError('Include anchor changed')
    text=text.replace(anchor,anchor+'#include "aaudio_clock.hpp"\n#include "android_performance_hint.hpp"\n#include "clock_trace.hpp"\n',1)
    anchor='    bool aec_capabilities=false,complex_pf_export_inputs=false;\n'
    if text.count(anchor)!=1:raise RuntimeError('CLI options anchor changed')
    text=text.replace(anchor,anchor+'    bool audio_clock_mode=false,mock_audio_clock=false,hint_requested=false,audio_clock_shared=false;\n',1)
    anchor='      else if(a=="--pace-16ms") paced=true;\n'
    if text.count(anchor)!=1:raise RuntimeError('Argument anchor changed')
    text=text.replace(anchor,anchor+'''      else if(a=="--aaudio-clock") audio_clock_mode=true;
      else if(a=="--audio-clock-test") mock_audio_clock=true;
      else if(a=="--performance-hint") hint_requested=true;
      else if(a=="--audio-clock-shared") audio_clock_shared=true;
''',1)
    # Check text identity of every DSP statement apart from inserted timing hooks.
    restored=core.replace(input_mark+'''      const auto clock_input_ready=itu_audio_clock::monotonic_ns();
      const auto clock_process_cpu_start=itu_audio_clock::thread_cpu_ns();
''',input_mark,1).replace(completed_mark+'''      const auto clock_process_cpu_end=itu_audio_clock::thread_cpu_ns();
      const auto clock_completed=itu_audio_clock::monotonic_ns();
''',completed_mark,1)
    if restored!=body.replace(prelude,'',1):raise RuntimeError('Unexpected DSP-body mutation')
    receipt={'source':str(SOURCE),'source_sha256':SOURCE_SHA,'source_untouched':SOURCE.read_bytes()==original,'DSP_body_identical_after_removing_timing_hooks':True,'original_body_sha256':hashlib.sha256(restored.encode()).hexdigest(),'generated_sha256':hashlib.sha256(text.encode()).hexdigest(),'scope':'same DSP order/inputs/states; optional silent AAudio output clock, not live mic capture/playback'}
    return text,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();text,receipt=transform();a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:f.write(text)
    with a.out.with_suffix('.source.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))
