set(ITU_WEBRTC_SOURCE_DIR "${CMAKE_SOURCE_DIR}/build/webrtc-src" CACHE PATH "Pinned upstream WebRTC source plus Chromium dependencies")
if(NOT EXISTS "${ITU_WEBRTC_SOURCE_DIR}/api/audio/builtin_audio_processing_builder.h")
  message(FATAL_ERROR "WebRTC source missing. Run python3 cpp/tools/fetch_webrtc.py first.")
endif()
set(wr "${ITU_WEBRTC_SOURCE_DIR}")
file(READ "${wr}/ITU_SOURCE_LOCK.json" wr_lock)
string(JSON wr_commit GET "${wr_lock}" webrtc_commit)
if(NOT wr_commit STREQUAL "52022620d7f1f97eed6bcfbfcb2744fb26acc86f")
  message(FATAL_ERROR "WebRTC source lock differs from the tested integration revision")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/webrtc-generated/experiments")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${wr}/experiments/field_trials.py" header
  --output "${CMAKE_BINARY_DIR}/webrtc-generated/experiments/registered_field_trials.h"
  COMMAND_ERROR_IS_FATAL ANY)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("${wr}/third_party/abseil-cpp" "${CMAKE_BINARY_DIR}/webrtc-absl" EXCLUDE_FROM_ALL)

# Compile only production audio modules from the pinned official source snapshot.
set(wr_dirs common_audio common_audio/resampler common_audio/signal_processing common_audio/vad
  common_audio/third_party/ooura/fft_size_128 common_audio/third_party/ooura/fft_size_256
  common_audio/third_party/spl_sqrt_floor modules/audio_processing
  modules/audio_processing/aec3 modules/audio_processing/agc modules/audio_processing/agc/legacy
  modules/audio_processing/agc2 modules/audio_processing/agc2/rnn_vad modules/audio_processing/ns
  modules/audio_processing/utility modules/audio_processing/logging modules/audio_processing/vad
  modules/audio_processing/capture_levels_adjuster modules/audio_processing/capture_mixer)
set(wr_sources)
foreach(dir IN LISTS wr_dirs)
  file(GLOB sources CONFIGURE_DEPENDS "${wr}/${dir}/*.cc" "${wr}/${dir}/*.c")
  list(FILTER sources EXCLUDE REGEX "(test|mock|benchmark|fuzz|allocation_counter|_avx2|_sse|_mips|_neon|wav_|sinusoidal_linear_chirp)")
  list(APPEND wr_sources ${sources})
endforeach()
set(wr_explicit
  api/audio/audio_processing.cc api/audio/audio_processing_statistics.cc api/audio/audio_frame.cc
  api/audio/channel_layout.cc api/audio/builtin_audio_processing_builder.cc api/audio/echo_canceller3_config.cc
  api/environment/environment_factory.cc api/environment/deprecated_global_field_trials.cc
  api/environment/force_test_environment.cc api/task_queue/task_queue_base.cc
  api/task_queue/default_task_queue_factory_stdlib.cc api/task_queue/pending_task_safety_flag.cc
  api/field_trials.cc api/field_trials_registry.cc api/rtc_event_log/rtc_event_log.cc
  modules/audio_processing/include/aec_dump.cc modules/audio_processing/aec_dump/null_aec_dump_factory.cc
  rtc_base/checks.cc rtc_base/cpu_info.cc rtc_base/event.cc rtc_base/event_tracer.cc rtc_base/denormal_disabler.cc
  rtc_base/logging.cc rtc_base/race_checker.cc rtc_base/random.cc rtc_base/string_encode.cc
  rtc_base/string_to_number.cc rtc_base/string_utils.cc rtc_base/strings/string_builder.cc
  rtc_base/memory/aligned_malloc.cc rtc_base/platform_thread.cc rtc_base/platform_thread_types.cc rtc_base/ref_counter.cc
  rtc_base/containers/flat_tree.cc rtc_base/experiments/field_trial_parser.cc
  rtc_base/synchronization/sequence_checker_internal.cc rtc_base/synchronization/yield_policy.cc
  rtc_base/synchronization/yield.cc rtc_base/system/file_wrapper.cc
  rtc_base/system_time.cc rtc_base/system/system_time.cc rtc_base/time_utils.cc rtc_base/zero_memory.cc rtc_base/task_queue_stdlib.cc
  system_wrappers/source/clock.cc system_wrappers/source/metrics.cc
  modules/audio_coding/codecs/isac/main/source/filter_functions.c
  modules/audio_coding/codecs/isac/main/source/isac_vad.c
  modules/audio_coding/codecs/isac/main/source/pitch_estimator.c
  modules/audio_coding/codecs/isac/main/source/pitch_filter.c
  third_party/pffft/src/pffft.c third_party/rnnoise/src/rnn_vad_weights.cc)
foreach(path IN LISTS wr_explicit)
  list(APPEND wr_sources "${wr}/${path}")
endforeach()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" OR ANDROID_ABI STREQUAL "arm64-v8a")
  set(wr_neon common_audio/fir_filter_neon.cc common_audio/resampler/sinc_resampler_neon.cc
    common_audio/third_party/ooura/fft_size_128/ooura_fft_neon.cc
    common_audio/signal_processing/cross_correlation_neon.c common_audio/signal_processing/downsample_fast_neon.c
    common_audio/signal_processing/min_max_operations_neon.c)
  foreach(path IN LISTS wr_neon)
    list(APPEND wr_sources "${wr}/${path}")
  endforeach()
  set(wr_arch WEBRTC_HAS_NEON WEBRTC_ARCH_ARM64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # WebRTC's generic APM code dispatches to these CPU-specific objects at
  # runtime.  Keep their upstream per-file ISA flags instead of compiling
  # the entire library with host-specific options.
  set(wr_x86_sse2
    common_audio/fir_filter_sse.cc
    common_audio/resampler/sinc_resampler_sse.cc
    common_audio/third_party/ooura/fft_size_128/ooura_fft_sse2.cc)
  set(wr_x86_avx2
    common_audio/fir_filter_avx2.cc
    common_audio/resampler/sinc_resampler_avx2.cc
    modules/audio_processing/aec3/adaptive_fir_filter_avx2.cc
    modules/audio_processing/aec3/adaptive_fir_filter_erl_avx2.cc
    modules/audio_processing/aec3/fft_data_avx2.cc
    modules/audio_processing/aec3/matched_filter_avx2.cc
    modules/audio_processing/aec3/vector_math_avx2.cc
    modules/audio_processing/agc2/rnn_vad/vector_math_avx2.cc)
  foreach(path IN LISTS wr_x86_sse2 wr_x86_avx2)
    list(APPEND wr_sources "${wr}/${path}")
  endforeach()
endif()
add_library(itu_webrtc STATIC ${wr_sources})
target_compile_features(itu_webrtc PUBLIC cxx_std_20)
target_include_directories(itu_webrtc PUBLIC "${wr}" "${wr}/third_party/abseil-cpp"
  "${CMAKE_BINARY_DIR}/webrtc-generated")
target_compile_definitions(itu_webrtc PUBLIC WEBRTC_POSIX WEBRTC_LINUX WEBRTC_APM_DEBUG_DUMP=0 ${wr_arch})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  foreach(path IN LISTS wr_x86_sse2)
    set_source_files_properties("${wr}/${path}" PROPERTIES COMPILE_OPTIONS "-msse2")
  endforeach()
  foreach(path IN LISTS wr_x86_avx2)
    set_source_files_properties("${wr}/${path}" PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endforeach()
endif()
if(ANDROID)
  target_compile_definitions(itu_webrtc PUBLIC WEBRTC_ANDROID)
  target_sources(itu_webrtc PRIVATE "${wr}/rtc_base/system/warn_current_thread_is_deadlocked.cc"
    "${wr}/sdk/android/native_api/stacktrace/stacktrace.cc")
  target_sources(itu_webrtc PRIVATE "${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c")
  target_include_directories(itu_webrtc PRIVATE "${ANDROID_NDK}/sources/android/cpufeatures")
  target_link_libraries(itu_webrtc PUBLIC log)
endif()
find_package(Threads REQUIRED)
target_link_libraries(itu_webrtc PUBLIC absl::base absl::strings absl::synchronization absl::flat_hash_map
  absl::raw_hash_set absl::optional absl::span absl::flags Threads::Threads ${CMAKE_DL_LIBS})
