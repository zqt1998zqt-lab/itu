#define AAUDIO_AAUDIO_H

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using aaudio_result_t = int32_t;
using aaudio_format_t = int32_t;
using aaudio_sharing_mode_t = int32_t;
using aaudio_performance_mode_t = int32_t;
using aaudio_stream_state_t = int32_t;
using aaudio_data_callback_result_t = int32_t;

struct AAudioStreamStruct;
struct AAudioStreamBuilderStruct;
using AAudioStream = AAudioStreamStruct;
using AAudioStreamBuilder = AAudioStreamBuilderStruct;

inline constexpr int32_t AAUDIO_OK = 0;
inline constexpr int32_t AAUDIO_ERROR_TIMEOUT = -916;
inline constexpr int32_t AAUDIO_ERROR_DISCONNECTED = -899;
inline constexpr int32_t AAUDIO_FORMAT_PCM_FLOAT = 2;
inline constexpr int32_t AAUDIO_DIRECTION_OUTPUT = 0;
inline constexpr int32_t AAUDIO_SHARING_MODE_EXCLUSIVE = 0;
inline constexpr int32_t AAUDIO_SHARING_MODE_SHARED = 1;
inline constexpr int32_t AAUDIO_PERFORMANCE_MODE_NONE = 10;
inline constexpr int32_t AAUDIO_PERFORMANCE_MODE_LOW_LATENCY = 12;
inline constexpr int32_t AAUDIO_STREAM_STATE_OPEN = 2;
inline constexpr int32_t AAUDIO_STREAM_STATE_STARTED = 4;
inline constexpr int32_t AAUDIO_STREAM_STATE_STOPPED = 10;
inline constexpr int32_t AAUDIO_STREAM_STATE_CLOSED = 12;
inline constexpr int32_t AAUDIO_STREAM_STATE_DISCONNECTED = 13;
inline constexpr int32_t AAUDIO_CALLBACK_RESULT_CONTINUE = 0;
inline constexpr int32_t AAUDIO_CALLBACK_RESULT_STOP = 1;

using AAudioStream_dataCallback = aaudio_data_callback_result_t (*)(
    AAudioStream*, void*, void*, int32_t);
using AAudioStream_errorCallback = void (*)(
    AAudioStream*, void*, aaudio_result_t);

aaudio_result_t AAudio_createStreamBuilder(AAudioStreamBuilder** builder);
void AAudioStreamBuilder_setDirection(AAudioStreamBuilder*, int32_t direction);
void AAudioStreamBuilder_setSampleRate(AAudioStreamBuilder*, int32_t rate);
void AAudioStreamBuilder_setChannelCount(AAudioStreamBuilder*, int32_t channels);
void AAudioStreamBuilder_setFormat(AAudioStreamBuilder*, aaudio_format_t format);
void AAudioStreamBuilder_setFramesPerDataCallback(
    AAudioStreamBuilder*, int32_t frames);
void AAudioStreamBuilder_setPerformanceMode(
    AAudioStreamBuilder*, aaudio_performance_mode_t mode);
void AAudioStreamBuilder_setSharingMode(
    AAudioStreamBuilder*, aaudio_sharing_mode_t mode);
void AAudioStreamBuilder_setDataCallback(
    AAudioStreamBuilder*, AAudioStream_dataCallback, void*);
void AAudioStreamBuilder_setErrorCallback(
    AAudioStreamBuilder*, AAudioStream_errorCallback, void*);
aaudio_result_t AAudioStreamBuilder_openStream(
    AAudioStreamBuilder*, AAudioStream** stream);
aaudio_result_t AAudioStreamBuilder_delete(AAudioStreamBuilder*);
int32_t AAudioStream_getSampleRate(AAudioStream*);
int32_t AAudioStream_getChannelCount(AAudioStream*);
aaudio_format_t AAudioStream_getFormat(AAudioStream*);
aaudio_performance_mode_t AAudioStream_getPerformanceMode(AAudioStream*);
aaudio_sharing_mode_t AAudioStream_getSharingMode(AAudioStream*);
int32_t AAudioStream_getFramesPerBurst(AAudioStream*);
int32_t AAudioStream_getFramesPerDataCallback(AAudioStream*);
int32_t AAudioStream_getBufferCapacityInFrames(AAudioStream*);
int32_t AAudioStream_getBufferSizeInFrames(AAudioStream*);
aaudio_result_t AAudioStream_setBufferSizeInFrames(
    AAudioStream*, int32_t frames);
aaudio_stream_state_t AAudioStream_getState(AAudioStream*);
aaudio_result_t AAudioStream_requestStart(AAudioStream*);
aaudio_result_t AAudioStream_requestStop(AAudioStream*);
aaudio_result_t AAudioStream_waitForStateChange(
    AAudioStream*, aaudio_stream_state_t input_state,
    aaudio_stream_state_t* next_state, int64_t timeout_nanoseconds);
aaudio_result_t AAudioStream_close(AAudioStream*);

struct FakeScenario {
    int32_t actual_sample_rate = 48000;
    int32_t actual_channel_count = 1;
    int32_t actual_format = AAUDIO_FORMAT_PCM_FLOAT;
    int32_t actual_performance_mode =
        AAUDIO_PERFORMANCE_MODE_LOW_LATENCY;
    int32_t actual_sharing_mode = AAUDIO_SHARING_MODE_EXCLUSIVE;
    int32_t actual_frames_per_burst = 192;
    int32_t actual_frames_per_callback = 768;
    int32_t callback_frames_override = 0;
    int32_t buffer_size_extra = 0;
    int callback_target = 3;
    bool error_after_callbacks = false;
};

struct AAudioStreamBuilderStruct {
    int32_t direction = AAUDIO_DIRECTION_OUTPUT;
    int32_t sample_rate = 0;
    int32_t channel_count = 0;
    int32_t format = 0;
    int32_t frames_per_callback = 0;
    int32_t performance_mode = AAUDIO_PERFORMANCE_MODE_NONE;
    int32_t sharing_mode = AAUDIO_SHARING_MODE_SHARED;
    AAudioStream_dataCallback data_callback = nullptr;
    AAudioStream_errorCallback error_callback = nullptr;
    void* user_data = nullptr;
};

struct AAudioStreamStruct {
    AAudioStreamBuilder settings{};
    int32_t state = AAUDIO_STREAM_STATE_OPEN;
    int32_t buffer_size = 0;
    int32_t buffer_capacity = 4096;
    std::array<float, 768> pcm{};
};

FakeScenario g_scenario;
float* g_current_audio = nullptr;
int g_fake_callback_invocations = 0;

aaudio_result_t AAudio_createStreamBuilder(AAudioStreamBuilder** builder) {
    if (builder == nullptr) {
        return -900;
    }
    *builder = new AAudioStreamBuilderStruct();
    return AAUDIO_OK;
}

void AAudioStreamBuilder_setDirection(
    AAudioStreamBuilder* builder, int32_t direction) {
    builder->direction = direction;
}

void AAudioStreamBuilder_setSampleRate(
    AAudioStreamBuilder* builder, int32_t rate) {
    builder->sample_rate = rate;
}

void AAudioStreamBuilder_setChannelCount(
    AAudioStreamBuilder* builder, int32_t channels) {
    builder->channel_count = channels;
}

void AAudioStreamBuilder_setFormat(
    AAudioStreamBuilder* builder, aaudio_format_t format) {
    builder->format = format;
}

void AAudioStreamBuilder_setFramesPerDataCallback(
    AAudioStreamBuilder* builder, int32_t frames) {
    builder->frames_per_callback = frames;
}

void AAudioStreamBuilder_setPerformanceMode(
    AAudioStreamBuilder* builder, aaudio_performance_mode_t mode) {
    builder->performance_mode = mode;
}

void AAudioStreamBuilder_setSharingMode(
    AAudioStreamBuilder* builder, aaudio_sharing_mode_t mode) {
    builder->sharing_mode = mode;
}

void AAudioStreamBuilder_setDataCallback(
    AAudioStreamBuilder* builder, AAudioStream_dataCallback callback,
    void* user_data) {
    builder->data_callback = callback;
    builder->user_data = user_data;
}

void AAudioStreamBuilder_setErrorCallback(
    AAudioStreamBuilder* builder, AAudioStream_errorCallback callback,
    void* user_data) {
    builder->error_callback = callback;
    builder->user_data = user_data;
}

aaudio_result_t AAudioStreamBuilder_openStream(
    AAudioStreamBuilder* builder, AAudioStream** stream) {
    if (builder == nullptr || stream == nullptr) {
        return -900;
    }
    auto* opened = new AAudioStreamStruct();
    opened->settings = *builder;
    opened->buffer_size = g_scenario.actual_frames_per_burst * 2;
    opened->pcm.fill(123.5f);
    *stream = opened;
    return AAUDIO_OK;
}

aaudio_result_t AAudioStreamBuilder_delete(AAudioStreamBuilder* builder) {
    delete builder;
    return AAUDIO_OK;
}

int32_t AAudioStream_getSampleRate(AAudioStream*) {
    return g_scenario.actual_sample_rate;
}

int32_t AAudioStream_getChannelCount(AAudioStream*) {
    return g_scenario.actual_channel_count;
}

aaudio_format_t AAudioStream_getFormat(AAudioStream*) {
    return g_scenario.actual_format;
}

aaudio_performance_mode_t AAudioStream_getPerformanceMode(AAudioStream*) {
    return g_scenario.actual_performance_mode;
}

aaudio_sharing_mode_t AAudioStream_getSharingMode(AAudioStream*) {
    return g_scenario.actual_sharing_mode;
}

int32_t AAudioStream_getFramesPerBurst(AAudioStream*) {
    return g_scenario.actual_frames_per_burst;
}

int32_t AAudioStream_getFramesPerDataCallback(AAudioStream*) {
    return g_scenario.actual_frames_per_callback;
}

int32_t AAudioStream_getBufferCapacityInFrames(AAudioStream* stream) {
    return stream->buffer_capacity;
}

int32_t AAudioStream_getBufferSizeInFrames(AAudioStream* stream) {
    return stream->buffer_size;
}

aaudio_result_t AAudioStream_setBufferSizeInFrames(
    AAudioStream* stream, int32_t frames) {
    stream->buffer_size = frames + g_scenario.buffer_size_extra;
    if (stream->buffer_size > stream->buffer_capacity) {
        stream->buffer_size = stream->buffer_capacity;
    }
    return stream->buffer_size;
}

aaudio_stream_state_t AAudioStream_getState(AAudioStream* stream) {
    return stream->state;
}

aaudio_result_t AAudioStream_requestStart(AAudioStream* stream) {
    stream->state = AAUDIO_STREAM_STATE_STARTED;
    const int32_t frames =
        g_scenario.callback_frames_override > 0
            ? g_scenario.callback_frames_override
            : g_scenario.actual_frames_per_callback;
    for (int32_t index = 0; index < g_scenario.callback_target; ++index) {
        ++g_fake_callback_invocations;
        g_current_audio = stream->pcm.data();
        const aaudio_data_callback_result_t result =
            stream->settings.data_callback(
                stream, stream->settings.user_data, stream->pcm.data(), frames);
        g_current_audio = nullptr;
        if (result == AAUDIO_CALLBACK_RESULT_STOP) {
            break;
        }
    }
    if (g_scenario.error_after_callbacks) {
        stream->state = AAUDIO_STREAM_STATE_DISCONNECTED;
        stream->settings.error_callback(
            stream, stream->settings.user_data,
            AAUDIO_ERROR_DISCONNECTED);
    }
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_requestStop(AAudioStream* stream) {
    stream->state = AAUDIO_STREAM_STATE_STOPPED;
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_waitForStateChange(
    AAudioStream* stream, aaudio_stream_state_t observed,
    aaudio_stream_state_t* next_state, int64_t) {
    if (stream->state != observed) {
        if (next_state != nullptr) {
            *next_state = stream->state;
        }
        return AAUDIO_OK;
    }
    return AAUDIO_ERROR_TIMEOUT;
}

aaudio_result_t AAudioStream_close(AAudioStream* stream) {
    delete stream;
    return AAUDIO_OK;
}

#include "../aaudio_clock.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void reset_fake() {
    g_scenario = FakeScenario{};
    g_current_audio = nullptr;
    g_fake_callback_invocations = 0;
}

itu_audio_clock::Report run_fallback_and_silence_contract() {
    reset_fake();
    g_scenario.actual_sharing_mode = AAUDIO_SHARING_MODE_SHARED;
    g_scenario.actual_performance_mode = AAUDIO_PERFORMANCE_MODE_NONE;
    g_scenario.buffer_size_extra = 64;
    g_scenario.callback_target = 8;

    int main_poll_calls = 0;
    int started_hook_calls = 0;
    int ended_hook_calls = 0;
    std::atomic<int32_t> callback_tid{0};
    std::array<float, 4> algorithm_pcm_sentinel{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> sentinel_before = algorithm_pcm_sentinel;
    bool all_output_zero = true;
    int64_t first_started = 0;
    int64_t first_expected = 0;

    itu_audio_clock::Config config;
    config.on_main_poll = [&] { ++main_poll_calls; };
    config.on_callback_thread = [&](int32_t tid) {
        callback_tid.store(tid, std::memory_order_release);
    };
    config.on_callback_started = [&] { ++started_hook_calls; };
    config.on_callback_ended = [&] { ++ended_hook_calls; };

    itu_audio_clock::Runner runner(
        config,
        [&](size_t index, int64_t release_expected_ns,
            int64_t callback_started_ns) {
            if (index == 0) {
                first_started = callback_started_ns;
                first_expected = release_expected_ns;
            }
            if (g_current_audio == nullptr) {
                all_output_zero = false;
            } else {
                for (size_t i = 0; i < 768; ++i) {
                    if (g_current_audio[i] != 0.0f) {
                        all_output_zero = false;
                        break;
                    }
                }
            }
            return index < 2;
        });
    itu_audio_clock::Report report = runner.run(0.1);

    require(report.api_available, "Android fake path must be available");
    require(report.setup_ok, "setup must pass");
    require(report.stream_started, "stream must start");
    require(report.done, "run must finish");
    require(report.completed, "algorithm false return must complete run");
    require(report.callback_returned_false, "false callback must be recorded");
    require(!report.stream_error, "no stream error expected");
    require(!report.callback_exception, "no callback exception expected");
    require(report.sharing_fallback, "shared fallback must be explicit");
    require(report.performance_fallback, "performance fallback must be explicit");
    require(!report.requested_modes_met, "fallback must not claim request met");
    require(!report.low_latency_guaranteed, "low latency is not guaranteed");
    require(report.geometry_ok, "open geometry must match");
    require(report.callback_geometry_ok, "callback geometry must match");
    require(report.callback_count == 3, "false at index 2 must yield 3 callbacks");
    require(all_output_zero, "AAudio buffer must be zero-filled before callback");
    require(algorithm_pcm_sentinel == sentinel_before,
            "harness must not overwrite the user algorithm PCM");
    require(report.callback_cadence_timebase_established,
            "first callback must establish callback cadence timebase");
    require(!report.hardware_presentation_time_available,
            "callback cadence is not a hardware presentation timestamp");
    require(first_started == report.first_callback_started_ns,
            "reported first callback time must match callback argument");
    require(first_expected == report.first_release_expected_ns,
            "first release expectation must equal first callback time");
    require(report.last_release_expected_ns - report.first_release_expected_ns ==
                2 * itu_audio_clock::kCallbackHopNanoseconds,
            "callback indices must advance by 16 ms");
    require(report.request_buffer_size_frames ==
                report.actual_frames_per_burst * 2,
            "buffer target must be 2 times actual burst");
    require(report.actual_buffer_size_frames ==
                report.request_buffer_size_frames + 64,
            "actual buffer return must be reported without pretending exactness");
    require(report.buffer_set_result == report.actual_buffer_size_frames,
            "AAudio setBuffer return must be preserved");
    require(main_poll_calls == report.main_poll_count && main_poll_calls > 0,
            "main poll hook must run on the polling thread");
    require(callback_tid.load(std::memory_order_acquire) != 0,
            "callback thread hook must receive a TID");
    require(report.callback_thread_hook_calls == 1,
            "unchanged callback TID must be registered once");
    require(started_hook_calls == 3 && ended_hook_calls == 3,
            "callback-local start/end hooks must bracket each algorithm call");
    require(report.callback_started_hook_calls == 3,
            "started hook count must be reported");
    require(report.callback_ended_hook_calls == 3,
            "ended hook count must be reported");
    require(report.stop_requested && report.stop_clean && report.closed,
            "callback completion must request stop, wait, and close");
    require(report.error == itu_audio_clock::RunError::None,
            "normal callback stop must not be an error");
    return report;
}

itu_audio_clock::Report run_rate_rejection() {
    reset_fake();
    g_scenario.actual_sample_rate = 44100;
    int callback_calls = 0;
    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [&](size_t, int64_t, int64_t) {
            ++callback_calls;
            return false;
        });
    itu_audio_clock::Report report = runner.run(0.1);
    require(!report.setup_ok, "wrong actual rate must reject setup");
    require(!report.stream_started, "rejected geometry must not start");
    require(report.error == itu_audio_clock::RunError::GeometryMismatch,
            "wrong rate must be an explicit geometry rejection");
    require(report.actual_sample_rate == 44100,
            "actual rejected rate must be reported");
    require(report.callback_count == 0 && callback_calls == 0,
            "callback must not run after geometry rejection");
    require(report.closed && report.callback_quiesced,
            "rejected stream must close without a callback race");
    return report;
}

itu_audio_clock::Report run_callback_geometry_rejection() {
    reset_fake();
    g_scenario.callback_frames_override = 512;
    int callback_calls = 0;
    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [&](size_t, int64_t, int64_t) {
            ++callback_calls;
            return true;
        });
    itu_audio_clock::Report report = runner.run(0.1);
    require(report.setup_ok, "open query geometry is valid");
    require(report.stream_started, "stream starts before runtime geometry check");
    require(!report.callback_geometry_ok,
            "runtime callback geometry mismatch must be recorded");
    require(report.geometry_errors == 1,
            "runtime callback geometry mismatch count must be one");
    require(report.error == itu_audio_clock::RunError::CallbackGeometryMismatch,
            "runtime callback geometry mismatch must stop with explicit error");
    require(report.callback_count == 1 && callback_calls == 0,
            "algorithm callback must not run on runtime geometry mismatch");
    require(report.stop_requested && report.closed,
            "runtime geometry failure must stop and close");
    return report;
}

itu_audio_clock::Report run_stream_error() {
    reset_fake();
    g_scenario.callback_target = 1;
    g_scenario.error_after_callbacks = true;
    int callback_calls = 0;
    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [&](size_t, int64_t, int64_t) {
            ++callback_calls;
            return true;
        });
    itu_audio_clock::Report report = runner.run(0.1);
    require(report.stream_error, "AAudio error callback must be recorded");
    require(!report.completed,
            "stream error must not be treated as algorithm completion");
    require(report.error == itu_audio_clock::RunError::StreamError,
            "stream error must have a distinct error code");
    require(report.stream_error_code == AAUDIO_ERROR_DISCONNECTED,
            "exact AAudio error code must be preserved");
    require(report.callback_count == 1 && callback_calls == 1,
            "error must arrive after one callback");
    require(report.stop_requested && report.closed,
            "stream error path must request stop and close");
    return report;
}

itu_audio_clock::Report run_callback_exception() {
    reset_fake();
    g_scenario.callback_target = 4;
    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [&](size_t index, int64_t, int64_t) -> bool {
            if (index == 1) {
                throw std::runtime_error("contract exception");
            }
            return true;
        });
    itu_audio_clock::Report report = runner.run(0.1);
    require(report.callback_exception,
            "callback exception must set a callback-local flag");
    require(!report.completed,
            "callback exception must not be reported as completion");
    require(report.error == itu_audio_clock::RunError::CallbackException,
            "callback exception must have an explicit error code");
    require(report.callback_count == 2,
            "exception at index one must stop after two callback entries");
    require(report.stop_requested && report.closed,
            "exception path must stop and close");
    return report;
}

}  // namespace

int main() {
    try {
        std::vector<std::pair<std::string, itu_audio_clock::Report>> reports;
        reports.emplace_back(
            "fallback_silence_endstop", run_fallback_and_silence_contract());
        reports.emplace_back("rate_reject", run_rate_rejection());
        reports.emplace_back(
            "callback_geometry_reject", run_callback_geometry_rejection());
        reports.emplace_back("stream_error", run_stream_error());
        reports.emplace_back("callback_exception", run_callback_exception());

        for (const auto& entry : reports) {
            std::cout << "JSON " << entry.first << ' '
                      << entry.second.to_json() << '\n';
        }
        std::cout << "RESULT PASS cases=" << reports.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RESULT FAIL: " << error.what() << '\n';
        return 1;
    }
}
