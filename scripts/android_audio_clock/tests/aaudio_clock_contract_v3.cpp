#define AAUDIO_AAUDIO_H

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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
inline constexpr int32_t AAUDIO_ERROR_INTERNAL = -895;
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
void AAudioStreamBuilder_setDirection(
    AAudioStreamBuilder*, int32_t direction);
void AAudioStreamBuilder_setSampleRate(AAudioStreamBuilder*, int32_t rate);
void AAudioStreamBuilder_setChannelCount(
    AAudioStreamBuilder*, int32_t channels);
void AAudioStreamBuilder_setFormat(
    AAudioStreamBuilder*, aaudio_format_t format);
void AAudioStreamBuilder_setFramesPerDataCallback(
    AAudioStreamBuilder*, int32_t frames);
void AAudioStreamBuilder_setPerformanceMode(
    AAudioStreamBuilder*, aaudio_performance_mode_t mode);
void AAudioStreamBuilder_setSharingMode(
    AAudioStreamBuilder*, aaudio_sharing_mode_t mode);
void AAudioStreamBuilder_setDataCallback(
    AAudioStreamBuilder*, AAudioStream_dataCallback callback, void* user_data);
void AAudioStreamBuilder_setErrorCallback(
    AAudioStreamBuilder*, AAudioStream_errorCallback callback, void* user_data);
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
int32_t AAudioStream_getXRunCount(AAudioStream*);

struct FakeScenario {
    int32_t actual_sample_rate = 48000;
    int32_t actual_channel_count = 1;
    int32_t actual_format = AAUDIO_FORMAT_PCM_FLOAT;
    int32_t actual_performance_mode =
        AAUDIO_PERFORMANCE_MODE_LOW_LATENCY;
    int32_t actual_sharing_mode = AAUDIO_SHARING_MODE_EXCLUSIVE;
    int32_t actual_frames_per_burst = 192;
    int32_t actual_frames_per_callback = 768;
    int32_t callback_target = 3;
    int32_t xrun_before = 3;
    int32_t xrun_after = 3;
    aaudio_result_t request_stop_result = AAUDIO_OK;
    aaudio_stream_state_t state_after_stop = AAUDIO_STREAM_STATE_STOPPED;
    aaudio_result_t close_result = AAUDIO_OK;
    bool wait_for_state_timeout = true;
    aaudio_stream_state_t wait_next_state = AAUDIO_STREAM_STATE_STARTED;
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
    aaudio_stream_state_t state = AAUDIO_STREAM_STATE_OPEN;
    int32_t buffer_size = 0;
    int32_t buffer_capacity = 4096;
    std::array<float, 768> pcm{};
};

struct FakeRuntime {
    FakeScenario scenario{};
    int callback_invocations = 0;
    int request_stop_calls = 0;
    int wait_for_state_calls = 0;
    int close_calls = 0;
    int xrun_reads = 0;
};

FakeRuntime g_fake;

void reset_fake() {
    g_fake = FakeRuntime{};
}

aaudio_result_t AAudio_createStreamBuilder(AAudioStreamBuilder** builder) {
    if (builder == nullptr) {
        return AAUDIO_ERROR_INTERNAL;
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
        return AAUDIO_ERROR_INTERNAL;
    }
    auto* opened = new AAudioStreamStruct();
    opened->settings = *builder;
    opened->buffer_size = g_fake.scenario.actual_frames_per_burst * 2;
    opened->pcm.fill(123.5f);
    *stream = opened;
    return AAUDIO_OK;
}

aaudio_result_t AAudioStreamBuilder_delete(AAudioStreamBuilder* builder) {
    delete builder;
    return AAUDIO_OK;
}

int32_t AAudioStream_getSampleRate(AAudioStream*) {
    return g_fake.scenario.actual_sample_rate;
}

int32_t AAudioStream_getChannelCount(AAudioStream*) {
    return g_fake.scenario.actual_channel_count;
}

aaudio_format_t AAudioStream_getFormat(AAudioStream*) {
    return g_fake.scenario.actual_format;
}

aaudio_performance_mode_t AAudioStream_getPerformanceMode(AAudioStream*) {
    return g_fake.scenario.actual_performance_mode;
}

aaudio_sharing_mode_t AAudioStream_getSharingMode(AAudioStream*) {
    return g_fake.scenario.actual_sharing_mode;
}

int32_t AAudioStream_getFramesPerBurst(AAudioStream*) {
    return g_fake.scenario.actual_frames_per_burst;
}

int32_t AAudioStream_getFramesPerDataCallback(AAudioStream*) {
    return g_fake.scenario.actual_frames_per_callback;
}

int32_t AAudioStream_getBufferCapacityInFrames(AAudioStream* stream) {
    return stream->buffer_capacity;
}

int32_t AAudioStream_getBufferSizeInFrames(AAudioStream* stream) {
    return stream->buffer_size;
}

aaudio_result_t AAudioStream_setBufferSizeInFrames(
    AAudioStream* stream, int32_t frames) {
    stream->buffer_size = frames;
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
    if (stream->settings.data_callback == nullptr) {
        return AAUDIO_OK;
    }
    for (int32_t index = 0;
         index < g_fake.scenario.callback_target; ++index) {
        ++g_fake.callback_invocations;
        const aaudio_data_callback_result_t result =
            stream->settings.data_callback(
                stream, stream->settings.user_data, stream->pcm.data(),
                g_fake.scenario.actual_frames_per_callback);
        if (result == AAUDIO_CALLBACK_RESULT_STOP) {
            break;
        }
    }
    stream->state = AAUDIO_STREAM_STATE_STARTED;
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_requestStop(AAudioStream* stream) {
    ++g_fake.request_stop_calls;
    stream->state = g_fake.scenario.state_after_stop;
    return g_fake.scenario.request_stop_result;
}

aaudio_result_t AAudioStream_waitForStateChange(
    AAudioStream* stream, aaudio_stream_state_t observed,
    aaudio_stream_state_t* next_state, int64_t timeout_nanoseconds) {
    ++g_fake.wait_for_state_calls;
    if (g_fake.scenario.wait_for_state_timeout) {
        (void)stream;
        (void)observed;
        (void)next_state;
        (void)timeout_nanoseconds;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return AAUDIO_ERROR_TIMEOUT;
    }
    if (next_state != nullptr) {
        *next_state = g_fake.scenario.wait_next_state;
    }
    return AAUDIO_OK;
}

aaudio_result_t AAudioStream_close(AAudioStream* stream) {
    ++g_fake.close_calls;
    const aaudio_result_t result = g_fake.scenario.close_result;
    if (result == AAUDIO_OK) {
        delete stream;
    }
    return result;
}

int32_t AAudioStream_getXRunCount(AAudioStream* stream) {
    (void)stream;
    const int read_index = g_fake.xrun_reads++;
    return read_index == 0 ? g_fake.scenario.xrun_before
                           : g_fake.scenario.xrun_after;
}

#include "../aaudio_clock_v3.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_json_fragment(
    const std::string& json, const std::string& fragment) {
    require(
        json.find(fragment) != std::string::npos,
        "JSON missing fragment: " + fragment + " in " + json);
}

void require_error(
    const itu_audio_clock::Report& report,
    itu_audio_clock::RunError expected) {
    require(
        report.error == expected,
        std::string("expected error ") +
            itu_audio_clock::to_string(expected) + ", got " +
            itu_audio_clock::to_string(report.error) + " (" +
            report.error_detail + ")");
}

itu_audio_clock::Report run_normal_xrun_contract() {
    reset_fake();
    g_fake.scenario.callback_target = 4;
    g_fake.scenario.xrun_before = 4;
    g_fake.scenario.xrun_after = 7;

    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [](size_t index, int64_t, int64_t) {
            return index < 2;
        });
    itu_audio_clock::Report report = runner.run(0.05);

    require(report.api_available, "fake Android path must be available");
    require(report.setup_ok, "setup must pass");
    require(report.stream_started, "stream must start");
    require(report.completed, "algorithm false return must complete");
    require(report.callback_returned_false, "false return must be recorded");
    require(report.stop_requested, "completion must request stop");
    require(report.callback_quiesced, "terminal state must quiesce callbacks");
    require(report.closed, "normal path must close");
    require(report.stop_clean, "normal path must have a clean stop");
    require(!report.forced_close_after_stop_timeout,
            "normal path must not force close");
    require(report.callback_count == 3,
            "false at callback index two must yield three callbacks");
    require(report.xrun_count_before == 4,
            "before xrun count must be preserved");
    require(report.xrun_count_after == 7,
            "after xrun count must be preserved");
    require(report.xrun_count_before >= 0 && report.xrun_count_after >= 0,
            "xrun counts must be non-negative");
    require(report.xrun_delta ==
                report.xrun_count_after - report.xrun_count_before,
            "xrun delta must equal after minus before");
    require(report.xrun_delta >= 0, "xrun delta must be non-negative");
    require(g_fake.request_stop_calls == 1,
            "normal path must call requestStop once");
    require(g_fake.close_calls == 1, "normal path must call close once");
    require_error(report, itu_audio_clock::RunError::None);
    return report;
}

itu_audio_clock::Report run_stop_request_failure_with_state(
    aaudio_stream_state_t terminal_state) {
    reset_fake();
    g_fake.scenario.callback_target = 3;
    g_fake.scenario.xrun_before = 6;
    g_fake.scenario.xrun_after = 6;
    g_fake.scenario.request_stop_result = AAUDIO_ERROR_INTERNAL;
    g_fake.scenario.state_after_stop = terminal_state;

    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [](size_t index, int64_t, int64_t) {
            return index < 2;
        });
    itu_audio_clock::Report report = runner.run(0.05);

    require(report.completed, "callback completion must be recorded");
    require(report.stop_requested, "stop must still be requested");
    require(report.stop_request_result != AAUDIO_OK,
            "failed requestStop result must be preserved");
    require(report.final_state == terminal_state,
            "terminal state must be preserved");
    require(report.callback_quiesced,
            "terminal state must quiesce callbacks");
    require(report.closed, "close may succeed after requestStop failure");
    require(!report.stop_clean,
            "requestStop failure must make stop_clean false");
    require(g_fake.close_calls == 1,
            "terminal requestStop failure must still close once");
    require_error(report, itu_audio_clock::RunError::StopRequestFailed);
    return report;
}

itu_audio_clock::Report run_stop_request_failure_contract() {
    const itu_audio_clock::Report stopped =
        run_stop_request_failure_with_state(AAUDIO_STREAM_STATE_STOPPED);
    const itu_audio_clock::Report closed =
        run_stop_request_failure_with_state(AAUDIO_STREAM_STATE_CLOSED);
    require(stopped.final_state == AAUDIO_STREAM_STATE_STOPPED,
            "STOPPED subcase must preserve STOPPED");
    require(closed.final_state == AAUDIO_STREAM_STATE_CLOSED,
            "CLOSED subcase must preserve CLOSED");
    return stopped;
}

itu_audio_clock::Report run_close_failure_contract() {
    reset_fake();
    g_fake.scenario.callback_target = 3;
    g_fake.scenario.xrun_before = 2;
    g_fake.scenario.xrun_after = 2;
    g_fake.scenario.request_stop_result = AAUDIO_OK;
    g_fake.scenario.state_after_stop = AAUDIO_STREAM_STATE_STOPPED;
    g_fake.scenario.close_result = AAUDIO_ERROR_INTERNAL;

    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [](size_t index, int64_t, int64_t) {
            return index < 2;
        });
    itu_audio_clock::Report report = runner.run(0.05);

    require(report.stop_requested, "close-failure path must request stop");
    require(report.stop_request_result == AAUDIO_OK,
            "close-failure setup requires requestStop success");
    require(report.callback_quiesced,
            "close must be attempted only after quiescence");
    require(report.close_result != AAUDIO_OK,
            "failed close result must be preserved");
    require(!report.closed, "failed close must not report closed");
    require(!report.stop_clean,
            "failed close must make stop_clean false");
    require(g_fake.close_calls == 1,
            "quiesced stream must attempt close exactly once");
    require_error(report, itu_audio_clock::RunError::CloseFailed);
    return report;
}

itu_audio_clock::Report run_stop_wait_failure_contract() {
    reset_fake();
    g_fake.scenario.callback_target = 3;
    g_fake.scenario.xrun_before = 0;
    g_fake.scenario.xrun_after = 0;
    g_fake.scenario.request_stop_result = AAUDIO_OK;
    g_fake.scenario.state_after_stop = AAUDIO_STREAM_STATE_STARTED;
    g_fake.scenario.wait_for_state_timeout = true;

    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [](size_t index, int64_t, int64_t) {
            return index < 2;
        });
    itu_audio_clock::Report report = runner.run(0.05);

    require(report.completed, "callback completion must be recorded");
    require(report.stop_requested, "stop must be requested");
    require(report.stop_request_result == AAUDIO_OK,
            "timeout subcase requires requestStop success");
    require(report.final_state == AAUDIO_STREAM_STATE_STARTED,
            "non-terminal stream state must be preserved");
    require(!report.callback_quiesced,
            "non-terminal timeout must not claim quiescence");
    require(!report.closed,
            "non-terminal timeout must not report closed");
    require(!report.stop_clean,
            "non-terminal timeout must make stop_clean false");
    require(!report.forced_close_after_stop_timeout,
            "v3 must not force close after stop timeout");
    require(g_fake.wait_for_state_calls > 0,
            "stop wait must call waitForStateChange");
    require(report.wait_for_state_result == AAUDIO_ERROR_TIMEOUT,
            "timeout result must be preserved");
    require(g_fake.close_calls == 0,
            "close must not be called without quiescence");
    require_error(report, itu_audio_clock::RunError::StopWaitFailed);
    return report;
}

itu_audio_clock::Report run_normal_json_contract() {
    reset_fake();
    g_fake.scenario.callback_target = 2;
    g_fake.scenario.xrun_before = 9;
    g_fake.scenario.xrun_after = 11;

    itu_audio_clock::Runner runner(
        itu_audio_clock::Config{},
        [](size_t index, int64_t, int64_t) {
            return index < 1;
        });
    itu_audio_clock::Report report = runner.run(0.05);

    require(report.completed, "normal end-stop must complete");
    require(report.callback_returned_false,
            "normal end-stop must come from false callback return");
    require(report.stop_requested && report.closed && report.stop_clean,
            "normal end-stop must request, close, and stop cleanly");
    require(report.xrun_count_before == 9 &&
                report.xrun_count_after == 11 &&
                report.xrun_delta == 2,
            "normal JSON scenario must preserve xrun values");

    const std::string json = report.to_json();
    require_json_fragment(json, "\"completed\":true");
    require_json_fragment(json, "\"callback_returned_false\":true");
    require_json_fragment(json, "\"stop_requested\":true");
    require_json_fragment(json, "\"closed\":true");
    require_json_fragment(json, "\"stop_clean\":true");
    require_json_fragment(json, "\"callback_quiesced\":true");
    require_json_fragment(
        json, "\"forced_close_after_stop_timeout\":false");
    require_json_fragment(json, "\"xrun_count_before\":9");
    require_json_fragment(json, "\"xrun_count_after\":11");
    require_json_fragment(json, "\"xrun_delta\":2");
    require_json_fragment(json, "\"error\":\"none\"");
    return report;
}

}  // namespace

int main() {
    struct Case {
        const char* name;
        std::function<itu_audio_clock::Report()> run;
    };

    const std::vector<Case> cases{
        {"normal_xrun_delta", run_normal_xrun_contract},
        {"stop_request_failed_terminal", run_stop_request_failure_contract},
        {"close_failed_after_quiesce", run_close_failure_contract},
        {"stop_wait_timeout_no_close", run_stop_wait_failure_contract},
        {"normal_end_stop_json", run_normal_json_contract},
    };

    int failures = 0;
    for (const Case& test : cases) {
        try {
            const itu_audio_clock::Report report = test.run();
            std::cout << "JSON " << test.name << ' '
                      << report.to_json() << '\n';
            std::cout << "FAKE " << test.name
                      << " callback_invocations="
                      << g_fake.callback_invocations
                      << " request_stop_calls="
                      << g_fake.request_stop_calls
                      << " wait_for_state_calls="
                      << g_fake.wait_for_state_calls
                      << " close_calls=" << g_fake.close_calls
                      << " xrun_reads=" << g_fake.xrun_reads << '\n';
            std::cout << "SCENARIO " << test.name << " PASS\n";
        } catch (const std::exception& error) {
            std::cout << "SCENARIO " << test.name << " FAIL: "
                      << error.what() << '\n';
            ++failures;
        } catch (...) {
            std::cout << "SCENARIO " << test.name
                      << " FAIL: unknown exception\n";
            ++failures;
        }
    }

    if (failures != 0) {
        std::cout << "RESULT FAIL cases=" << cases.size()
                  << " failures=" << failures << '\n';
        return 1;
    }
    std::cout << "RESULT PASS cases=" << cases.size() << '\n';
    return 0;
}
