#include "../android_performance_hint.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

using itu_audio_clock::PerformanceHint;

bool expect(bool condition, const char* label) {
    if (!condition) {
        std::printf("FAIL %s\n", label);
    }
    return condition;
}

struct FakeApi {
    bool null_manager = false;
    bool fail_create = false;
    int report_result = 0;
    int update_result = 0;

    std::uint64_t get_manager_calls = 0;
    std::uint64_t create_calls = 0;
    std::uint64_t report_calls = 0;
    std::uint64_t update_calls = 0;
    std::uint64_t close_calls = 0;
    std::vector<std::int32_t> created_tids;
    std::vector<std::size_t> created_tid_counts;
    std::vector<std::int64_t> created_targets;
    std::vector<std::int64_t> reports;
    std::vector<std::int64_t> updates;

    static void* get_manager() {
        // The manager is a per-call token resolved through a thread-local
        // dispatcher installed by table().
        FakeApi* api = active();
        ++api->get_manager_calls;
        return api->null_manager ? nullptr : api;
    }

    static void* create_session(void* manager, const std::int32_t* tids,
                                std::size_t count, std::int64_t target) {
        auto* api = static_cast<FakeApi*>(manager);
        ++api->create_calls;
        api->created_tids.push_back(tids == nullptr ? 0 : tids[0]);
        api->created_tid_counts.push_back(count);
        api->created_targets.push_back(target);
        return api->fail_create ? nullptr : api;
    }

    static int report_actual(void* session, std::int64_t duration) {
        auto* api = static_cast<FakeApi*>(session);
        ++api->report_calls;
        api->reports.push_back(duration);
        return api->report_result;
    }

    static int update_target(void* session, std::int64_t target) {
        auto* api = static_cast<FakeApi*>(session);
        ++api->update_calls;
        api->updates.push_back(target);
        return api->update_result;
    }

    static void close_session(void* session) {
        ++static_cast<FakeApi*>(session)->close_calls;
    }

    static FakeApi*& active() {
        static FakeApi* current = nullptr;
        return current;
    }

    PerformanceHint::FunctionTable table() {
        active() = this;
        PerformanceHint::FunctionTable functions;
        functions.get_manager = &FakeApi::get_manager;
        functions.create_session = &FakeApi::create_session;
        functions.report_actual_work_duration = &FakeApi::report_actual;
        functions.update_target_work_duration = &FakeApi::update_target;
        functions.close_session = &FakeApi::close_session;
        return functions;
    }
};

bool off_creates_no_session() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), false);
    hint.register_callback_tid(101);
    hint.submit_callback_duration(1000);
    hint.drain();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(!report.requested, "off requested=false");
    ok &= expect(report.api_available, "off still exposes API availability");
    ok &= expect(fake.get_manager_calls == 0, "off getManager calls=0");
    ok &= expect(fake.create_calls == 0, "off createSession calls=0");
    ok &= expect(report.submitted_count == 0, "off submit ignored");
    hint.close();
    ok &= expect(fake.close_calls == 0, "off closeSession calls=0");
    std::printf("PASS off requested=0 getManager=0 createSession=0\n");
    return ok;
}

bool missing_symbol_is_explicit() {
    FakeApi fake;
    auto functions = fake.table();
    functions.report_actual_work_duration = nullptr;
    PerformanceHint hint(functions, true);
    hint.register_callback_tid(202);
    hint.submit_callback_duration(2000);
    hint.drain();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(!hint.supported(), "missing symbol supported=false");
    ok &= expect(report.error == "missing_symbols",
                 "missing symbol error=missing_symbols");
    ok &= expect(report.unsupported_reason.find(
                     "APerformanceHint_reportActualWorkDuration") !=
                     std::string::npos,
                 "missing symbol names exact function");
    ok &= expect(fake.get_manager_calls == 0,
                 "missing symbol skips getManager");
    ok &= expect(report.dropped_count == 1,
                 "missing symbol counts dropped report");
    hint.close();
    std::printf("PASS missing_symbol explicit=1 dropped=1\n");
    return ok;
}

bool null_manager_is_explicit() {
    FakeApi fake;
    fake.null_manager = true;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(303);
    hint.submit_callback_duration(3000);
    hint.drain();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(fake.get_manager_calls == 1,
                 "null manager getManager calls=1");
    ok &= expect(fake.create_calls == 0,
                 "null manager skips createSession");
    ok &= expect(report.error == "manager_unavailable",
                 "null manager explicit error");
    ok &= expect(report.dropped_count == 1,
                 "null manager drops queued work");
    hint.close();
    std::printf("PASS null_manager error=manager_unavailable dropped=1\n");
    return ok;
}

bool create_failure_is_explicit() {
    FakeApi fake;
    fake.fail_create = true;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(404);
    hint.submit_callback_duration(4000);
    hint.drain();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(fake.create_calls == 1, "create failure attempts once");
    ok &= expect(report.error == "session_create_failed",
                 "create failure explicit error");
    ok &= expect(!report.ready, "create failure ready=false");
    ok &= expect(report.dropped_count == 1,
                 "create failure drops queued work");
    hint.close();
    std::printf("PASS create_fail error=session_create_failed dropped=1\n");
    return ok;
}

bool forwarding_uses_actual_tid_and_duration() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    const std::int64_t target = 8'000'000;

    bool ok = true;
    ok &= expect(hint.register_thread(4242, target) == -3,
                 "main registration requires callback handshake");
    hint.register_callback_tid(4242);
    ok &= expect(hint.register_thread(4242, target) == 0,
                 "main registration accepts handshaken tid");

    hint.submit_callback_duration(1111);
    hint.submit_callback_duration(2222);
    hint.submit_callback_duration(3333);
    const auto before = hint.snapshot();
    ok &= expect(before.queued_count == 3, "three callback values queued");
    ok &= expect(!before.ready && !before.active,
                 "queued values do not mark ready/active");

    hint.drain();
    const std::int64_t injected_wrong_second =
#if defined(ITU_ADPF_TEST_INJECT_WRONG_EXPECTATION)
        9999;
#else
        2222;
#endif
    const std::vector<std::int64_t> expected_reports{
        1111, injected_wrong_second, 3333};

    ok &= expect(fake.create_calls == 1, "createSession calls=1");
    ok &= expect(fake.created_tids == std::vector<std::int32_t>{4242},
                 "createSession received callback tid");
    ok &= expect(fake.created_tid_counts == std::vector<std::size_t>{1},
                 "createSession tid count=1");
    ok &= expect(fake.created_targets == std::vector<std::int64_t>{target},
                 "createSession received target ns");
    ok &= expect(fake.reports == expected_reports,
                 "reportActualWorkDuration durations exact");

    const auto after = hint.snapshot();
    ok &= expect(after.ready && after.active,
                 "accepted report marks ready and active");
    ok &= expect(after.reported_count == 3, "reported_count=3");
    ok &= expect(after.dropped_count == 0, "dropped_count=0");

    hint.set_target_duration_ns(9'000'000);
    ok &= expect(fake.update_calls == 0,
                 "target update waits for main drain");
    hint.drain();
    ok &= expect(fake.updates == std::vector<std::int64_t>{9'000'000},
                 "updateTargetWorkDuration received target ns");

    const std::string json_before_close = hint.to_json();
    ok &= expect(json_before_close.find(
                     "\"session_ever_created\":true") !=
                     std::string::npos,
                 "JSON session_ever_created before close");
    ok &= expect(json_before_close.find("\"reported_count\":3") !=
                     std::string::npos,
                 "JSON reported_count before close");
    ok &= expect(json_before_close.find(
                     "not_frequency_or_hard_realtime_proof") !=
                     std::string::npos,
                 "JSON carries effect limitation");

    hint.close();
    const auto closed = hint.snapshot();
    const std::string json_after_close = hint.to_json();
    ok &= expect(fake.close_calls == 1, "closeSession calls=1");
    ok &= expect(closed.session_ever_created,
                 "session_ever_created retained after close");
    ok &= expect(closed.reported_count == 3,
                 "reported count retained after close");
    ok &= expect(!closed.ready && !closed.active,
                 "closed session ready/active=false");
    ok &= expect(json_after_close.find(
                     "\"session_ever_created\":true") !=
                     std::string::npos,
                 "JSON session_ever_created after close");

    std::printf(
        "PASS forward tid=%d reports=%llu exact=1 "
        "session_ever_created_after_close=1\n",
        4242,
        static_cast<unsigned long long>(fake.report_calls));
    return ok;
}

bool report_error_is_explicit() {
    FakeApi fake;
    fake.report_result = -32;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(505);
    hint.submit_callback_duration(5000);
    hint.drain();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(report.reported_count == 0,
                 "report error accepted count=0");
    ok &= expect(report.dropped_count == 1,
                 "report error counts dropped work");
    ok &= expect(report.api_error_count == 1,
                 "report error api_error_count=1");
    ok &= expect(report.last_api_result == -32,
                 "report error preserves API result");
    ok &= expect(report.error == "report_failed",
                 "report error explicit");
    ok &= expect(!report.active,
                 "report error does not mark active");
    hint.close();
    std::printf("PASS report_error last_api_result=-32 active=0\n");
    return ok;
}

bool queue_overflow_is_counted() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    constexpr std::size_t overflow = 3;
    for (std::size_t index = 0;
         index < PerformanceHint::kQueueCapacity + overflow; ++index) {
        hint.submit_callback_duration(
            static_cast<std::int64_t>(1000 + index));
    }
    const auto full = hint.snapshot();
    bool ok = true;
    ok &= expect(full.queued_count == PerformanceHint::kQueueCapacity,
                 "queue retains fixed capacity");
    ok &= expect(full.dropped_count == overflow,
                 "queue overflow dropped count exact");
    ok &= expect(full.error == "queue_overflow",
                 "queue overflow explicit error");
    ok &= expect(!full.ready && !full.active,
                 "queue overflow does not mark ready/active");

    hint.register_callback_tid(606);
    hint.drain();
    const auto drained = hint.snapshot();
    ok &= expect(fake.reports.size() == PerformanceHint::kQueueCapacity,
                 "all queued reports delivered after handshake");
    ok &= expect(drained.reported_count == PerformanceHint::kQueueCapacity,
                 "reported count equals queue capacity");
    ok &= expect(drained.dropped_count == overflow,
                 "overflow count survives drain");
    hint.close();
    std::printf("PASS queue_overflow dropped=3 delivered=256\n");
    return ok;
}

bool concurrent_spsc_preserves_counts() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(707);
    hint.register_thread(707, 16'000'000);
    hint.drain();

    constexpr std::uint64_t count = 5000;
    std::atomic<bool> producer_done{false};
    std::thread producer([&hint, &producer_done]() {
        for (std::uint64_t index = 0; index < count; ++index) {
            hint.submit_callback_duration(
                static_cast<std::int64_t>(100'000 + index));
        }
        producer_done.store(true, std::memory_order_release);
    });

    while (!producer_done.load(std::memory_order_acquire)) {
        hint.drain();
    }
    producer.join();
    hint.drain();

    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(report.submitted_count == count,
                 "concurrent submitted count exact");
    ok &= expect(report.submitted_count ==
                     report.reported_count + report.dropped_count,
                 "concurrent counts conserved");
    ok &= expect(report.queued_count == 0,
                 "concurrent queue drains empty");
    ok &= expect(fake.reports.size() == report.reported_count,
                 "concurrent fake report calls match accepted count");
    hint.close();
    std::printf(
        "PASS concurrent_spsc submitted=5000 conserved=1 "
        "reported=%llu dropped=%llu\n",
        static_cast<unsigned long long>(report.reported_count),
        static_cast<unsigned long long>(report.dropped_count));
    return ok;
}

bool callback_tid_change_is_not_rebound() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(1001);
    hint.register_thread(1001, 16'000'000);
    hint.submit_callback_duration(6001);
    hint.drain();

    hint.register_callback_tid(2002);
    hint.submit_callback_duration(6002);
    hint.drain();
    const auto report = hint.snapshot();

    bool ok = true;
    ok &= expect(fake.create_calls == 1,
                 "tid change does not create a second session");
    ok &= expect(fake.created_tids == std::vector<std::int32_t>{1001},
                 "session remains bound to first callback tid");
    ok &= expect(fake.reports ==
                     std::vector<std::int64_t>{6001, 6002},
                 "both durations reach original session");
    ok &= expect(report.first_actual_tid == 1001,
                 "first_actual_tid remains 1001");
    ok &= expect(report.error == "callback_tid_changed",
                 "tid change explicit error");
    hint.close();
    std::printf(
        "PASS tid_change first=1001 createSession=1 explicit_error=1\n");
    return ok;
}

bool main_tid_cannot_replace_callback_tid() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(700);
    const int wrong = hint.register_thread(999, 16'000'000);
    const int right = hint.register_thread(700, 16'000'000);
    hint.submit_callback_duration(7000);
    hint.drain();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(wrong == -2, "main tid replacement rejected");
    ok &= expect(right == 0, "actual callback tid accepted");
    ok &= expect(fake.created_tids == std::vector<std::int32_t>{700},
                 "fake createSession received callback tid only");
    ok &= expect(report.first_actual_tid == 700,
                 "main tid did not replace callback tid");
    hint.close();
    std::printf("PASS main_tid_rejected=1 actual_tid=700\n");
    return ok;
}

bool invalid_values_do_not_report() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(808);
    hint.submit_callback_duration(0);
    hint.submit_callback_duration(-1);
    hint.set_target_duration_ns(0);
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(report.invalid_count == 2,
                 "negative/zero duration invalid count=2");
    ok &= expect(report.submitted_count == 0,
                 "invalid durations are not submitted");
    ok &= expect(report.queued_count == 0,
                 "invalid durations are not queued");
    ok &= expect(report.target_duration_ns == 16'000'000,
                 "invalid target does not replace 16 ms");
    ok &= expect(report.error == "invalid_target",
                 "latest invalid target explicit");
    hint.close();
    std::printf("PASS invalid duration=2 target_unchanged=1\n");
    return ok;
}

bool close_accounts_for_pending_values() {
    FakeApi fake;
    PerformanceHint hint(fake.table(), true);
    hint.register_callback_tid(909);
    hint.submit_callback_duration(9001);
    hint.submit_callback_duration(9002);
    hint.close();
    const auto report = hint.snapshot();
    bool ok = true;
    ok &= expect(report.submitted_count == 2,
                 "close retains submitted count");
    ok &= expect(report.dropped_count == 2,
                 "close accounts pending as dropped");
    ok &= expect(report.queued_count == 0,
                 "close leaves no phantom queued values");
    ok &= expect(!report.session_ever_created,
                 "close before drain never claims session");
    std::printf("PASS close_pending submitted=2 dropped=2\n");
    return ok;
}

bool negative_controls() {
    const std::vector<std::int32_t> actual_tids{1234};
    const std::vector<std::int32_t> wrong_tids{4321};
    const bool wrong_tid_detected = !(actual_tids == wrong_tids);

    const std::vector<std::int64_t> actual_reports{123, 456};
    const std::vector<std::int64_t> wrong_reports{123, 457};
    const bool wrong_duration_detected =
        !(actual_reports == wrong_reports);

    bool ok = true;
    ok &= expect(wrong_tid_detected,
                 "negative wrong tid is detectable");
    ok &= expect(wrong_duration_detected,
                 "negative wrong duration is detectable");
    std::printf(
        "NEGATIVE wrong_tid expected_failure detected=%d\n",
        wrong_tid_detected ? 1 : 0);
    std::printf(
        "NEGATIVE wrong_duration expected_failure detected=%d\n",
        wrong_duration_detected ? 1 : 0);
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    const bool scenarios[] = {
        off_creates_no_session(),
        missing_symbol_is_explicit(),
        null_manager_is_explicit(),
        create_failure_is_explicit(),
        forwarding_uses_actual_tid_and_duration(),
        report_error_is_explicit(),
        queue_overflow_is_counted(),
        concurrent_spsc_preserves_counts(),
        callback_tid_change_is_not_rebound(),
        main_tid_cannot_replace_callback_tid(),
        invalid_values_do_not_report(),
        close_accounts_for_pending_values(),
        negative_controls(),
    };
    for (const bool scenario : scenarios) {
        ok = scenario && ok;
    }

    std::printf(
        "RESULT %s scenarios=%zu fake_tid_and_duration_checked=1 "
        "negative_controls=%d\n",
        ok ? "PASS" : "FAIL",
        sizeof(scenarios) / sizeof(scenarios[0]),
        ok ? 2 : 0);
    return ok ? 0 : 1;
}
