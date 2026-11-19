"""Analyze one completed AAudio/timer device trial without changing evidence.

The process exit code is only a transport signal. This script separately checks
the frozen DSP contract, the real AAudio stream contract, optional ADPF status,
and optional Perfetto receipt. It never converts a timer replay or silent output
callback into an AudioRecord/AudioTrack end-to-end claim.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def locate_collected(run_dir: Path, plan: dict) -> Path:
    root = run_dir / "collected"
    remote_name = Path(str(plan.get("remote", ""))).name
    candidate = root / remote_name
    return candidate if candidate.is_dir() else root


def find_one(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def read_int(root: Path, name: str) -> int | None:
    path = find_one(root, name)
    if path is None:
        return None
    text = path.read_text().strip()
    try:
        return int(text)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    plan_path = run_dir / "PLAN.json"
    if not plan_path.is_file():
        raise RuntimeError(f"PLAN.json missing: {plan_path}")
    plan = load_json(plan_path)
    collected = locate_collected(run_dir, plan)
    if not collected.is_dir():
        raise RuntimeError(f"collected directory missing: {collected}")

    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime(
        "%Y%m%dT%H%M%S%f"
    )
    out_dir = (args.out_dir or (run_dir / f"analysis-{stamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    checks: list[dict] = []

    def check(
        name: str,
        passed: bool,
        detail: str,
        *,
        critical: bool = True,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else ("FAIL" if critical else "WARN"),
                "critical": critical,
                "detail": detail,
            }
        )

    exitcode = read_int(collected, "exitcode")
    algorithm_exitcode = read_int(collected, "algorithm_exitcode")
    check("runner_exitcode", exitcode == 0, f"exitcode={exitcode}")
    check(
        "algorithm_exitcode",
        algorithm_exitcode == 0,
        f"algorithm_exitcode={algorithm_exitcode}",
    )

    clock_sidecar_path = find_one(collected, "out.audio-clock.json")
    failure_sidecar_path = find_one(
        collected, "out.audio-clock.failure.json"
    )
    algorithm_path = find_one(collected, "out.json")
    trace_path = find_one(collected, "out.audio-clock.frames.csv")

    clock_sidecar = (
        load_json(clock_sidecar_path) if clock_sidecar_path else None
    )
    failure_sidecar = (
        load_json(failure_sidecar_path) if failure_sidecar_path else None
    )
    algorithm = load_json(algorithm_path) if algorithm_path else None
    audio_stream = (
        clock_sidecar.get("audio_stream") if clock_sidecar else None
    )
    hint = clock_sidecar.get("performance_hint") if clock_sidecar else None

    mode = plan.get("mode")
    check(
        "success_report_present",
        algorithm is not None,
        str(algorithm_path) if algorithm_path else "missing out.json",
    )
    check(
        "failure_sidecar_absent_on_success",
        algorithm is None or failure_sidecar is None,
        str(failure_sidecar_path)
        if failure_sidecar_path
        else "no failure sidecar",
    )

    if mode == "aaudio":
        check(
            "clock_sidecar_present",
            clock_sidecar is not None,
            str(clock_sidecar_path)
            if clock_sidecar_path
            else "missing out.audio-clock.json",
        )
        check(
            "hardware_clock_executed",
            bool(
                clock_sidecar
                and clock_sidecar.get("hardware_audio_clock_executed")
            ),
            str(
                clock_sidecar.get("hardware_audio_clock_executed")
                if clock_sidecar
                else None
            ),
        )
        check(
            "mock_clock_disabled",
            bool(clock_sidecar and not clock_sidecar.get("mock_clock")),
            str(clock_sidecar.get("mock_clock") if clock_sidecar else None),
        )
        if isinstance(audio_stream, dict):
            for key in [
                "api_available",
                "setup_ok",
                "stream_started",
                "done",
                "completed",
                "callback_returned_false",
                "callback_quiesced",
                "closed",
                "stop_clean",
                "geometry_ok",
                "callback_geometry_ok",
                "silent_output",
                "full_algorithm_callback_contract",
            ]:
                check(
                    f"aaudio_{key}",
                    audio_stream.get(key) is True,
                    f"{key}={audio_stream.get(key)!r}",
                )
            for key in [
                "timed_out",
                "stream_error",
                "callback_exception",
                "main_poll_exception",
                "forced_close_after_stop_timeout",
            ]:
                check(
                    f"aaudio_{key}_false",
                    audio_stream.get(key) is False,
                    f"{key}={audio_stream.get(key)!r}",
                )
            check(
                "aaudio_error_none",
                audio_stream.get("error") == "none",
                f"error={audio_stream.get('error')!r} "
                f"detail={audio_stream.get('error_detail')!r}",
            )
            actual = {
                "sample_rate": audio_stream.get("actual_sample_rate"),
                "channels": audio_stream.get("actual_channel_count"),
                "format": audio_stream.get("actual_format"),
                "frames_per_callback": audio_stream.get(
                    "actual_frames_per_callback"
                ),
            }
            check(
                "aaudio_actual_geometry",
                actual
                == {
                    "sample_rate": 48000,
                    "channels": 1,
                    "format": 2,
                    "frames_per_callback": 768,
                },
                json.dumps(actual, sort_keys=True),
            )
            burst = audio_stream.get("actual_frames_per_burst")
            buffer_frames = audio_stream.get("actual_buffer_size_frames")
            check(
                "aaudio_positive_burst_buffer",
                isinstance(burst, int)
                and burst > 0
                and isinstance(buffer_frames, int)
                and buffer_frames > 0,
                f"burst={burst} buffer_frames={buffer_frames}",
            )
            check(
                "aaudio_callback_thread_observed",
                isinstance(audio_stream.get("callback_tid"), int)
                and audio_stream.get("callback_tid") > 0
                and audio_stream.get("callback_thread_hook_calls", 0) >= 1,
                f"tid={audio_stream.get('callback_tid')} "
                f"hook_calls={audio_stream.get('callback_thread_hook_calls')}",
            )
            before = audio_stream.get("xrun_count_before")
            after = audio_stream.get("xrun_count_after")
            delta = audio_stream.get("xrun_delta")
            if isinstance(before, int) and before >= 0:
                check(
                    "aaudio_xrun_delta",
                    isinstance(after, int)
                    and after >= before
                    and delta == after - before,
                    f"before={before} after={after} delta={delta}",
                )
            else:
                check(
                    "aaudio_xrun_available",
                    False,
                    f"before={before!r}; platform returned unavailable",
                    critical=False,
                )
            requested_modes_met = bool(
                audio_stream.get("requested_modes_met")
            )
            check(
                "requested_sharing_and_performance_modes",
                requested_modes_met,
                f"requested_sharing={audio_stream.get('request_sharing_mode_name')} "
                f"actual_sharing={audio_stream.get('actual_sharing_mode_name')} "
                f"requested_performance={audio_stream.get('request_performance_mode_name')} "
                f"actual_performance={audio_stream.get('actual_performance_mode_name')}",
                critical=False,
            )
        else:
            check(
                "aaudio_stream_report_present",
                False,
                repr(audio_stream),
            )

        if plan.get("hint_requested") == "on":
            if isinstance(hint, dict):
                check(
                    "adpf_api_available",
                    hint.get("api_available") is True,
                    f"api_available={hint.get('api_available')!r}",
                    critical=False,
                )
                check(
                    "adpf_session_created",
                    hint.get("session_ever_created") is True,
                    f"session_ever_created={hint.get('session_ever_created')!r}",
                    critical=False,
                )
                check(
                    "adpf_reports_accepted",
                    hint.get("reported_count", 0) > 0,
                    f"submitted={hint.get('submitted_count')} "
                    f"reported={hint.get('reported_count')} "
                    f"dropped={hint.get('dropped_count')} "
                    f"api_errors={hint.get('api_error_count')}",
                    critical=False,
                )
                check(
                    "adpf_no_api_errors",
                    hint.get("api_error_count") == 0
                    and hint.get("last_api_result") == 0,
                    f"api_error_count={hint.get('api_error_count')} "
                    f"last_api_result={hint.get('last_api_result')}",
                    critical=False,
                )
                check(
                    "adpf_not_claimed_as_hard_realtime",
                    hint.get("semantics")
                    == "api_accepted_only;not_frequency_or_hard_realtime_proof",
                    f"semantics={hint.get('semantics')!r}",
                )
            else:
                check(
                    "adpf_report_present",
                    False,
                    repr(hint),
                    critical=False,
                )
    else:
        check(
            "timer_mode_not_hardware_clock",
            bool(
                clock_sidecar
                and not clock_sidecar.get("hardware_audio_clock_executed")
            ),
            "timer comparison must not claim hardware callback execution",
        )

    if algorithm is not None:
        seconds = int(plan["seconds"])
        expected_input = seconds * 16000
        expected_callbacks = expected_input // 256 + math.ceil(513 / 256)
        expected_checks = {
            "input_samples_per_channel": expected_input,
            "dsp_delay_samples": 513,
            "aec_localvqe_inferences": expected_callbacks - 1,
            "ns_inference_calls": expected_callbacks,
            "aec_localvqe_fallback_frames": 0,
            "ns_external_fallback_frames": 0,
            "aec_model_sha256": "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731",
        }
        for key, expected in expected_checks.items():
            check(
                f"dsp_{key}",
                algorithm.get(key) == expected,
                f"actual={algorithm.get(key)!r} expected={expected!r}",
            )
        expected_provider = plan.get("new_assets", {}).get("library", {}).get(
            "sha256"
        )
        check(
            "dsp_provider_identity",
            algorithm.get("aec_provider_sha256") == expected_provider,
            f"actual={algorithm.get('aec_provider_sha256')!r} "
            f"expected={expected_provider!r}",
        )
        output_hashes = {}
        for suffix in [
            "after.wav",
            "aec.wav",
            "aec-input.wav",
            "bf.wav",
        ]:
            path = find_one(collected, f"out.{suffix}")
            output_hashes[suffix] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path is not None
                else None
            )
            check(
                f"output_{suffix}",
                path is not None and path.stat().st_size > 44,
                str(path) if path else "missing",
            )
    else:
        output_hashes = {}

    trace_metrics = {}
    if trace_path is not None:
        with trace_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        wall_ms = [int(row["process_wall_ns"]) / 1e6 for row in rows]
        cpu_ms = [
            int(row["process_thread_cpu_ns"]) / 1e6 for row in rows
        ]
        callback_cpu_ms = [
            int(row["callback_thread_cpu_ns"]) / 1e6 for row in rows
        ]
        lateness_ms = [
            int(row["release_lateness_ns"]) / 1e6 for row in rows
        ]
        completion_from_start_ms = [
            max(
                0.0,
                (
                    int(row["completed_ns"])
                    - int(row["callback_start_ns"])
                    - 16_000_000
                )
                / 1e6,
            )
            for row in rows
        ]
        start_ns = [int(row["callback_start_ns"]) for row in rows]
        intervals_ms = [
            (start_ns[index + 1] - start_ns[index]) / 1e6
            for index in range(len(start_ns) - 1)
        ]
        trace_metrics = {
            "rows": len(rows),
            "clock_errors": sum(row["clocks_valid"] != "1" for row in rows),
            "completion_late": sum(row["completion_late"] == "1" for row in rows),
            "completion_late_from_callback_start": sum(
                value > 0 for value in completion_from_start_ms
            ),
            "process_wall_ms": {
                "mean": statistics.fmean(wall_ms) if wall_ms else None,
                "p50": nearest_rank(wall_ms, 0.50),
                "p95": nearest_rank(wall_ms, 0.95),
                "p99": nearest_rank(wall_ms, 0.99),
                "max": max(wall_ms) if wall_ms else None,
            },
            "process_thread_cpu_ms": {
                "mean": statistics.fmean(cpu_ms) if cpu_ms else None,
                "p95": nearest_rank(cpu_ms, 0.95),
                "max": max(cpu_ms) if cpu_ms else None,
            },
            "callback_thread_cpu_ms": {
                "mean": statistics.fmean(callback_cpu_ms)
                if callback_cpu_ms
                else None,
                "p95": nearest_rank(callback_cpu_ms, 0.95),
                "max": max(callback_cpu_ms) if callback_cpu_ms else None,
            },
            "release_lateness_ms": {
                "p50": nearest_rank(lateness_ms, 0.50),
                "p95": nearest_rank(lateness_ms, 0.95),
                "p99": nearest_rank(lateness_ms, 0.99),
                "max": max(lateness_ms) if lateness_ms else None,
            },
            "completion_over_16ms_from_callback_start_ms": {
                "p50": nearest_rank(completion_from_start_ms, 0.50),
                "p95": nearest_rank(completion_from_start_ms, 0.95),
                "p99": nearest_rank(completion_from_start_ms, 0.99),
                "max": max(completion_from_start_ms)
                if completion_from_start_ms
                else None,
            },
            "callback_interval_ms": {
                "mean": statistics.fmean(intervals_ms)
                if intervals_ms
                else None,
                "p50": nearest_rank(intervals_ms, 0.50),
                "p95": nearest_rank(intervals_ms, 0.95),
                "p99": nearest_rank(intervals_ms, 0.99),
                "max": max(intervals_ms) if intervals_ms else None,
            },
        }
        expected_callbacks = (
            audio_stream.get("callback_count")
            if isinstance(audio_stream, dict)
            else (clock_sidecar.get("callbacks") if clock_sidecar else None)
        )
        check(
            "trace_rows_match_callback_count",
            isinstance(expected_callbacks, int)
            and len(rows) == expected_callbacks,
            f"rows={len(rows)} callback_count={expected_callbacks}",
        )
        check(
            "trace_clocks_valid",
            trace_metrics["clock_errors"] == 0,
            f"clock_errors={trace_metrics['clock_errors']}",
        )
        check(
            "hard_realtime_16ms_callback_budget",
            trace_metrics["completion_late_from_callback_start"] == 0,
            f"callbacks_over_16ms_from_callback_start="
            f"{trace_metrics['completion_late_from_callback_start']}/"
            f"{trace_metrics['rows']}",
            critical=False,
        )
    elif mode == "aaudio":
        check("trace_present", False, "missing audio-clock frames CSV")

    trace_requested = bool(plan.get("trace_requested"))
    perfetto_exitcode = read_int(collected, "perfetto_exitcode")
    pftrace = find_one(collected, "scheduler.pftrace")
    if trace_requested:
        check(
            "perfetto_exitcode",
            perfetto_exitcode == 0,
            f"perfetto_exitcode={perfetto_exitcode}",
            critical=False,
        )
        check(
            "perfetto_trace_nonempty",
            pftrace is not None and pftrace.stat().st_size > 0,
            f"{pftrace} size={pftrace.stat().st_size if pftrace else None}",
            critical=False,
        )

    critical_failures = [
        item["name"] for item in checks if item["status"] == "FAIL"
    ]
    warnings = [
        item["name"] for item in checks if item["status"] == "WARN"
    ]
    status = (
        "failed"
        if critical_failures
        else ("passed_with_observations" if warnings else "passed")
    )
    result = {
        "schema": "itu-aaudio-device-analysis-v1",
        "status": status,
        "run_dir": str(run_dir),
        "collected_dir": str(collected),
        "mode": mode,
        "seconds": plan.get("seconds"),
        "hint_requested": plan.get("hint_requested"),
        "trace_requested": trace_requested,
        "exitcode": exitcode,
        "algorithm_exitcode": algorithm_exitcode,
        "critical_failures": critical_failures,
        "warnings": warnings,
        "checks": checks,
        "trace_metrics": trace_metrics,
        "output_sha256": output_hashes,
        "scope": {
            "frozen_file_input": True,
            "aaudio_output_callback_pacing": mode == "aaudio",
            "silent_clock_stream": mode == "aaudio",
            "microphone_capture_verified": False,
            "processed_audio_playback_verified": False,
            "end_to_end_acoustic_latency_verified": False,
            "formal_itu_compliance_verified": False,
            "hardware_presentation_deadline_verified": False,
            "hard_real_time_verified": False,
        },
        "artifacts": {
            "plan": str(plan_path),
            "clock_sidecar": str(clock_sidecar_path)
            if clock_sidecar_path
            else None,
            "failure_sidecar": str(failure_sidecar_path)
            if failure_sidecar_path
            else None,
            "algorithm_report": str(algorithm_path) if algorithm_path else None,
            "frames_csv": str(trace_path) if trace_path else None,
            "perfetto_trace": str(pftrace) if pftrace else None,
        },
    }
    result_path = out_dir / "RESULT.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    summary = [
        f"# AAudio trial analysis: {status}",
        "",
        f"- mode: {mode}",
        f"- seconds: {plan.get('seconds')}",
        f"- hint requested: {plan.get('hint_requested')}",
        f"- critical failures: {', '.join(critical_failures) or 'none'}",
        f"- warnings: {', '.join(warnings) or 'none'}",
    ]
    if isinstance(audio_stream, dict):
        summary.extend(
            [
                f"- AAudio error: {audio_stream.get('error')}",
                f"- callbacks: {audio_stream.get('callback_count')}",
                f"- actual sharing: {audio_stream.get('actual_sharing_mode_name')}",
                f"- actual performance: {audio_stream.get('actual_performance_mode_name')}",
                f"- xrun delta: {audio_stream.get('xrun_delta')}",
            ]
        )
    if trace_metrics:
        process = trace_metrics["process_wall_ms"]
        summary.append(
            "- process wall ms: "
            f"mean={process['mean']:.6f} p95={process['p95']:.6f} "
            f"p99={process['p99']:.6f} max={process['max']:.6f}"
        )
        summary.append(
            "- completion over 16 ms from callback start: "
            f"{trace_metrics['completion_late_from_callback_start']}"
        )
        summary.append(
            "- completion late against first-start grid: "
            f"{trace_metrics['completion_late']}"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "critical_failures": critical_failures,
                "warnings": warnings,
                "result": str(result_path),
                "summary": str(out_dir / "SUMMARY.md"),
            },
            ensure_ascii=False,
        )
    )
    if critical_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
