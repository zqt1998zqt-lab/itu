"""Compare completed timer/AAudio analyses without claiming causality."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite: {args.out}")
    args.out.mkdir(parents=True)

    trials = []
    for path in args.analysis:
        result = json.loads((path / "RESULT.json").read_text())
        trials.append(
            {
                "analysis_dir": str(path.resolve()),
                "status": result["status"],
                "mode": result["mode"],
                "seconds": result["seconds"],
                "hint_requested": result["hint_requested"],
                "critical_failures": result["critical_failures"],
                "warnings": result["warnings"],
                "trace_metrics": result.get("trace_metrics", {}),
                "output_sha256": result.get("output_sha256", {}),
                "checks": {
                    item["name"]: item for item in result.get("checks", [])
                },
            }
        )

    seconds = {trial["seconds"] for trial in trials}
    if len(seconds) != 1:
        raise RuntimeError(f"duration mismatch: {sorted(seconds)}")
    seconds_value = next(iter(seconds))

    output_names = sorted(
        set.intersection(
            *[
                set(trial["output_sha256"])
                for trial in trials
                if trial["output_sha256"]
            ]
        )
    )
    output_parity = {}
    for name in output_names:
        hashes = [trial["output_sha256"].get(name) for trial in trials]
        output_parity[name] = {
            "hashes": hashes,
            "all_identical": None not in hashes and len(set(hashes)) == 1,
        }

    def metric(trial: dict, group: str, key: str):
        return trial["trace_metrics"].get(group, {}).get(key)

    comparisons = []
    timer_trials = [trial for trial in trials if trial["mode"] == "timer"]
    aaudio_trials = [
        trial for trial in trials if trial["mode"] == "aaudio"
    ]
    for audio in aaudio_trials:
        for timer in timer_trials:
            delta = {}
            for group in [
                "process_wall_ms",
                "process_thread_cpu_ms",
                "callback_thread_cpu_ms",
                "release_lateness_ms",
            ]:
                for key in ["mean", "p50", "p95", "p99", "max"]:
                    audio_value = metric(audio, group, key)
                    timer_value = metric(timer, group, key)
                    delta[f"{group}.{key}"] = (
                        audio_value - timer_value
                        if isinstance(audio_value, (int, float))
                        and isinstance(timer_value, (int, float))
                        else None
                    )
            comparisons.append(
                {
                    "audio_label": Path(audio["analysis_dir"]).parent.name,
                    "audio": audio["analysis_dir"],
                    "timer": timer["analysis_dir"],
                    "audio_hint_requested": audio["hint_requested"],
                    "audio_minus_timer": delta,
                    "completion_late": {
                        "audio": audio["trace_metrics"].get(
                            "completion_late_from_callback_start"
                        ),
                        "timer": timer["trace_metrics"].get(
                            "completion_late_from_callback_start"
                        ),
                    },
                }
            )

    group_summary = {}
    groups = {
        "timer": timer_trials,
        "aaudio_hint_off": [
            trial for trial in aaudio_trials if trial["hint_requested"] == "off"
        ],
        "aaudio_hint_on": [
            trial for trial in aaudio_trials if trial["hint_requested"] == "on"
        ],
    }
    for group_name, group_trials in groups.items():
        group_summary[group_name] = {"trial_count": len(group_trials)}
        for metric_name in [
            "process_wall_ms.mean",
            "process_wall_ms.p95",
            "process_wall_ms.p99",
            "process_wall_ms.max",
            "completion_late_from_callback_start",
        ]:
            if metric_name == "completion_late_from_callback_start":
                values = [
                    trial["trace_metrics"].get(metric_name)
                    for trial in group_trials
                ]
            else:
                first, second = metric_name.split(".")
                values = [
                    trial["trace_metrics"].get(first, {}).get(second)
                    for trial in group_trials
                ]
            values = [value for value in values if isinstance(value, (int, float))]
            group_summary[group_name][metric_name] = (
                sum(values) / len(values) if values else None
            )

    required_aaudio_checks = [
        "hardware_clock_executed",
        "aaudio_error_none",
        "aaudio_completed",
        "aaudio_callback_quiesced",
        "aaudio_closed",
        "aaudio_stop_clean",
        "aaudio_actual_geometry",
        "trace_clocks_valid",
        "dsp_aec_localvqe_inferences",
        "dsp_ns_inference_calls",
        "dsp_aec_localvqe_fallback_frames",
        "dsp_ns_external_fallback_frames",
    ]
    aaudio_contract = {}
    for trial in aaudio_trials:
        label = (
            f"aaudio_hint_{trial['hint_requested']}:{trial['analysis_dir']}"
        )
        aaudio_contract[label] = {
            name: trial["checks"].get(name, {}).get("status")
            for name in required_aaudio_checks
        }

    result = {
        "schema": "itu-aaudio-trial-comparison-v1",
        "created_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "seconds": seconds_value,
        "trials": trials,
        "output_parity": output_parity,
        "aaudio_contract": aaudio_contract,
        "group_summary": group_summary,
        "comparisons": comparisons,
        "interpretation_limits": [
            "Runs are sequential and not randomized; deltas are observational.",
            "The AAudio output stream is silent and only supplies callback pacing.",
            "No microphone capture or processed-audio playback was exercised.",
            "Callback cadence is not a DAC presentation deadline.",
            "ADPF API acceptance is not proof of CPU frequency or hard real time.",
        ],
    }
    result_path = args.out / "COMPARISON.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Timer vs AAudio callback trial comparison",
        "",
        f"- duration: {seconds_value}s",
        f"- trials: {len(trials)}",
    ]
    for name, parity in output_parity.items():
        lines.append(
            f"- output `{name}` identical: {parity['all_identical']}"
        )
    for comparison in comparisons:
        label = comparison["audio_label"]
        delta = comparison["audio_minus_timer"]
        lines.extend(
            [
                "",
                f"## {label} minus timer",
                "",
                f"- process mean delta: {delta.get('process_wall_ms.mean')} ms",
                f"- process p95 delta: {delta.get('process_wall_ms.p95')} ms",
                f"- process p99 delta: {delta.get('process_wall_ms.p99')} ms",
                f"- process max delta: {delta.get('process_wall_ms.max')} ms",
                f"- callbacks over 16 ms from callback start: "
                f"{comparison['completion_late']}",
            ]
        )
    lines.extend(["", "## Group means", ""])
    for group_name, summary in group_summary.items():
        lines.append(
            f"- `{group_name}` n={summary['trial_count']} "
            f"process_mean={summary.get('process_wall_ms.mean')} ms "
            f"p95={summary.get('process_wall_ms.p95')} ms "
            f"over16ms={summary.get('completion_late_from_callback_start')}"
        )
    lines.extend(
        [
            "",
            "These are sequential observational comparisons, not a causal "
            "speedup or hard-real-time certification.",
        ]
    )
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "comparison": str(result_path),
                "summary": str(args.out / "SUMMARY.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
