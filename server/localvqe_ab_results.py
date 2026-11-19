"""Validate and summarize a frozen, full-duration external-MDF A/B report."""
from __future__ import annotations

import math
from typing import Any

SCHEMA = "itu-mdf-localvqe-mask-ab-v2"
CATEGORIES = {
    "near_single": "近端单讲",
    "far_single": "远端单讲",
    "double_talk": "双讲",
    "path_change": "回声路径突变",
}
ROUTES = {
    "input_mic": "麦克风混合输入",
    "mdf_residual_only": "仅 VCP MDF 残差",
    "localvqe_full": "LocalVQE 自身 DAF + 203K mask",
    "mdf_residual_mask": "VCP MDF 残差 → 203K mask",
    "mdf_residual_wrong_ref": "错配对照：用播放信号代替回声估计",
}
METRICS = ("output_rms", "output_attenuation_db", "near_si_sdr_db", "near_total_error_rms", "near_target_gain_db")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(f"MDF / LocalVQE A/B evidence invalid: {detail}")


def _mean(values: list) -> float | None:
    # An unavailable measurement must remain unavailable in the aggregate.
    if not values or any(value is None for value in values):
        return None
    _require(all(_finite(value) for value in values), "non-finite aggregate")
    return round(sum(values) / len(values), 6)


def summarize_ab(document: dict) -> dict:
    _require(isinstance(document, dict) and document.get("schema") == SCHEMA, "schema")
    rows = document.get("cases")
    _require(isinstance(rows, list) and bool(rows), "empty cases")
    _require(document.get("case_count") == len(rows), "case count")
    _require(document.get("duration_s") == 12 and document.get("sample_rate") == 16000, "timeline")
    _require(document.get("formal_compliance") is False, "formal compliance boundary")
    _require(document.get("controls_passed") is True, "top-level controls status")
    alignment = document.get("alignment", {})
    _require(alignment.get("fixed_codec_delay_samples") == 256 and alignment.get("flush_output") is True
             and alignment.get("fitted_delay_for_scoring") is False
             and alignment.get("state_reset_at_path_switch") is False, "alignment/flush/reset")
    categories = {}
    case_ids = set()
    coverage = []
    for group, label in CATEGORIES.items():
        cases = [row for row in rows if row.get("category") == group]
        _require(bool(cases), f"missing {group}")
        controls = [row["controls"] for row in cases if "controls" in row]
        _require(bool(controls), f"missing independent controls for {group}")
        for control in controls:
            _require(control.get("passed") is True, "controls flag")
            for field, limit in (("daf_replay_max_abs_error", 1e-6), ("original_full_max_abs_error", 1e-6),
                                 ("codec_reference_max_abs_error", 2e-6)):
                value = control.get(field)
                _require(_finite(value) and 0 <= value < limit, field)
            _require(control.get("codec_raw_lag_samples") == 256 and control.get("partial_hop_output_samples")
                     == control.get("partial_hop_input_samples") == 8193, "codec delay/partial tail")
            _require(control.get("zero_mask_output_max_abs") == 0 and control.get("muted_near_rejected") is True
                     and control.get("bypass_far_rejected") is True, "negative controls")
        for row in cases:
            identifier = row.get("case_id")
            _require(isinstance(identifier, str) and identifier not in case_ids, "case ID")
            case_ids.add(identifier)
            _require(row.get("samples") == 192000 and row.get("sample_rate") == 16000, "truncated case")
            status = row.get("measurement_status")
            _require(status in {"measured", "invalid_stimulus"}, "measurement validity")
            _require(status == "measured" or bool(row.get("measurement_reason")), "missing invalid reason")
            activity = row.get("activity", {})
            if status == "measured":
                _require(activity.get("measurement_s", 0) >= .05, "insufficient observed activity")
                if group == "double_talk":
                    _require(activity.get("design_window_coverage", 0) >= .05, "unverified double-talk")
            scene, mdf = row.get("scene", {}), row.get("mdf", {})
            _require(scene.get("duration_s") == 12 and scene.get("kind") == "synthetic_matched_post_rx_reconstruction",
                     "matched RX scene")
            _require(scene.get("rx_speaker_pcm_sha256") == scene.get("tx_echo_pcm_sha256")
                     and isinstance(scene.get("rx_speaker_pcm_sha256"), str), "RX/TX reference mismatch")
            report = mdf.get("report", {})
            _require(report.get("aec_backend") == "speex_mdf" and report.get("speex_echo_tail_ms") == 500
                     and report.get("speex_native_conditioning") is True
                     and report.get("speex_residual_suppression") is False
                     and report.get("speex_coherence_residual_suppression") is False, "VCP raw MDF configuration")
            _require(mdf.get("capture_reconstruction_max_abs_error") == 0, "x=e+yhat quantization domain")
            if group == "path_change":
                events = scene.get("path_change_times_s", [])
                _require(events == [6.0], "path switch not inside the measurement")
                segments = scene.get("echo_path_segments", [])
                _require(len(segments) == 2 and segments[0].get("end_s") == 6
                         and segments[1].get("start_s") == 6 and segments[1].get("end_s") == 12,
                         "path timeline")
                _require(segments[0].get("paths", [{}])[0] != segments[1].get("paths", [{}])[0], "unchanged FIR")
            _require(set(row.get("routes", {})) == set(ROUTES), "route coverage")
            for name, route in row["routes"].items():
                _require(route.get("artifact", {}).get("samples") == 192000, "short output")
                metrics = route.get("metrics", {})
                _require((_finite(metrics.get("output_rms")) and metrics["output_rms"] >= 0)
                         if status == "measured" else metrics.get("samples") == 0, "output measurement validity")
                _require(all(metrics.get(field) is None or _finite(metrics[field]) for field in METRICS), "metric finiteness")
                if name in ("localvqe_full", "mdf_residual_mask", "mdf_residual_wrong_ref"):
                    timing = route.get("timing", {})
                    _require(timing.get("codec_delay_samples") == 256 and timing.get("reset_sample_offsets") == [0]
                             and timing.get("processed_hops", 0) >= 751, "stream state/tail")
                    _require(_finite(timing.get("process_ms")) and timing["process_ms"] > 0, "runtime")
                if group == "path_change":
                    windows = metrics.get("path_windows", {})
                    _require(set(windows) == {"before_switch", "after_switch_0_1s", "after_switch_2_6s"}, "recovery windows")
                    _require(all(window.get("samples", 0) > 0 for window in windows.values()), "empty path window")
        route_summaries = []
        measured_cases = [row for row in cases if row["measurement_status"] == "measured"]
        _require(bool(measured_cases), f"no measurable {group}")
        for name, route_label in ROUTES.items():
            values = [row["routes"][name] for row in measured_cases]
            summary = {"route": name, "label": route_label,
                       **{metric: _mean([v["metrics"].get(metric) for v in values]) for metric in METRICS},
                       "process_ms": _mean([(v.get("timing") or {}).get("process_ms") for v in values]),
                       "process_rtf": _mean([(v.get("timing") or {}).get("process_rtf") for v in values])}
            if group == "path_change":
                summary["path_windows"] = {window: _mean([v["metrics"]["path_windows"][window]["output_attenuation_db"]
                    for v in values]) for window in ("before_switch", "after_switch_0_1s", "after_switch_2_6s")}
            route_summaries.append(summary)
        categories[group] = {"label": label, "case_count": len(cases), "routes": route_summaries,
                             "measured_case_count": len(measured_cases),
                             "invalid_cases": [{"case_id": row["case_id"], "reason": row["measurement_reason"]}
                                               for row in cases if row["measurement_status"] != "measured"],
                             "cases": [{"case_id": row["case_id"], "source_category": row["scene"].get("source_category"),
                                        "measurement_status": row["measurement_status"],
                                        "audio": {route: f"/api/localvqe/mdf-mask-ab/cases/{row['case_id']}/{route}.wav"
                                                  for route in ROUTES}} for row in cases]}
        coverage.append(len(cases))
    _require(sum(coverage) == len(rows), "unknown cases")
    external_near = next(route for route in categories["near_single"]["routes"] if route["route"] == "mdf_residual_mask")
    near_gain = external_near["near_target_gain_db"]
    over_suppressed = near_gain is not None and near_gain < -6.0
    return {"schema": SCHEMA, "created_at": document.get("created_at"), "case_count": len(rows),
            "duration_s": 12, "sample_rate": 16000, "categories": categories,
            "negative_controls_passed": True, "controls": {group: {key: value for key, value in
                next(row["controls"] for row in rows if row["category"] == group and "controls" in row).items()}
                for group in CATEGORIES},
            "model": document.get("identity", {}).get("model"),
            "assessment": {"status": "not_recommended" if over_suppressed else "needs_review",
                           "label": "暂不建议替换：检测到近端语音误抑制" if over_suppressed else "实验完成，需工程复核",
                           "near_target_gain_db": near_gain},
            "boundary": document.get("boundary"), "formal_compliance": False}
