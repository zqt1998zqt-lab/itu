#!/usr/bin/env python3
"""Independent completeness, provenance, gate and publication audit for the chain tab."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def independent_gate(row, baseline, thresholds):
    actual, control = row["measurements"], baseline["measurements"]
    expected_failures = []
    for name, after in actual["speech_windows"].items():
        before = control["speech_windows"][name]
        values = [after["sisdr_db"], before["sisdr_db"], after["stoi"], before["stoi"]]
        if any(value is None or not math.isfinite(value) for value in values):
            expected_failures.append(f"{name}:invalid_speech")
            continue
        sd, st = after["sisdr_db"] - before["sisdr_db"], after["stoi"] - before["stoi"]
        check(math.isclose(row["gate"]["speech"][name]["sisdr_delta_db"], sd, abs_tol=1e-12), "SI-SDR delta drift")
        check(math.isclose(row["gate"]["speech"][name]["stoi_delta"], st, abs_tol=1e-12), "STOI delta drift")
        if sd < thresholds["speech_sisdr_delta_min_db"]:
            expected_failures.append(f"{name}:sisdr")
        if st < thresholds["speech_stoi_delta_min"]:
            expected_failures.append(f"{name}:stoi")
        gain = after["projection_gain_db"]
        if gain is None or not thresholds["projection_gain_min_db"] <= gain <= thresholds["projection_gain_max_db"]:
            expected_failures.append(f"{name}:gain")
    noise = actual["noise_attenuation_db"] - control["noise_attenuation_db"]
    check(math.isclose(noise, row["gate"]["noise_delta_db"], abs_tol=1e-12), "noise delta drift")
    if noise < thresholds["noise_attenuation_delta_min_db"]:
        expected_failures.append("noise_attenuation")
    if actual["far_attenuation_db"] is not None:
        far = actual["far_attenuation_db"] - control["far_attenuation_db"]
        check(math.isclose(far, row["gate"]["far_delta_db"], abs_tol=1e-12), "far delta drift")
        if far < thresholds["far_attenuation_delta_min_db"]:
            expected_failures.append("far_attenuation")
    if actual["peak_dbfs"] > thresholds["peak_max_dbfs"]:
        expected_failures.append("peak")
    if actual["clipping_percent"] > thresholds["clipping_max_percent"]:
        expected_failures.append("clipping")
    check(row["gate"]["failures"] == expected_failures, f"gate failure list drift: {row['variant']}/{row['scene_id']}")
    return not expected_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    folder = args.root.resolve()
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    config = json.loads((folder / "artifacts/enhancement-chain-configs.json").read_text())
    route_count = len(config["presets"])
    expected_rows = 72 * route_count
    check(report["case_count"] == 72 and report["cli_runs"] == len(rows) == expected_rows, "incomplete matrix cannot be published")
    check(len(report["variants"]) == route_count, "incomplete route set")
    check(report["standard_compliance"] is False, "engineering matrix must not claim formal compliance")
    for field, file in [("cli_sha256", folder / "artifacts/itu_audio_cli"), ("config_sha256", ROOT / "src/features/enhancement-chain-configs.json"), ("results_sha256", folder / "results.jsonl"), ("manifest_sha256", folder / "artifacts/inputs-manifest.json")]:
        check(report[field] == sha(file), f"provenance mismatch: {field}")
    baseline = {row["scene_id"]: row for row in rows if row["variant"] == "array_control"}
    check(len(baseline) == 72, "missing or duplicate DAS controls")
    for row in rows:
        check(sha(ROOT / row["output_file"]) == row["output_sha256"], "output file hash mismatch")
        check(row["output_changed"] == (row["output_sha256"] != baseline[row["scene_id"]]["output_sha256"]), "output-change attribution mismatch")
        check(independent_gate(row, baseline[row["scene_id"]], report["gates"]) == row["gate"]["pass"], "pass/fail mismatch")
        prefix = (ROOT / row["output_file"]).with_name("result.json")
        engine = json.loads(prefix.read_text())
        route = next(item for item in report["variants"] if item["id"] == row["variant"])
        selected = route["selection"]
        check(engine["beamforming_before_aec"] and engine["beamforming_enabled"], "BF was not executed before AEC")
        check(route["profile"] != "vcp" or engine["eq_before_agc"], "EQ order mismatch")
        check(engine["aec_enabled"] == (selected["aec"] != "off"), "AEC enable mismatch")
        check(engine["ns_enabled"] == (selected["ns"] != "off"), "NS enable mismatch")
        check(engine["tx_drc_enabled"] == (selected["drc"] != "off"), "DRC enable mismatch")
        check(engine["rx_processing_frames"] == 0 and engine["reference_source"] == "provided_playback_pcm", "frozen playback was unexpectedly reprocessed")
        check(engine["agc2_speech_gain_policy"] == (selected["agc"] != "off"), "AGC execution mismatch")
        check(len(engine["output_eq_bands"]) == (0 if selected["eq"] == "off" else 3), "EQ bands not bound to the real executor")
        if selected["aec"] != "off":
            expected_aec = {
                "speex": "speex_mdf",
                "speex_fused": "speex_mdf_nlms_fused",
                "aec3": "webrtc_aec3",
            }[selected["aec"]]
            check(engine["aec_backend"] == expected_aec, "wrong AEC backend executed")
        if selected["ns"] == "unified":
            check(engine["ns_backend"] == "unified", "Unified NS backend was not executed")
            check(engine["unified_eq_agc"] is True, "Unified EQ/AGC fusion was not enabled")
    for variant in report["variants"]:
        selected = [row for row in rows if row["variant"] == variant["id"]]
        check(len(selected) == len({row["scene_id"] for row in selected}) == 72, "unpaired or duplicate route cases")
        check(sum(row["gate"]["pass"] for row in selected) == variant["pass"], "summary pass count drift")
        check(dict(Counter(reason for row in selected for reason in row["gate"]["failures"])) == variant["failure_counts"], "failure counter drift")
        check(sum(row["measurements"]["noise_attenuation_lower_bound"] for row in selected) == variant["noise_censored_cases"], "quantization censor count drift")
        check(sum(row["output_changed"] for row in selected) == variant["changed"], "changed output summary drift")
        for group in variant["caseGroups"]:
            row = next(item for item in selected if item["scene_id"] == group["caseAudio"]["id"])
            check(row["gate"]["pass"] == (group["id"] == "pass"), "typical-case status mismatch")
            check(group["caseAudio"]["afterFile"] == row["output_file"], "typical-case output mismatch")
    controls = json.loads((folder / "negative-controls.json").read_text())
    check(len(controls) == len({item["scene_id"] for item in controls}) == 72, "incomplete mute negative controls")
    check(all(item["rejected"] and any(reason.endswith(":invalid_speech") for reason in item["failures"]) for item in controls), "mute passed or failed for the wrong reason")
    result = {"rows": len(rows), "unique_cases_per_route": 72, "routes": route_count, "gates_recomputed": len(rows), "output_hashes_checked": len(rows), "mute_rejected": len(controls), "report_sha256": sha(folder / "report.json")}
    (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.publish:
        destination = ROOT / "public/evaluation/enhancement-chain"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "report.json", destination / "report.json.tmp")
        (destination / "report.json.tmp").replace(destination / "report.json")
        shutil.copy2(folder / "verification.json", destination / "verification.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
