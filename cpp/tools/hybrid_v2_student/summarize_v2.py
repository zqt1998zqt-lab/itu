#!/usr/bin/env python3
"""Summarize frozen Hybrid-v2 eval manifests without rerunning NS or scorers.

The script is intentionally read-only over evaluation data.  A config file can
replace the default suite map; superseded roots are recorded in the output but
their manifests are never loaded as active evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

METRICS = (
    "raw_G", "raw_S", "raw_N",
    "minus30_G", "minus30_S", "minus30_N",
    "pesq", "stoi", "sisdr_db",
    "energy_ratio_vs_input_db", "weak_speech_energy_ratio_vs_input_db",
)
SPEECH_METRICS = set(METRICS) - {"energy_ratio_vs_input_db", "weak_speech_energy_ratio_vs_input_db"}

DEFAULT_CONFIG = {
    "schema_version": 1,
    "suites": [
        {
            "id": "nativefixed-oldloss256",
            "root": "evaluation/hybridv2-nativefixed-oldloss256-20261115",
            "expected_variants": ["oldloss256"],
            "expected_runs": 176,
            "expected_groups": [
                "v1speech48", "v1clean16", "v1continuous4", "v1guards",
                "highway120-speech48", "highway120-continuous4",
                "highway120-noise2", "newtest48",
            ],
        },
        {
            "id": "nativefixed-newloss256",
            "root": "evaluation/hybridv2-nativefixed-newloss256-20261115",
            "expected_variants": ["newloss256"],
            "expected_runs": 176,
            "expected_groups": [
                "v1speech48", "v1clean16", "v1continuous4", "v1guards",
                "highway120-speech48", "highway120-continuous4",
                "highway120-noise2", "newtest48",
            ],
        },
        {
            "id": "nativefixed-newloss320",
            "root": "evaluation/hybridv2-nativefixed-newloss320-20261115",
            "expected_variants": ["newloss320"],
            "expected_runs": 176,
            "expected_groups": [
                "v1speech48", "v1clean16", "v1continuous4", "v1guards",
                "highway120-speech48", "highway120-continuous4",
                "highway120-noise2", "newtest48",
            ],
        },
        {
            "id": "nativefixed-newloss320-pitch05",
            "root": "evaluation/hybridv2-nativefixed-newloss320-pitch05-20261115",
            "expected_variants": ["newloss320-pitch05"],
            "expected_runs": 176,
            "expected_groups": [
                "v1speech48", "v1clean16", "v1continuous4", "v1guards",
                "highway120-speech48", "highway120-continuous4",
                "highway120-noise2", "newtest48",
            ],
        },
        {
            "id": "new48-legacy-baselines",
            "root": "evaluation/hybridv2-new48-baselines-20261115-v2",
            "expected_variants": ["statistical-21"],
            "expected_runs": 48,
            "expected_groups": ["newtest48"],
        },
        {
            "id": "new48-dpdf",
            "manifest_glob": "evaluation/*new48-dpdf*/references/*/*/scored/run_manifest.json",
            "expected_variants": ["dpdfnet2"],
            "expected_runs": 48,
            "expected_groups": ["newtest48"],
        },
    ],
    "superseded": [
        {
            "root": "evaluation/hybridv2-formal-oldnew-20261115",
            "status": "superseded_invalid_native_projection_offset",
            "active": False,
            "reason": "native projection offset bug invalidates candidate quality; evidence retained for audit only",
        },
        {
            "root": "evaluation/hybridv2-formal-newloss320-20261115",
            "status": "superseded_invalid_native_projection_offset",
            "active": False,
            "reason": "native projection offset bug invalidates candidate quality; evidence retained for audit only",
        },
    ],
    "historical_manifests": [
        *[
            {"dataset": f"v1{group}", "source": "dpdfnet2", "role": "dpdf", "path": f"../ns-next-20261115/dpdfnet-{group}-scored/run_manifest.json"}
            for group in ("speech48", "clean16", "continuous4", "guards")
        ],
        *[
            {"dataset": f"highway120-{group}", "source": "dpdfnet2", "role": "dpdf", "path": f"../ns-highway120-20261115/runs/dpdfnet2-{group}/run_manifest.json"}
            for group in ("speech48", "continuous4", "noise2")
        ],
        *[
            {"dataset": f"v1{group}", "source": "statistical-21", "role": "stat21", "path": f"../ns-distilled-c64-20261116/evaluation/stat21-frozen/variants/statistical-21/{group}/scored/run_manifest.json"}
            for group in ("speech48", "clean16", "continuous4", "guards")
        ],
        *[
            {"dataset": f"highway120-{group}", "source": "statistical-21", "role": "stat21", "path": f"../ns-distilled-c64-20261116/evaluation/stat21-frozen/variants/statistical-21/highway120-{group}/scored/run_manifest.json"}
            for group in ("speech48", "continuous4", "noise2")
        ],
        *[
            {"dataset": f"v1{group}", "source": "kd21", "role": "oldcontrol", "path": f"../ns-distilled-c64-20261116/evaluation/kd21-frozen/variants/kd21/{group}/scored/run_manifest.json"}
            for group in ("speech48", "clean16", "continuous4", "guards")
        ],
        *[
            {"dataset": f"highway120-{group}", "source": "kd21", "role": "oldcontrol", "path": f"../ns-distilled-c64-20261116/evaluation/kd21-frozen/variants/kd21/highway120-{group}/scored/run_manifest.json"}
            for group in ("speech48", "continuous4", "noise2")
        ],
        {"dataset": "v1speech48", "source": "c4eb", "role": "c4eb", "variant_filter": "adaptive", "path": "../ns-next-20261115/evaluation/runs/speech48-final-20261114T163202Z/run_manifest.json"},
        {"dataset": "v1clean16", "source": "c4eb", "role": "c4eb", "variant_filter": "adaptive", "path": "../ns-next-20261115/evaluation/runs/clean16-final-20261114T163743Z/run_manifest.json"},
        {"dataset": "v1continuous4", "source": "c4eb", "role": "c4eb", "variant_filter": "adaptive", "path": "../ns-next-20261115/evaluation/runs/continuous4-final-20261114T163814Z/run_manifest.json"},
        {"dataset": "v1guards", "source": "c4eb", "role": "c4eb", "variant_filter": "adaptive", "path": "../ns-next-20261115/evaluation/runs/guards-final-20261114T163819Z/run_manifest.json"},
        *[
            {"dataset": f"highway120-{group}", "source": "c4eb", "role": "c4eb", "variant_filter": "adaptive", "path": f"../ns-highway120-20261115/runs/native-{group}/run_manifest.json"}
            for group in ("speech48", "continuous4", "noise2")
        ],
    ],
    "pairs": [
        {"id": "newloss256_vs_oldloss256", "candidate": "newloss256", "baseline": "oldloss256"},
        {"id": "newloss320_vs_newloss256", "candidate": "newloss320", "baseline": "newloss256"},
        {"id": "pitch05_vs_newloss320", "candidate": "newloss320-pitch05", "baseline": "newloss320"},
        {"id": "candidate_vs_dpdf", "candidate": "newloss320", "baseline": "dpdfnet2"},
        {"id": "candidate_vs_stat21", "candidate": "newloss320", "baseline": "statistical-21"},
        {"id": "candidate_vs_oldcontrol", "candidate": "newloss320", "baseline": "oldloss256"},
        {"id": "candidate_vs_nomodel", "candidate": "newloss320", "baseline": "probe-negative-base9"},
        {"id": "newloss256_vs_c4eb", "candidate": "newloss256", "baseline": "c4eb"},
        {"id": "newloss320_vs_c4eb", "candidate": "newloss320", "baseline": "c4eb"},
        {"id": "pitch05_vs_c4eb", "candidate": "newloss320-pitch05", "baseline": "c4eb"},
    ],
}


def finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def mean(values):
    vals = [finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return fmean(vals) if vals else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def mic_sha(case: dict) -> str | None:
    raw = case.get("raw") or {}
    components = case.get("components") or raw.get("components") or {}
    return ((components.get("mic.wav") or {}).get("sha256")
            or case.get("mic_sha256") or case.get("input_mic_sha256"))


def case_key(case: dict) -> tuple:
    window = tuple(case.get("window_s") or case.get("raw", {}).get("window_s") or (0.0, None))
    return case.get("id"), mic_sha(case), window


def case_cluster(case: dict) -> str:
    raw = case.get("raw") or {}
    return str(case.get("speaker_cluster") or case.get("speakercluster") or raw.get("speaker_cluster")
               or raw.get("speakercluster") or case.get("speaker_id") or raw.get("speaker_id") or "none")


def case_group(case: dict) -> str:
    raw = case.get("raw") or {}
    return str(case.get("group") or raw.get("group") or case.get("kind") or ("noise" if case.get("pure_noise") else "speech"))


def case_refscope(case: dict, dataset: str) -> str:
    raw = case.get("raw") or {}
    explicit = case.get("reference_scope") or case.get("refscope") or raw.get("reference_scope") or raw.get("refscope")
    if explicit:
        return str(explicit)
    if dataset.endswith("clean16"):
        return "clean16_supplied_reference"
    if dataset == "newtest48":
        return "independent_cv_synthetic_mix_clean_reference"
    if dataset.endswith("continuous4"):
        return "constructed_continuous_clean_reference"
    if dataset.endswith("noise2") or dataset.endswith("guards"):
        return "noise_guard_clean_reference"
    return "original_voice_reference_not_guaranteed_noise_free"


def discover_suite_paths(spec: dict) -> list[Path]:
    paths = []
    if spec.get("root"):
        path = resolve(spec["root"])
        if path.is_dir():
            paths.append(path)
    if spec.get("root_glob"):
        paths.extend(sorted(path for path in ROOT.glob(spec["root_glob"]) if path.is_dir()))
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def discover_suite_manifests(spec: dict, suite_roots: list[Path]) -> list[Path]:
    paths = []
    for suite_root in suite_roots:
        paths.extend(sorted(suite_root.glob("variants/*/*/scored/run_manifest.json")))
    if spec.get("manifest_glob"):
        paths.extend(sorted(ROOT.glob(spec["manifest_glob"])))
    for item in spec.get("manifests", []):
        path = resolve(item)
        if path.is_file():
            paths.append(path)
    unique = []
    for path in paths:
        if path.is_file() and path not in unique:
            unique.append(path)
    return unique


def stage_channel(stage: dict) -> dict | None:
    return next((item for item in stage.get("channels_scored", []) if item.get("channel") == 0), None)


def receipt_for(manifest_path: Path) -> Path:
    group = manifest_path.parent.parent
    for candidate in (group / "candidate/receipt.json", group / "receipt.json", manifest_path.parent / "receipt.json"):
        if candidate.is_file():
            return candidate
    return group / "candidate/receipt.json"


def load_manifest_entries(path: Path, suite_id: str, dataset: str, source_hint: str | None, role_hint: str | None, historical: bool, variant_filter: str | None = None):
    try:
        document = load_json(path)
    except Exception as error:
        return [], {"manifest": str(path), "status": "unreadable", "error": f"{type(error).__name__}: {error}"}
    runs = document.get("runs") or []
    if variant_filter is not None:
        runs = [run for run in runs if run.get("variant") == variant_filter]
    receipt_path = receipt_for(path)
    receipt = load_json(receipt_path) if receipt_path.is_file() else {}
    manifest_sha = sha256(path)
    receipt_sha = sha256(receipt_path) if receipt_path.is_file() else None
    probe_path = Path(str(receipt.get("probe_path"))) if receipt.get("probe_path") else None
    probe_sha = sha256(probe_path) if probe_path else None
    stage_tool = document.get("stage_tool")
    if isinstance(stage_tool, dict):
        stage_tool_path = Path(stage_tool["path"]) if stage_tool.get("path") else None
        stage_tool_sha = stage_tool.get("sha256") or (sha256(stage_tool_path) if stage_tool_path else None)
    else:
        stage_tool_sha = sha256(Path(stage_tool)) if stage_tool else None
    records = []
    for run in runs:
        case = run.get("case") or {}
        source = source_hint or str(run.get("variant") or "unknown")
        if role_hint:
            role = role_hint
        elif source == "probe-negative-base9":
            role = "nomodel"
        elif source == "statistical-21":
            role = "stat21"
        elif source == "dpdfnet2":
            role = "dpdf"
        elif historical and source == "kd21":
            role = "oldcontrol"
        else:
            role = "candidate"
        base = {
            "suite_id": suite_id,
            "dataset": dataset,
            "source": source,
            "role": role,
            "historical": historical,
            "manifest_path": str(path),
            "manifest_sha256": manifest_sha,
            "manifest_status": document.get("status"),
            "receipt_path": str(receipt_path) if receipt_path.is_file() else None,
            "receipt_sha256": receipt_sha,
            "case_manifest_sha256": receipt.get("cases_manifest_sha256"),
            "model_sha256": receipt.get("model_sha256"),
            "probe_sha256": probe_sha,
            "liveweight": receipt.get("model_sha256"),
            "stage_tool_sha256": stage_tool_sha,
            "scorer_protocol": document.get("protocol") or run.get("scoring_protocol"),
            "case_id": case.get("id") or run.get("case_id"),
            "mic_sha256": mic_sha(case) or run.get("hashes_before", {}).get("input"),
            "window_s": case.get("window_s") or run.get("window_s"),
            "language": case.get("language") or "unknown",
            "snr_db": case.get("snr_db"),
            "group": case_group(case),
            "pure_noise": bool(case.get("pure_noise")),
            "speaker_cluster": case_cluster(case),
            "refscope": case_refscope(case, dataset),
            "status": run.get("status"),
            "engine_exit_code": run.get("engine_exit_code"),
            "stages": {},
            "clean": run.get("clean") or {},
        }
        for stage in run.get("scored_stages") or []:
            channel = stage_channel(stage)
            if channel:
                base["stages"][str(stage.get("id"))] = channel
        records.append(base)
    summary = {
        "manifest": str(path), "sha256": manifest_sha, "status": document.get("status"),
        "runs": len(runs), "failed_runs": document.get("failed_runs"), "dataset": dataset,
        "sources": sorted({record["source"] for record in records}),
        "receipt": str(receipt_path) if receipt_path.is_file() else None,
        "receipt_sha256": receipt_sha,
    }
    return records, summary


def collect(config: dict):
    records, manifests, suite_status = [], [], []
    for spec in config.get("suites", []):
        paths = discover_suite_paths(spec)
        manifest_paths = discover_suite_manifests(spec, paths)
        suite = {
            "id": spec.get("id"), "active": True,
            "paths": [str(path) for path in paths],
            "manifests": [str(path) for path in manifest_paths],
            "expected_groups": spec.get("expected_groups", []),
            "expected_variants": spec.get("expected_variants", []),
            "expected_runs": spec.get("expected_runs"),
            "status": "pending", "invalid_reason": None,
        }
        if not manifest_paths:
            suite["status"] = "pending_missing_root"
            suite_status.append(suite)
            continue
        found_groups, found_variants = set(), set()
        complete_groups, complete_variants = set(), set()
        observed_runs, complete_runs = 0, 0
        failed_manifests = []
        for manifest_path in manifest_paths:
            dataset = manifest_path.parents[1].name
            variant = manifest_path.parents[2].name
            found_groups.add(dataset)
            found_variants.add(variant)
            loaded, summary = load_manifest_entries(manifest_path, suite["id"], dataset, None, None, False)
            records.extend(loaded)
            manifests.append(summary)
            observed_runs += int(summary.get("runs") or 0)
            if summary.get("status") == "complete" and summary.get("failed_runs") == 0:
                complete_groups.add(dataset)
                complete_variants.add(variant)
                complete_runs += int(summary.get("runs") or 0)
            else:
                failed_manifests.append({"manifest": summary.get("manifest"), "status": summary.get("status"), "failed_runs": summary.get("failed_runs")})
        expected_groups = set(suite["expected_groups"])
        expected_variants = set(suite["expected_variants"])
        suite.update({
            "found_groups": sorted(found_groups), "found_variants": sorted(found_variants),
            "complete_groups": sorted(complete_groups), "complete_variants": sorted(complete_variants),
            "missing_groups": sorted(expected_groups - complete_groups),
            "missing_variants": sorted(expected_variants - complete_variants),
            "observed_runs": observed_runs, "complete_runs": complete_runs,
            "missing_runs": max(0, int(suite["expected_runs"] or 0) - complete_runs) if suite.get("expected_runs") else None,
            "failed_manifests": failed_manifests,
        })
        if expected_groups or expected_variants:
            runs_complete = suite.get("expected_runs") is None or complete_runs >= int(suite["expected_runs"])
            suite["status"] = "complete" if expected_groups <= complete_groups and expected_variants <= complete_variants and runs_complete else "pending"
        else:
            suite["status"] = "complete" if manifests and all(item.get("status") == "complete" for item in manifests) else "pending"
        suite_status.append(suite)
    for spec in config.get("historical_manifests", []):
        path = resolve(spec["path"])
        if not path.is_file():
            manifests.append({"manifest": str(path), "status": "missing_historical", "dataset": spec["dataset"], "sources": [spec["source"]]})
            continue
        loaded, summary = load_manifest_entries(path, "historical", spec["dataset"], spec["source"], spec.get("role"), True, spec.get("variant_filter"))
        records.extend(loaded)
        manifests.append(summary)
    invalid = []
    for item in config.get("superseded", []):
        invalid.append({**item, "path": str(resolve(item["root"])) if item.get("root") else None, "receipt": "invalid_reason"})
    return records, manifests, suite_status, invalid


def metric_value(record: dict, stage: str, metric: str):
    container = record["clean"] if stage == "clean_reference" else record["stages"].get(stage, {})
    return finite(container.get(metric))


def stage_rows(records: list[dict]):
    rows = []
    for record in records:
        for stage in ["clean_reference", *sorted(record["stages"])]:
            for metric in METRICS:
                value = metric_value(record, stage, metric)
                applicable = not (record["pure_noise"] and metric in SPEECH_METRICS)
                rows.append({
                    **{key: record.get(key) for key in (
                        "suite_id", "dataset", "source", "role", "historical", "case_id", "mic_sha256",
                        "window_s", "language", "snr_db", "group", "pure_noise", "speaker_cluster", "refscope",
                        "manifest_path", "manifest_sha256", "receipt_path", "receipt_sha256", "case_manifest_sha256",
                        "model_sha256", "probe_sha256", "liveweight", "stage_tool_sha256", "status", "engine_exit_code",
                    )},
                    "stage": stage, "metric": metric, "value": value, "available": value is not None,
                    "metric_applicable": applicable,
                })
    return rows


def stage_coverage(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        for stage in ["clean_reference", *sorted(record["stages"])]:
            grouped[(record["dataset"], record["source"], stage)].append(record)
    rows = []
    for (dataset, source, stage), selected in sorted(grouped.items()):
        row = {"dataset": dataset, "source": source, "stage": stage, "n_expected": len(selected), "n_present": 0}
        for metric in ("raw_G", "raw_S", "raw_N", "minus30_G", "minus30_S", "minus30_N", "pesq", "stoi", "energy_ratio_vs_input_db"):
            row[f"n_{metric}"] = sum(metric_value(record, stage, metric) is not None for record in selected)
        row["n_present"] = sum(stage == "clean_reference" or stage in record["stages"] for record in selected)
        rows.append(row)
    return rows


def subset(rows: list[dict], language=None, snr=None, group=None):
    result = rows
    if language is not None:
        result = [row for row in result if row.get("language") == language]
    if snr is not None:
        result = [row for row in result if row.get("snr_db") == snr]
    if group is not None:
        result = [row for row in result if row.get("group") == group]
    return result


def bootstrap(rows: list[dict], metric: str, samples: int, seed: int, language: str | None, snr: float | None):
    if samples < 2:
        return None, None
    by_language = defaultdict(list)
    for row in rows:
        value = finite(row.get(metric))
        if value is not None:
            by_language[row.get("language") or "unknown"].append((row.get("speaker_cluster") or "none", value))
    if language is not None:
        by_language = {language: by_language.get(language, [])}
    by_language = {key: value for key, value in by_language.items() if value}
    if not by_language:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        language_means = []
        for values in by_language.values():
            clusters = defaultdict(list)
            for cluster, value in values:
                clusters[cluster].append(value)
            ids = sorted(clusters)
            picked = [rng.choice(ids) for _ in ids]
            language_means.append(fmean(fmean(clusters[cluster]) for cluster in picked))
        if language_means:
            draws.append(fmean(language_means))
    if not draws:
        return None, None
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def summarize(records: list[dict], samples: int, seed: int):
    summaries = []
    by_source = defaultdict(list)
    for record in records:
        by_source[(record["dataset"], record["source"], record["suite_id"])].append(record)
    for (dataset, source, suite_id), source_records in sorted(by_source.items()):
        languages = sorted({record["language"] for record in source_records})
        snrs = sorted({record["snr_db"] for record in source_records if record["snr_db"] is not None})
        groups = sorted({record["group"] for record in source_records})
        scopes = [("overall", None, None, None)]
        scopes += [("language", language, None, None) for language in languages]
        scopes += [("language_snr", language, snr, None) for language in languages for snr in snrs]
        scopes += [("group", None, None, group) for group in groups]
        scopes += [("snr", None, snr, None) for snr in snrs]
        for stage in ["clean_reference", *sorted({stage for record in source_records for stage in record["stages"]})]:
            for scope, language, snr, group in scopes:
                selected_records = subset(source_records, language, snr, group)
                expected = len(selected_records)
                for metric in METRICS:
                    values = [metric_value(record, stage, metric) for record in selected_records]
                    valid = sum(value is not None for value in values)
                    applicable = not (selected_records and all(record["pure_noise"] for record in selected_records) and metric in SPEECH_METRICS)
                    rows = [{**record, metric: metric_value(record, stage, metric)} for record in selected_records]
                    ci_scope = stage in {"clean_reference", "tx_output"} and scope in {"overall", "language", "language_snr"}
                    lo, hi = bootstrap(rows, metric, samples, seed, language, snr) if applicable and valid >= 2 and ci_scope else (None, None)
                    summaries.append({
                        "suite_id": suite_id, "dataset": dataset, "source": source, "stage": stage,
                        "scope": scope, "language": language, "snr_db": snr, "group": group,
                        "metric": metric, "metric_applicable": applicable, "mean": mean(values),
                        "ci95_low": lo, "ci95_high": hi, "n_expected": expected, "n_valid": valid,
                        "ci_status": "computed" if lo is not None else ("not_computed" if ci_scope else "not_requested_for_scope"),
                        "n_missing": expected - valid, "coverage": valid / expected if expected else None,
                        "n_speaker_clusters": len({record["speaker_cluster"] for record in selected_records}),
                        "bootstrap_unit": "one_speaker_draw_per_language_all_snr_synchronized",
                        "seed": seed, "bootstrap_samples": samples,
                        "refscope_values": sorted({record["refscope"] for record in selected_records}),
                    })
    return summaries


def reference_facts(summaries: list[dict]) -> dict:
    def clean_values(dataset: str) -> dict:
        result = {}
        for metric in ("minus30_G", "minus30_S", "minus30_N"):
            row = next((item for item in summaries if item["dataset"] == dataset and item["stage"] == "clean_reference"
                        and item["scope"] == "overall" and item["metric"] == metric), None)
            result[metric] = row.get("mean") if row else None
        return result
    return {
        "newtest48_clean_reference": clean_values("newtest48"),
        "v1clean16_clean_reference": clean_values("v1clean16"),
        "newtest48_reference_scope": "independent CV synthetic-mix reference includes real residual noise/codec; not pure clean",
        "dpdf_not_quality_ceiling": True,
        "candidate_vs_stat21_generalization": "not established: one extension set with 16 synthetic timelines; do not compare v1 and newtest means",
        "cross_dataset_mean_comparison": False,
    }


def headline(summaries: list[dict]) -> list[dict]:
    datasets = {"v1speech48", "highway120-speech48", "newtest48"}
    metrics = {"minus30_G", "minus30_S", "minus30_N"}
    return [
        {key: row.get(key) for key in ("dataset", "source", "stage", "metric", "mean", "ci95_low", "ci95_high", "n_valid", "n_expected", "coverage", "refscope_values")}
        for row in summaries
        if row["dataset"] in datasets and row["scope"] == "overall"
        and row["stage"] in {"clean_reference", "tx_ns", "tx_output"} and row["metric"] in metrics
    ]


def processing_coverage(suites: list[dict], records: list[dict]) -> dict:
    active = [suite for suite in suites if str(suite.get("id", "")).startswith("nativefixed-")]
    baseline = [suite for suite in suites if str(suite.get("id", "")).startswith("new48-")]
    active_manifests = sum(len(suite.get("manifests", [])) for suite in active)
    active_observed = sum(int(suite.get("complete_runs") or 0) for suite in active)
    baseline_observed = sum(int(suite.get("complete_runs") or 0) for suite in baseline)
    complete = bool(active and baseline and all(suite.get("status") == "complete" for suite in active + baseline))
    historical = [record for record in records if record["historical"]]
    c4eb = [record for record in historical if record["source"] == "c4eb"]
    return {
        "schema_version": 1,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "active_fixed": {
            "suite_count": len(active), "manifest_count": active_manifests,
            "expected_runs": 704, "observed_complete_runs": active_observed,
            "groups_per_suite": 8,
        },
        "legacy_baselines": {
            "suite_count": len(baseline), "manifest_count": sum(len(suite.get("manifests", [])) for suite in baseline),
            "expected_runs": 96, "observed_complete_runs": baseline_observed,
        },
        "total_new_processing": {"expected_runs": 800, "observed_complete_runs": active_observed + baseline_observed},
        "historical_reference_records": len(historical),
        "historical_reference_records_excluding_c4eb": len(historical) - len(c4eb),
        "historical_c4eb_records": len(c4eb),
        "historical_references_excluded_from_processing_count": True,
        "shared_eval_coverage_used": False,
        "source": "independent run_manifest aggregation",
    }


def paired_rows(records: list[dict], pairs: list[dict]):
    output = []
    datasets = sorted({record["dataset"] for record in records})
    for dataset in datasets:
        for pair in pairs:
            candidates = [record for record in records if record["dataset"] == dataset and record["source"] == pair["candidate"]]
            baselines = [record for record in records if record["dataset"] == dataset and record["source"] == pair["baseline"]]
            if not candidates or not baselines:
                continue
            baseline_by_key = {case_key({**record, "id": record["case_id"], "window_s": record["window_s"]}): record for record in baselines}
            for candidate in candidates:
                key = case_key({"id": candidate["case_id"], "window_s": candidate["window_s"], "mic_sha256": candidate["mic_sha256"]})
                baseline = baseline_by_key.get(key)
                if baseline is None:
                    baseline = next((item for item in baselines if item["case_id"] == candidate["case_id"]), None)
                input_match = bool(baseline and candidate["mic_sha256"] and candidate["mic_sha256"] == baseline["mic_sha256"] and tuple(candidate["window_s"] or []) == tuple(baseline["window_s"] or []))
                stages = ["clean_reference", *sorted(set(candidate["stages"]) | set(baseline["stages"] if baseline else {}))]
                for stage in stages:
                    for metric in METRICS:
                        candidate_value = metric_value(candidate, stage, metric)
                        baseline_value = metric_value(baseline, stage, metric) if baseline else None
                        eligible = bool(input_match and candidate_value is not None and baseline_value is not None)
                        output.append({
                            "pair_id": pair["id"], "dataset": dataset, "candidate": pair["candidate"], "baseline": pair["baseline"],
                            "case_id": candidate["case_id"], "language": candidate["language"], "snr_db": candidate["snr_db"],
                            "group": candidate["group"], "speaker_cluster": candidate["speaker_cluster"], "stage": stage, "metric": metric,
                            "candidate_value": candidate_value, "baseline_value": baseline_value,
                            "candidate_mic_sha256": candidate["mic_sha256"], "baseline_mic_sha256": baseline["mic_sha256"] if baseline else None,
                            "candidate_window_s": candidate["window_s"], "baseline_window_s": baseline["window_s"] if baseline else None,
                            "input_match": input_match, "diff_eligible": eligible,
                            "delta": candidate_value - baseline_value if eligible else None,
                            "candidate_manifest": candidate["manifest_path"], "baseline_manifest": baseline["manifest_path"] if baseline else None,
                        })
    return output


def plot(records, summaries, deltas, output: Path):
    if not records:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    overall = [row for row in summaries if row["scope"] == "overall" and row["stage"] in {"clean_reference", "tx_output"}]
    families = {
        "v1": lambda name: name.startswith("v1"),
        "highway120": lambda name: name.startswith("highway120-"),
        "newtest": lambda name: name == "newtest48",
    }
    for prefix in ("minus30", "raw"):
        metrics = [f"{prefix}_G", f"{prefix}_S", f"{prefix}_N"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), squeeze=False)
        for axis, (family, predicate) in zip(axes[0], families.items()):
            selected = [row for row in overall if predicate(row["dataset"])]
            datasets = sorted({row["dataset"] for row in selected})
            sources = sorted({row["source"] for row in selected if row["stage"] == "tx_output"})
            if not datasets:
                axis.set_visible(False)
                continue
            width = 0.8 / max(1, len(sources) + 1)
            for index, dataset in enumerate(datasets):
                for source_index, source in enumerate(sources):
                    values = [next((row["mean"] or 0 for row in selected if row["dataset"] == dataset and row["source"] == source and row["stage"] == "tx_output" and row["metric"] == metric), 0) for metric in metrics]
                    offset = (source_index - (len(sources) - 1) / 2) * width
                    axis.bar([index + offset + (metric_index - 1) * width / 3 for metric_index in range(3)], values, width=width / 3, label=source if index == 0 else None)
                clean_values = [next((row["mean"] or 0 for row in selected if row["dataset"] == dataset and row["stage"] == "clean_reference" and row["metric"] == metric), 0) for metric in metrics]
                axis.bar([index + 0.38 + (metric_index - 1) * width / 3 for metric_index in range(3)], clean_values, width=width / 3, color="black", alpha=0.65, label="clean_ref" if index == 0 else None)
            axis.set_xticks(range(len(datasets)), datasets, rotation=25, ha="right")
            axis.set_ylabel(f"{prefix} G / S / N")
            axis.set_title(f"{family}: {prefix} GSN overall (G,S,N)")
            axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = output / f"gsn-{prefix}-overall.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    speech = [row for row in summaries if row["scope"] == "overall" and row["stage"] == "tx_output" and row["metric"] in {"pesq", "stoi"}]
    if speech:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for axis, metric in zip(axes, ("pesq", "stoi")):
            selected = [row for row in speech if row["metric"] == metric]
            labels = [f"{row['dataset']}/{row['source']}" for row in selected]
            axis.bar(labels, [row["mean"] or 0 for row in selected])
            axis.set_title(metric.upper())
            axis.tick_params(axis="x", rotation=45, labelsize=7)
        fig.tight_layout()
        path = output / "clean-pesq-stoi.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    pitch = [row for row in deltas if row["pair_id"] == "pitch05_vs_newloss320" and row["diff_eligible"] and row["stage"] == "tx_output" and row["metric"] in {"minus30_G", "minus30_S", "minus30_N"}]
    if pitch:
        datasets = sorted({row["dataset"] for row in pitch})
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
        for axis, metric in zip(axes[0], ("minus30_G", "minus30_S", "minus30_N")):
            values = [mean([row["delta"] for row in pitch if row["dataset"] == dataset and row["metric"] == metric]) or 0 for dataset in datasets]
            axis.bar(datasets, values)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"pitch 0.5 Δ{metric[-1]}")
            axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = output / "pitch-ablation-delta-gsn.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--out", type=Path, default=REPORTS)
    args = parser.parse_args(argv)
    if args.allow_partial and args.strict:
        parser.error("--allow-partial and --strict are mutually exclusive")
    if args.bootstrap_samples < 2:
        parser.error("--bootstrap-samples must be >= 2")
    config = load_json(args.config) if args.config else DEFAULT_CONFIG
    records, manifests, suites, invalid = collect(config)
    rows = stage_rows(records)
    coverage_rows = stage_coverage(records)
    deltas = paired_rows(records, config.get("pairs", []))
    summaries = summarize(records, args.bootstrap_samples, args.seed)
    out = args.out.expanduser().resolve()
    write_csv(out / "stage-scores.csv", rows)
    write_csv(out / "paired-deltas.csv", deltas)
    figures = plot(records, summaries, deltas, out / "figures")
    pending = any(item["status"] != "complete" for item in suites)
    overall = "partial" if pending else "complete"
    full_coverage = processing_coverage(suites, records)
    document = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "coverage": {
            "status": "pending" if pending else "complete",
            "complete": not pending,
            "active_suites": suites,
            "manifests": manifests,
            "full800": full_coverage,
            "note": "pending is not complete; datasets and source-specific groups are never pooled",
        },
        "invalid_suites": invalid,
        "policy": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "one speaker draw per language, synchronized across SNR",
            "datasets_never_pooled": True,
            "refscope_required": True,
            "same_source_input_required_for_delta": True,
            "mos_status": "frozen_scores_complete",
            "newtest48_bootstrap": {
                "cluster_unit": "scene", "n_clusters": 16,
                "snr_not_independent": True,
                "reference_scope": "independent_cv_synthetic_mix_clean_reference_not_pure_original",
            },
            "pitch_noise_scope": {
                "pure_noise_case_is_not_deployment_truth": True,
                "periodic_multitone_noise_can_activate_pitch": True,
                "report_real_noise_effects": True,
            },
            "promotion_decision": "not_established; do not change default from evidence alone",
            "new_processing_counts": {
                "active_fixed_manifests": 32, "active_fixed_runs": 704,
                "legacy_newtest_baseline_runs": 96, "total_new_processing_runs": 800,
                "candidate_run_decomposition": {"v1": 74, "highway120": 54, "newtest48": 48},
                "historical_references_not_added_to_processing_count": True,
            },
            "split_integrity": {"train_dev_test_raw_speaker_intersection": 0},
            "raw_image_text": "not applicable",
        },
        "counts": {
            "records": len(records),
            "candidate_records": sum(1 for record in records if not record["historical"] and record["role"] == "candidate"),
            "legacy_baseline_records": sum(1 for record in records if not record["historical"] and record["role"] in {"stat21", "dpdf"}),
            "historical_records": sum(1 for record in records if record["historical"]),
            "historical_c4eb_records": sum(1 for record in records if record["historical"] and record["source"] == "c4eb"),
            "stage_rows": len(rows), "paired_rows": len(deltas), "summary_rows": len(summaries),
        },
        "stage_coverage": coverage_rows,
        "reference_facts": reference_facts(summaries),
        "headline": headline(summaries),
        "summaries": summaries,
        "figures": [str(path) for path in figures],
    }
    atomic_json(out / "score-summary.json", document)
    final_coverage = {
        "schema_version": 1,
        "status": full_coverage["status"],
        "complete": full_coverage["complete"],
        "total_new_processing_runs": full_coverage["total_new_processing"]["expected_runs"],
        "observed_complete_new_processing_runs": full_coverage["total_new_processing"]["observed_complete_runs"],
        "active_fixed": full_coverage["active_fixed"],
        "legacy_baselines": full_coverage["legacy_baselines"],
        "historical_reference_records": full_coverage["historical_reference_records"],
        "historical_reference_records_excluding_c4eb": full_coverage["historical_reference_records_excluding_c4eb"],
        "historical_c4eb_records": full_coverage["historical_c4eb_records"],
        "historical_references_excluded_from_processing_count": True,
        "shared_eval_coverage_used": False,
        "source": "reports/score-summary.json#coverage.full800",
        "score_summary_sha256": sha256(out / "score-summary.json"),
    }
    atomic_json(out / "final-coverage.json", final_coverage)
    print(json.dumps({
        "status": overall, "records": len(records), "stage_rows": len(rows),
        "paired_rows": len(deltas), "summary_rows": len(summaries),
        "figures": [str(path) for path in figures], "invalid_suites": len(invalid),
    }, ensure_ascii=False))
    return 0 if overall == "complete" or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
