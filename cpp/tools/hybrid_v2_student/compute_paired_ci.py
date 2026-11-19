#!/usr/bin/env python3
"""Build bounded exploratory paired CIs from frozen v2 summary artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

# Import the exact cluster bootstrap used by summarize_v2.py without creating
# any additional files outside this script's two-path write contract.
sys.dont_write_bytecode = True
from summarize_v2 import bootstrap  # noqa: E402


ROOT = Path(__file__).resolve().parent
PAIRED_PATH = ROOT / "reports/paired-deltas.csv"
SUMMARY_PATH = ROOT / "reports/score-summary.json"
OUTPUT_PATH = ROOT / "reports/paired-ci.json"
SUMMARIZER_PATH = ROOT / "summarize_v2.py"

DATASETS = ("v1speech48", "highway120-speech48", "newtest48")
STAGE = "tx_ns"
METRICS = ("minus30_G", "minus30_S", "minus30_N")
CONTROL_STAGES = ("clean_reference", "tx_input")
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20261117
MEAN_TOLERANCE = 1e-12
EXPECTED_CASES = 48
EXPECTED_CLUSTERS = {
    "v1speech48": 12,
    "highway120-speech48": 8,
    "newtest48": 16,
}
CLUSTER_UNITS = {
    "v1speech48": "speaker",
    "highway120-speech48": "speaker",
    "newtest48": "scene",
}

# candidate, baseline, candidate-row source pair, baseline-row source pair.
# Identical source pair ids are direct comparisons; the other three are
# same-case joins over the already completed paired CSV.
COMPARISONS = (
    ("newloss256", "oldloss256", "newloss256_vs_oldloss256", "newloss256_vs_oldloss256"),
    ("newloss256", "statistical-21", "newloss256_vs_oldloss256", "candidate_vs_stat21"),
    ("newloss320", "oldloss256", "candidate_vs_oldcontrol", "candidate_vs_oldcontrol"),
    ("newloss320", "statistical-21", "candidate_vs_stat21", "candidate_vs_stat21"),
    ("newloss320-pitch05", "oldloss256", "pitch05_vs_newloss320", "newloss256_vs_oldloss256"),
    ("newloss320-pitch05", "statistical-21", "pitch05_vs_newloss320", "candidate_vs_stat21"),
    ("newloss320-pitch05", "newloss320", "pitch05_vs_newloss320", "pitch05_vs_newloss320"),
)
SOURCE_PAIR_IDS = frozenset(
    pair_id for comparison in COMPARISONS for pair_id in comparison[2:]
)
SOURCE_PAIR_ENDPOINTS = {
    "newloss256_vs_oldloss256": ("newloss256", "oldloss256"),
    "candidate_vs_oldcontrol": ("newloss320", "oldloss256"),
    "candidate_vs_stat21": ("newloss320", "statistical-21"),
    "pitch05_vs_newloss320": ("newloss320-pitch05", "newloss320"),
}
SELECTED_STAGES = frozenset((STAGE, *CONTROL_STAGES))
SELECTED_METRICS = frozenset(METRICS)
SELECTED_DATASETS = frozenset(DATASETS)
CSV_FIELDS = frozenset(
    (
        "pair_id",
        "dataset",
        "candidate",
        "baseline",
        "case_id",
        "language",
        "snr_db",
        "speaker_cluster",
        "stage",
        "metric",
        "candidate_value",
        "baseline_value",
        "candidate_mic_sha256",
        "baseline_mic_sha256",
        "candidate_window_s",
        "baseline_window_s",
        "input_match",
        "diff_eligible",
        "delta",
    )
)


class ContractError(RuntimeError):
    """Raised when a frozen input no longer satisfies the analysis contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ContractError(f"{label}: expected True/False, got {value!r}")


def parse_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label}: expected finite float, got {value!r}") from exc
    require(math.isfinite(result), f"{label}: expected finite float, got {value!r}")
    return result


def parse_window(value: str, label: str) -> tuple[float, ...]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label}: invalid window JSON {value!r}") from exc
    require(isinstance(decoded, list), f"{label}: window must be a list")
    return tuple(parse_float(str(item), label) for item in decoded)


def load_summary(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {path}: {exc}") from exc
    require(isinstance(document, dict), "score summary root must be an object")
    require(isinstance(document.get("summaries"), list), "score summary summaries must be a list")
    return document


def load_selected_paired_rows(path: Path) -> dict[tuple[str, str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "paired CSV has no header")
        missing = CSV_FIELDS.difference(reader.fieldnames)
        require(not missing, f"paired CSV missing columns: {sorted(missing)}")
        for row_number, raw in enumerate(reader, start=2):
            if (
                raw["dataset"] not in SELECTED_DATASETS
                or raw["stage"] not in SELECTED_STAGES
                or raw["metric"] not in SELECTED_METRICS
                or raw["pair_id"] not in SOURCE_PAIR_IDS
            ):
                continue
            label = f"paired CSV row {row_number}"
            row = {
                "pair_id": raw["pair_id"],
                "dataset": raw["dataset"],
                "candidate": raw["candidate"],
                "baseline": raw["baseline"],
                "case_id": raw["case_id"],
                "language": raw["language"] or "unknown",
                "snr_db": parse_float(raw["snr_db"], f"{label} snr_db"),
                "speaker_cluster": raw["speaker_cluster"] or "none",
                "stage": raw["stage"],
                "metric": raw["metric"],
                "candidate_value": parse_float(raw["candidate_value"], f"{label} candidate_value"),
                "baseline_value": parse_float(raw["baseline_value"], f"{label} baseline_value"),
                "candidate_mic_sha256": raw["candidate_mic_sha256"],
                "baseline_mic_sha256": raw["baseline_mic_sha256"],
                "candidate_window_s": parse_window(raw["candidate_window_s"], f"{label} candidate_window_s"),
                "baseline_window_s": parse_window(raw["baseline_window_s"], f"{label} baseline_window_s"),
                "input_match": parse_bool(raw["input_match"], f"{label} input_match"),
                "diff_eligible": parse_bool(raw["diff_eligible"], f"{label} diff_eligible"),
                "delta": parse_float(raw["delta"], f"{label} delta"),
            }
            groups[(row["pair_id"], row["dataset"], row["stage"], row["metric"])].append(row)
    require(groups, "no selected rows found in paired CSV")
    return groups


def validate_source_rows(rows: list[dict], expected_candidate: str, expected_baseline: str, label: str) -> None:
    require(len(rows) == EXPECTED_CASES, f"{label}: expected {EXPECTED_CASES} rows, got {len(rows)}")
    require(len({row["case_id"] for row in rows}) == EXPECTED_CASES, f"{label}: duplicate case ids")
    require(all(row["candidate"] == expected_candidate for row in rows), f"{label}: candidate drift")
    require(all(row["baseline"] == expected_baseline for row in rows), f"{label}: baseline drift")
    require(all(row["input_match"] for row in rows), f"{label}: input_match is not true for every row")
    require(all(row["diff_eligible"] for row in rows), f"{label}: diff_eligible is not true for every row")
    require(all(row["candidate_mic_sha256"] == row["baseline_mic_sha256"] for row in rows),
            f"{label}: candidate/baseline mic SHA mismatch")
    require(all(row["candidate_window_s"] == row["baseline_window_s"] for row in rows),
            f"{label}: candidate/baseline window mismatch")


def paired_rows_for_metric(
    comparison: tuple[str, str, str, str],
    dataset: str,
    stage: str,
    metric: str,
    groups: dict[tuple[str, str, str, str], list[dict]],
) -> tuple[list[dict], list[str]]:
    candidate, baseline, candidate_pair, baseline_pair = comparison
    candidate_rows = groups.get((candidate_pair, dataset, stage, metric), [])
    source_candidate, source_baseline = SOURCE_PAIR_ENDPOINTS[candidate_pair]
    validate_source_rows(candidate_rows, source_candidate, source_baseline,
                         f"{dataset}/{stage}/{metric}/{candidate_pair}")
    if candidate_pair == baseline_pair:
        require(candidate_pair in {"candidate_vs_oldcontrol", "candidate_vs_stat21", "pitch05_vs_newloss320"}
                or candidate_pair == "newloss256_vs_oldloss256", "unexpected direct pair")
        rows = [
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "snr_db": row["snr_db"],
                "speaker_cluster": row["speaker_cluster"],
                "delta": row["delta"],
            }
            for row in candidate_rows
        ]
        for row in candidate_rows:
            require(abs(row["delta"] - (row["candidate_value"] - row["baseline_value"])) <= 1e-15,
                    f"{dataset}/{stage}/{metric}/{row['case_id']}: direct delta/value mismatch")
        return rows, [candidate_pair]

    baseline_rows = groups.get((baseline_pair, dataset, stage, metric), [])
    source_candidate, source_baseline = SOURCE_PAIR_ENDPOINTS[baseline_pair]
    validate_source_rows(baseline_rows, source_candidate, source_baseline,
                         f"{dataset}/{stage}/{metric}/{baseline_pair}")
    candidate_by_case = {row["case_id"]: row for row in candidate_rows}
    baseline_by_case = {row["case_id"]: row for row in baseline_rows}
    require(candidate_by_case.keys() == baseline_by_case.keys(),
            f"{dataset}/{stage}/{metric}: derived pair case ids differ")
    rows = []
    for candidate_row in candidate_rows:
        baseline_row = baseline_by_case[candidate_row["case_id"]]
        label = f"{dataset}/{stage}/{metric}/{candidate_row['case_id']}"
        require(candidate_row["language"] == baseline_row["language"], f"{label}: language differs")
        require(candidate_row["snr_db"] == baseline_row["snr_db"], f"{label}: SNR differs")
        require(candidate_row["speaker_cluster"] == baseline_row["speaker_cluster"], f"{label}: cluster differs")
        require(candidate_row["candidate_mic_sha256"] == baseline_row["baseline_mic_sha256"],
                f"{label}: derived input mic SHA differs")
        require(candidate_row["candidate_window_s"] == baseline_row["baseline_window_s"],
                f"{label}: derived input window differs")
        rows.append(
            {
                "case_id": candidate_row["case_id"],
                "language": candidate_row["language"],
                "snr_db": candidate_row["snr_db"],
                "speaker_cluster": candidate_row["speaker_cluster"],
                "delta": candidate_row["candidate_value"] - baseline_row["baseline_value"],
            }
        )
    return rows, [candidate_pair, baseline_pair]


def index_summary(summary: dict) -> dict[tuple[str, str, str, str], dict]:
    wanted_sources = {comparison[0] for comparison in COMPARISONS} | {comparison[1] for comparison in COMPARISONS}
    wanted = {
        (dataset, source, stage, metric)
        for dataset in DATASETS
        for source in wanted_sources
        for stage in SELECTED_STAGES
        for metric in METRICS
    }
    indexed: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in summary["summaries"]:
        if row.get("scope") != "overall" or row.get("language") is not None:
            continue
        if row.get("snr_db") is not None or row.get("group") is not None:
            continue
        key = (row.get("dataset"), row.get("source"), row.get("stage"), row.get("metric"))
        if key in wanted:
            indexed[key].append(row)
    result = {}
    for key in wanted:
        matches = indexed.get(key, [])
        require(len(matches) == 1, f"summary lookup {key}: expected one overall row, got {len(matches)}")
        row = matches[0]
        require(row.get("n_expected") == EXPECTED_CASES, f"summary lookup {key}: n_expected != {EXPECTED_CASES}")
        require(row.get("n_valid") == EXPECTED_CASES, f"summary lookup {key}: n_valid != {EXPECTED_CASES}")
        require(row.get("metric_applicable") is True, f"summary lookup {key}: metric not applicable")
        require(math.isfinite(float(row.get("mean"))), f"summary lookup {key}: non-finite mean")
        result[key] = row
    return result


def summary_mean(indexed: dict[tuple[str, str, str, str], dict], dataset: str, source: str, stage: str, metric: str) -> float:
    return float(indexed[(dataset, source, stage, metric)]["mean"])


def cluster_count(rows: list[dict]) -> int:
    return len({row["speaker_cluster"] for row in rows})


def main() -> int:
    require(PAIRED_PATH.is_file(), f"missing {PAIRED_PATH}")
    require(SUMMARY_PATH.is_file(), f"missing {SUMMARY_PATH}")
    require(SUMMARIZER_PATH.is_file(), f"missing {SUMMARIZER_PATH}")

    paired_sha256 = sha256_file(PAIRED_PATH)
    summary_sha256 = sha256_file(SUMMARY_PATH)
    summarizer_sha256 = sha256_file(SUMMARIZER_PATH)
    summary = load_summary(SUMMARY_PATH)
    indexed_summary = index_summary(summary)
    groups = load_selected_paired_rows(PAIRED_PATH)

    comparisons_out = []
    mismatch_count = 0
    max_mean_error = 0.0
    metric_count = 0
    derived_comparisons = []

    for comparison in COMPARISONS:
        candidate, baseline, candidate_pair, baseline_pair = comparison
        comparison_id = f"{candidate}_vs_{baseline}"
        if candidate_pair != baseline_pair:
            derived_comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "candidate_rows_from": candidate_pair,
                    "baseline_rows_from": baseline_pair,
                    "join_keys": ["dataset", "case_id", "stage", "metric"],
                }
            )
        dataset_outputs = []
        for dataset in DATASETS:
            metrics_out = {}
            comparison_case_ids = None
            comparison_languages = set()
            comparison_snrs = set()
            for metric in METRICS:
                rows, source_pair_ids = paired_rows_for_metric(comparison, dataset, STAGE, metric, groups)
                require(len(rows) == EXPECTED_CASES, f"{comparison_id}/{dataset}/{metric}: expected {EXPECTED_CASES} rows")
                case_ids = {row["case_id"] for row in rows}
                require(len(case_ids) == EXPECTED_CASES, f"{comparison_id}/{dataset}/{metric}: expected {EXPECTED_CASES} unique cases")
                if comparison_case_ids is None:
                    comparison_case_ids = case_ids
                else:
                    require(case_ids == comparison_case_ids, f"{comparison_id}/{dataset}: metric case sets differ")
                comparison_languages.update(row["language"] for row in rows)
                comparison_snrs.update(row["snr_db"] for row in rows)
                n_clusters = cluster_count(rows)
                require(n_clusters == EXPECTED_CLUSTERS[dataset],
                        f"{comparison_id}/{dataset}/{metric}: expected {EXPECTED_CLUSTERS[dataset]} clusters, got {n_clusters}")
                mean_delta = fmean(row["delta"] for row in rows)
                ci_low, ci_high = bootstrap(rows, "delta", BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, None, None)
                require(ci_low is not None and ci_high is not None, f"{comparison_id}/{dataset}/{metric}: CI not computed")
                candidate_mean = summary_mean(indexed_summary, dataset, candidate, STAGE, metric)
                baseline_mean = summary_mean(indexed_summary, dataset, baseline, STAGE, metric)
                summary_delta = candidate_mean - baseline_mean
                mean_error = abs(mean_delta - summary_delta)
                max_mean_error = max(max_mean_error, mean_error)
                if mean_error > MEAN_TOLERANCE:
                    mismatch_count += 1
                metrics_out[metric] = {
                    "n": len(rows),
                    "n_clusters": n_clusters,
                    "n_languages": len({row["language"] for row in rows}),
                    "n_snrs": len({row["snr_db"] for row in rows}),
                    "mean_delta": mean_delta,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "summary_candidate_mean": candidate_mean,
                    "summary_baseline_mean": baseline_mean,
                    "summary_delta": summary_delta,
                    "mean_check_abs_error": mean_error,
                    "row_source_pair_ids": source_pair_ids,
                }
                metric_count += 1
            comparisons_out.append(
                {
                    "comparison_id": comparison_id,
                    "dataset": dataset,
                    "candidate": candidate,
                    "baseline": baseline,
                    "cluster_unit": CLUSTER_UNITS[dataset],
                    "n_cases": EXPECTED_CASES,
                    "n_clusters": EXPECTED_CLUSTERS[dataset],
                    "languages": sorted(comparison_languages),
                    "snr_db": sorted(comparison_snrs),
                    "metrics": metrics_out,
                }
            )

    controls = {}
    for control_stage in CONTROL_STAGES:
        case_metric_deltas = 0
        nonzero_delta_count = 0
        max_abs_delta = 0.0
        control_mismatch_count = 0
        control_max_mean_error = 0.0
        for comparison in COMPARISONS:
            candidate, baseline = comparison[:2]
            for dataset in DATASETS:
                for metric in METRICS:
                    rows, _ = paired_rows_for_metric(comparison, dataset, control_stage, metric, groups)
                    require(len(rows) == EXPECTED_CASES, f"control {control_stage}/{dataset}/{metric}: expected {EXPECTED_CASES} rows")
                    for row in rows:
                        delta = row["delta"]
                        case_metric_deltas += 1
                        nonzero_delta_count += int(delta != 0.0)
                        max_abs_delta = max(max_abs_delta, abs(delta))
                    candidate_mean = summary_mean(indexed_summary, dataset, candidate, control_stage, metric)
                    baseline_mean = summary_mean(indexed_summary, dataset, baseline, control_stage, metric)
                    control_mean_error = abs((candidate_mean - baseline_mean) - fmean(row["delta"] for row in rows))
                    control_max_mean_error = max(control_max_mean_error, control_mean_error)
                    if control_mean_error > MEAN_TOLERANCE:
                        control_mismatch_count += 1
        controls[control_stage] = {
            "comparisons": len(COMPARISONS),
            "datasets": len(DATASETS),
            "metrics": len(METRICS),
            "case_metric_deltas": case_metric_deltas,
            "nonzero_delta_count": nonzero_delta_count,
            "max_abs_delta": max_abs_delta,
            "summary_mean_mismatch_count": control_mismatch_count,
            "max_abs_summary_mean_mismatch": control_max_mean_error,
        }

    require(len(comparisons_out) == len(COMPARISONS) * len(DATASETS), "comparison count mismatch")
    require(metric_count == len(COMPARISONS) * len(DATASETS) * len(METRICS), "metric count mismatch")
    require(mismatch_count == 0, f"summary mean mismatch count is {mismatch_count}")
    require(all(item["nonzero_delta_count"] == 0 for item in controls.values()), "negative control has a nonzero delta")
    require(all(item["summary_mean_mismatch_count"] == 0 for item in controls.values()),
            "negative control summary mean mismatch")

    output = {
        "schema_version": 1,
        "status": "complete",
        "analysis_class": "exploratory_paired_ci",
        "promotion_gate": False,
        "multiplicity_correction": "none",
        "source_receipt": {
            "score_summary": {
                "path": "reports/score-summary.json",
                "sha256": summary_sha256,
                "bytes": SUMMARY_PATH.stat().st_size,
                "generated_utc": summary.get("generated_utc"),
                "schema_version": summary.get("schema_version"),
                "status": summary.get("status"),
            },
            "paired_deltas": {
                "path": "reports/paired-deltas.csv",
                "sha256": paired_sha256,
                "bytes": PAIRED_PATH.stat().st_size,
            },
            "bootstrap_implementation": {
                "path": "summarize_v2.py",
                "symbol": "bootstrap",
                "sha256": summarizer_sha256,
            },
            "filters": {
                "datasets": list(DATASETS),
                "stage": STAGE,
                "metrics": list(METRICS),
                "comparisons": [f"{item[0]}_vs_{item[1]}" for item in COMPARISONS],
                "paired_input_match": True,
                "paired_diff_eligible": True,
                "datasets_pooled": False,
                "control_stages": list(CONTROL_STAGES),
                "source_pair_ids": sorted(SOURCE_PAIR_IDS),
                "mos_or_model_rerun": False,
            },
            "derived_comparisons": derived_comparisons,
        },
        "bootstrap": {
            "implementation": "summarize_v2.bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "percentiles": [2.5, 97.5],
            "percentile_index_rule": "int(p * (len(draws) - 1)) after sorting",
            "resample_with_replacement": True,
            "stratified_by_language": True,
            "cluster_unit_by_dataset": CLUSTER_UNITS,
            "cluster_sampling_synchronized_across_snr": True,
        },
        "checks": {
            "comparison_count": len(comparisons_out),
            "metric_count": metric_count,
            "cases_per_comparison": EXPECTED_CASES,
            "clusters_per_dataset": EXPECTED_CLUSTERS,
            "input_match_all": True,
            "diff_eligible_all": True,
            "summary_mean_mismatch_count": mismatch_count,
            "summary_mean_tolerance": MEAN_TOLERANCE,
            "max_abs_summary_mean_mismatch": max_mean_error,
            "negative_controls": controls,
        },
        "interpretation": {
            "use": "exploratory paired uncertainty only",
            "multiple_comparison_warning": "No multiplicity correction; individual 95% intervals are exploratory.",
            "promotion_decision": "not established; this report is not a promotion gate",
            "datasets_never_pooled": True,
        },
        "comparisons": comparisons_out,
    }

    payload = (json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT_PATH.write_bytes(payload)
    output_sha256 = hashlib.sha256(payload).hexdigest()

    controls_clean = controls["clean_reference"]
    controls_input = controls["tx_input"]
    print("status=complete")
    print(f"comparisons={len(comparisons_out)} metrics={metric_count} mean_mismatches={mismatch_count}")
    print("clusters=" + ",".join(f"{dataset}:{EXPECTED_CLUSTERS[dataset]}" for dataset in DATASETS))
    print(f"clean_reference_nonzero={controls_clean['nonzero_delta_count']} tx_input_nonzero={controls_input['nonzero_delta_count']}")
    for comparison in comparisons_out:
        if comparison["dataset"] in {"v1speech48", "newtest48"} and comparison["comparison_id"] == "newloss320-pitch05_vs_newloss320":
            metric = comparison["metrics"]["minus30_G"]
            print(
                f"control {comparison['dataset']} pitch-vs-320 delta_G={metric['mean_delta']:.16g} "
                f"ci=[{metric['ci95_low']:.16g},{metric['ci95_high']:.16g}]"
            )
    print(f"output=reports/paired-ci.json bytes={len(payload)} sha256={output_sha256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
