#!/usr/bin/env python3
"""Bind scored hybrid-v2 manifests to the preserved candidate receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import BUILD_ROOT, atomic_json, require_file, sha256_file, utc_now


VERSIONS = {
    "oldloss256": "hybridv2-nativefixed-oldloss256-20261115",
    "newloss256": "hybridv2-nativefixed-newloss256-20261115",
    "newloss320": "hybridv2-nativefixed-newloss320-20261115",
    "newloss320-pitch05": "hybridv2-nativefixed-newloss320-pitch05-20261115",
}
EXPECTED = {
    "v1speech48": 48,
    "v1clean16": 16,
    "v1continuous4": 4,
    "v1guards": 6,
    "highway120-speech48": 48,
    "highway120-continuous4": 4,
    "highway120-noise2": 2,
    "newtest48": 48,
}
GSN_FIELDS = tuple(f"{prefix}_{metric}" for prefix in ("raw", "minus30") for metric in ("G", "S", "N"))


class ProvenanceError(RuntimeError):
    """A scored manifest is not bound to a valid candidate receipt."""


def load(path: Path, label: str) -> dict[str, Any]:
    document = json.loads(require_file(path, label).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ProvenanceError(f"{label} root is not an object: {path}")
    return document


def build_report(evaluation_root: Path) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    manifest_count = 0
    for variant, version in VERSIONS.items():
        version_root = evaluation_root / version
        progress = load(version_root / "progress.json", f"{variant} progress")
        if progress.get("status") != "complete" or progress.get("failures"):
            raise ProvenanceError(f"{variant} suite is not complete: {progress.get('status')}")
        groups = progress.get("artifacts", {}).get("variants", {}).get(variant)
        if not isinstance(groups, dict) or set(groups) != set(EXPECTED):
            raise ProvenanceError(f"{variant} group coverage mismatch: {sorted(groups or {})}")
        variant_groups: dict[str, Any] = {}
        for group, expected_runs in EXPECTED.items():
            artifact = groups[group]
            candidate_path = Path(str(artifact.get("candidate_receipt", ""))).resolve()
            scored_path = Path(str(artifact.get("scored_manifest", ""))).resolve()
            candidate = load(candidate_path, f"{variant}/{group} candidate receipt")
            scored = load(scored_path, f"{variant}/{group} scored manifest")
            if candidate.get("status") != "complete" or int(candidate.get("failed", 0)) != 0:
                raise ProvenanceError(f"candidate receipt incomplete: {candidate_path}")
            if scored.get("status") != "complete" or int(scored.get("failed_runs", 0)) != 0:
                raise ProvenanceError(f"scored manifest incomplete: {scored_path}")
            candidate_runs = candidate.get("runs", [])
            scored_runs = scored.get("runs", [])
            if len(candidate_runs) != expected_runs or len(scored_runs) != expected_runs:
                raise ProvenanceError(
                    f"{variant}/{group} coverage candidate={len(candidate_runs)} scored={len(scored_runs)} expected={expected_runs}"
                )
            candidate_map = {str(run["case"]["id"]): run for run in candidate_runs}
            scored_map = {str(run["case"]["id"]): run for run in scored_runs}
            if len(candidate_map) != expected_runs or set(candidate_map) != set(scored_map):
                raise ProvenanceError(f"{variant}/{group} case IDs differ between candidate and scored manifests")
            candidate_sha = sha256_file(candidate_path)
            scored_sha = sha256_file(scored_path)
            for case_id in sorted(candidate_map):
                candidate_run = candidate_map[case_id]
                scored_run = scored_map[case_id]
                if candidate_run.get("status") != "ok" or scored_run.get("status") != "ok":
                    raise ProvenanceError(f"{variant}/{group}/{case_id} is not ok")
                if candidate_run.get("upstream_input_exact") is not True:
                    raise ProvenanceError(f"{variant}/{group}/{case_id} lacks upstream_input_exact=true")
                if candidate_run.get("clean_used_for_processing") is not False:
                    raise ProvenanceError(f"{variant}/{group}/{case_id} lacks clean_used_for_processing=false")
                if str(scored_run.get("variant")) != variant:
                    raise ProvenanceError(f"{variant}/{group}/{case_id} scored variant mismatch")
                candidate_window = candidate_run.get("case", {}).get("window_s")
                scored_window = scored_run.get("case", {}).get("window_s")
                if candidate_window != scored_window:
                    raise ProvenanceError(f"{variant}/{group}/{case_id} window mismatch")
                candidate_input_sha = candidate_run.get("input_sha256")
                scored_input_sha = scored_run.get("hashes_before", {}).get("input")
                if not candidate_input_sha or candidate_input_sha != scored_input_sha:
                    raise ProvenanceError(f"{variant}/{group}/{case_id} input SHA binding mismatch")
                pure_noise = bool(
                    candidate_run.get(
                        "pure_noise", scored_run.get("case", {}).get("pure_noise", False)
                    )
                )
                noise_gsn_all_none = True
                stage_rows = 0
                for stage in scored_run.get("scored_stages", []):
                    for row in stage.get("channels_scored", []):
                        stage_rows += 1
                        if any(field not in row for field in GSN_FIELDS):
                            raise ProvenanceError(f"{variant}/{group}/{case_id} omitted GSN fields")
                        if pure_noise and any(row.get(field) is not None for field in GSN_FIELDS):
                            noise_gsn_all_none = False
                if pure_noise and not noise_gsn_all_none:
                    raise ProvenanceError(f"{variant}/{group}/{case_id} pure-noise MOS changed from None")
                all_runs.append(
                    {
                        "score_case_id": case_id,
                        "case_id": case_id,
                        "variant": variant,
                        "group": group,
                        "window_s": scored_window,
                        "input_sha256": candidate_input_sha,
                        "candidate_input_float_sha256": candidate_run.get("candidate_input_float_sha256"),
                        "upstream_input_sha256": candidate_run.get("upstream_input_sha256"),
                        "upstream_input_exact": True,
                        "clean_used_for_processing": False,
                        "pure_noise": pure_noise,
                        "noise_gsn_fields_all_none": noise_gsn_all_none if pure_noise else None,
                        "scored_status": scored_run.get("status"),
                        "stage_rows": stage_rows,
                        "probe_sha256": candidate_run.get("native_sha256"),
                        "model_sha256": candidate_run.get("model_sha256"),
                        "candidate_receipt": str(candidate_path),
                        "candidate_receipt_sha256": candidate_sha,
                        "scored_manifest": str(scored_path),
                        "scored_manifest_sha256": scored_sha,
                    }
                )
            variant_groups[group] = {
                "status": "verified",
                "expected_runs": expected_runs,
                "candidate_receipt": str(candidate_path),
                "candidate_receipt_sha256": candidate_sha,
                "scored_manifest": str(scored_path),
                "scored_manifest_sha256": scored_sha,
            }
            manifest_count += 1
        variants[variant] = {
            "version": version,
            "version_root": str(version_root),
            "runs": sum(EXPECTED.values()),
            "status": "verified",
            "groups": variant_groups,
        }
    if manifest_count != 32 or len(all_runs) != 704:
        raise ProvenanceError(f"expected 32 manifests/704 runs, got {manifest_count}/{len(all_runs)}")
    probe_shas = sorted({str(run["probe_sha256"]) for run in all_runs})
    if probe_shas != ["cca8c9ab5955b2bdde7912203c1c61a1ffdcbd45b124c26c78d952ab7540c152"]:
        raise ProvenanceError(f"unexpected probe SHA set: {probe_shas}")
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "verified",
        "scored_manifest_count": manifest_count,
        "runs": len(all_runs),
        "all_upstream_input_exact": all(run["upstream_input_exact"] for run in all_runs),
        "all_clean_used_for_processing_false": all(not run["clean_used_for_processing"] for run in all_runs),
        "all_pure_noise_gsn_none": all(
            run["noise_gsn_fields_all_none"] is True for run in all_runs if run["pure_noise"]
        ),
        "pure_noise_runs": sum(1 for run in all_runs if run["pure_noise"]),
        "probe_sha256": probe_shas[0],
        "scorer_outputs_modified": False,
        "variants": variants,
        "cases": all_runs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evaluation-root", type=Path, default=BUILD_ROOT / "evaluation")
    parser.add_argument("--output", type=Path, default=BUILD_ROOT / "reports/scored-provenance.json")
    args = parser.parse_args(argv)
    report = build_report(args.evaluation_root.expanduser().resolve())
    atomic_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "manifests": report["scored_manifest_count"],
                "runs": report["runs"],
                "pure_noise_runs": report["pure_noise_runs"],
                "all_upstream_input_exact": report["all_upstream_input_exact"],
                "all_clean_used_for_processing_false": report["all_clean_used_for_processing_false"],
                "all_pure_noise_gsn_none": report["all_pure_noise_gsn_none"],
                "output": str(args.output.expanduser().resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

