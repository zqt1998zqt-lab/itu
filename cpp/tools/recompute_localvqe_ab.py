#!/usr/bin/env python3
"""Verify retained artifact hashes and re-score actual PCM in a new report root."""
import argparse
from collections import Counter
import json
from pathlib import Path

import run_localvqe_ab as ab
import run_audio_eval as ev
import check_audio_acceptance as ac


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    source, root, dataset = a.source.resolve(), a.root.resolve(), a.dataset.resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads((source / "report.json").read_text())
    manifest_path = dataset / "inputs-manifest.json"
    if ev.digest(manifest_path) != old["identity"]["manifest_sha256"] or ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen inputs or thresholds changed")
    manifest = json.loads(manifest_path.read_text())
    scenes = {s["id"]: s for s in manifest["scenes"]}
    rules = json.loads(ac.RULES.read_text())
    for artifact in old["identity"]["artifacts"].values():
        if ev.digest(source / artifact["path"]) != artifact["sha256"]:
            raise ValueError("original executable/library/model hash mismatch")
    keys = [(r["id"], r["rate"], r["variant"]) for r in old["rows"]]
    if len(keys) != len(set(keys)) or len(keys) != len(old["identity"]["scene_ids"])*len(old["identity"]["rates"])*len(old["identity"]["variants"]):
        raise ValueError("incomplete or duplicate original output matrix")
    rows = []
    for old_row in old["rows"]:
        scene = scenes[old_row["id"]]
        inputs = dataset / "cases" / scene["id"]
        for filename, record in scene["files"].items():
            if ev.digest(inputs / filename) != record["sha256"]:
                raise ValueError("original source PCM changed")
        folder = source / f"{scene['id']}-r{old_row['rate']}" / old_row["variant"]
        for filename, sha in old_row["files"].items():
            if ev.digest(folder / filename) != sha:
                raise ValueError("original result/report hash mismatch")
        invocation = json.loads((folder / "invocation.json").read_text())
        if invocation["exit_code"] != 0 and not old_row.get("execution_failed"):
            raise ValueError("unreported original invocation failure")
        for filename, sha in invocation["input_sha256"].items():
            if ev.digest(folder.parent / filename) != sha:
                raise ValueError("resampled input hash mismatch")
        if old_row.get("execution_failed"):
            rows.append(dict(old_row,evidence_directory=str(folder)))
            continue
        checked, measured = ab.measure(scene, inputs, folder / "result", old_row["rate"])
        verdict = ac.check_scene(checked, measured, rules)
        def passed(gates):
            return all(g["status"] == "pass" or g["status"] == "unavailable" and g["optional"] for g in gates)
        row = dict(old_row, measurements=measured, acceptance=verdict,
            quality_passed=passed([g for g in verdict["gates"] if g["segment"] != "runtime"]),
            runtime_passed=passed([g for g in verdict["gates"] if g["segment"] == "runtime"]),
            evidence_directory=str(folder))
        rows.append(row)
    groups = {}
    for key in sorted({(r["rate"], r["variant"]) for r in rows}):
        selected = [r for r in rows if (r["rate"], r["variant"]) == key]
        groups[f"{key[0]}-{key[1]}"] = dict(rows=len(selected), quality_pass=sum(r["quality_passed"] for r in selected),
            runtime_pass=sum(r["runtime_passed"] for r in selected), all_pass=sum(r["acceptance"]["passed"] for r in selected),
            failed_metrics=dict(Counter(g["metric"] for r in selected for g in r["acceptance"]["gates"] if g["status"] != "pass" and not g["optional"])))
    identity = dict(old["identity"], original_run_root=str(source),
        original_report_sha256=ev.digest(source / "report.json"),
        recompute_runner_sha256=ev.digest(Path(__file__)), measurement_runner_sha256=ev.digest(Path(ab.__file__)),
        remeasurement_scope="retained output PCM, native I/O attenuation and PCM16 censoring; no audio reprocessing or fitted lag",
        frontend_scope="frontend alone AEC-only; frontend_webrtc/ulunas include separate 16k NS/AGC",
        metadata_audit="earlier probe used incorrect architecture key; models independently pinned by hashes; codec delay unchanged; current probe checks general.architecture")
    report = dict(identity=identity, rows=rows, groups=groups, negative_controls=ab.negative_control_summary(rows))
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(groups=groups, negative_controls=report["negative_controls"]), indent=2))


if __name__ == "__main__":
    main()
