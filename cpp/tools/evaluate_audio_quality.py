#!/usr/bin/env python3
"""Evaluate a JSON manifest of aligned clean/input/multi-algorithm audio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from server.quality.engine import evaluate
from server.quality.evaluators import DEFAULT_SCOREQ_ASSETS, create_evaluators
from server.quality.metrics import load_audio, sha
from cpp.tools.ecnr_perceptual import save_json


def run(manifest_path: Path, output: Path, evaluators: list[str], cache: Path,
        scoreq_assets: Path = DEFAULT_SCOREQ_ASSETS) -> dict:
    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise ValueError("require schema_version=1 and nonempty cases")
    if (output / "report.json").exists():
        raise ValueError("report already exists; choose a new output directory to preserve evidence")
    names = [c["id"] for c in manifest["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("case ids must be unique")
    scorers = create_evaluators(evaluators, cache, scoreq_assets)
    completed, bindings = [], {str(manifest_path):sha(manifest_bytes)}
    for index, case in enumerate(manifest["cases"]):
        def read(spec):
            path = (manifest_path.parent / spec["path"]).resolve()
            payload = path.read_bytes()
            digest = sha(payload)
            if not spec.get("sha256") or digest != spec["sha256"]:
                raise ValueError(f"input file SHA mismatch: {path}")
            bindings[str(path)] = digest
            return load_audio(payload, channel=spec.get("channel"),
                              offset_samples=spec.get("offset_samples", 0), pcm16=spec.get("pcm16", False))
        before = read(case["before"])
        clean = read(case["clean"]) if case.get("clean") else None
        outputs = {name:read(spec) for name,spec in case["outputs"].items()}
        report = evaluate(case["id"], before, outputs, clean=clean, windows=case.get("windows"),
                          evaluators=scorers, mos_deadband=manifest.get("diagnostic_mos_deadband", .1))
        path = output / "cases" / f"{index:06d}.json"
        save_json(path, report)
        completed.append(dict(id=case["id"], state=report["state"], path=str(path.resolve()),
                              sha256=sha(path.read_bytes()), algorithms=report["algorithms"],
                              review_queue=report["review_queue"], coverage=report["coverage"],
                              cross_checked_comparisons=report["cross_checked_comparisons"]))
        print(f"{index+1}/{len(names)} {case['id']} {report['state']}", flush=True)
    if any(sha(Path(path).read_bytes()) != digest for path,digest in bindings.items()):
        raise ValueError("input files changed during evaluation")
    summary = dict(schema_version=1, state="completed" if all(c["state"] == "completed" for c in completed) else "completed_partial",
                   manifest_sha256=sha(manifest_bytes), cases=completed,
                   evaluator_identities={s.name:s.identity for s in scorers},
                   cross_checked_comparisons=sum(c["cross_checked_comparisons"] for c in completed),
                   review_queue=[r for c in completed for r in c["review_queue"]],
                   source_bindings=bindings, cli_sha256=sha(Path(__file__).read_bytes()),
                   scope="Independent quality diagnostics; existing DSP acceptance results unchanged")
    save_json(output / "report.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluators", default="dnsmos,scoreq")
    parser.add_argument("--cache", type=Path, default=PROJECT / "build/quality-evaluations/cache")
    parser.add_argument("--scoreq-assets", type=Path, default=DEFAULT_SCOREQ_ASSETS)
    parser.add_argument("--allow-partial", action="store_true", help="allow exit 0 when multi-model coverage is incomplete")
    args = parser.parse_args()
    result = run(args.manifest, args.out, args.evaluators.split(","), args.cache, args.scoreq_assets)
    print(json.dumps(dict(state=result["state"], cases=len(result["cases"]),
                          cross_checked=result["cross_checked_comparisons"], report=str(args.out / "report.json"))))
    return 0 if result["state"] == "completed" or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
