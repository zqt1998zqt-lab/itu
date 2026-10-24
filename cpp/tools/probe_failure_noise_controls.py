#!/usr/bin/env python3
"""Replay noise-attenuation failures using the existing frozen ablation runner."""
import json
from pathlib import Path

import analyze_failure_ablations as ab

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "build/failure-cause-analysis-20261017"


def select_noise_cases():
    candidates = {rate: [] for rate in (16000, 24000, 48000)}
    for run, manifest in ab.SOURCES:
        source = PROJECT / "build" / run
        core = {s["id"] for s in json.loads(
            (PROJECT / "datasets" / manifest / "manifest.json").read_text()
        )["scenes"] if s["track"] == "spectrum_screened_core"}
        for line in (source / "results.jsonl").open():
            row = json.loads(line)
            scene = row["scene"]
            if scene["id"] not in core or scene["family"] != "ns" or scene["mics"] != 1:
                continue
            failed = [g for g in row["acceptance"]["gates"] if g["status"] == "fail"]
            # Select the worst attenuation among cases with this sole failed gate.
            if len(failed) != 1:
                continue
            gate = failed[0]
            if (gate["segment"], gate["metric"]) != ("noise_only", "total_attenuation_db"):
                continue
            candidates[scene["sample_rate"]].append(
                (gate["value"], str(source), scene["id"], source, row)
            )
    choices = []
    for rate, values in candidates.items():
        if not values:
            raise ValueError(f"no noise-only failure candidate at {rate}")
        _, _, _, source, row = min(values, key=lambda v: v[:3])
        choices.append(("noise_attenuation", source, row))
    return choices


def main():
    receipt = ROOT / "noise-controls-driver-identity.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    bound = {str(p): ab.ctl.ev.digest(p) for p in (Path(__file__), Path(ab.__file__))}
    ab.ctl.legacy.save(receipt, {"bindings": bound, "scope": "diagnostic, not candidate acceptance"})
    ab.OUT = ROOT / "noise-controls"
    ab.select = select_noise_cases
    ab.main()
    changed = [p for p, digest in bound.items() if ab.ctl.ev.digest(Path(p)) != digest]
    ab.ctl.legacy.save(ROOT / "noise-controls-driver-verification.json", {
        "identity_verified": not changed, "changed_dependencies": changed,
        "bindings": bound,
    })
    if changed:
        raise ValueError(f"driver identity changed: {changed}")


if __name__ == "__main__":
    main()
