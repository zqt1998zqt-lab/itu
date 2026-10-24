#!/usr/bin/env python3
"""Full paired validation of fixed rumble filtering, with unchanged NS gates."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from diagnose_traditional_ns_remaining import BASE, FROZEN, P, counterfactual, key, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = BASE / "full-r3-fixed-results.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [r for r in rows if r["treatment"] == "off" and r["variant"] != "webrtc_v5_bounded09"]
    counts = collections.Counter(r["variant"] for r in rows)
    if len(counts) != 3 or any(n != 692 for n in counts.values()):
        raise ValueError("incomplete fixed-rumble input coverage")
    scenes = {(r["run"], r["scene"]["id"]): r["scene"] for r in
              json.loads((P / "build/ns-gtcrn-direct-20261018/full/identity.json").read_text())["selection"]}
    identity = dict(binary_sha256=sha(FROZEN / "ns_vehicle_probe"), baseline_sha256=sha(source),
                    diagnostic_script_sha256=sha(Path(__file__).with_name("diagnose_traditional_ns_remaining.py")),
                    script_sha256=sha(Path(__file__)), total=len(rows),
                    change="rumble_kind adaptive -> linear; all other argv unchanged",
                    thresholds_changed=False)
    save(out / "identity.json", identity)
    tasks = [(out, r, scenes[key(r)], "baseline_passed_control" if r["passed"] else "baseline_failed", "fixed_rumble")
             for r in rows]
    results = []
    with ProcessPoolExecutor(args.jobs) as pool, (out / "results.jsonl").open("x") as stream:
        for i, future in enumerate(as_completed([pool.submit(counterfactual, task) for task in tasks]), 1):
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            if i % 100 == 0 or i == len(tasks):
                save(out / "report.json", dict(state="running", completed=i, total=len(tasks)))
                print(i, "/", len(tasks), flush=True)
    if sha(FROZEN / "ns_vehicle_probe") != identity["binary_sha256"] or sha(source) != identity["baseline_sha256"]:
        raise ValueError("input artifacts changed during validation")
    summary = {}
    for variant in counts:
        group = [r for r in results if r["variant"] == variant]
        unique = {key(r) for r in group}
        if len(unique) != len(group) or len(group) != 692:
            raise ValueError("duplicate/missing full results")
        summary[variant] = dict(total=len(group), before_passed=sum(r["baseline_passed"] for r in group),
             after_passed=sum(r["passed"] for r in group),
             recovered=sum(not r["baseline_passed"] and r["passed"] for r in group),
             regressed=sum(r["baseline_passed"] and not r["passed"] for r in group),
             noise_failed=sum(any("attenuation" in f for f in r["failures"]) for r in group),
             speech_failed=sum(any("attenuation" not in f for f in r["failures"]) for r in group),
             noise_classes={kind: dict(total=sum(r["noise_class"] == kind for r in group),
                 passed=sum(r["passed"] and r["noise_class"] == kind for r in group))
                 for kind in ("steady", "transient")})
    save(out / "summary.json", summary)
    save(out / "report.json", dict(state="completed", completed=len(results), total=len(tasks),
         thresholds_changed=False, full_quality_validation=True, realtime_validation=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
