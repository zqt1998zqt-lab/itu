#!/usr/bin/env python3
"""Serial CPU-pinned and optional paced replay of the speech-band profile."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from diagnose_traditional_ns_remaining import token
from run_ns_speechband_full import sha

P = Path(__file__).resolve().parents[2]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(fraction * (len(ordered) - 1))))]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    root, out = args.quality_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (root / "results.jsonl").open()
            if json.loads(line)["recovery"] == 1]
    chosen = []
    for kind in ("steady", "transient"):
        group = sorted((row for row in rows if row["noise_class"] == kind),
                       key=lambda row: (row["snr_db"], row["run"], row["id"]))
        chosen.append(group[len(group) // 2] if args.paced else group[0])
        chosen.append(group[len(group) // 2] if args.paced else group[-1])
    chosen = list({(row["run"], row["id"]): row for row in chosen}.values())
    identity = json.loads((root / "identity.json").read_text())
    binary = Path(next(iter(identity["bindings"]))).resolve()
    original_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {min(original_affinity)})
    save(out / "identity.json", dict(
        quality_bindings=identity["bindings"],
        quality_results_sha256=sha(root / "results.jsonl"),
        machine=platform.machine(), cpu=min(original_affinity),
        original_affinity=original_affinity, paced=args.paced,
        parallel_workers=1, target_device_validation=False))
    measurements = []
    for row in chosen:
        folder = out / token(row) / ("paced" if args.paced else "unpaced")
        folder.mkdir(parents=True, exist_ok=False)
        scored = root / "recovery-1p0" / token(row) / "invocation.json"
        command = json.loads(scored.read_text())["argv"]
        command[4] = str(folder / "result")
        if args.paced:
            command.append("--pace-10ms")
        started = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True)
        save(folder / "invocation.json", dict(
            argv=command, returncode=proc.returncode, stderr=proc.stderr,
            elapsed_seconds=time.monotonic() - started))
        proc.check_returncode()
        engine = json.loads((folder / "result.json").read_text())
        expected = root / "recovery-1p0" / token(row) / "result.after.wav"
        if (sha(folder / "result.after.wav") != sha(expected)
                or engine["core_rate"] != 16000 or engine["src_delay_samples"] != 0
                or engine["delay_samples"] != row["delay_samples"]):
            raise ValueError("runtime replay differs from scored PCM or timebase")
        if args.paced and (
                not engine["paced"] or engine["paced_deadline_misses"] != 0):
            raise ValueError("paced replay missed a 10 ms completion deadline")
        with (folder / "result.csv").open() as stream:
            times = [float(item["process_ms"]) for item in csv.DictReader(stream)]
        if len(times) != engine["frames"]:
            raise ValueError("incomplete frame timing evidence")
        measurements.append(dict(
            run=row["run"], id=row["id"], noise_class=row["noise_class"],
            snr_db=row["snr_db"], first_ms=times[0], steady_ms=times[1:],
            paced_deadline_misses=engine["paced_deadline_misses"],
            output_sha256=sha(folder / "result.after.wav"), engine=engine))
        print(row["id"], "max_ms", max(times), "over_10ms", sum(t > 10 for t in times),
              "deadline_misses", engine["paced_deadline_misses"], flush=True)
    times = [value for row in measurements for value in row["steady_ms"]]
    summary = dict(
        streams=len(measurements), steady_frames=len(times),
        first_max_ms=max(row["first_ms"] for row in measurements),
        steady_mean_ms=statistics.fmean(times),
        steady_p95_ms=percentile(times, .95),
        steady_p99_ms=percentile(times, .99),
        steady_max_ms=max(times),
        over_10ms=sum(value > 10 for value in times),
        first_over_10ms=sum(row["first_ms"] > 10 for row in measurements),
        paced_deadline_misses=sum(row["paced_deadline_misses"] for row in measurements))
    save(out / "measurements.json", measurements)
    save(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
