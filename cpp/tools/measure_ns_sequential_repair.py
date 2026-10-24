#!/usr/bin/env python3
"""Serial CPU-pinned replay of scored native-16k NS profiles.

Use --paced for one steady and one transient real stream per profile; output
PCM must match the quality run. This measures this host, not a target device.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from audit_traditional_ns_residual import read_rows, sha
from diagnose_traditional_ns_remaining import BASE, original_folder, token
from measure_traditional_ns_residual import choose_cpu, percentile, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    root, out = args.quality_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = json.loads((root / "report.json").read_text())
    if status["state"] != "completed":
        raise ValueError("quality batch must finish before serial runtime replay")
    identity = json.loads((root / "identity.json").read_text())
    profiles = identity["plan"]
    rows = read_rows(root / "results.jsonl")
    reference = {r["profile"]: {} for r in rows}
    for row in rows:
        reference[row["profile"]][(row["run"], row["id"])] = row
    baseline = {(r["variant"], r["run"], r["id"]): r
                for r in read_rows(BASE / "full-r3-fixed-results.jsonl") if r["treatment"] == "off"}
    cpu, loads = choose_cpu(None)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    save(out / "identity.json", dict(quality_bindings=identity["bindings"],
        quality_results_sha256=sha(root / "results.jsonl"), machine=platform.machine(),
        cpu=cpu, original_affinity=affinity, sampled_cpu_load=loads,
        paced=args.paced, parallel_workers=1, target_device_validation=False))
    measurements = []
    for profile in profiles:
        name = profile["name"]
        group = list(reference[name].values())
        chosen = []
        for kind in ("steady", "transient"):
            available = sorted((r for r in group if r["noise_class"] == kind),
                               key=lambda r: (r["snr_db"], r["run"], r["id"]))
            for index in ([len(available) // 2] if args.paced else [0, len(available) - 1]):
                chosen.append(available[index])
        for row in chosen:
            for treatment in (["candidate"] if args.paced else ["off", "candidate"]):
                folder = out / name / token(row) / treatment
                folder.mkdir(parents=True, exist_ok=False)
                scored_invocation = json.loads((root / name / token(row) / "invocation.json").read_text())
                binary = Path(scored_invocation["argv"][0])
                old = baseline[(row["variant"], row["run"], row["id"])]
                command = json.loads((original_folder(old) / "invocation.json").read_text())["argv"]
                command[0], command[4] = str(binary), str(folder / "result")
                if treatment == "candidate":
                    command.extend(profile.get("options", []))
                if args.paced:
                    command.append("--pace-10ms")
                input_path = Path(command[3])
                input_hash = sha(input_path)
                if sha(binary) != identity["bindings"][str(binary)]:
                    raise ValueError("runtime binary drift")
                started = time.monotonic()
                proc = subprocess.run(command, capture_output=True, text=True)
                save(folder / "invocation.json", dict(argv=command, returncode=proc.returncode,
                    stderr=proc.stderr, elapsed_seconds=time.monotonic()-started, input_sha256=input_hash))
                proc.check_returncode()
                engine = json.loads((folder / "result.json").read_text())
                expected = row if treatment == "candidate" else old
                if (sha(folder / "result.after.wav") != expected["output_sha256"]
                        or sha(input_path) != input_hash
                        or engine["core_rate"] != 16000 or engine["src_delay_samples"] != 0
                        or engine["delay_samples"] != expected["engine"]["delay_samples"]):
                    raise ValueError("serial replay differs from the scored PCM or timebase")
                with (folder / "result.csv").open() as stream:
                    times = [float(r["process_ms"]) for r in csv.DictReader(stream)]
                if len(times) != engine["frames"] or len(times) < 2:
                    raise ValueError("incomplete frame timing evidence")
                measurements.append(dict(profile=name, treatment=treatment, run=row["run"], id=row["id"],
                    first_ms=times[0], steady_ms=times[1:], engine=engine,
                    output_sha256=expected["output_sha256"]))
                print(name, treatment, "max_ms", max(times), "over_10ms", sum(t>10 for t in times),
                      "deadline_misses", engine["paced_deadline_misses"], flush=True)
    summaries = {}
    for name in reference:
        for treatment in (["candidate"] if args.paced else ["off", "candidate"]):
            group = [r for r in measurements if r["profile"] == name and r["treatment"] == treatment]
            times = [t for r in group for t in r["steady_ms"]]
            summaries[name+"."+treatment] = dict(streams=len(group), steady_frames=len(times),
                first_max_ms=max(r["first_ms"] for r in group), steady_mean_ms=statistics.fmean(times),
                steady_p95_ms=percentile(times,.95), steady_p99_ms=percentile(times,.99),
                steady_max_ms=max(times), over_10ms=sum(t>10 for t in times),
                first_over_10ms=sum(r["first_ms"]>10 for r in group),
                paced_deadline_misses=sum(r["engine"]["paced_deadline_misses"] for r in group))
    for filename,digest in identity["bindings"].items():
        if sha(Path(filename)) != digest:
            raise ValueError("quality dependency drift during serial replay")
    save(out / "measurements.json", measurements)
    save(out / "summary.json", summaries)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == "__main__":
    main()
