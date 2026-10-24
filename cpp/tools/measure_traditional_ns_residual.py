#!/usr/bin/env python3
"""Serial, CPU-pinned replay of retained NS invocations; never re-score gates.

Uses the same frozen executable for off, uniform and SNR-gated treatments.
Real input WAVs, arguments and output hashes are retained for every replay.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import wave


PROJECT = Path(__file__).resolve().parents[2]
BRANCHES = {
    "webrtc_v5_bounded09": "webrtc",
    "speex_v5_steadydual": "speex",
    "omlsa_imcra_v2_mild": "omlsa",
    "super_gaussian_balanced": "super",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def counters():
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        key, *fields = line.split()
        if key.startswith("cpu") and key[3:].isdigit():
            values = [int(x) for x in fields[:8]]
            result[int(key[3:])] = (sum(values), values[3] + values[4])
    return result


def choose_cpu(explicit):
    allowed = os.sched_getaffinity(0)
    if explicit is not None:
        if explicit not in allowed:
            raise ValueError("requested CPU is outside current affinity")
        return explicit, {}
    before = counters()
    time.sleep(.3)
    after = counters()
    loads = {}
    for cpu in allowed:
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        loads[cpu] = 1 - idle / max(total, 1)
    return min(loads, key=loads.get), loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    frozen = args.frozen_root.resolve()
    baseline = args.baseline_root.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = frozen / "ns_vehicle_probe"
    binary_hash = sha(executable)
    identity = json.loads((frozen / "identity.json").read_text())
    if binary_hash != identity["binary_sha256"]:
        raise ValueError("frozen executable identity drift")
    rows = [json.loads(line) for line in
            (baseline / "full-r3-fixed-results.jsonl").read_text().splitlines()]
    expected = {(r["variant"], r["run"], r["id"],
                 "off" if r["treatment"] == "off" else "uniform3"):
                r["output_sha256"] for r in rows}
    gated = [json.loads(line) for line in
             (frozen / "full-gate-results.jsonl").read_text().splitlines()]
    expected.update({(r["variant"], r["run"], r["id"], "gated3"):
                     r["output_sha256"] for r in gated})
    reference = [r for r in rows if r["variant"] == next(iter(BRANCHES))
                 and r["treatment"] == "off"]
    # Deterministic real scenes covering steady/transient and both SNR extremes.
    selection = {}
    for kind in sorted({r["noise_class"] for r in reference}):
        group = sorted((r for r in reference if r["noise_class"] == kind),
                       key=lambda r: (r["snr_db"], r["run"], r["id"]))
        positions = [len(group) // 2] if args.paced else [0, len(group) // 2, len(group) - 1]
        for index in positions:
            row = group[index]
            selection[(row["run"], row["id"])] = row
    cpu, loads = choose_cpu(args.cpu)
    original_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    manifest = dict(binary_sha256=binary_hash, binary=str(executable),
                    machine=platform.machine(), cpu=cpu,
                    original_affinity=original_affinity,
                    sampled_cpu_load=loads, paced=args.paced,
                    parallel_workers=1, selection=[dict(run=r["run"], id=r["id"],
                    noise_class=r["noise_class"], snr_db=r["snr_db"])
                    for r in selection.values()])
    save(output / "identity.json", manifest)
    measurements = []
    for variant, branch in BRANCHES.items():
        for (run, case_id), scene in selection.items():
            token = hashlib.sha256((run + "/" + case_id).encode()).hexdigest()[:20]
            previous = baseline / "full-r3-fixed" / branch / "off" / token / "invocation.json"
            invocation = json.loads(previous.read_text())
            if invocation["returncode"]:
                raise ValueError("baseline invocation failed")
            for treatment in ("off", "uniform3", "gated3"):
                folder = output / branch / token / treatment
                folder.mkdir(parents=True, exist_ok=False)
                command = invocation["argv"][:]
                if any(x.startswith("--residual-") for x in command[19:]):
                    raise ValueError("baseline off invocation includes residual options")
                command[0] = str(executable)
                command[4] = str(folder / "result")
                if treatment != "off":
                    command += ["--residual-max-db=3", "--residual-speech-guard=.75"]
                if treatment == "gated3":
                    command += ["--residual-gate-snr-db=-12", "--residual-gate-width-db=3"]
                if args.paced:
                    command += ["--pace-10ms"]
                source = Path(command[3])
                input_hash = sha(source)
                started = time.monotonic()
                proc = subprocess.run(command, capture_output=True, text=True)
                elapsed = time.monotonic() - started
                save(folder / "invocation.json", dict(argv=command,
                     returncode=proc.returncode, stderr=proc.stderr,
                     input_sha256=input_hash, elapsed_seconds=elapsed))
                proc.check_returncode()
                engine = json.loads((folder / "result.json").read_text())
                if sha(source) != input_hash or sha(executable) != binary_hash:
                    raise ValueError("input or binary changed during replay")
                if engine["io_rate"] != 16000 or engine["src_delay_samples"] != 0:
                    raise ValueError("runtime comparison must use native 16 kHz")
                if engine["residual_max_db"] != (0 if treatment == "off" else 3):
                    raise ValueError("residual option did not reach engine")
                with (folder / "result.csv").open() as stream:
                    timings = [float(row["process_ms"]) for row in csv.DictReader(stream)]
                if len(timings) != engine["frames"] or len(timings) < 2:
                    raise ValueError("incomplete frame timing evidence")
                audio_path = folder / "result.after.wav"
                output_hash = sha(audio_path)
                if output_hash != expected[(variant, run, case_id, treatment)]:
                    raise ValueError("serial replay differs from scored PCM")
                with wave.open(str(audio_path)) as audio:
                    if (audio.getnchannels() != 1 or audio.getframerate() != 16000
                            or audio.getnframes() != engine["samples"] + engine["delay_samples"]):
                        raise ValueError("serial output shape or declared delay differs")
                row = dict(variant=variant, treatment=treatment, run=run, id=case_id,
                           noise_class=scene["noise_class"], snr_db=scene["snr_db"],
                           input_sha256=input_hash, engine=engine,
                           output_sha256=output_hash,
                           first_ms=timings[0], steady_ms=timings[1:])
                measurements.append(row)
                print(f"{len(measurements)} {branch} {treatment} first={timings[0]:.4f} "
                      f"max={max(timings):.4f} over10={sum(t > 10 for t in timings)}", flush=True)
    summary = {}
    for variant in BRANCHES:
        for treatment in ("off", "uniform3", "gated3"):
            group = [r for r in measurements if r["variant"] == variant
                     and r["treatment"] == treatment]
            steady = [t for r in group for t in r["steady_ms"]]
            summary[variant + "." + treatment] = dict(
                streams=len(group), steady_frames=len(steady),
                first_max_ms=max(r["first_ms"] for r in group),
                steady_mean_ms=statistics.fmean(steady),
                steady_p50_ms=percentile(steady, .5), steady_p95_ms=percentile(steady, .95),
                steady_p99_ms=percentile(steady, .99), steady_max_ms=max(steady),
                steady_over_10ms=sum(t > 10 for t in steady),
                first_over_10ms=sum(r["first_ms"] > 10 for r in group),
                paced_deadline_misses=sum(r["engine"]["paced_deadline_misses"] for r in group))
    save(output / "measurements.json", measurements)
    save(output / "summary.json", summary)
    save(output / "report.json", dict(state="completed", runs=len(measurements),
         identity_verified=True, cpu=cpu, paced=args.paced,
         interpretation="serial pinned-host observation; no device real-time certification"))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
