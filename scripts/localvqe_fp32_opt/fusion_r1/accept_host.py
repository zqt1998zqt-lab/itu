"""Compare experimental modes to frozen D: five PCM scenes, then serial timing."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import wave

from build_fusion_v2 import BUILD, HOST_CLOCK, OUT, ROOT, guard, save, sha

MODES = ("baseline", "norm", "tiled", "epilogue")
NODE_COUNTS = {
    "baseline": (0, 0, 0, 32, 37, 23),
    "norm": (29, 0, 0, 32, 37, 23),
    "tiled": (0, 1, 0, 31, 36, 23),
    "epilogue": (0, 0, 19, 32, 37, 4),
}
INVENTORY = re.compile(
    r"^itu_fusion graph mode=(\S+) norm=(\d+) tiled=(\d+) epilogue=(\d+) "
    r"im2col=(\d+) mul_mat=(\d+) silu=(\d+) accepted=1$", re.M)
TAPS = (".bf.wav", ".aec-input.wav", ".aec.wav", ".after.wav")


def need(test: bool, message: str) -> None:
    if not test:
        raise RuntimeError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"mean": statistics.fmean(values),
            **{f"p{p:g}": ordered[math.ceil(len(ordered) * p / 100) - 1]
               for p in (50, 95, 99, 99.9)},
            "max": ordered[-1], "over16ms": sum(value > 16 for value in values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES[1:], default=list(MODES[1:]))
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    build = read(args.manifest)
    need(build["platform"] == "host", "host runner needs the host manifest")
    modes = ("baseline", *args.modes)
    need(len(set(modes)) == len(modes) and set(args.modes) <= set(build["enabled_candidates"]),
         "duplicate or unbuilt experiment modes")
    old = read(HOST_CLOCK / "MANIFEST.json")["targets"]["host"]
    frozen = (Path(old["binaries"]["callback"]), Path(old["D_library"]))
    exp = (Path(build["binary"]["path"]), Path(build["runtime"]["path"]))
    for path, expect in zip((*frozen, *exp), (old["binaries_sha256"]["callback"],
                            old["D_library_sha256"], build["binary"]["sha256"],
                            build["runtime"]["sha256"]), strict=True):
        need(sha(path) == expect, f"runtime drift {path}")

    assets = build["model_paths"]
    for name, path in assets.items():
        need(sha(Path(path)) == build["model_sha256"][name], f"model drift: {name}")
    scenes = read(BUILD / "localvqe-opt-20261117/INPUTS.json")["cases"]
    need({s["condition"] for s in scenes} ==
         {"near_single", "far_single", "double_talk", "transition", "noise_only"},
         "five frozen scenes required")
    for s in scenes:
        for f in s["files"].values():
            need(sha(Path(f["path"])) == f["sha256"], f"input drift: {f['path']}")

    guard()
    out = OUT / ("host-accept-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    print(f"EVIDENCE {out}", flush=True)

    def run(label: str, scene: dict, mode: str) -> dict:
        guard(out)
        destination = out / label
        destination.mkdir()
        prefix = destination / "out"
        cli, lib = frozen if mode == "frozen" else exp
        provider = old["D_library_sha256"] if mode == "frozen" else build["runtime"]["sha256"]
        env = {"ITU_LVQ_FUSION": "baseline" if mode == "frozen" else mode,
               "ITU_LOCALVQE_LIBRARY": str(lib), "ITU_LOCALVQE_MODEL": assets["localvqe"],
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        argv = [str(cli), "--profile", "vcp", "--no-rx-processing", "--aec-backend",
                "localvqe_aec200k", "--ns-backend", "webrtc", "--eq-preamp-db", "-2",
                "--eq-band", "low_shelf,180,-2,0.707", "--eq-band", "peak,1800,2,1",
                "--eq-band", "high_shelf,4500,-1,0.707", "--tx-drc",
                "--localvqe-reference-channel", "0", "--beam-delays", "1.0,0.0",
                "--native16-cofi-model", assets["cofi"], "--native16-cofi-window",
                assets["window"], "--mic", scene["files"]["mic"]["path"], "--echo",
                scene["files"]["echo"]["path"], "--profile-stages", "--out", str(prefix)]
        save(destination / "COMMAND.json", json.dumps({"argv": argv, "env": env}, indent=2))
        with (destination / "stdout").open("xb") as stdout, (destination / "stderr").open("xb") as stderr:
            process = subprocess.run(argv, cwd=ROOT, env={**os.environ, **env},
                                     stdout=stdout, stderr=stderr, timeout=90)
        log = (destination / "stderr").read_text()
        if re.search(r"No space left on device|Out of memory|std::bad_alloc", log, re.I):
            raise RuntimeError("resource stop; evidence retained")
        save(destination / "EXITCODE", str(process.returncode) + "\n")
        need(process.returncode == 0, f"run failed: {destination}")
        matches = INVENTORY.findall(log)
        if mode == "frozen":
            need(not matches, "frozen control unexpectedly contains fusion")
        else:
            need(bool(matches) and all(m[0] == mode and tuple(map(int, m[1:])) == NODE_COUNTS[mode]
                                       for m in matches), f"wrong executed graph identity: {destination}")
        report = read(prefix.with_suffix(".json"))
        expected = {"aec_provider_sha256": provider,
                    "aec_model_sha256": build["model_sha256"]["localvqe"],
                    "input_samples_per_channel": 192000, "dsp_delay_samples": 513,
                    "frame_size": 256, "aec_localvqe_inferences": 752,
                    "aec_localvqe_fallback_frames": 0, "ns_inference_calls": 753,
                    "ns_external_fallback_frames": 0, "native16_in_process_ort": True}
        need(all(report.get(k) == v for k, v in expected.items()),
             f"chain receipt mismatch: {destination}: "
             f"{ {k: report.get(k) for k,v in expected.items() if report.get(k) != v} }")
        with wave.open(str(prefix.with_suffix(".after.wav"))) as wav:
            need((wav.getnframes(), wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) ==
                 (192513, 16000, 1, 2), "final PCM extent, rate or format changed")
        with prefix.with_suffix(".audio-clock.frames.csv").open(newline="") as handle:
            timing = list(csv.DictReader(handle))
        need(len(timing) == 753 and all(int(r["frame"]) == i and r["clocks_valid"] == "1"
                                        for i, r in enumerate(timing)),
             f"clock ledger invalid: {destination}")
        with prefix.with_suffix(".timing.csv").open(newline="") as handle:
            stages = list(csv.DictReader(handle))
        need(len(stages) == 753 and all(int(r["frame"]) == i for i, r in enumerate(stages)),
             f"stage timing misaligned: {destination}")
        wavs = {suffix: sha(prefix.with_suffix(suffix)) for suffix in TAPS}
        return {"mode": mode, "scene": scene["condition"], "directory": str(destination),
                "inputs": {name: value["sha256"] for name, value in scene["files"].items()},
                "provider": provider, "graphs": matches, "contract": expected, "wav_sha256": wavs,
                "thread_cpu_ms": describe([int(r["process_thread_cpu_ns"]) / 1e6 for r in timing[:750]]),
                "processing_wall_ms": describe([int(r["process_wall_ns"]) / 1e6 for r in timing[:750]]),
                "localvqe_wall_ms": describe([float(r["localvqe_process_ms"]) for r in stages[:750]])}

    quality = {}
    for scene in scenes:
        name = scene["condition"]
        control = run(f"{name}-frozen", scene, "frozen")
        quality[name] = [control]
        for mode in modes:
            result = run(f"{name}-{mode}", scene, mode)
            need(result["wav_sha256"] == control["wav_sha256"], f"PCM mismatch: {name}/{mode}")
            quality[name].append(result)
            print(f"PCM_PASS {name} {mode}", flush=True)

    rounds = []
    if args.timing:
        # All modes take each run position once; never measure concurrently.
        scene = next(s for s in scenes if s["condition"] == "double_talk")
        for i in range(len(modes)):
            order = modes[i:] + modes[:i]
            for mode in order:
                result = run(f"timing-{i + 1}-{mode}", scene, mode)
                need(result["wav_sha256"] == quality["double_talk"][0]["wav_sha256"],
                     f"timing run changed PCM: {mode}")
                rounds.append(result)
                print("TIMING", i + 1, mode, result["thread_cpu_ms"]["mean"],
                      result["processing_wall_ms"]["p99.9"], flush=True)
    summary = {}
    for mode in modes:
        relevant = [r for r in rounds if r["mode"] == mode]
        if relevant:
            summary[mode] = {"runs": len(relevant),
                             "median_run_cpu_mean_ms": statistics.median(
                                 r["thread_cpu_ms"]["mean"] for r in relevant),
                             "median_run_wall_p99_ms": statistics.median(
                                 r["processing_wall_ms"]["p99"] for r in relevant),
                             "median_run_wall_p99_9_ms": statistics.median(
                                 r["processing_wall_ms"]["p99.9"] for r in relevant)}
    report = {"manifest": str(args.manifest.resolve()), "binary_sha256": build["binary"]["sha256"],
              "scope": "Unpaced AArch64 host DSP. Wall tails are not device deadlines.",
              "quality_runs": quality, "byte_equal_wav_comparisons": 5 * len(modes) * len(TAPS),
              "timing_runs": rounds, "summary": summary}
    save(out / "RESULT.json", json.dumps(report, indent=2) + "\n")
    print("PASS", out / "RESULT.json", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
