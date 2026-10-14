#!/usr/bin/env python3
"""Rate-adapter regression on frozen 16 kHz speech, not fullband source certification."""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import run_audio_eval as ev


def convert(x, source, target):
    divisor = math.gcd(source, target)
    return resample_poly(x, target // divisor, source // divisor, axis=0)


def sisdr(target, observed):
    target = target - np.mean(target)
    observed = observed - np.mean(observed)
    projected = target * (np.dot(target, observed) / max(np.dot(target, target), ev.EPS))
    return float(10 * np.log10(max(np.dot(projected, projected), ev.EPS)
                              / max(np.dot(observed - projected, observed - projected), ev.EPS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--pairs", default="0,2")
    parser.add_argument("--backends", default="webrtc,rnnoise,speex")
    parser.add_argument("--speex-suppression-db", type=int, default=-12)
    parser.add_argument("--cli-option", action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "inputs-manifest.json").read_text())
    args.root.mkdir(parents=True, exist_ok=False)
    cli = args.root.resolve() / "itu_audio_cli"
    shutil.copy2(args.cli, cli)
    rows = []
    for pair in map(int, args.pairs.split(",")):
        matches = [s for s in manifest["scenes"] if s["pair"] == pair and s["mics"] == 1
                   and s["family"] == "ns" and s["noise"] == "tcar" and s["snr_db"] == 30]
        if len(matches) != 1:
            raise ValueError("unique startup input required")
        scene = matches[0]
        folder = args.dataset / "cases" / scene["id"]
        for filename in ("before.wav", "clean.wav", "ref.wav"):
            if ev.digest(folder / filename) != scene["files"][filename]["sha256"]:
                raise ValueError("frozen input changed")
        clean = ev.read(folder / "clean.wav")[32000:144000, 0]
        before = ev.read(folder / "before.wav")[32000:144000]
        ref = ev.read(folder / "ref.wav")[32000:144000]
        for rate in (16000, 24000, 48000):
            target = convert(clean, 16000, rate)
            directory = args.root / f"p{pair:02d}-{rate}"
            directory.mkdir()
            sf.write(directory / "before.wav", convert(before, 16000, rate), rate, subtype="PCM_16")
            sf.write(directory / "ref.wav", convert(ref, 16000, rate), rate, subtype="PCM_16")
            for backend in args.backends.split(","):
                if backend not in ("webrtc", "rnnoise", "speex"):
                    raise ValueError("unknown backend")
                prefix = directory / backend
                command = [str(cli), "--profile", "vehicle_stream", "--mic", str(directory / "before.wav"),
                           "--echo", str(directory / "ref.wav"), "--out", str(prefix), "--ns-backend", backend,
                           "--speex-suppression-db", str(args.speex_suppression_db)]
                command.extend(args.cli_option)
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                result.check_returncode()
                engine = json.loads(prefix.with_suffix(".json").read_text())
                output, actual_rate = sf.read(prefix.with_suffix(".after.wav"))
                delay = engine["dsp_delay_samples"]
                if actual_rate != rate or len(output) != len(target) + delay:
                    raise ValueError("rate/length contract failed")
                output = output[delay:]
                # SI-SDR at the actual I/O rate; STOI uses its standard 16 kHz adapter.
                score = sisdr(target, output)
                quality = ev.quality(clean, convert(output, rate, 16000))
                row = dict(pair=pair, sample_rate=rate, backend=backend, sisdr_db=score,
                           stoi_delta=quality["stoi"] - ev.quality(clean, before[:, 0])["stoi"],
                           latency_samples=delay, engine=engine, command=command,
                           output_sha256=ev.digest(prefix.with_suffix(".after.wav")))
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("pair", "sample_rate", "backend", "sisdr_db", "stoi_delta", "latency_samples")}), flush=True)
    (args.root / "report.json").write_text(json.dumps(dict(
        scope="real speech from frozen 16k sources resampled for interface regression; no native highband claim",
        cli_sha256=ev.digest(cli), source_manifest_sha256=ev.digest(args.dataset / "inputs-manifest.json"),
        rows=rows), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
