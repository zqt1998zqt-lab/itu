#!/usr/bin/env python3
"""Local real recording regression, deliberately without reference scores."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(__file__).resolve().parents[2]
source = root / "build/capture/android-real-20261008/capture.before.wav"
destination = root / "datasets/itu-eval-v1/real-unpaired"
destination.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, destination / "before.wav")
cli = root / "build/webrtc-host/cpp/itu_audio_cli"
before, rate = sf.read(source)
rows = []
for mode, flags in {"ns_agc": [], "ns_only": ["--no-agc"], "agc_only": ["--no-ns"],
                    "bypass": ["--no-ns", "--no-agc", "--no-limiter"]}.items():
    prefix = destination / mode
    args = [str(cli), "--mic", str(source), "--no-aec", "--out", str(prefix)] + flags
    run = subprocess.run(args, capture_output=True, text=True, check=True)
    report = json.loads(prefix.with_suffix(".json").read_text())
    output, sr = sf.read(prefix.with_suffix(".after.wav"))
    delay = report["dsp_delay_samples"]
    assert sr == rate and len(output) == len(before) + delay
    aligned = output[delay:delay + len(before)]
    rms = lambda data: float(10 * np.log10(max(float(np.mean(data * data)), 1e-20)))
    rows.append({"mode": mode, "argv": args, "input_rms_dbfs": rms(before), "output_rms_dbfs": rms(aligned),
                 "overall_level_reduction_db": rms(before) - rms(aligned),
                 "output_peak_dbfs": float(20 * np.log10(max(float(abs(aligned).max()), 1e-10))),
                 "output_clipping_percent": float(np.mean(np.abs(aligned) >= .999) * 100),
                 "dsp_delay_samples": delay, "process_p95_ms": report["process_p95_ms"],
                 "sisdr": None, "stoi": None, "snr_improvement": None, "erle": None, "mos": None,
                 "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "output_sha256": hashlib.sha256(prefix.with_suffix(".after.wav").read_bytes()).hexdigest()})
    if mode == "bypass":
        assert np.array_equal(before, aligned)
result = {"input_kind": "real Android microphone recording, replayed on local host",
          "reference_status": "no paired clean target, actual playback reference, or confirmed noise windows",
          "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(), "modes": rows}
(destination / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
