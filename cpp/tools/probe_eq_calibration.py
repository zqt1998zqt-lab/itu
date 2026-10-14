#!/usr/bin/env python3
"""Three-rate C++ calibration loop; synthetic diagnostic masks, never ITU limits."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

import run_audio_eval as ev
from measure_itut_pcm import transfer_response
from tune_calibrated_eq import fit_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    rows = []
    for rate in (16000, 24000, 48000):
        folder = args.root / str(rate)
        folder.mkdir()
        source = np.random.default_rng(rate).normal(0, .03, rate * 4)
        input_path = folder / "excitation.wav"
        sf.write(input_path, source, rate, subtype="PCM_16")
        source, _ = sf.read(input_path)
        mask = dict(measurement_id=f"synthetic-resonance-{rate}", mask_source="synthetic +/-0.25 dB test; not an ITU mask",
                    normalized_at_hz=1000, frequency_hz=np.unique(np.r_[np.geomspace(80, 6000, 100), 1000]).tolist())
        mask.update(lower_db=[-.25] * len(mask["frequency_hz"]), upper_db=[.25] * len(mask["frequency_hz"]))
        (folder / "mask.json").write_text(json.dumps(mask, indent=2) + "\n")

        def process(path, prefix, options):
            command = [str(args.cli.resolve()), "--mic", str(path), "--out", str(prefix), "--no-aec",
                       "--no-ns", "--no-agc", "--no-limiter", "--no-low-cut"] + options
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            prefix.with_suffix(".invocation.json").write_text(json.dumps(dict(argv=command, exit_code=run.returncode,
                                           stderr=run.stderr, cli_sha256=ev.digest(args.cli)), indent=2) + "\n")
            run.check_returncode()
            metadata = json.loads(prefix.with_suffix(".json").read_text())
            delay = metadata["dsp_delay_samples"]
            values, actual_rate = sf.read(prefix.with_suffix(".after.wav"))
            if actual_rate != rate or len(values) != len(source) + delay:
                raise ValueError("EQ diagnostic format changed")
            return values[delay:delay + len(source)]

        resonant = process(input_path, folder / "resonant", ["--eq-preamp-db", "-6", "--eq-band", "peak,250,6,1"])
        measured = transfer_response(source[rate:], resonant[rate:], rate, mask)
        (folder / "measured-response.json").write_text(json.dumps(measured, indent=2) + "\n")
        candidate = fit_response(measured)
        (folder / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
        aligned_path = folder / "resonant-aligned.wav"
        sf.write(aligned_path, resonant, rate, subtype="PCM_16")
        corrected = process(aligned_path, folder / "corrected", candidate["cli_options"])
        replay = transfer_response(source[rate:], corrected[rate:], rate, mask)
        (folder / "remeasured-response.json").write_text(json.dumps(replay, indent=2) + "\n")
        passed = (not measured["shape_mask_passed"] and measured["worst_shape_margin_db"] < -4 and
                  candidate["accepted_candidate"] and replay["shape_mask_passed"])
        rows.append(dict(sample_rate=rate, before_margin_db=measured["worst_shape_margin_db"],
                         predicted_margin_db=candidate["after_shape_margin_db"],
                         remeasured_margin_db=replay["worst_shape_margin_db"], passed=bool(passed)))
    report = dict(cli_sha256=ev.digest(args.cli), rows=rows, passed=all(row["passed"] for row in rows),
                  standard_compliance=False, scope="known synthetic digital resonance through real C++ EQ; not measured cabin tuning")
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
