#!/usr/bin/env python3
"""Audit the full speech-band regression without trusting stored gate flags."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluate_gtcrn_direct import score_aligned
from run_ns_speechband_full import FROZEN, IDENTITY, PROFILE, RESULTS, sha, token

P = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--off-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, off_root, out = args.root.resolve(), args.off_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "identity.json").read_text())
    off_identity = json.loads((off_root / "identity.json").read_text())
    for name, digest in identity["bindings"].items():
        if sha(Path(name)) != digest:
            raise ValueError("bound dependency drift: " + name)
    frozen_identity = json.loads(IDENTITY.read_text())
    baseline_rows = [json.loads(line) for line in RESULTS.open()
                     if json.loads(line)["profile"] == PROFILE]
    scenes = {(r["run"], r["scene"]["id"]): r["scene"]
              for r in json.loads((P / "build/ns-gtcrn-direct-20261018/full/identity.json").read_text())["selection"]}
    rows = [json.loads(line) for line in (root / "results.jsonl").open()]
    off_rows = [json.loads(line) for line in (off_root / "results.jsonl").open()]
    if len(rows) != 2 * len(baseline_rows) or len(rows) != len(off_rows):
        raise ValueError("incomplete comparison coverage")
    def key(row):
        return row["run"], row["id"]
    baseline = {key(row): row for row in baseline_rows}
    candidates = {key(row): row for row in rows if row["recovery"] == 1}
    disabled = {key(row): row for row in off_rows if row["recovery"] == 0}
    if set(baseline) != set(disabled) or set(baseline) != set(candidates):
        raise ValueError("unpaired comparison selection")
    checked = dict(rows=0, input_hashes=0, output_hashes=0,
                   off_bitexact=0, on_changed=0, gates_recomputed=0)
    changes = []
    for row in candidates.values():
        scene_key = key(row)
        original = baseline[scene_key]
        off_row = disabled[scene_key]
        scene = scenes[scene_key]
        folder = root / "recovery-1p0" / token(row)
        off_folder = off_root / "recovery-0p0" / token(row)
        invocation = json.loads((folder / "invocation.json").read_text())
        if invocation["returncode"]:
            raise ValueError("candidate invocation failed")
        origin = P / "build" / row["run"] / "cases" / row["id"]
        before, before_rate = sf.read(origin / "before.wav")
        clean, clean_rate = sf.read(origin / "clean.wav")
        before = np.clip(np.rint(before * 32768), -32768, 32767) / 32768
        if before_rate != 16000 or clean_rate != 16000 or before.shape != clean.shape:
            raise ValueError("input shape/rate drift")
        checked["input_hashes"] += 2
        output, rate = sf.read(folder / "result.after.wav")
        engine = json.loads((folder / "result.json").read_text())
        delay = engine["delay_samples"]
        if (rate != 16000 or len(output) != len(clean) + delay
                or sha(folder / "result.after.wav") != row["output_sha256"]
                or sha(off_folder / "result.after.wav") != original["output_sha256"]
                or off_row["output_sha256"] != original["output_sha256"]
                or engine["core_rate"] != 16000 or engine["src_delay_samples"] != 0
                or delay != original["engine"]["delay_samples"]):
            raise ValueError("output identity or timebase drift")
        scored = score_aligned(clean, before, output[delay:], scene, "full",
                               engine["clipped_samples_before_write"])
        if scored["failures"] != row["failures"] or scored["passed"] != row["passed"]:
            raise ValueError("stored gate differs from PCM recomputation")
        checked["gates_recomputed"] += 1
        checked["output_hashes"] += 2
        checked["off_bitexact"] += 1
        checked["on_changed"] += row["output_sha256"] != original["output_sha256"]
        if original["passed"] != row["passed"]:
            changes.append(dict(run=row["run"], id=row["id"],
                                transition="recovered" if row["passed"] else "regressed",
                                baseline_failures=original["failures"],
                                candidate_failures=row["failures"]))
        checked["rows"] += 1
    if any(change["transition"] == "regressed" for change in changes):
        raise ValueError("full comparison has a passing-scene regression")
    control = json.loads(json.dumps(next(iter(candidates.values()))))
    near = next(label for label in control["segments"] if label.startswith("near"))
    control["segments"][near]["after"]["projection_gain_db"] = -4
    if near + ":projection_gain" not in {
            label + ":projection_gain"
            for label, segment in control["segments"].items()
            if label.startswith("near")
            and not -3 <= segment["after"]["projection_gain_db"] <= 18}:
        raise ValueError("known-bad projection negative control was not detected")
    report = dict(state="completed", checked=checked, changes=changes,
                  thresholds_changed=False, fitted_alignment=False,
                  waveform_metrics_recomputed=True,
                  known_bad_gate_detected=True,
                  limitations=["STOI/SI-SDR recomputed by the unchanged scorer",
                               "input/output WAV identities verified"],
                  bindings=identity["bindings"],
                  off_bindings=off_identity["bindings"],
                  frozen_binary_sha256=frozen_identity["bindings"].get(
                      str(P / "build/ns-rumble-adaptive-20261019/build-r3/ns_vehicle_probe")),
                  audit_sha256=sha(Path(__file__)))
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(checked=checked, changes=changes), indent=2))


if __name__ == "__main__":
    main()
