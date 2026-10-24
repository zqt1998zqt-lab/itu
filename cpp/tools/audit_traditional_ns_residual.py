#!/usr/bin/env python3
"""Audit retained paired residual-NS results without rerunning or changing gates."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

import soundfile as sf

PROJECT = Path(__file__).resolve().parents[2]
BRANCHES = {
    "webrtc_v5_bounded09": "webrtc", "speex_v5_steadydual": "speex",
    "omlsa_imcra_v2_mild": "omlsa", "super_gaussian_balanced": "super",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(row):
    return row["variant"], row["run"], row["id"]


def unique(rows):
    result = {key(row): row for row in rows}
    if len(rows) != len(result):
        raise ValueError("duplicate variant/run/id")
    return result


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def failures(row):
    result = []
    for name, segment in row["segments"].items():
        if name.startswith("near"):
            b, a = segment["before"], segment["after"]
            if a["sisdr_db"] is None:
                continue
            if a["sisdr_db"] < min(b["sisdr_db"], 20) - 1:
                result.append(name + ":raw_sisdr")
            if not -3 <= a["projection_gain_db"] <= 18:
                result.append(name + ":projection_gain")
            if a["stoi"] is not None and b["stoi"] is not None and a["stoi"] < b["stoi"] - .02:
                result.append(name + ":stoi")
        elif name.startswith("noise_only"):
            attenuation = segment.get("total_attenuation_db")
            if attenuation is None:
                attenuation = segment.get("attenuation_lower_bound_db")
            if attenuation is not None and attenuation < 6:
                result.append(name + ":attenuation")
    if row["engine"]["clipped_samples_before_write"]:
        result.append("clipping")
    return result


def aggregate(rows):
    result = dict(total=len(rows), passed=sum(r["passed"] for r in rows),
                  noise_failed=sum(any("attenuation" in f for f in r["failures"]) for r in rows),
                  speech_failed=sum(any("attenuation" not in f for f in r["failures"]) for r in rows))
    for field in ("mean_noise_attenuation_db", "mean_stoi_improvement",
                  "mean_sisdr_improvement_db", "mean_projection_gain_db"):
        values = [r[field] for r in rows if r[field] is not None]
        result[field] = statistics.fmean(values) if values else None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    candidate, baseline, out = args.candidate_root.resolve(), args.baseline_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    previous = read_rows(baseline / "full-r3-fixed-results.jsonl")
    off = unique([r for r in previous if r["treatment"] == "off"])
    uniform = unique([r for r in previous if r["treatment"] == "r3"])
    gated = unique(read_rows(candidate / "full-gate-results.jsonl"))
    history = unique([r for r in read_rows(PROJECT / "build/ns-all-family-comparison-20261019/matched-results.jsonl")
                      if r["variant"] in BRANCHES and r["scope"] == "full"])
    if not off.keys() == uniform.keys() == gated.keys() == history.keys():
        raise ValueError("paired scenario sets differ")
    expected_counts = {variant: 692 for variant in BRANCHES}
    if dict(collections.Counter(k[0] for k in off)) != expected_counts:
        raise ValueError("incomplete branch coverage")
    identity = json.loads((candidate / "identity.json").read_text())
    if sha(candidate / "ns_vehicle_probe") != identity["binary_sha256"]:
        raise ValueError("candidate executable drift")
    selection = json.loads((PROJECT / "build/ns-gtcrn-direct-20261018/full/identity.json").read_text())["selection"]
    scenes = {(s["run"], s["scene"]["id"]): s["scene"] for s in selection}
    verified_inputs = set()
    input_lengths = {}
    checked = collections.Counter()
    for k, row in gated.items():
        variant, run, case_id = k
        scene = scenes[(run, case_id)]
        if (run, case_id) not in verified_inputs:
            source = PROJECT / "build" / run / "cases" / case_id
            lengths = []
            for name in ("before.wav", "clean.wav"):
                path = source / name
                if sha(path) != scene["files"][name]["sha256"]:
                    raise ValueError("frozen input hash drift")
                info = sf.info(path)
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError("non-native input in 16 kHz comparison")
                lengths.append(info.frames)
                checked["input_files"] += 1
            if lengths[0] != lengths[1]:
                raise ValueError("input and clean lengths differ")
            input_lengths[(run, case_id)] = lengths[0]
            verified_inputs.add((run, case_id))
        token = hashlib.sha256((run + "/" + case_id).encode()).hexdigest()[:20]
        folder = candidate / "full-gate" / BRANCHES[variant] / "r3" / token
        invocation = json.loads((folder / "invocation.json").read_text())
        options = invocation["argv"][19:]
        expected_options = ["--residual-max-db=3", "--residual-speech-guard=.75",
                            "--residual-gate-snr-db=-12", "--residual-gate-width-db=3"]
        if invocation["returncode"] or not all(option in options for option in expected_options):
            raise ValueError("candidate invocation or treatment differs")
        if invocation["argv"][0] != str(candidate / "ns_vehicle_probe"):
            raise ValueError("invocation used another executable")
        if sha(folder / "result.after.wav") != row["output_sha256"]:
            raise ValueError("scored output hash drift")
        info = sf.info(folder / "result.after.wav")
        engine = row["engine"]
        if (info.samplerate != 16000 or info.channels != 1
                or info.frames != engine["samples"] + engine["delay_samples"]
                or engine["samples"] != input_lengths[(run, case_id)]
                or engine["src_delay_samples"] != 0):
            raise ValueError("output length, rate or alignment differs")
        for table in (off, uniform, gated):
            item = table[k]
            if not finite(item) or failures(item) != item["failures"] or item["passed"] != (not item["failures"]):
                raise ValueError("nonfinite metric or score/gate inconsistency")
        if off[k]["output_sha256"] != history[k]["output_sha256"]:
            raise ValueError("disabled residual changed historical output")
        for name, segment in row["segments"].items():
            if name.startswith("near") and segment["before"] != off[k]["segments"][name]["before"]:
                raise ValueError("before-metric drift in paired comparison")
        checked["candidate_invocations"] += 1
        checked["candidate_output_hashes"] += 1
        checked["candidate_output_shapes"] += 1
        checked["historical_off_hash_matches"] += 1
    summary, changed = {}, []
    for variant in BRANCHES:
        group = [k for k in off if k[0] == variant]
        branch = {name: aggregate([table[k] for k in group])
                  for name, table in (("off", off), ("uniform3", uniform), ("gated3", gated))}
        for name, table in (("uniform3", uniform), ("gated3", gated)):
            branch[name]["recovered"] = sum(not off[k]["passed"] and table[k]["passed"] for k in group)
            branch[name]["regressed"] = sum(off[k]["passed"] and not table[k]["passed"] for k in group)
            branch[name]["noise_classes"] = {}
            for noise_class in ("steady", "transient"):
                subset = [k for k in group if table[k]["noise_class"] == noise_class]
                branch[name]["noise_classes"][noise_class] = dict(total=len(subset),
                    before_passed=sum(off[k]["passed"] for k in subset),
                    after_passed=sum(table[k]["passed"] for k in subset),
                    recovered=sum(not off[k]["passed"] and table[k]["passed"] for k in subset),
                    regressed=sum(off[k]["passed"] and not table[k]["passed"] for k in subset))
            if name == "gated3":
                for k in group:
                    if off[k]["passed"] != table[k]["passed"]:
                        r = table[k]
                        changed.append(dict(variant=variant, run=r["run"], id=r["id"],
                            recovered=r["passed"], noise=r["noise"], snr_db=r["snr_db"],
                            before_failures=off[k]["failures"], after_failures=r["failures"],
                            before_segments=off[k]["segments"], after_segments=r["segments"]))
        summary[variant] = branch
    for name, value in (("summary", summary), ("changed-cases", changed),
                        ("report", dict(state="completed", checked=dict(checked),
                         unique_scenes=len(verified_inputs), identity_verified=True,
                         thresholds_changed=False, duplicate_or_missing_rows=0,
                         input_nonfinite_samples="not independently decoded; runner validated PCM"))):
        (out / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for variant, branch in summary.items():
        print(variant, "off", branch["off"]["passed"],
              "gate", branch["gated3"]["passed"],
              "recovered", branch["gated3"]["recovered"],
              "regressed", branch["gated3"]["regressed"])
    print(dict(checked), flush=True)


if __name__ == "__main__":
    main()
