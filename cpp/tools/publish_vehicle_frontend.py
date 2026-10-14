#!/usr/bin/env python3
"""Publish a complete, identified diagnostic batch separately from formal ITU tests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from run_vehicle_stream import VEHICLE_TIMELINE


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--publish-to", type=Path, required=True)
    p.add_argument("--sequence-root", type=Path, help="Completed native, method, FIR and stream batches")
    p.add_argument("--stream-dataset", type=Path, help="Frozen array input for the sequence matrix")
    a = p.parse_args()
    if bool(a.sequence_root) != bool(a.stream_dataset):
        p.error("sequence-root and stream-dataset must be supplied together")
    source = a.root / "report.json"
    report = json.loads(source.read_text())
    manifest = json.loads((a.dataset / "inputs-manifest.json").read_text())
    audit = json.loads(a.audit.read_text())
    if not audit["passed"]:
        raise ValueError("input audit failed")
    import check_audio_acceptance as ac
    if digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen thresholds changed")
    expected_identity = dict(cli_sha256=digest(a.root / "itu_audio_cli"),
                             model_sha256=digest(a.model),
                             manifest_sha256=digest(a.dataset / "inputs-manifest.json"),
                             rules_sha256=digest(ac.RULES))
    for name, value in expected_identity.items():
        if report["identity"][name] != value:
            raise ValueError(f"identity changed: {name}")
    for key, name in (("runner_sha256", "run_vehicle_field_ab.py"), ("target_runner_sha256", "evaluation_targets.py")):
        if key in report["identity"] and report["identity"][key] != digest(Path(__file__).with_name(name)):
            raise ValueError(f"measurement source changed: {name}")
    if audit["manifest_sha256"] != expected_identity["manifest_sha256"]:
        raise ValueError("audit belongs to another manifest")
    expected = set()
    for scene in manifest["scenes"]:
        for name, record in scene["files"].items():
            if digest(a.dataset / "cases" / scene["id"] / name) != record["sha256"]:
                raise ValueError(f"input changed: {scene['id']} {name}")
        for rate in (16000, 24000, 48000):
            if rate != 16000 and scene["snr_db"] != 15:
                continue
            variants = ["baseline", "control"]
            if scene["mics"] > 1:
                variants.append("spatial")
            if scene["snr_db"] == 15:
                variants += ["rnnoise", "ulunas"]
            if scene.get("reference_channels", 1) > 1:
                variants.append("single_render_negative")
            expected.update((scene["id"], rate, variant) for variant in variants)
    rows = report["rows"]
    actual = [(r["id"], r["rate"], r["variant"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("batch incomplete or duplicated")
    for row in rows:
        folder = a.root / f"{row['id']}-r{row['rate']}" / row["variant"]
        invocation = json.loads((folder / "invocation.json").read_text())
        if invocation["exit_code"] != 0:
            raise ValueError("algorithm invocation failed")
        if Path(invocation["argv"][0]).resolve() != (a.root / "itu_audio_cli").resolve():
            raise ValueError("invocation did not use frozen binary")
        if json.loads((folder / "measurements.json").read_text()) != row:
            raise ValueError("summary differs from per-run evidence")
    destination = a.publish_to
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(complete=True, standard_compliance=False, total_runs=len(rows),
                   scene_rates=len({(r["id"], r["rate"]) for r in rows}),
                   identity=report["identity"], groups=report["groups"],
                   coverage=dict(scenes=len(manifest["scenes"]),
                                 near_speakers=len({s["near_speaker"] for s in manifest["scenes"]}),
                                 snrs_db=sorted({s["snr_db"] for s in manifest["scenes"]}),
                                 field_labels=sorted({s["field_label"] for s in manifest["scenes"]})),
                   input_quality=dict(passed=True, audit_sha256=digest(a.audit)),
                   failed_metric_counts=dict(Counter(g["metric"] for r in rows for g in r["acceptance"]["gates"] if g["status"] == "fail")))
    extra_files = {}
    if a.sequence_root:
        batches = {name: json.loads((a.sequence_root / folder / "report.json").read_text()) for name, folder in
                   (("native", "final-native-highband"), ("method", "final-method"),
                    ("fir", "final-fir-paired"), ("streams", "final-stream-matrix"))}
        for name in ("native", "method", "streams"):
            if batches[name]["identity"]["cli_sha256"] != expected_identity["cli_sha256"]:
                raise ValueError(f"sequence binary changed: {name}")
        if (batches["method"]["identity"]["input_manifest_sha256"] != expected_identity["manifest_sha256"] or
                batches["streams"]["identity"]["manifest_sha256"] != digest(a.stream_dataset / "inputs-manifest.json")):
            raise ValueError("sequence input identity changed")
        for name in ("native", "streams"):
            if batches[name]["identity"]["rules_sha256"] != expected_identity["rules_sha256"]:
                raise ValueError("sequence thresholds changed")
        native = batches["native"]
        expected_native = {(asset["id"], rate, snr, variant) for asset in native["assets"]
                           for rate in (16000, 24000, 48000) for snr in (0, 15, 30) for variant in ("baseline", "control")}
        actual_native = [(r["id"], r["rate"], r["snr_db"], r["variant"]) for r in native["rows"]]
        if len(actual_native) != 144 or len(actual_native) != len(set(actual_native)) or set(actual_native) != expected_native:
            raise ValueError("native batch incomplete or duplicated")
        for asset in native["assets"]:
            for kind in ("clean", "noisy"):
                if digest(asset[f"{kind}_path"]) != asset[f"{kind}_sha256"]:
                    raise ValueError("native source changed")
        method = batches["method"]
        if len(method["rows"]) != 12 or len({r["id"] for r in method["rows"]}) != 12:
            raise ValueError("method batch incomplete or duplicated")
        streams = batches["streams"]
        expected_streams = {(rate, mics, variant) for rate in (16000, 24000, 48000) for mics in (1, 2, 4)
                            for variant in (["control"] if mics == 1 else ["control", "spatial"])}
        actual_streams = [(r["rate"], r["microphones"], r["variant"]) for r in streams["rows"]]
        if (streams["complete"] is not True or set(actual_streams) != expected_streams or len(actual_streams) != 15 or
                streams["quality_phases"] != 15 * 2 * len(VEHICLE_TIMELINE) or
                any(r["sequences"] != 2 or r["phases"] != 2 * len(VEHICLE_TIMELINE) for r in streams["rows"]) or
                {r["rate"] for r in streams["clock_negative_controls"]} != {16000, 24000, 48000} or not streams["paced"]):
            raise ValueError("stream matrix incomplete")
        for filename, checksum in streams["identity"]["runner_sha256"].items():
            if digest(Path(__file__).with_name(filename)) != checksum:
                raise ValueError(f"stream runner changed: {filename}")
        fir = batches["fir"]
        if len(fir["rows"]) != 6 or any(r["cli_sha256"] != expected_identity["cli_sha256"] for r in fir["rows"] if r["variant"] == "block"):
            raise ValueError("FIR paired batch incomplete or changed")
        for name, folder in (("native", "final-native-highband"), ("method", "final-method"), ("streams", "final-stream-matrix")):
            for invocation in (a.sequence_root / folder).rglob("*.invocation.json"):
                if json.loads(invocation.read_text())["exit_code"] != 0:
                    raise ValueError(f"sequence invocation failed: {invocation}")
        for path in (a.sequence_root / "final-stream-matrix").rglob("stream-evaluation.json"):
            stream = json.loads(path.read_text())
            if (stream["identity"]["cli_sha256"] != expected_identity["cli_sha256"] or
                    stream["identity"]["engine_resets"] != 0 or stream["engine"]["stream_resets"] != 0):
                raise ValueError("stream binary changed or engine reset")
            invocation = json.loads(path.with_name("invocation.json").read_text())
            if invocation["exit_code"] != 0 or digest(invocation["argv"][0]) != expected_identity["cli_sha256"]:
                raise ValueError("stream invocation changed or failed")
            for filename, checksum in stream["identity"]["files"].items():
                if digest(path.with_name(filename)) != checksum:
                    raise ValueError("stream PCM changed after processing")
        for row in method["rows"]:
            folder = a.sequence_root / "final-method" / row["id"]
            for mode, source_hashes in row["source_hashes"].items():
                for kind, checksum in source_hashes.items():
                    if digest(folder / f"{mode}.{kind}.wav") != checksum:
                        raise ValueError("method paired PCM changed after processing")
        for row in native["rows"]:
            invocation_path = a.sequence_root / "final-native-highband" / f"{row['id']}-r{row['rate']}-snr{row['snr_db']}" / f"{row['variant']}.invocation.json"
            invocation = json.loads(invocation_path.read_text())
            if invocation["exit_code"] != 0 or digest(invocation["argv"][0]) != expected_identity["cli_sha256"]:
                raise ValueError("native invocation changed or failed")
        native_groups = {}
        for rate in (16000, 24000, 48000):
            for variant in ("baseline", "control"):
                subset = [r for r in native["rows"] if r["rate"] == rate and r["variant"] == variant]
                native_groups[f"{rate}-{variant}"] = dict(runs=len(subset), passed=sum(r["acceptance"]["passed"] for r in subset))
        summary["sequence"] = dict(native=dict(runs=144, groups=native_groups), method=method["summary"],
                                   streams=dict(phases=streams["quality_phases"], quality_passed=streams["quality_passed_phases"],
                                                seconds_per_sequence=streams["identity"]["seconds_per_sequence"], paced=streams["paced"]))
        for name, batch in batches.items():
            filename = f"{name}-report.json"
            write_json(destination / filename, batch)
            extra_files[filename] = digest(destination / filename)
    for src, name in ((source, "report.json"), (a.audit, "field-quality.json"), (a.document, "report.md")):
        temporary = destination / (name + ".tmp")
        shutil.copyfile(src, temporary)
        temporary.replace(destination / name)
    write_json(destination / "summary.json", summary)
    write_json(destination / "publication.json", dict(source_root=str(a.root.resolve()),
               standard_compliance=False, post_run_inputs_verified=True,
               files={**extra_files, **{name: digest(destination / name) for name in ("report.json", "field-quality.json", "report.md", "summary.json")}}))
    print(json.dumps(dict(total_runs=len(rows), scene_rates=summary["scene_rates"], groups=summary["groups"]), indent=2))


if __name__ == "__main__":
    main()
