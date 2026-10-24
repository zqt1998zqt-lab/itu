#!/usr/bin/env python3
"""Read-only PCM audit of active, array, and native-rate simulation inventories.

Writes new audit artifacts only. Energy masks are engineering proxies, not VAD
or an ITU active-speech-level measurement. Does not change acceptance rules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

LSB = 1 / 32768


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def db(value):
    return float(10 * np.log10(max(float(value), 1e-20)))


def power(value):
    return float(np.mean(np.square(value))) if value.size else 0.0


def quantiles(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return dict(zip(("min", "p05", "p50", "p95", "max"),
                    [float(v) for v in np.percentile(values, (0, 5, 50, 95, 100))])) if values else {}


def read_inventory(project):
    work = []
    bindings = {}
    for label, directory in (("active", "datasets/itu-eval-current"),
                             ("array", "datasets/itu-vehicle-array-current")):
        dataset = (project / directory).resolve()
        manifest = dataset / "inputs-manifest.json"
        bindings[str(manifest)] = digest(manifest)
        for scene in json.loads(manifest.read_text())["scenes"]:
            folder = Path(scene.get("source_dataset", dataset)) / "cases" / scene.get("source_case_id", scene["id"])
            work.append((label, folder, scene))
    native = project / "build/native-rate-replay-v5-20261017"
    for filename in ("plan.json", "identity.json", "results.jsonl"):
        bindings[str(native / filename)] = digest(native / filename)
    for line in (native / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        scene = row["scene"]
        work.append(("native", native / "cases" / scene["id"], scene))
    if len({(label, scene["id"]) for label, _, scene in work}) != len(work):
        raise ValueError("nonunique inventory IDs")
    return work, bindings


def check_scene(item):
    label, folder, scene = item
    errors, warnings, audio = [], [], {}
    result = dict(dataset=label, id=scene["id"], family=scene["family"],
                  sample_rate=scene["sample_rate"], microphones=scene["mics"],
                  near_speaker=scene["near_speaker"], noise=scene["noise"],
                  snr_db=scene.get("snr_db"), errors=errors, warnings=warnings)
    try:
        rate, count = scene["sample_rate"], scene["mics"]
        n = round(scene["seconds"] * rate)
        # Equal identities inside a case share immutable decoded arrays.
        cache = {}
        file_peaks, clipping = {}, {}
        checked_files = 0
        for name, identity in scene["files"].items():
            path = folder / name
            if not path.is_file():
                errors.append("missing:" + name)
                continue
            checked_files += 1
            if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
                errors.append("identity:" + name)
            if path.suffix != ".wav":
                continue
            sha = identity["sha256"]
            if sha not in cache:
                data, actual = sf.read(path, dtype="float64", always_2d=True)
                cache[sha] = (data, actual)
            data, actual = cache[sha]
            channels = count if name in ("before.wav", "clean-mics.wav", "echo-component.wav", "noise-component.wav") else scene.get("reference_channels", 1) if name in ("ref.wav", "playback.wav", "rx-input.wav") else 1
            if actual != rate or data.shape != (n, channels) or not np.isfinite(data).all():
                errors.append("geometry_or_finite:" + name)
                continue
            audio[name] = data
            file_peaks[name] = float(np.max(np.abs(data)))
            clipping[name] = int(np.count_nonzero(np.abs(data) >= .999))
            if clipping[name]:
                errors.append("clipped:" + name)
        result["files_declared"] = len(scene["files"])
        result["files_checked"] = checked_files
        result["peak"] = max(file_peaks.values(), default=0.0)
        result["clipped_samples_by_file"] = {k: v for k, v in clipping.items() if v}
        required = {"before.wav", "clean.wav", "clean-mics.wav", "noise-component.wav", "echo-component.wav", "ref.wav", "playback.wav"}
        if required - audio.keys():
            errors.append("required:" + str(sorted(required - audio.keys())))
            return result
        before, clean, noise, echo = (audio[k] for k in ("before.wav", "clean-mics.wav", "noise-component.wav", "echo-component.wav"))
        reconstruction = float(np.max(np.abs(before - clean - noise - echo)))
        result["mixture_max_error_lsb"] = reconstruction / LSB
        if reconstruction > 2 * LSB:
            errors.append("mixture_reconstruction_over_2_lsb")
        if not np.array_equal(clean[:, :1], audio["clean.wav"]):
            errors.append("clean_channel0_mismatch")
        if scene.get("playback_reference_available") and not np.array_equal(audio["ref.wav"], audio["playback.wav"]):
            errors.append("available_reference_not_playback_pcm")
        scheduled = np.zeros(n, dtype=bool)
        for lo, hi in scene["near_schedule"]:
            scheduled[round(lo * rate):round(hi * rate)] = True
        calibration = scheduled.copy()
        if scene.get("noise_steps"):
            calibration[round(min(e["time_s"] for e in scene["noise_steps"]) * rate):] = False
        c, noise0 = clean[:, 0], noise[:, 0]
        result["near_scheduled_rms_dbfs"] = db(power(c[scheduled]))
        if scene["noise"] != "none" and power(noise0[scheduled]) > 0:
            snr = db(power(c[scheduled]) / power(noise0[scheduled]))
            calibrated = db(power(c[calibration]) / power(noise0[calibration]))
            result["scheduled_snr_db"] = snr
            result["calibrated_snr_db"] = calibrated
            result["nominal_snr_error_db"] = calibrated - scene["snr_db"]
            result["recorded_snr_error_db"] = snr - scene["measured_input_snr_db"]
            if abs(result["recorded_snr_error_db"]) > .001:
                errors.append("measured_snr_metadata_error_over_0.001_db")
            if count == 1 and abs(result["nominal_snr_error_db"]) > .05:
                errors.append("mono_nominal_snr_error_over_0.05_db")
        hop = rate // 50
        size = n // hop * hop
        cp = np.mean(c[:size].reshape(-1, hop) ** 2, axis=1)
        npow = np.mean(noise0[:size].reshape(-1, hop) ** 2, axis=1)
        frames_scheduled = scheduled[:size].reshape(-1, hop).all(axis=1)
        candidate = cp[frames_scheduled]
        active = frames_scheduled & (cp > max(float(candidate.max(initial=0)) * .001, 1e-6))
        result["scheduled_low_energy_20ms_percent"] = float(np.mean(candidate < 1e-6) * 100) if candidate.size else None
        result["scheduled_energy_active_percent"] = float(active.sum() / max(frames_scheduled.sum(), 1) * 100)
        if scene["noise"] != "none" and active.any() and npow[active].sum() > 0:
            energy_active_snr = db(cp[active].sum() / npow[active].sum())
            result["energy_active_snr_db"] = energy_active_snr
            result["energy_active_minus_scheduled_snr_db"] = energy_active_snr - result["scheduled_snr_db"]
            result["frame_snr_db"] = quantiles(10 * np.log10(np.maximum(cp[active], 1e-20) / np.maximum(npow[active], 1e-20)))
        window_stats = {}
        for name, (lo, hi) in scene["windows"].items():
            start, stop = round(lo * rate), round(hi * rate)
            if start < 0 or stop > n or stop <= start:
                errors.append("invalid_window:" + name)
                continue
            cw, ew = c[start:stop], echo[start:stop, 0]
            row = dict(seconds=(stop - start) / rate, clean_rms_dbfs=db(power(cw)),
                       echo_rms_dbfs=db(power(ew)), noise_rms_dbfs=db(power(noise0[start:stop])))
            if name.startswith("near") or name == "double_talk":
                if np.max(np.abs(cw)) <= LSB:
                    errors.append("no_target_speech:" + name)
                h = cw[:len(cw) // hop * hop].reshape(-1, hop)
                row["low_energy_20ms_percent"] = float(np.mean(np.mean(h ** 2, axis=1) < 1e-6) * 100) if len(h) else None
            if name in ("noise_only", "noise_only_before_step", "noise_only_after_step", "far_only", "far_startup", "far_recovery") and np.max(np.abs(cw)) > LSB:
                errors.append("target_leaks_into:" + name)
            if (name.startswith("near") or name.startswith("noise_only")) and np.max(np.abs(ew)) > 2 * LSB:
                errors.append("echo_leaks_into:" + name)
            window_stats[name] = row
        result["windows"] = window_stats
        if result["scheduled_low_energy_20ms_percent"] is not None and result["scheduled_low_energy_20ms_percent"] > 50:
            warnings.append("over_half_scheduled_frames_below_minus60_dbfs")
    except Exception as error:
        errors.append(type(error).__name__ + ":" + str(error))
    return result


def negative_controls(item):
    label, folder, scene = item
    correct = check_scene(item)
    if correct["errors"]:
        raise ValueError("negative control requires a valid original: " + str(correct["errors"]))
    results = []
    for mode in ("bad_hash", "snr_plus_3db", "speech_contaminated_noise_window"):
        altered = json.loads(json.dumps(scene))
        if mode == "bad_hash":
            altered["files"]["before.wav"]["sha256"] = "0" * 64
            expected = "identity:before.wav"
        elif mode == "snr_plus_3db":
            altered["snr_db"] += 3
            expected = "mono_nominal_snr_error_over_0.05_db"
        else:
            altered["windows"]["noise_only"] = altered["windows"]["near_only"]
            expected = "target_leaks_into:noise_only"
        errors = check_scene((label, folder, altered))["errors"]
        if expected not in errors:
            raise AssertionError((mode, expected, errors))
        results.append(dict(control=mode, detected=True, expected_error=expected, observed_errors=errors))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--resolve-missing-from", type=Path,
                        help="Read displaced scene copies under this cases directory; still verify original SHA256")
    parser.add_argument("--only-relocated", action="store_true",
                        help="Audit only missing paths resolved through --resolve-missing-from")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.root / "audit-tool-frozen.py")
    work, bindings = read_inventory(args.project.resolve())
    resolutions = []
    selected = []
    for label, folder, scene in work:
        relocated = False
        if not folder.is_dir() and args.resolve_missing_from is not None:
            alternate = args.resolve_missing_from.resolve() / scene["id"]
            if alternate.is_dir() and all((alternate / name).is_file() for name in scene["files"]):
                resolutions.append(dict(dataset=label, id=scene["id"], original=str(folder), alternate=str(alternate)))
                folder, relocated = alternate, True
        if not args.only_relocated or relocated:
            selected.append((label, folder, scene))
    work = selected
    if not work:
        raise ValueError("audit selection must be nonempty")
    started = time.monotonic()
    controls = negative_controls(next(x for x in work if x[2]["mics"] == 1 and x[2]["family"] == "ns"))
    rows = []
    with (args.root / "scenes.jsonl").open("w") as out, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, row in enumerate(pool.map(check_scene, work), 1):
            out.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if index % 500 == 0:
                out.flush()
                print(json.dumps(dict(completed=index, total=len(work), elapsed_s=round(time.monotonic() - started, 1),
                                      scenes_with_errors=sum(bool(r["errors"]) for r in rows))), flush=True)
    groups = {}
    for label in ("active", "array", "native"):
        values = [r for r in rows if r["dataset"] == label]
        groups[label] = dict(scenes=len(values), checked_files=sum(r.get("files_checked", 0) for r in values),
            scenes_with_errors=sum(bool(r["errors"]) for r in values),
            error_counts=dict(Counter(e for r in values for e in r["errors"])),
            warning_counts=dict(Counter(e for r in values for e in r["warnings"])),
            mixture_max_error_lsb=max((r.get("mixture_max_error_lsb", 0) for r in values), default=0),
            snr_metadata_error_db=quantiles([r.get("recorded_snr_error_db") for r in values]),
            nominal_snr_error_db=quantiles([r.get("nominal_snr_error_db") for r in values]),
            energy_active_minus_scheduled_snr_db=quantiles([r.get("energy_active_minus_scheduled_snr_db") for r in values]),
            scheduled_low_energy_20ms_percent=quantiles([r.get("scheduled_low_energy_20ms_percent") for r in values]),
            near_scheduled_rms_dbfs=quantiles([r.get("near_scheduled_rms_dbfs") for r in values]))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), state="completed", scenes=len(rows),
                  elapsed_s=time.monotonic() - started, audit_tool_sha256=digest(__file__), input_bindings=bindings,
                  negative_controls=controls, groups=groups,
                  relocated_scenes=resolutions,
                  screening_rules=dict(mixture_error_lsb=2, nominal_mono_snr_tolerance_db=.05,
                    metadata_snr_tolerance_db=.001, clipping_abs=.999, energy_frame_ms=20,
                    energy_active_rule="scheduled frame power > max(-60 dBFS, scheduled peak frame power -30dB); proxy, not VAD or P.56",
                    noise_step_snr_calibration="scheduled samples before first step; full-window SNR also reported"),
                  failures=[r for r in rows if r["errors"]])
    (args.root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("failures", "input_bindings", "relocated_scenes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
