#!/usr/bin/env python3
"""Audit frozen inputs independently of DSP output and acceptance scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from run_vehicle_stream import select_timeline

LSB = 1 / 32768


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def power_db(data):
    return float(10 * np.log10(max(float(np.mean(np.square(data))), 1e-20)))


def audio_stats(data, rate):
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError("nonempty finite channel matrix required")
    hop = max(1, rate // 50)
    frames = data[:len(data) // hop * hop].reshape(-1, hop, data.shape[1])
    energies = np.mean(frames * frames, axis=1)
    return dict(samples=len(data), channels=data.shape[1], seconds=len(data) / rate,
                rms_dbfs=power_db(data), peak=float(np.max(np.abs(data))),
                clipping_percent=float(np.mean(np.abs(data) >= .999) * 100),
                dc=float(np.mean(data)), zero_lsb_percent=float(np.mean(np.abs(data) <= LSB) * 100),
                low_energy_20ms_percent=float(np.mean(energies < 1e-6) * 100))


def ratio_db(a, b):
    pa, pb = float(np.mean(a * a)), float(np.mean(b * b))
    return float(10 * np.log10(pa / pb)) if pa > 0 and pb > 0 else None


def target_spectrum(data, rate):
    x = data[:, 0] - np.mean(data[:, 0])
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    weights = np.full(len(spectrum), 2.)
    weights[0] = 1
    if len(x) % 2 == 0:
        weights[-1] = 1
    spectrum *= weights
    frequencies = np.fft.rfftfreq(len(x), 1 / rate)
    total = float(np.sum(spectrum))
    return dict(ac_below_60hz_percent=float(np.sum(spectrum[(frequencies > 0) & (frequencies < 60)]) / total * 100)
                if total > 0 else None, dc_dbfs=float(20 * np.log10(max(abs(float(np.mean(data))), 1e-10))))


def window(data, limits, rate):
    lo, hi = (round(value * rate) for value in limits)
    if lo < 0 or hi > len(data) or hi <= lo:
        raise ValueError("invalid/nonempty evaluation window")
    return data[lo:hi]


def audit_scene(item):
    dataset, scene = item
    errors = []
    folder = Path(scene.get("source_dataset", dataset)) / "cases" / scene.get("source_case_id", scene["id"])
    rate, count = scene["sample_rate"], scene["mics"]
    samples = round(scene["seconds"] * rate)
    audio = {}
    for name, identity in scene["files"].items():
        path = folder / name
        if not path.is_file():
            errors.append(f"missing:{name}")
            continue
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            errors.append(f"identity:{name}")
        if path.suffix == ".wav":
            data, actual_rate = sf.read(path, always_2d=True, dtype="float64")
            channels = count if name in ("before.wav", "clean-mics.wav", "echo-component.wav", "noise-component.wav") else 1
            if actual_rate != rate or data.shape != (samples, channels) or not np.isfinite(data).all():
                errors.append(f"geometry_or_finite:{name}")
                continue
            if np.any(np.abs(data) >= .999):
                errors.append(f"clipped:{name}")
            audio[name] = data
    required = ("before.wav", "clean.wav", "clean-mics.wav", "echo-component.wav", "noise-component.wav", "ref.wav", "playback.wav")
    missing = set(required) - set(audio)
    if missing:
        return dict(dataset=str(dataset), id=scene["id"], errors=errors + [f"required:{sorted(missing)}"])
    clean, echo, noise, before = (audio[k] for k in ("clean-mics.wav", "echo-component.wav", "noise-component.wav", "before.wav"))
    reconstruction = float(np.max(np.abs(before - clean - echo - noise)) / LSB)
    if reconstruction > 2:
        errors.append("mixture_reconstruction_over_2_lsb")
    if not np.array_equal(clean[:, :1], audio["clean.wav"]):
        errors.append("clean_channel0_mismatch")
    mask = np.zeros(samples, dtype=bool)
    for lo, hi in scene["near_schedule"]:
        mask[round(lo * rate):round(hi * rate)] = True
    snr = ratio_db(clean[mask, 0], noise[mask, 0]) if scene["noise"] != "none" else None
    snr_error = abs(snr - scene["snr_db"]) if snr is not None else None
    # Fractional steering interpolation and sensor noise change the array's source SNR.
    if snr is not None and count == 1 and snr_error > .05:
        errors.append("measured_snr_outside_0.05_db")
    if snr is not None and abs(snr - scene["measured_input_snr_db"]) > 1e-8:
        errors.append("snr_metadata_mismatch")
    if scene.get("playback_reference_available") and not np.array_equal(audio["ref.wav"], audio["playback.wav"]):
        errors.append("available_reference_not_playback_pcm")
    windows = {}
    for name, limits in scene["windows"].items():
        c, e, n = (window(value[:, :1], limits, rate) for value in (clean, echo, noise))
        stats = dict(clean=audio_stats(c, rate), echo_rms_dbfs=power_db(e), noise_rms_dbfs=power_db(n))
        if name in ("noise_only", "far_only", "far_startup", "far_recovery") and np.any(np.abs(c) > LSB):
            errors.append(f"target_leaks_into:{name}")
        if name.startswith("near") and np.any(np.abs(e) > 2 * LSB):
            errors.append(f"echo_leaks_into:{name}")
        if name == "noise_only" and np.any(np.abs(e) > 2 * LSB):
            errors.append("echo_leaks_into:noise_only")
        if name.startswith("near") and not np.any(np.abs(c) > LSB):
            errors.append(f"no_target_speech:{name}")
        windows[name] = stats
    return dict(dataset=str(dataset), id=scene["id"], pair=scene["pair"], family=scene["family"],
                near_speaker=scene["near_speaker"], noise=scene["noise"], snr_db=scene["snr_db"],
                microphones=count, errors=errors, max_reconstruction_error_lsb=reconstruction,
                measured_snr_db=snr, snr_error_db=snr_error,
                files={name: audio_stats(audio[name], rate) for name in required}, windows=windows)


def inventory(dataset, manifest):
    scenes = manifest["scenes"]
    ids = [scene["id"] for scene in scenes]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty inventory required")
    ns = [s for s in scenes if s["family"] == "ns"]
    sweeps = defaultdict(list)
    for s in ns:
        sweeps[(s["near_speaker"], s["noise"])].append(s)
    ranges = [max(s["effective_near_dbfs"] for s in rows) - min(s["effective_near_dbfs"] for s in rows)
              for rows in sweeps.values()]
    people = {s["near_speaker"]: s.get("near_sex", "unknown") for s in scenes}
    return dict(path=str(dataset), manifest_sha256=digest(dataset / "inputs-manifest.json"), scenes=len(scenes),
                family_counts=dict(Counter(s["family"] for s in scenes)), near_speakers=len(people),
                sex_counts=dict(Counter(people.values())), languages=sorted({s.get("language", "unknown") for s in scenes}),
                source_sample_rates=sorted({s["sample_rate"] for s in scenes}),
                noise_types=sorted({s["noise"] for s in scenes if s["noise"] != "none"}),
                snr_db=sorted({s["snr_db"] for s in ns}),
                ns_sweeps=len(sweeps), ns_sweeps_with_level_range_over_1db=sum(v > 1 for v in ranges),
                max_ns_sweep_level_range_db=max(ranges, default=0),
                peak_scaled_scenes=sum(s["common_gain"] < 1 for s in scenes),
                historical_radio_as_noise=sum(s["noise"] == "car_radio" for s in scenes),
                legacy_snr_on_no_noise=sum(s["noise"] == "none" and s.get("measured_input_snr_db") is not None for s in scenes))


def source_audit(dataset, manifest):
    root = dataset / "sources"
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError(f"source manifest missing: {path}")
    actual_hash = digest(path)
    if manifest.get("source_manifest_sha256") != actual_hash:
        raise ValueError("frozen source inventory identity mismatch")
    source = json.loads(path.read_text())
    rows, errors = [], []
    seen = defaultdict(list)
    for entry in source["sources"]:
        wav = root / entry["path"]
        if digest(wav) != entry["sha256"]:
            errors.append(f"source_identity:{entry['path']}")
        data, rate = sf.read(wav, always_2d=True)
        if rate != entry["sample_rate"] or data.shape[1] != entry["channels"]:
            errors.append(f"source_geometry:{entry['path']}")
        rows.append(dict(path=str(wav), kind=entry["kind"], speaker=entry.get("speaker"),
                         scene=entry.get("scene"), sha256=entry["sha256"], **audio_stats(data, rate)))
        seen[entry["sha256"]].append(entry["path"])
    return dict(manifest_sha256=actual_hash, entries=len(rows), errors=errors,
                duplicate_source_files=[v for v in seen.values() if len(v) > 1],
                near_full_scale_sources=[dict(path=r["path"], percentage=r["clipping_percent"])
                                         for r in rows if r["clipping_percent"] > 0],
                speech_seconds=sum(r["seconds"] for r in rows if r["kind"] == "clean_speech"), rows=rows)


def build_report(datasets, scope, jobs):
    loaded = [(p.resolve(), json.loads((p / "inputs-manifest.json").read_text())) for p in datasets]
    inventories = [inventory(p, m) for p, m in loaded]
    sources = {}
    for p, m in loaded:
        key = str((p / "sources/manifest.json").resolve())
        if key not in sources:
            sources[key] = source_audit(p, m)
        elif m.get("source_manifest_sha256") != sources[key]["manifest_sha256"]:
            raise ValueError("supplement source identity mismatch")
    selected = {}
    if scope == "all":
        selected = {(str(p), s["id"]): (p, s) for p, m in loaded for s in m["scenes"]}
    else:
        # Select the exact continuous-test inputs for every speaker, preferring the radio supplement.
        scenes = [dict(s, audit_dataset=str(p)) for p, m in loaded for s in m["scenes"]
                  if p == loaded[0][0] or s.get("playback_source") == "radio"]
        for pair in sorted({s["pair"] for s in scenes if s["mics"] == 1}):
            for _, s, _, _ in select_timeline(scenes, pair):
                p = Path(s["audit_dataset"])
                selected[(str(p), s["id"])] = (p, s)
    if not selected:
        raise ValueError("PCM audit selection must not be empty")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(audit_scene, selected.values()))
    failures = [dict(dataset=r["dataset"], id=r["id"], errors=r["errors"]) for r in rows if r["errors"]]
    error_counts = Counter(e for r in rows for e in r["errors"])
    near = [dict(id=r["id"], pair=r["pair"], window=name, **stats["clean"])
            for r in rows if "pair" in r for name, stats in r["windows"].items()
            if name.startswith("near") or name == "double_talk"]
    target_diagnostics = []
    for p, s in selected.values():
        if s["family"] == "ns" and s["noise"] == "tcar" and s["snr_db"] == 30:
            folder = Path(s.get("source_dataset", p)) / "cases" / s.get("source_case_id", s["id"])
            data, rate = sf.read(folder / "clean.wav", always_2d=True)
            target_diagnostics.append(dict(id=s["id"], pair=s["pair"], speaker=s["near_speaker"],
                window_s=s["windows"]["near_only"],
                **target_spectrum(window(data, s["windows"]["near_only"], rate), rate)))
    return dict(schema_version=1, scope=scope, audit_tool_sha256=digest(__file__),
                checks=dict(max_mixture_reconstruction_error_lsb=2, max_mono_snr_error_db=.05,
                            array_snr_scope="check actual PCM against recorded measured SNR; nominal is pre-steering",
                            window_forbidden_component_max_lsb=2,
                            energy_proxy_20ms_dbfs=-60, energy_proxy_is_not_vad=True),
                inventories=inventories, sources=sources, audited_pcm_scenes=len(rows),
                input_integrity_passed=not failures and all(not s["errors"] for s in sources.values()),
                error_counts=dict(error_counts), failures=failures,
                max_mono_snr_error_db=max((r["snr_error_db"] for r in rows
                    if r.get("microphones") == 1 and r["snr_error_db"] is not None), default=None),
                max_array_nominal_snr_difference_db=max((r["snr_error_db"] for r in rows
                    if r.get("microphones", 1) > 1 and r["snr_error_db"] is not None), default=None),
                worst_low_energy_near_windows=sorted(near, key=lambda r:r["low_energy_20ms_percent"], reverse=True)[:20],
                cold_target_spectrum=target_diagnostics,
                representative_for_vehicle_deployment=False,
                limitations=["16 kHz source audio; upsampling is not native 24/48 kHz acoustic coverage",
                             "English read speech only; no Chinese/in-car spontaneous speech coverage",
                             "DEMAND TCAR plus controlled interference, no measured vehicle/speed/HVAC diversity",
                             "Near target is dry, echo RIR is a room model, not measured cabin acoustics",
                             "Fixed source excerpts reused across SNR/noise; scene count is not independent recording count",
                             "SNR uses scheduled windows including natural pauses, not speech-active-frame power",
                             "Array nominal SNR precedes fractional steering and added sensor noise",
                             "No frozen disjoint development/held-out speaker or vehicle split",
                             "Unreferenced passenger speech lacks target identity or spatial cue in mono",
                             "Clean corpus labels do not prove noise-free targets; DC and sub-60Hz power require review",
                             "Frozen v2 car_radio background labels are historical and unsuitable for AEC certification"],
                scenes=rows)


def write_report(report, root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "quality-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Dataset Quality Audit", "", f"PCM scope: {report['scope']}; audited scenes: {report['audited_pcm_scenes']}.",
             f"Input integrity passed: {report['input_integrity_passed']}. Vehicle deployment coverage: incomplete.", "",
             "This audit does not change samples, DSP thresholds, or prior acceptance results.", "",
             "## Inventory", "", "| Dataset | Scenes | Speakers | Source Rates | SNR Sweep Level Range >1 dB |",
             "| --- | ---: | ---: | --- | ---: |"]
    for r in report["inventories"]:
        lines.append(f"| {Path(r['path']).name} | {r['scenes']} | {r['near_speakers']} | {r['source_sample_rates']} | {r['ns_sweeps_with_level_range_over_1db']}/{r['ns_sweeps']} |")
    lines += ["", "## Input Errors", "", json.dumps(report["error_counts"], ensure_ascii=False), "", "## Coverage Limits", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    lines += ["", "Low-energy fractions use a fixed -60 dBFS/20 ms energy proxy. They are not a speech detector or proof of clean source SNR."]
    (root / "QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "vehicle"), default="vehicle")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.root.exists():
        raise ValueError("audit output must be a new directory")
    report = build_report(args.dataset, args.scope, args.jobs)
    write_report(report, args.root)
    print(json.dumps({k:report[k] for k in ("scope", "audited_pcm_scenes", "input_integrity_passed", "error_counts")}, indent=2))


if __name__ == "__main__":
    main()
