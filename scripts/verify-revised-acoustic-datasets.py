#!/usr/bin/env python3
"""Verify all 48 revised stores before publishing an activation receipt."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guard(path: Path) -> None:
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if shutil.disk_usage(path).free < 100 * 1024**3 or available < 5 * 1024**3:
        raise RuntimeError("resource safety floor reached; preserve all evidence")


def validate_all() -> tuple[dict, dict]:
    import numpy as np
    import soundfile as sf
    from server.acoustic.dataset_revisions_v2 import (
        DATASET_REVISION, install_revision_profiles,
        revision_scene_profile, revision_stress_profile,
    )
    from server.acoustic.acceptance_scope import ACCEPTANCE_MODES
    from server.acoustic.acceptance_scene_condition_api import _load_dataset
    from server.acoustic.acoustic_stress_api import load_stress

    install_revision_profiles(activate=False)
    profiles = [
        ("scene", revision_scene_profile(scene, mode))
        for scene in ("sce1", "sce2", "sce3", "sce4")
        for mode in sorted(ACCEPTANCE_MODES)
    ] + [
        ("stress", revision_stress_profile(scope, mode))
        for scope in ("echo", "double_talk")
        for mode in sorted(ACCEPTANCE_MODES)
    ]
    datasets = []
    source_rates = Counter()
    scope_counts = Counter()
    near_echo_levels = defaultdict(set)
    highband = []
    short_snr_errors = []
    assets_checked = 0
    source_files_checked = set()
    for kind, profile in profiles:
        _guard(PROJECT)
        dataset = _load_dataset(profile.dataset_id) if kind == "scene" else load_stress(profile.dataset_id)
        root = Path(dataset["root"])
        assert len(dataset["cases"]) == 100
        scope_key = profile.scene if kind == "scene" else profile.scope
        scope_counts[scope_key] += len(dataset["cases"])
        expected_rate = 48_000 if profile.mode.startswith("SWB-") else 8_000 if profile.mode.startswith("NB-") else 16_000
        for case in dataset["cases"]:
            provenance = case["provenance"]
            assert provenance["dataset_revision"] == DATASET_REVISION
            design = provenance["render_metadata"]
            old = design.get("ser_db_measured")
            near_ser = design.get("near_to_echo_ser_db_measured")
            if old is not None:
                assert abs(float(old) + float(near_ser)) < 1e-8
            else:
                assert near_ser is None
            observed_ser = design.get("near_to_echo_overlap_ser_db_measured")
            if observed_ser is not None:
                assert abs(float(observed_ser) + float(design["echo_to_near_overlap_esr_db_measured"])) < 1e-8
                near_echo_levels[f"{scope_key}:{profile.mode}"].add(round(float(observed_ser), 2))
            transport = design["transport"]
            assert transport["mode"] == profile.mode
            assert transport["codec_conformance"] is False
            assert transport["reference_or_noise_modified"] is False
            if profile.mode.endswith("-wired"):
                assert transport["input_sha256_f32le"] == transport["output_sha256_f32le"]
            if scope_key == "sce3":
                noise_info = provenance["noise"]
                assert "TCAR" not in noise_info["source_scenes"]
                assert all("TCAR" not in component["path"] for component in noise_info["components"])
                assert noise_info.get("actual_vehicle_speed_recording") is False
            for name in ("near_speech", "far_input"):
                for segment in provenance[name].get("segments", []):
                    source_rate = int(segment["source_sample_rate_hz"])
                    source_rates[str(source_rate)] += 1
                    if expected_rate == 48_000:
                        assert source_rate == 48_000
                    source = (PROJECT / segment["path"]).resolve(strict=True)
                    if source not in source_files_checked:
                        assert source.is_relative_to(PROJECT)
                        assert _hash(source) == segment["sha256"]
                        assert sf.info(source).samplerate == source_rate
                        source_files_checked.add(source)
            samples_by_role = {}
            for role, relative in case["signals"].items():
                path = root / relative
                assert _hash(path) == case["signals_sha256"][role]
                audio, rate = sf.read(path, always_2d=True)
                assert rate == expected_rate and len(audio) == rate * 12
                assert np.isfinite(audio).all() and float(np.max(np.abs(audio))) < 0.7
                samples_by_role[role] = audio[:, 0]
                assets_checked += 1
                if expected_rate == 48_000 and role in {"reference", "far_end"} and np.any(audio):
                    spectrum = np.abs(np.fft.rfft(audio[:, 0])) ** 2
                    freq = np.fft.rfftfreq(len(audio), 1 / rate)
                    fraction = float(spectrum[(freq >= 8000) & (freq < 14000)].sum() / max(spectrum.sum(), 1e-30))
                    highband.append(10 * np.log10(max(fraction, 1e-30)))
            if kind == "stress" and provenance["workload_category"] == "short_overlap":
                for role in ("near_speech", "far_input"):
                    selection_info = provenance[role]["observation_source_selection"]
                    assert selection_info["post_band_rms"] >= 0.01
                selection = slice(round(6.5 * expected_rate), round(7.0 * expected_rate))
                nrms = np.sqrt(np.mean(samples_by_role["reference"][selection] ** 2))
                noise_rms = np.sqrt(np.mean(samples_by_role["noise"][selection] ** 2))
                assert nrms > 0 and noise_rms > 0
                measured = 20 * np.log10(nrms / noise_rms)
                error = abs(float(measured) - float(design["noise_snr_db_target"]))
                assert error < 0.1, f"short overlap SNR drift: {profile.mode}/{case['case_id']} {error}"
                short_snr_errors.append(error)
        if kind == "scene":
            assert min(near_echo_levels[f"{scope_key}:{profile.mode}"]) < 0
            assert max(near_echo_levels[f"{scope_key}:{profile.mode}"]) > 0
        artifact_names = (
            "manifest.json", "assets.json", "quality.json", "summary.json",
            "runtime-contract-binding.json",
        )
        datasets.append({
            "dataset_id": profile.dataset_id, "kind": kind, "mode": profile.mode,
            "scene" if kind == "scene" else "scope": scope_key,
            "artifacts": {name: _hash(root / name) for name in artifact_names},
        })
        print(f"VERIFIED {kind}/{scope_key}/{profile.mode}: 100 cases", flush=True)
    receipt = {"revision": DATASET_REVISION, "datasets": datasets}
    report = {
        "revision": DATASET_REVISION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(datasets), "case_count": sum(scope_counts.values()),
        "scope_counts": dict(scope_counts), "referenced_audio_checked": assets_checked,
        "source_segment_rate_counts": dict(source_rates),
        "unique_source_files_sha_header_checked": len(source_files_checked),
        "near_to_echo_ser_db_levels": {key: sorted(value) for key, value in near_echo_levels.items()},
        "swb_8_14khz_energy_db": {
            "count": len(highband), "min": float(min(highband)),
            "median": float(np.median(highband)), "max": float(max(highband)),
        },
        "short_overlap_pcm_snr_max_error_db": float(max(short_snr_errors)),
        "standard_compliance": False, "formal_itu_t_compliance_status": "not_measured",
    }
    return receipt, report


def main() -> int:
    from server.acoustic.dataset_revisions_v2 import ACTIVATION_PATH, install_from_activation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    _guard(PROJECT)
    receipt, report = validate_all()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if args.activate:
        _guard(PROJECT)
        ACTIVATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ACTIVATION_PATH.exists():
            if json.loads(ACTIVATION_PATH.read_text()) != receipt:
                raise RuntimeError("existing activation differs; do not overwrite it")
        else:
            with ACTIVATION_PATH.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        install_from_activation()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
