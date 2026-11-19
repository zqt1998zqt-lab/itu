from __future__ import annotations

import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.acoustic import (
    acceptance_batch,
    acceptance_scene_condition_api,
    acoustic_stress_batch,
    acoustic_stress_models,
    acoustic_stress_store,
    scene_condition_batch,
    scene_condition_batch_api,
    service,
)
from server.acoustic.acceptance_contract import required_conditions
from server.acoustic.acoustic_stress_profiles import (
    ACCEPTANCE_MODES,
    CATEGORIES,
    SCOPES,
    stress_profile,
)


SCENES = ("sce1",)
RATE_BY_MODE = {"NB-": 8_000, "WB-": 16_000, "SWB-": 48_000}
FRAME_SECONDS = 12


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _rate(mode: str) -> int:
    if mode.startswith("SWB-"):
        return 48_000
    if mode.startswith("NB-"):
        return 8_000
    return 16_000


def _segments(scope: str, category: str) -> list[dict[str, object]]:
    if scope == "echo":
        if category == "convergence":
            return [
                {"start_s": 0.0, "end_s": 0.5, "label": "quiet"},
                {"start_s": 0.5, "end_s": 12.0, "label": "far_only"},
            ]
        return [
            {"start_s": 0.0, "end_s": 12.0, "label": "far_only"},
        ]
    if category in {"balanced", "near_dominant", "far_dominant"}:
        return [
            {"start_s": 0.0, "end_s": 12.0, "label": "double_talk"},
        ]
    if category == "transition":
        return [
            {"start_s": 0.0, "end_s": 0.5, "label": "quiet"},
            {"start_s": 0.5, "end_s": 2.0, "label": "far_only"},
            {"start_s": 2.0, "end_s": 5.0, "label": "near_only"},
            {"start_s": 5.0, "end_s": 8.0, "label": "far_only"},
            {"start_s": 8.0, "end_s": 11.0, "label": "double_talk"},
            {"start_s": 11.0, "end_s": 12.0, "label": "noise_only"},
        ]
    return [
        {"start_s": 0.0, "end_s": 0.5, "label": "quiet"},
        {"start_s": 0.5, "end_s": 6.5, "label": "near_only"},
        {"start_s": 6.5, "end_s": 7.0, "label": "double_talk"},
        {"start_s": 7.0, "end_s": 12.0, "label": "far_only"},
    ]


_ECHO_TAP_OFFSETS_MS = {
    "steady": {
        0: (0.0, 11.5, 27.0, 53.0),
        1: (0.0, 15.5, 34.0, 61.0),
    },
    "late": {
        0: (0.0, 17.0, 39.0, 71.0),
        1: (0.0, 21.0, 46.0, 83.0),
    },
    "alternate": {
        0: (0.0, 5.0, 39.0, 67.0),
        1: (0.0, 8.0, 43.0, 79.0),
    },
}
_ECHO_RAW_GAINS = {
    "steady": {
        0: (1.0, 0.45, 0.24, 0.11),
        1: (0.96, 0.36, 0.29, 0.15),
    },
    "late": {
        0: (0.82, 0.51, 0.31, 0.17),
        1: (0.79, 0.47, 0.35, 0.21),
    },
    "alternate": {
        0: (0.78, 0.22, 0.61, 0.33),
        1: (0.71, 0.55, 0.18, 0.46),
    },
}


def _echo_path_segments(
    scope: str,
    category: str,
    mode: str,
) -> list[dict[str, object]]:
    if scope != "echo":
        return []
    if category in {"delay_change", "path_change"}:
        if category == "delay_change":
            windows = (
                (0.0, 6.0, "steady", (20.0, 25.0)),
                (6.0, 12.0, "late", (80.0, 90.0)),
            )
        else:
            windows = (
                (0.0, 6.0, "steady", (20.0, 25.0)),
                (6.0, 12.0, "alternate", (20.0, 25.0)),
            )
    elif category == "convergence":
        windows = ((0.5, 12.0, "steady", (20.0, 25.0)),)
    else:
        windows = ((0.0, 12.0, "steady", (20.0, 25.0)),)
    rate = _rate(mode)
    result = []
    for start, end, variant, delays in windows:
        paths = []
        for channel in (0, 1):
            delay_ms = delays[channel]
            offsets = _ECHO_TAP_OFFSETS_MS[variant][channel]
            raw_gains = _ECHO_RAW_GAINS[variant][channel]
            normalizer = sum(abs(gain) for gain in raw_gains)
            gains = [float(gain / normalizer) for gain in raw_gains]
            paths.append(
                {
                    "channel": channel,
                    "variant": variant,
                    "delay_ms": delay_ms,
                    "tap_samples": [
                        int(round(delay_ms * rate / 1000.0))
                        + int(round(offset * rate / 1000.0))
                        for offset in offsets
                    ],
                    "tap_offsets_ms": [
                        round(delay_ms + offset, 9)
                        for offset in offsets
                    ],
                    "tap_gains": gains,
                    "echo_gain_linear": 1.0,
                    "tap_l1": float(sum(abs(gain) for gain in gains)),
                }
            )
        result.append(
            {
                "start_s": start,
                "end_s": end,
                "paths": paths,
            }
        )
    return result


def _signals(scope: str, mode: str, root: Path) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, object]],
]:
    rate = _rate(mode)
    frames = rate * FRAME_SECONDS
    time = np.arange(frames, dtype=np.float64) / rate
    if scope == "echo":
        reference = np.zeros(frames, dtype=np.float64)
        far = np.where(
            time >= 0.5,
            0.05 * np.sin(2 * np.pi * 300 * time),
            0.0,
        )
        noise = 0.0005 * np.sin(2 * np.pi * 700 * time)
        microphone = far * 0.35 + noise
    else:
        reference = 0.08 * np.sin(2 * np.pi * 220 * time)
        far = 0.05 * np.sin(2 * np.pi * 330 * time)
        noise = 0.0005 * np.sin(2 * np.pi * 700 * time)
        microphone = reference * 0.8 + far * 0.4 + noise
    signals: dict[str, str] = {}
    hashes: dict[str, str] = {}
    assets: dict[str, dict[str, object]] = {}
    for role in ("microphone", "reference", "far_end", "noise"):
        relative = f"assets/audio/{role}.wav"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            np.column_stack((microphone, microphone * 0.5))
            if role == "microphone"
            else reference
            if role == "reference"
            else far
            if role == "far_end"
            else noise
        )
        sf.write(path, payload, rate, subtype="PCM_16")
        digest = _sha_file(path)
        signals[role] = relative
        hashes[role] = digest
        assets[relative] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    return signals, hashes, assets


def _build_stress_dataset(build: Path, scope: str, mode: str) -> dict:
    profile = stress_profile(scope, mode)
    root = build / "acoustic-stress-datasets" / profile.dataset_id
    root.mkdir(parents=True, exist_ok=False)
    signals, hashes, assets = _signals(scope, mode, root)
    contracts = profile.contract_rows
    family_by_id = {row["id"]: row["family"] for row in contracts}
    cases = []
    for category in CATEGORIES[scope]:
        for index in range(20):
            contract = contracts[(index + len(category)) % len(contracts)]
            activity_segments = _segments(scope, category)
            cases.append(
                {
                    "case_id": f"{scope}-{mode}-{category}-{index:02d}",
                    "conditions": required_conditions(contract),
                    "families": [family_by_id[contract["id"]]],
                    "contract_item_ids": [contract["id"]],
                    "signals": copy.deepcopy(signals),
                    "signals_sha256": copy.deepcopy(hashes),
                    "provenance": {
                        "scope": scope,
                        "mode": mode,
                        "scene": "baseline",
                        "workload_category": category,
                        "category_index": index + 1,
                        "conversation_mode": (
                            "far_single"
                            if scope == "echo"
                            else "transition"
                            if category == "transition"
                            else "double_talk"
                        ),
                        "contract_item_id": contract["id"],
                        "conditions": copy.deepcopy(
                            required_conditions(contract)
                        ),
                        "analysis_start_s": (
                            0.5
                            if category
                            in {"convergence", "transition", "short_overlap"}
                            else 0.0
                        ),
                        "activity_segments": activity_segments,
                        "render_metadata": {
                            "workload_scope": scope,
                            "workload_category": category,
                            "sample_rate_hz": _rate(mode),
                            "activity_segments": copy.deepcopy(
                                activity_segments
                            ),
                            "analysis_start_s": (
                                0.5
                                if category
                                in {
                                    "convergence",
                                    "transition",
                                    "short_overlap",
                                }
                                else 0.0
                            ),
                            "echo_path_segments": _echo_path_segments(
                                scope,
                                category,
                                mode,
                            )
                        },
                        "engineering_preview": True,
                        "engineering_combination": True,
                        "independent_source_recordings": False,
                        "actual_vehicle_recording": False,
                        "within_case_source_looping": False,
                    },
                }
            )
    manifest = {
        "schema_version": "acoustic-stress-preview-v1",
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": "acoustic_stress_preview",
        "test_scope": scope,
        "mode": mode,
        "scene": "baseline",
        "contract_item_count": profile.contract_item_count,
        "preview_condition_group_count": 100,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "cases": cases,
    }
    _json_write(root / "manifest.json", manifest)
    _json_write(root / "assets.json", assets)
    _json_write(root / "summary.json", {"summary": "test fixture"})
    _json_write(root / "quality.json", {"quality": "test fixture"})
    project = Path(__file__).resolve().parents[1]
    source_paths = {
        "server/acoustic/acoustic_stress_profiles.py": project
        / "server/acoustic/acoustic_stress_profiles.py",
        "server/acoustic/acceptance_scope.py": project
        / "server/acoustic/acceptance_scope.py",
        "server/acoustic/acceptance_contract.py": project
        / "server/acoustic/acceptance_contract.py",
        "server/acoustic/acceptance_scene_conditions.py": project
        / "server/acoustic/acceptance_scene_conditions.py",
        "scripts/acoustic_stress_signal_design.py": project
        / "scripts/acoustic_stress_signal_design.py",
        "scripts/create-acoustic-stress-datasets.py": project
        / "scripts/create-acoustic-stress-datasets.py",
        "scripts/create-background-noise-conversation-dataset.py": project
        / "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py": project
        / "scripts/create-background-noise-scene-condition-dataset.py",
        "scripts/create-acceptance-measurement-dataset.py": project
        / "scripts/create-acceptance-measurement-dataset.py",
        "scripts/acceptance_signal_design.py": project
        / "scripts/acceptance_signal_design.py",
        "cpp/evaluation/nv200-test-contract.json": project
        / "cpp/evaluation/nv200-test-contract.json",
    }
    source_sha256 = {
        relative: _sha_file(path)
        for relative, path in source_paths.items()
    }
    artifacts = {
        name: _sha_file(root / name)
        for name in ("manifest.json", "assets.json", "summary.json", "quality.json")
    }
    _json_write(
        root / "runtime-contract-binding.json",
        {
            "schema_version": "acoustic-stress-runtime-binding-v1",
            "dataset_id": profile.dataset_id,
            "status": "reconciled",
            "source_sha256": source_sha256,
            "artifact_sha256": artifacts,
        },
    )
    return {
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": "acoustic_stress_preview",
        "schema_version": profile.schema_version,
        "test_scope": scope,
        "mode": mode,
        "scene": "baseline",
        "root": root,
        "manifest": manifest,
        "assets": assets,
        "cases": cases,
        "contracts": contracts,
        "contract_by_id": {row["id"]: row for row in contracts},
        "dataset_manifest_sha256": "1" * 64,
        "dataset_assets_sha256": "2" * 64,
        "counts": {"cases": 100, "observations": 100},
        "category_counts": profile.category_counts,
        "contract_item_count": profile.contract_item_count,
        "contract_item_ids": profile.contract_item_ids,
        "preview_condition_group_count": 100,
        "observation_count": 100,
        "case_count": 100,
    }


@pytest.fixture(scope="module")
def stress_build(tmp_path_factory):
    build = tmp_path_factory.mktemp("stress-build") / "acoustic-qualification"
    datasets = {
        (scope, mode): _build_stress_dataset(build, scope, mode)
        for scope, mode in (
            ("echo", "WB-wired"),
            ("echo", "SWB-wired"),
            ("double_talk", "WB-wired"),
            ("double_talk", "SWB-wired"),
        )
    }
    return build, datasets


def _tx_for_case(dataset: dict, case: dict, destination: Path) -> Path:
    source = dataset["root"] / case["signals"]["microphone"]
    return _tx_from_source(source, destination)


def _tx_from_source(source: Path, destination: Path) -> Path:
    samples, rate = sf.read(source, dtype="float64", always_2d=True)
    tx = destination / "tx.after.wav"
    sf.write(tx, samples, rate, subtype="PCM_16")
    return tx


def test_all_stress_profiles_expose_real_contracts() -> None:
    assert len(SCOPES) == 2
    for scope in SCOPES:
        for mode in ACCEPTANCE_MODES:
            profile = stress_profile(scope, mode)
            assert profile.scope == scope
            assert profile.mode == mode
            assert profile.contract_item_count in {2, 3}
            assert len(profile.contract_item_ids) == profile.contract_item_count
            for row in profile.contract_rows:
                assert required_conditions(row)["mode"] == mode


@pytest.mark.parametrize(
    ("scope", "mode"),
    (
        ("echo", "WB-wired"),
        ("echo", "SWB-wired"),
        ("double_talk", "WB-wired"),
        ("double_talk", "SWB-wired"),
    ),
)
def test_stress_store_returns_100_cases_and_category_counts(
    stress_build,
    monkeypatch,
    scope: str,
    mode: str,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    expected = datasets[(scope, mode)]

    loaded = acoustic_stress_store.load_stress_dataset(
        expected["dataset_id"]
    )

    assert loaded["mode"] == mode
    assert loaded["scene"] == "baseline"
    assert loaded["test_scope"] == scope
    assert loaded["counts"]["cases"] == 100
    assert loaded["counts"]["observations"] == 100
    assert loaded["category_counts"] == expected["category_counts"]
    assert len(loaded["cases"]) == 100
    assert all(len(case["contract_item_ids"]) == 1 for case in loaded["cases"])
    assert set(loaded["contract_by_id"]) == set(expected["contract_item_ids"])
    assert set(loaded["hashes"]) >= {
        "manifest",
        "assets",
        "runtime_binding",
    }


def _diagnostic_inputs(dataset: dict, case: dict) -> dict[str, Path]:
    return {
        role: dataset["root"] / case["signals"][role]
        for role in ("microphone", "reference", "far_end", "noise")
    }


def _write_diagnostic_inputs(
    root: Path,
    *,
    rate: int,
    microphone: np.ndarray,
    reference: np.ndarray,
    far_end: np.ndarray,
    noise: np.ndarray,
    tx: np.ndarray,
) -> tuple[dict[str, Path], Path]:
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "microphone": microphone,
        "reference": reference,
        "far_end": far_end,
        "noise": noise,
    }
    paths: dict[str, Path] = {}
    for role, samples in values.items():
        path = root / f"{role}.wav"
        sf.write(path, np.asarray(samples, dtype=np.float64), rate)
        paths[role] = path
    tx_path = root / "tx.wav"
    sf.write(tx_path, np.asarray(tx, dtype=np.float64), rate)
    return paths, tx_path


def test_echo_diagnostic_primary_is_far_over_mic_proxy(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    case = dataset["cases"][0]
    tx = _tx_for_case(dataset, case, tmp_path)
    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        _diagnostic_inputs(dataset, case),
        tx,
        profile=stress_profile("echo", "WB-wired"),
    )
    segments = acoustic_stress_models.validate_activity_segments(
        case["provenance"],
        scope="echo",
        mode="WB-wired",
    )
    acoustic_stress_models.validate_stress_diagnostics(
        diagnostics,
        expected_case_id=case["case_id"],
        expected_tx_sha256=_sha_file(tx),
        expected_scope="echo",
        expected_mode="WB-wired",
        expected_category=case["provenance"]["workload_category"],
        expected_segments=segments,
    )

    assert diagnostics["status"] == "review_required"
    assert diagnostics["formal_compliance"] is False
    assert diagnostics["primary"]["value"] is not None
    assert diagnostics["primary"]["name"] == "echo_primary_proxy_db"
    assert diagnostics["segments"][-1]["label"] == "far_only"
    assert diagnostics["segments"][-1]["mode"] == "far"
    json.dumps(diagnostics, allow_nan=False)


def test_double_talk_diagnostic_has_overlap_ratio_and_cleanref_projection(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("double_talk", "WB-wired")]
    case = dataset["cases"][0]
    tx = _tx_for_case(dataset, case, tmp_path)
    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        _diagnostic_inputs(dataset, case),
        tx,
        profile=stress_profile("double_talk", "WB-wired"),
    )
    overlap = next(
        segment
        for segment in diagnostics["segments"]
        if segment["label"] == "double_talk"
    )

    assert diagnostics["status"] == "review_required"
    assert diagnostics["primary"]["name"] == (
        "double_talk_energy_ratio_proxy_percent"
    )
    assert 0.0 <= diagnostics["primary"]["value"] <= 150.0
    assert overlap["mode"] == "double_talk"
    assert overlap["metrics"]["cleanref_correlation"] is not None
    assert overlap["metrics"]["cleanref_projection_gain_db"] is not None
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("active_seconds", (1 / 16_000, 0.12))
def test_echo_activity_gate_rejects_one_sample_and_99_percent_silence(
    stress_build,
    monkeypatch,
    tmp_path,
    active_seconds: float,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    case = dataset["cases"][0]
    rate = 16_000
    frames = rate * FRAME_SECONDS
    far = np.zeros(frames, dtype=np.float64)
    count = max(1, int(round(active_seconds * rate)))
    far[:count] = 0.05 * np.sin(
        2 * np.pi * 300 * np.arange(count, dtype=np.float64) / rate
    )
    microphone = far * 0.35
    zeros = np.zeros(frames, dtype=np.float64)
    paths, tx = _write_diagnostic_inputs(
        tmp_path / "inputs",
        rate=rate,
        microphone=microphone,
        reference=zeros,
        far_end=far,
        noise=zeros,
        tx=microphone,
    )

    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        paths,
        tx,
        profile=stress_profile("echo", "WB-wired"),
    )

    assert diagnostics["primary"]["value"] is None
    assert diagnostics["status"] == "not_measurable"
    assert diagnostics["segments"][0]["activity"]["valid"] is False
    assert diagnostics["segments"][0]["metrics"]["primary"] is None


def test_double_talk_activity_gate_rejects_non_overlapping_full_label(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("double_talk", "WB-wired")]
    case = dataset["cases"][0]
    rate = 16_000
    frames = rate * FRAME_SECONDS
    time = np.arange(frames, dtype=np.float64) / rate
    near = np.where(
        time < 2.0,
        0.08 * np.sin(2 * np.pi * 220 * time),
        0.0,
    )
    far = np.where(
        (time >= 2.0) & (time < 4.0),
        0.05 * np.sin(2 * np.pi * 330 * time),
        0.0,
    )
    microphone = near * 0.8 + far * 0.4
    zeros = np.zeros(frames, dtype=np.float64)
    paths, tx = _write_diagnostic_inputs(
        tmp_path / "inputs",
        rate=rate,
        microphone=microphone,
        reference=near,
        far_end=far,
        noise=zeros,
        tx=microphone,
    )

    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        paths,
        tx,
        profile=stress_profile("double_talk", "WB-wired"),
    )
    overlap = diagnostics["segments"][0]

    assert overlap["activity"]["joint_active_samples"] < int(0.05 * rate)
    assert overlap["activity"]["valid"] is False
    assert overlap["status"] == "not_measurable"
    assert diagnostics["primary"]["value"] is None


def test_echo_activity_gate_rejects_near_contamination(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    case = dataset["cases"][0]
    rate = 16_000
    frames = rate * FRAME_SECONDS
    time = np.arange(frames, dtype=np.float64) / rate
    far = 0.05 * np.sin(2 * np.pi * 300 * time)
    near = np.where(
        (time >= 4.0) & (time < 4.1),
        0.08 * np.sin(2 * np.pi * 220 * time),
        0.0,
    )
    microphone = far * 0.35 + near * 0.8
    zeros = np.zeros(frames, dtype=np.float64)
    paths, tx = _write_diagnostic_inputs(
        tmp_path / "inputs",
        rate=rate,
        microphone=microphone,
        reference=near,
        far_end=far,
        noise=zeros,
        tx=microphone,
    )

    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        paths,
        tx,
        profile=stress_profile("echo", "WB-wired"),
    )

    assert diagnostics["segments"][0]["activity"]["near_active_samples"] > 0
    assert diagnostics["segments"][0]["activity"]["valid"] is False
    assert diagnostics["primary"]["value"] is None


@pytest.mark.parametrize(
    ("active_seconds", "expected_finite"),
    ((0.040, False), (0.050, True)),
)
def test_activity_frame_and_50ms_gate_are_rate_aware_at_48k(
    stress_build,
    monkeypatch,
    tmp_path,
    active_seconds: float,
    expected_finite: bool,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("double_talk", "SWB-wired")]
    case = next(
        row
        for row in dataset["cases"]
        if row["provenance"]["workload_category"] == "short_overlap"
    )
    rate = 48_000
    frames = rate * FRAME_SECONDS
    near = np.zeros(frames, dtype=np.float64)
    far = np.zeros(frames, dtype=np.float64)
    start = int(round(6.5 * rate))
    count = int(round(active_seconds * rate))
    phase = np.arange(count, dtype=np.float64) / rate
    near[start:start + count] = 0.08 * np.sin(2 * np.pi * 220 * phase)
    far[start:start + count] = 0.05 * np.sin(2 * np.pi * 330 * phase)
    microphone = near * 0.8 + far * 0.4
    zeros = np.zeros(frames, dtype=np.float64)
    paths, tx = _write_diagnostic_inputs(
        tmp_path / "inputs",
        rate=rate,
        microphone=microphone,
        reference=near,
        far_end=far,
        noise=zeros,
        tx=microphone,
    )

    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        paths,
        tx,
        profile=stress_profile("double_talk", "SWB-wired"),
    )
    overlap = next(
        segment
        for segment in diagnostics["segments"]
        if segment["label"] == "double_talk"
    )

    assert overlap["activity"]["frame_ms"] == 20
    assert overlap["activity"]["hop_ms"] == 10
    assert (overlap["activity"]["joint_active_samples"] >= 2_400) is expected_finite
    assert (overlap["activity"]["valid"]) is expected_finite
    assert (diagnostics["primary"]["value"] is not None) is expected_finite


def test_echo_input_coupling_survives_muted_tx_without_tx_residual(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    case = dataset["cases"][0]
    rate = 16_000
    frames = rate * FRAME_SECONDS
    time = np.arange(frames, dtype=np.float64) / rate
    far = 0.05 * np.sin(2 * np.pi * 300 * time)
    microphone = far * 0.35
    zeros = np.zeros(frames, dtype=np.float64)
    paths, tx = _write_diagnostic_inputs(
        tmp_path / "inputs",
        rate=rate,
        microphone=microphone,
        reference=zeros,
        far_end=far,
        noise=zeros,
        tx=zeros,
    )

    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        paths,
        tx,
        profile=stress_profile("echo", "WB-wired"),
    )
    far_only = diagnostics["segments"][0]

    assert diagnostics["tx_muted"] is True
    assert diagnostics["status"] == "review_required"
    assert diagnostics["primary"]["value"] is not None
    assert diagnostics["convergence_db"] is None
    assert far_only["status"] == "review_required"
    assert far_only["metrics"]["primary"] is not None
    assert far_only["metrics"]["tx_residual_reduction_db"] is None
    assert far_only["metrics"]["tx_level_dbfs"] is None


def test_diagnostics_v1_is_legacy_and_v2_requires_activity_gate(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    case = dataset["cases"][0]
    tx = _tx_for_case(dataset, case, tmp_path)
    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        _diagnostic_inputs(dataset, case),
        tx,
        profile=stress_profile("echo", "WB-wired"),
    )
    segments = acoustic_stress_models.validate_activity_segments(
        case["provenance"],
        scope="echo",
        mode="WB-wired",
    )

    legacy = copy.deepcopy(diagnostics)
    legacy["schema_version"] = "acoustic-stress-diagnostics-v1"
    legacy.pop("activity_gate")
    for segment in legacy["segments"]:
        segment.pop("activity")
    acoustic_stress_models.validate_stress_diagnostics(
        legacy,
        expected_case_id=case["case_id"],
        expected_tx_sha256=_sha_file(tx),
        expected_scope="echo",
        expected_mode="WB-wired",
        expected_category=case["provenance"]["workload_category"],
        expected_segments=segments,
    )

    missing_gate = copy.deepcopy(diagnostics)
    missing_gate.pop("activity_gate")
    with pytest.raises(ValueError):
        acoustic_stress_models.validate_stress_diagnostics(
            missing_gate,
            expected_case_id=case["case_id"],
            expected_tx_sha256=_sha_file(tx),
            expected_scope="echo",
            expected_mode="WB-wired",
            expected_category=case["provenance"]["workload_category"],
            expected_segments=segments,
        )


def test_diagnostics_reject_tx_mute_or_all_not_applicable_primary(
    stress_build,
    monkeypatch,
    tmp_path,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    echo_dataset = datasets[("echo", "WB-wired")]
    echo_case = echo_dataset["cases"][0]
    (tmp_path / "echo").mkdir()
    echo_tx = _tx_for_case(echo_dataset, echo_case, tmp_path / "echo")
    echo_diagnostics = acoustic_stress_models.diagnose_stress_case(
        echo_case,
        _diagnostic_inputs(echo_dataset, echo_case),
        echo_tx,
        profile=stress_profile("echo", "WB-wired"),
    )
    echo_segments = acoustic_stress_models.validate_activity_segments(
        echo_case["provenance"],
        scope="echo",
        mode="WB-wired",
    )
    all_not_applicable = copy.deepcopy(echo_diagnostics)
    all_not_applicable["tx_duration_s"] = 1.0
    for segment in all_not_applicable["segments"]:
        segment["status"] = "not_applicable"
        segment["metrics"] = {
            key: None for key in acoustic_stress_models.METRIC_KEYS
        }
    all_not_applicable["convergence_db"] = None
    with pytest.raises(ValueError):
        acoustic_stress_models.validate_stress_diagnostics(
            all_not_applicable,
            expected_case_id=echo_case["case_id"],
            expected_tx_sha256=_sha_file(echo_tx),
            expected_scope="echo",
            expected_mode="WB-wired",
            expected_category=echo_case["provenance"]["workload_category"],
            expected_segments=echo_segments,
        )

    double_dataset = datasets[("double_talk", "WB-wired")]
    double_case = double_dataset["cases"][0]
    (tmp_path / "double").mkdir()
    double_tx = _tx_for_case(double_dataset, double_case, tmp_path / "double")
    double_diagnostics = acoustic_stress_models.diagnose_stress_case(
        double_case,
        _diagnostic_inputs(double_dataset, double_case),
        double_tx,
        profile=stress_profile("double_talk", "WB-wired"),
    )
    double_segments = acoustic_stress_models.validate_activity_segments(
        double_case["provenance"],
        scope="double_talk",
        mode="WB-wired",
    )
    muted = copy.deepcopy(double_diagnostics)
    muted["tx_muted"] = True
    with pytest.raises(ValueError):
        acoustic_stress_models.validate_stress_diagnostics(
            muted,
            expected_case_id=double_case["case_id"],
            expected_tx_sha256=_sha_file(double_tx),
            expected_scope="double_talk",
            expected_mode="WB-wired",
            expected_category=double_case["provenance"]["workload_category"],
            expected_segments=double_segments,
        )


def test_activity_validation_uses_exact_category_timeline_and_paths(
    stress_build,
) -> None:
    _, datasets = stress_build
    echo = datasets[("echo", "WB-wired")]
    delay_case = next(
        case
        for case in echo["cases"]
        if case["provenance"]["workload_category"] == "delay_change"
    )
    wrong_delay = copy.deepcopy(delay_case["provenance"])
    wrong_delay["activity_segments"] = [
        {"start_s": 0.0, "end_s": 6.0, "label": "far_only"},
        {"start_s": 6.0, "end_s": 12.0, "label": "far_only"},
    ]
    with pytest.raises(ValueError):
        acoustic_stress_models.validate_activity_segments(
            wrong_delay,
            scope="echo",
            mode="WB-wired",
        )
    missing_paths = copy.deepcopy(delay_case["provenance"])
    missing_paths["render_metadata"]["echo_path_segments"] = (
        missing_paths["render_metadata"]["echo_path_segments"][:1]
    )
    with pytest.raises(ValueError):
        acoustic_stress_models.validate_activity_segments(
            missing_paths,
            scope="echo",
            mode="WB-wired",
        )

    double = datasets[("double_talk", "WB-wired")]
    short = next(
        case
        for case in double["cases"]
        if case["provenance"]["workload_category"] == "short_overlap"
    )
    wrong_short = copy.deepcopy(short["provenance"])
    wrong_short["activity_segments"] = [
        {"start_s": 0.0, "end_s": 12.0, "label": "double_talk"},
    ]
    with pytest.raises(ValueError):
        acoustic_stress_models.validate_activity_segments(
            wrong_short,
            scope="double_talk",
            mode="WB-wired",
        )


@pytest.mark.parametrize("short_tx", (False, True))
def test_mute_or_short_tx_is_not_measured(
    stress_build,
    monkeypatch,
    tmp_path,
    short_tx: bool,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("double_talk", "WB-wired")]
    case = dataset["cases"][0]
    tx = tmp_path / "tx.wav"
    if short_tx:
        sf.write(tx, np.zeros(1_600, dtype=np.float64), 16_000)
    else:
        sf.write(tx, np.zeros(16_000 * 12, dtype=np.float64), 16_000)
    diagnostics = acoustic_stress_models.diagnose_stress_case(
        case,
        _diagnostic_inputs(dataset, case),
        tx,
        profile=stress_profile("double_talk", "WB-wired"),
    )
    segments = acoustic_stress_models.validate_activity_segments(
        case["provenance"],
        scope="double_talk",
        mode="WB-wired",
    )
    acoustic_stress_models.validate_stress_diagnostics(
        diagnostics,
        expected_case_id=case["case_id"],
        expected_tx_sha256=_sha_file(tx),
        expected_scope="double_talk",
        expected_mode="WB-wired",
        expected_category=case["provenance"]["workload_category"],
        expected_segments=segments,
    )
    assert diagnostics["status"] == "not_measurable"
    assert diagnostics["primary"]["value"] is None
    assert all(
        value is None
        for segment in diagnostics["segments"]
        for value in segment["metrics"].values()
    )


def _fake_visual(source_sha256: str) -> dict:
    return {
        "source_sha256": source_sha256,
        "duration_s": 0.1,
        "sample_rate": 16_000,
        "channels": 1,
        "peak": 0.0,
        "wave_scale": 1.0,
        "wave_min_i16": base64.b64encode(b"\0" * 1440).decode(),
        "wave_max_i16": base64.b64encode(b"\0" * 1440).decode(),
        "mel_i16": base64.b64encode(b"\0" * (40 * 240 * 2)).decode(),
        "mel_ceiling_db": -120.0,
    }


def _complete_first_stress_case(
    stress_build,
    monkeypatch,
    *,
    scope: str = "echo",
    mode: str = "WB-wired",
) -> tuple[dict, dict, dict, list[tuple[object, tuple[object, ...]]]]:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[(scope, mode)]
    monkeypatch.setattr(acceptance_batch, "_ACTIVE", set())
    monkeypatch.setattr(acceptance_batch, "_LEGACY_ACTIVE", False)
    monkeypatch.setattr(
        acceptance_scene_condition_api,
        "_visual_track",
        lambda path, source_sha256: _fake_visual(source_sha256),
    )
    submissions = []

    class Executor:
        def submit(self, function, *args):
            submissions.append((function, args))

    monkeypatch.setattr(acceptance_batch, "_EXECUTOR", Executor())
    cli = build / "fake-vcp"
    cli.write_bytes(b"fake-vcp")
    import server.alignment_runner as alignment_runner

    monkeypatch.setattr(alignment_runner, "resolve_cli", lambda: cli)
    monkeypatch.setattr(
        alignment_runner,
        "resolve_algorithm_cluster",
        lambda cluster: (
            {"profile": "vcp_default"},
            {},
            {"profile": "vcp_default"},
            "e" * 64,
        ),
    )
    queued = scene_condition_batch.start_batch(
        dataset["dataset_id"],
        "vcp_default",
        mode=mode,
        scene="baseline",
    )

    def run_cli(case, paths, destination, cluster):
        tx = _tx_from_source(paths["microphone"], destination)
        inputs = {
            role: _sha_file(path)
            for role, path in paths.items()
        }
        output = {"tx": _sha_file(tx)}
        return {"tx": tx}, {
            "source": "controlled-stress-cli",
            "algorithm_cluster_id": queued["algorithm_cluster_id"],
            "configuration": copy.deepcopy(queued["configuration"]),
            "catalog_sha256": queued["catalog_sha256"],
            "cli_sha256": queued["cli_sha256"],
            "input_sha256": inputs,
            "output_sha256": output,
            "output_alignment": {
                "tx": {"analysis_filename": tx.name}
            },
        }

    monkeypatch.setattr(acceptance_batch, "_run_cli", run_cli)
    snapshot = scene_condition_batch.get_batch(queued["batch_id"])
    assert snapshot is not None
    case_record = snapshot["cases"][0]
    item_by_key = {
        (row["case_id"], row["item_id"]): row
        for row in snapshot["items"]
    }
    scene_condition_batch._run_one_case(
        snapshot,
        dataset,
        case_record,
        item_by_key,
    )
    assert case_record["status"] == "completed", case_record.get("error")
    completed = scene_condition_batch.get_batch(queued["batch_id"])
    assert completed is not None
    assert completed["cases"][0]["status"] == "completed"
    return queued, completed, dataset, submissions


def test_stress_start_dispatch_and_one_case_use_shared_engine(
    stress_build,
    monkeypatch,
) -> None:
    queued, completed, _, submissions = _complete_first_stress_case(
        stress_build,
        monkeypatch,
    )
    assert queued["status"] == "queued"
    assert queued["stress_scope"] == "echo"
    assert queued["case_count"] == 100
    assert queued["observation_count"] == 100
    assert submissions[0][0] is scene_condition_batch.run_batch
    case_record = completed["cases"][0]
    assert isinstance(case_record["stress_diagnostics"], dict)
    assert case_record["stress_diagnostics"]["scope"] == "echo"
    row = next(
        item
        for item in completed["items"]
        if item["case_id"] == case_record["case_id"]
    )
    assert row["status"] in {"review_required", "not_measured"}
    assert row["status"] not in {"pass", "fail"}
    scene_condition_batch.validate_snapshot(completed)


def test_stress_snapshot_rejects_cross_scope_contract_row(
    stress_build,
    monkeypatch,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    snapshot = acoustic_stress_batch.build_snapshot(
        dataset,
        dataset["contract_by_id"],
        "a" * 32,
        "vcp_default",
        {"profile": "vcp_default"},
        "b" * 64,
        "c" * 64,
    )
    acoustic_stress_batch.validate_snapshot(snapshot)
    tampered = copy.deepcopy(snapshot)
    tampered["cases"][0]["contract_item_ids"] = ["WB-wired:B19"]
    with pytest.raises(ValueError):
        acoustic_stress_batch.validate_snapshot(tampered)


def test_stress_snapshot_rejects_weak_hashes_signals_and_terminal_state(
    stress_build,
    monkeypatch,
) -> None:
    build, datasets = stress_build
    monkeypatch.setattr(service, "BUILD", build)
    dataset = datasets[("echo", "WB-wired")]
    snapshot = acoustic_stress_batch.build_snapshot(
        dataset,
        dataset["contract_by_id"],
        "a" * 32,
        "vcp_default",
        {"profile": "vcp_default"},
        "b" * 64,
        "c" * 64,
    )

    mutations = (
        lambda value: value.__setitem__("dataset_manifest_sha256", "bad"),
        lambda value: value["cases"][0]["signals"].pop("noise"),
        lambda value: value["cases"][0].__setitem__("report_id", "bad"),
        lambda value: value.__setitem__("status", "completed"),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(snapshot)
        mutate(tampered)
        with pytest.raises(ValueError):
            acoustic_stress_batch.validate_snapshot(tampered)


@pytest.mark.parametrize(
    "mutation",
    ("stress_diagnostics", "provenance", "output_sha256"),
)
def test_stress_assessment_rejects_tampered_binding(
    stress_build,
    monkeypatch,
    tmp_path,
    mutation: str,
) -> None:
    _, completed, _, _ = _complete_first_stress_case(
        stress_build,
        monkeypatch,
    )
    case = copy.deepcopy(completed["cases"][0])
    original_path = scene_condition_batch._case_directory(
        completed["batch_id"],
        case["case_id"],
    ) / "assessment.json"
    assessment = json.loads(original_path.read_text(encoding="utf-8"))
    if mutation == "stress_diagnostics":
        assessment["stress_diagnostics"]["reason"] = "tampered diagnostic"
    elif mutation == "provenance":
        assessment["provenance"]["workload_category"] = "tampered"
    else:
        assessment["output_sha256"] = {}
    case["assessment_sha256"] = scene_condition_batch._write_once_json(
        tmp_path / "assessment.json",
        assessment,
    )

    with pytest.raises(ValueError, match="identity differs"):
        acoustic_stress_batch.validate_assessment(
            completed,
            case,
            tmp_path,
        )


def test_stress_tx_endpoint_serves_only_the_bound_case(
    stress_build,
    monkeypatch,
) -> None:
    queued, completed, _, _ = _complete_first_stress_case(
        stress_build,
        monkeypatch,
    )
    case_id = completed["cases"][0]["case_id"]
    expected_path, expected_sha256 = scene_condition_batch.output_path(
        queued["batch_id"],
        case_id,
    )
    app = FastAPI()
    app.include_router(scene_condition_batch_api.router)
    client = TestClient(app)

    response = client.get(
        "/api/alignment/acceptance/scene-condition-batches/"
        f"{queued['batch_id']}/cases/{case_id}/outputs/tx"
    )
    assert response.status_code == 200
    assert response.content == expected_path.read_bytes()
    assert response.headers["etag"] == f'"{expected_sha256}"'

    wrong_case = client.get(
        "/api/alignment/acceptance/scene-condition-batches/"
        f"{queued['batch_id']}/cases/not-a-case/outputs/tx"
    )
    assert wrong_case.status_code == 404


def test_stress_tx_endpoint_rejects_tampered_assessment(
    stress_build,
    monkeypatch,
) -> None:
    queued, completed, _, _ = _complete_first_stress_case(
        stress_build,
        monkeypatch,
    )
    source_case = completed["cases"][0]
    case_id = source_case["case_id"]
    source_directory = scene_condition_batch._case_directory(
        completed["batch_id"],
        case_id,
    )
    identifier = "d" * 32
    destination = scene_condition_batch._case_directory(identifier, case_id)
    destination.mkdir(parents=True, exist_ok=False)
    tx_name = (
        source_case["execution"]
        .get("output_alignment", {})
        .get("tx", {})
        .get("analysis_filename", "tx.after.wav")
    )
    (destination / tx_name).write_bytes(
        (source_directory / tx_name).read_bytes()
    )
    assessment = json.loads(
        (source_directory / "assessment.json").read_text(encoding="utf-8")
    )
    assessment["batch_id"] = identifier
    assessment["stress_diagnostics"]["reason"] = "tampered diagnostic"
    clone = copy.deepcopy(completed)
    clone["batch_id"] = identifier
    clone["cases"][0]["assessment_sha256"] = (
        scene_condition_batch._write_once_json(
            destination / "assessment.json",
            assessment,
        )
    )
    monkeypatch.setattr(
        scene_condition_batch,
        "get_batch",
        lambda requested_identifier: clone
        if requested_identifier == identifier
        else None,
    )
    app = FastAPI()
    app.include_router(scene_condition_batch_api.router)

    response = TestClient(app).get(
        "/api/alignment/acceptance/scene-condition-batches/"
        f"{identifier}/cases/{case_id}/outputs/tx"
    )

    assert response.status_code == 409
    assert "assessment" in response.json()["detail"]
