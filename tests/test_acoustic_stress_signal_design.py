from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SIGNAL_PATH = PROJECT / "scripts/acoustic_stress_signal_design.py"
GENERATOR_PATH = PROJECT / "scripts/create-acoustic-stress-datasets.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signal_module():
    return _load(SIGNAL_PATH, "itu_acoustic_stress_signal_test")


def _generator_module():
    return _load(GENERATOR_PATH, "itu_acoustic_stress_generator_test")


def _source(
    rate: int = 16_000,
    seconds: float = 12.0,
    *,
    seed: int = 909,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = int(round(rate * seconds))
    time = np.arange(frames, dtype=np.float64) / rate
    near = (
        0.05 * np.sin(2.0 * math.pi * 310.0 * time)
        + 0.03 * np.sin(2.0 * math.pi * 733.0 * time + 0.2)
    )
    far = (
        0.04 * np.sin(2.0 * math.pi * 510.0 * time + 0.3)
        + 0.02 * np.sin(2.0 * math.pi * 1_170.0 * time)
    )
    noise = np.random.default_rng(seed).normal(size=frames)
    return near, far, noise


def _render(
    module,
    scope: str,
    category: str,
    *,
    rate: int = 16_000,
    bandwidth: str = "WB",
    volume_gain_db: float = 0.0,
    case_index: int = 1,
    seed: int = 71,
):
    near, far, noise = _source(rate, seed=seed)
    return module.render_stress_signals(
        near,
        far,
        noise,
        rate,
        scope=scope,
        category=category,
        volume_gain_db=volume_gain_db,
        seed=seed,
        case_index=case_index,
        bandwidth=bandwidth,
    )


def _encode(module, signals):
    decoded = {}
    for role in ("microphone", "reference", "far_end", "noise"):
        _, decoded[role] = module.CONVERSATION.encode(
            signals[role],
            int(round(len(signals["reference"]) / 12.0)),
        )
    return decoded


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def test_profile_matrix_has_sixteen_datasets_and_real_contract_rows():
    from server.acoustic.acoustic_stress_profiles import (
        CATEGORIES,
        PROFILES,
        SCOPES,
    )

    assert len(PROFILES) == 16
    assert set(SCOPES) == {"echo", "double_talk"}
    for (scope, mode), profile in PROFILES.items():
        assert profile.case_count == 100
        assert profile.observation_count == 100
        assert profile.schema_version == "acoustic-stress-preview-v1"
        assert profile.dataset_kind == "acoustic_stress_preview"
        assert profile.scene == "baseline"
        assert set(profile.category_counts) == set(CATEGORIES[scope])
        assert set(profile.category_counts.values()) == {20}
        expected_rows = 3 if scope == "echo" and mode.startswith("SWB-") else 2
        assert profile.contract_item_count == expected_rows
        assert len(profile.contract_rows) == expected_rows
        assert all(
            profile.conditions_for_item(item_id)["scene"] == "baseline"
            for item_id in profile.contract_item_ids
        )


def test_all_sixteen_case_plans_cycle_real_volume_order_per_category():
    module = _generator_module()
    from server.acoustic.acceptance_contract import required_conditions
    from server.acoustic.acoustic_stress_profiles import PROFILES

    assert len(module.stress_matrix_targets()) == 16
    assert len(set(module.stress_matrix_targets())) == 16
    for scope, mode in module.stress_matrix_targets():
        module.configure_dataset(scope, mode)
        plan = module._case_plan()
        profile = PROFILES[(scope, mode)]
        assert len(plan) == 100
        assert Counter(row["category"] for row in plan) == {
            category: 20 for category in module.STRESS.SCOPE_CATEGORIES[scope]
        }
        assert all(row["item_id"] in profile.contract_item_ids for row in plan)
        assert module.OUTPUT_ROOT == (
            module.service.BUILD / "acoustic-stress-datasets" / profile.dataset_id
        )
        for category in module.STRESS.SCOPE_CATEGORIES[scope]:
            rows = [row for row in plan if row["category"] == category]
            assert [row["item_id"] for row in rows] == [
                profile.contract_item_ids[index % len(profile.contract_item_ids)]
                for index in range(20)
            ]
            for row in rows:
                assert row["conditions"] == required_conditions(row["contract_row"])
                assert row["volume_gain_db"] == {
                    "nominal": 0.0,
                    "recommended": -2.0,
                    "maximum": 6.0,
                }[row["conditions"]["volume"]]


def test_echo_categories_have_zero_reference_and_real_dynamic_paths():
    module = _signal_module()
    rendered = {}
    for index, category in enumerate(module.SCOPE_CATEGORIES["echo"], 1):
        signals, metadata = _render(
            module,
            "echo",
            category,
            case_index=index,
            seed=100 + index,
        )
        rendered[category] = (signals, metadata)
        assert metadata["kind"] == "speech"
        assert metadata["workload_category"] == category
        assert np.count_nonzero(signals["reference"]) == 0
        assert _rms(signals["far_end"]) > 0.01
        assert metadata["analysis_start_s"] == (
            0.5 if category == "convergence" else 0.0
        )
        assert metadata["echo_path_segments"]

    convergence = rendered["convergence"][1]
    assert convergence["activity_segments"] == [
        {"start_s": 0.0, "end_s": 0.5, "label": "quiet"},
        {"start_s": 0.5, "end_s": 12.0, "label": "far_only"},
    ]
    delay = rendered["delay_change"][1]
    assert len(delay["echo_path_segments"]) == 2
    assert delay["echo_path_segments"][0]["paths"][0]["delay_ms"] == 20.0
    assert delay["echo_path_segments"][1]["paths"][0]["delay_ms"] == 80.0
    assert delay["echo_path_segments"][0]["paths"][1]["delay_ms"] == 25.0
    assert delay["echo_path_segments"][1]["paths"][1]["delay_ms"] == 90.0

    path_change = rendered["path_change"][1]
    first_paths = path_change["echo_path_segments"][0]["paths"]
    second_paths = path_change["echo_path_segments"][1]["paths"]
    assert first_paths[0]["tap_samples"] != second_paths[0]["tap_samples"]
    assert first_paths[1]["tap_samples"] != second_paths[1]["tap_samples"]
    assert first_paths[0]["tap_gains"] != second_paths[0]["tap_gains"]
    assert first_paths[1]["tap_gains"] != second_paths[1]["tap_gains"]

    steady_noise_signals, _ = _render(
        module,
        "echo",
        "steady_echo",
        case_index=5,
        seed=105,
    )
    steady_noise = _rms(steady_noise_signals["noise"])
    noisy_noise = _rms(rendered["noisy_echo"][0]["noise"])
    assert noisy_noise > steady_noise


def test_double_talk_categories_have_real_full_and_overlap_activity():
    module = _signal_module()
    rendered = {}
    for index, category in enumerate(
        module.SCOPE_CATEGORIES["double_talk"],
        1,
    ):
        signals, metadata = _render(
            module,
            "double_talk",
            category,
            case_index=index,
            seed=200 + index,
        )
        rendered[category] = (signals, metadata)
        assert _rms(signals["reference"]) > 0.01
        assert _rms(signals["far_end"]) > 0.01
        assert metadata["echo_path_segments"]

    expected_ser = {
        "balanced": 0.0,
        "near_dominant": -10.0,
        "far_dominant": 10.0,
    }
    for category, ser_db in expected_ser.items():
        metadata = rendered[category][1]
        assert metadata["activity_segments"] == [
            {"start_s": 0.0, "end_s": 12.0, "label": "double_talk"}
        ]
        assert metadata["ser_db_nominal_target"] == ser_db
        assert metadata["ser_db_measured"] == pytest.approx(ser_db, abs=1e-8)

    transition = rendered["transition"][1]
    assert {
        segment["label"] for segment in transition["activity_segments"]
    } == {
        "quiet",
        "far_only",
        "near_only",
        "double_talk",
        "noise_only",
    }
    short = rendered["short_overlap"][1]
    assert short["activity_segments"] == [
        {"start_s": 0.0, "end_s": 0.5, "label": "quiet"},
        {"start_s": 0.5, "end_s": 6.5, "label": "near_only"},
        {"start_s": 6.5, "end_s": 7.0, "label": "double_talk"},
        {"start_s": 7.0, "end_s": 12.0, "label": "far_only"},
    ]


def test_stress_pcm_reconstruction_uses_dynamic_paths_and_theoretical_bound():
    module = _signal_module()
    for scope, categories in module.SCOPE_CATEGORIES.items():
        for index, category in enumerate(categories, 1):
            signals, metadata = _render(
                module,
                scope,
                category,
                case_index=index,
                seed=300 + index,
            )
            decoded = _encode(module, signals)
            check = module.verify_stress_mixture(signals, decoded, metadata)
            activity = module.assert_stress_activity(
                decoded,
                metadata,
                scope=scope,
                category=category,
            )
            assert max(check["pre_pcm_residuals_max_abs"]) <= 1e-12
            assert check["source_mix_residual_max_abs"] <= 1e-12
            for residual, bound in zip(
                check["pcm_component_residuals_max_abs"],
                check["theoretical_bounds"],
                strict=True,
            ):
                assert residual <= bound["theoretical_max_abs_residual"]
            if scope == "echo":
                assert activity["near_active_seconds"] == 0.0
            if category == "short_overlap":
                assert activity["near_active_seconds"] == 6.5
                assert activity["far_active_seconds"] == 5.5


@pytest.mark.parametrize("mutation", ("wrong_path", "wrong_near", "wrong_far"))
def test_stress_mixture_negative_controls_reject_wrong_components(mutation: str):
    module = _signal_module()
    signals, metadata = _render(
        module,
        "double_talk",
        "short_overlap",
        case_index=3,
        seed=505,
    )
    decoded = _encode(module, signals)
    if mutation == "wrong_path":
        bad_design = deepcopy(metadata)
        bad_design["echo_path_segments"][0]["paths"][0]["tap_samples"][1] += 7
        with pytest.raises(ValueError, match="pre-PCM"):
            module.verify_stress_mixture(signals, decoded, bad_design)
        return

    bad_decoded = {
        role: values.copy()
        for role, values in decoded.items()
    }
    if mutation == "wrong_near":
        start = int(round(0.5 * 16_000))
        bad_decoded["reference"][start] += 0.01
    else:
        start = int(round(7.0 * 16_000))
        bad_decoded["far_end"][start] += 0.01
    with pytest.raises(ValueError, match="theoretical bound"):
        module.verify_stress_mixture(signals, bad_decoded, metadata)


def test_swb_uses_48khz_and_applies_profile_band_limit():
    module = _signal_module()
    signals, metadata = _render(
        module,
        "echo",
        "steady_echo",
        rate=48_000,
        bandwidth="SWB",
        seed=606,
    )
    assert metadata["sample_rate_hz"] == 48_000
    assert metadata["bandwidth"] == "SWB"
    assert metadata["band_limit"]["far"]["applied_edges_hz"] == [125.0, 12_500.0]
    assert len(signals["reference"]) == 48_000 * 12
    decoded = _encode(module, signals)
    module.verify_stress_mixture(signals, decoded, metadata)


def test_swb_source_crop_resamples_16k_to_48k_without_looping(monkeypatch):
    module = _generator_module()
    module.configure_dataset("echo", "SWB-wired")
    source_16k = np.zeros(module.RATE_16K * 12, dtype=np.float64)
    source_16k[::400] = 0.1

    def fake_crop(bank, records, *, group_id, role):
        assert bank is None
        assert records == []
        return source_16k, {
            "source_kind": "clean_speech_crop",
            "segments": [{
                "path": "speech/fake.wav",
                "sha256": "a" * 64,
                "selected_audio_sha256_f32le": "b" * 64,
                "repeated": False,
            }],
            "crop_recipe_sha256": "c" * 64,
            "repeated": False,
        }

    monkeypatch.setattr(
        module.CONVERSATION.OLD,
        "_build_speech_crop",
        fake_crop,
    )
    samples, info = module._crop_16k(
        None,
        [],
        group_id=1,
        role="far_input",
    )
    assert samples.shape == (module.RATE_SWB * 12,)
    assert info["target_sample_rate_hz"] == module.RATE_SWB
    assert info["resampling"] == {
        "method": "scipy.signal.resample_poly",
        "up": 3,
        "down": 1,
        "source_rate_hz": module.RATE_16K,
        "target_rate_hz": module.RATE_SWB,
    }
    assert info["within_case_source_repetition"] is False


def test_generator_manifest_and_runtime_binding_contracts():
    module = _generator_module()
    module.configure_dataset("echo", "SWB-wired")
    manifest = module._build_manifest([{} for _ in range(100)])
    assert set(manifest) == set(module.legacy_contract._MANIFEST_FIELDS)
    assert manifest["schema_version"] == "acoustic-stress-preview-v1"
    assert manifest["dataset_kind"] == "acoustic_stress_preview"
    assert manifest["test_scope"] == "echo"
    assert manifest["mode"] == "SWB-wired"
    assert manifest["scene"] == "baseline"
    assert manifest["contract_item_count"] == 3
    assert manifest["preview_condition_group_count"] == 100

    artifacts = {
        "manifest.json": "a" * 64,
        "assets.json": "b" * 64,
        "summary.json": "c" * 64,
        "quality.json": "d" * 64,
    }
    binding = module._runtime_binding(
        artifact_sha256=artifacts,
        source_sha256=module._source_hash_snapshot(),
    )
    assert binding["schema_version"] == "acoustic-stress-runtime-binding-v1"
    assert binding["dataset_id"] == module.DATASET_ID
    assert binding["status"] == "reconciled"
    assert binding["artifact_sha256"] == artifacts
    assert {
        "server/acoustic/acoustic_stress_profiles.py",
        "scripts/create-acoustic-stress-datasets.py",
        "scripts/acoustic_stress_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "scripts/acceptance_signal_design.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
    }.issubset(binding["source_sha256"])


def test_case_provenance_contract_is_serializable_and_explicit():
    module = _signal_module()
    signals, metadata = _render(
        module,
        "double_talk",
        "short_overlap",
        case_index=7,
        seed=707,
    )
    assert metadata["engineering_preview"] is True
    assert metadata["activity_segments"]
    assert metadata["echo_path_segments"]
    assert metadata["analysis_start_s"] == 0.5
    assert metadata["beam_delays_ms"] == [1000.0 / 16_000, 0.0]
    json.dumps(metadata, allow_nan=False)
    assert set(signals) == {"microphone", "reference", "far_end", "noise"}
