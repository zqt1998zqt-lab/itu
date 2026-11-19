from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.acceptance_signal_design import render_signals


PROJECT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    PROJECT / "scripts/create-background-noise-conversation-dataset.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "itu_conversation_generator_test",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(
    rate: int = 16_000,
    seconds: float = 12.0,
    *,
    seed: int = 1701,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = int(round(rate * seconds))
    time = np.arange(frames, dtype=np.float64) / rate
    near = (
        0.045 * np.sin(2.0 * math.pi * 310.0 * time)
        + 0.025 * np.sin(2.0 * math.pi * 733.0 * time + 0.4)
    )
    far = (
        0.040 * np.sin(2.0 * math.pi * 510.0 * time + 0.2)
        + 0.020 * np.sin(2.0 * math.pi * 1_170.0 * time)
    )
    noise = np.random.default_rng(seed).normal(size=frames)
    return near, far, noise


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def _slice(
    values: np.ndarray,
    rate: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    return values[
        int(round(start_s * rate)):int(round(end_s * rate))
    ]


def _reconstruct_echo(
    far_end: np.ndarray,
    path: dict,
) -> np.ndarray:
    echo = np.zeros(len(far_end), dtype=np.float64)
    for position, gain in zip(
        path["tap_samples"],
        path["tap_gains"],
        strict=True,
    ):
        echo[position:] += far_end[:len(far_end) - position] * gain
    return echo * float(path["echo_gain_linear"])


def _reconstruct_noise_1(
    noise_0: np.ndarray,
    recipe: dict,
) -> np.ndarray:
    noise_scale = float(recipe["noise_channel_scale_linear"])
    unit_noise_0 = noise_0 / noise_scale
    independent_2 = np.random.default_rng(recipe["seed2"]).normal(
        size=len(noise_0)
    )
    noise_1 = (
        float(recipe["rho"]) * unit_noise_0
        + math.sqrt(1.0 - float(recipe["rho"]) ** 2) * independent_2
    )
    noise_1 /= float(recipe["normalizer"])
    return noise_1 * noise_scale


def _render(
    mode: str,
    *,
    seed: int = 23,
) -> tuple[dict[str, np.ndarray], dict]:
    near, far, noise = _source(seed=seed)
    return render_signals(
        near,
        far,
        noise,
        16_000,
        direction="send",
        kind=mode,
        snr_db=10.0,
        ser_db=-5.0,
        echo_delay_ms=23.0,
        seed=seed,
        bandwidth="WB",
    )


def test_near_single_has_full_near_speech_and_no_far_or_echo():
    signals, metadata = _render("near_single")
    assert metadata["kind"] == "speech"
    assert metadata["stimulus_kind"] == "near_single"
    assert metadata["analysis_start_s"] == 0.0
    assert metadata["snr_db_target_applicable"] is True
    assert metadata["snr_db_measured"] == pytest.approx(10.0, abs=1e-9)
    assert metadata["ser_db_target_applicable"] is False
    assert metadata["ser_db_measured"] is None
    assert metadata["echo_paths"] == []
    assert np.count_nonzero(signals["far_end"]) == 0
    assert _rms(_slice(signals["reference"], 16_000, 0.1, 11.9)) > 0.01
    assert np.array_equal(
        signals["microphone"][:, 0],
        signals["reference"] + signals["noise"],
    )
    assert metadata["activity_masks"]["near"] == [
        {"start_s": 0.0, "end_s": 12.0}
    ]
    assert metadata["activity_masks"]["far"] == []


def test_far_single_has_strict_zero_reference_and_two_source_echoes():
    signals, metadata = _render("far_single")
    assert metadata["kind"] == "speech"
    assert metadata["stimulus_kind"] == "far_single"
    assert metadata["snr_db_target_applicable"] is False
    assert metadata["ser_db_target_applicable"] is False
    assert len(metadata["echo_paths"]) == 2
    assert np.count_nonzero(signals["reference"]) == 0
    assert _rms(signals["far_end"]) > 0.01
    noise_recipe = metadata["noise_spatial"]["channel_1_recipe"]
    noise_1 = _reconstruct_noise_1(signals["noise"], noise_recipe)
    for channel, path in enumerate(metadata["echo_paths"]):
        expected = (
            _reconstruct_echo(signals["far_end"], path)
            + (
                signals["noise"]
                if channel == 0
                else noise_1
            )
        )
        assert np.allclose(
            signals["microphone"][:, channel],
            expected,
            rtol=0.0,
            atol=1e-15,
        )
    assert metadata["source_mix_check"]["residual_max_abs"] <= 1e-15


def test_double_talk_is_real_twelve_second_overlap_with_measured_ser():
    signals, metadata = _render("double_talk")
    assert metadata["kind"] == "speech"
    assert metadata["stimulus_kind"] == "double_talk"
    assert metadata["double_talk_overlap_s"] == 12.0
    assert metadata["snr_db_target_applicable"] is True
    assert metadata["ser_db_target_applicable"] is True
    assert metadata["snr_db_measured"] == pytest.approx(10.0, abs=1e-9)
    assert metadata["ser_db_measured"] == pytest.approx(-5.0, abs=1e-9)
    assert len(metadata["echo_paths"]) == 2
    assert _rms(_slice(signals["reference"], 16_000, 0.1, 11.9)) > 0.01
    assert _rms(_slice(signals["far_end"], 16_000, 0.1, 11.9)) > 0.01
    assert metadata["activity_masks"]["double_talk"] == [
        {"start_s": 0.0, "end_s": 12.0}
    ]


def test_transition_really_contains_far_near_double_and_noise_only_windows():
    signals, metadata = _render("transition")
    assert metadata["kind"] == "speech"
    assert metadata["stimulus_kind"] == "transition"
    assert metadata["analysis_start_s"] == 0.5
    assert metadata["double_talk_overlap_s"] == 3.0
    assert metadata["snr_db_target_applicable"] is True
    assert metadata["ser_db_target_applicable"] is True
    assert len(metadata["echo_paths"]) == 2
    assert {
        segment["label"] for segment in metadata["segments"]
    } >= {
        "quiet",
        "far_only",
        "near_only",
        "double_talk",
        "noise_only",
    }

    far_only_reference = _slice(signals["reference"], 16_000, 0.7, 1.8)
    far_only_far = _slice(signals["far_end"], 16_000, 0.7, 1.8)
    near_only_reference = _slice(signals["reference"], 16_000, 2.1, 4.9)
    near_only_far = _slice(signals["far_end"], 16_000, 2.1, 4.9)
    double_reference = _slice(signals["reference"], 16_000, 8.1, 10.9)
    double_far = _slice(signals["far_end"], 16_000, 8.1, 10.9)
    noise_only_reference = _slice(signals["reference"], 16_000, 11.1, 11.9)
    noise_only_far = _slice(signals["far_end"], 16_000, 11.1, 11.9)
    noise_only_noise = _slice(signals["noise"], 16_000, 11.1, 11.9)
    assert _rms(far_only_reference) == 0.0
    assert _rms(far_only_far) > 0.01
    assert _rms(near_only_reference) > 0.01
    assert _rms(near_only_far) == 0.0
    assert _rms(double_reference) > 0.01
    assert _rms(double_far) > 0.01
    assert _rms(noise_only_reference) == 0.0
    assert _rms(noise_only_far) == 0.0
    assert _rms(noise_only_noise) > 0.0


def test_noise_only_is_cng_with_two_second_preheat_and_four_to_twelve_observation():
    signals, metadata = _render("noise_only")
    assert metadata["kind"] == "cng"
    assert metadata["stimulus_kind"] == "noise_only"
    assert metadata["analysis_start_s"] == 4.0
    assert metadata["cng_measurement_start_s"] == 4.0
    assert metadata["cng_preheat_window_s"] == [0.0, 2.0]
    assert metadata["noise_observation_window_s"] == [4.0, 12.0]
    assert metadata["snr_db_target_applicable"] is True
    assert metadata["ser_db_target_applicable"] is False
    assert metadata["echo_paths"] == []
    assert _rms(_slice(signals["reference"], 16_000, 0.1, 1.9)) > 0.01
    assert np.count_nonzero(_slice(signals["reference"], 16_000, 2.0, 12.0)) == 0
    assert np.count_nonzero(signals["far_end"]) == 0
    assert _rms(_slice(signals["noise"], 16_000, 4.0, 12.0)) > 0.0
    assert metadata["activity_masks"]["near"] == [
        {"start_s": 0.0, "end_s": 2.0}
    ]
    assert metadata["activity_masks"]["noise_only"] == [
        {"start_s": 2.0, "end_s": 12.0}
    ]
    json.dumps(metadata, allow_nan=False)


def test_conversation_stimuli_reject_receive_direction():
    near, far, noise = _source(seconds=2.0)
    with pytest.raises(ValueError, match="send direction"):
        render_signals(
            near,
            far,
            noise,
            16_000,
            direction="receive",
            kind="near_single",
            snr_db=10.0,
        )


def test_legacy_kind_metadata_does_not_gain_conversation_fields():
    near, far, noise = _source()
    _, metadata = render_signals(
        near,
        far,
        noise,
        16_000,
        direction="send",
        kind="speech",
        snr_db=10.0,
        seed=31,
    )
    assert metadata["kind"] == "speech"
    assert "stimulus_kind" not in metadata
    assert "activity_masks" not in metadata


def test_profile_plan_exposes_twenty_cases_per_mode_and_variations():
    module = _load_generator()
    profile = module.SCENE_PROFILE
    assert profile.schema_version == "scene-condition-preview-v2"
    assert profile.dataset_kind == "background_noise_conversation_preview"
    assert profile.case_count == 100
    assert profile.speech_case_count == 80
    assert profile.cng_case_count == 20
    assert profile.observation_count == 280

    plan = module.case_plan()
    assert len(plan) == 100
    assert Counter(row["mode"] for row in plan) == {
        "near_single": 20,
        "far_single": 20,
        "double_talk": 20,
        "transition": 20,
        "noise_only": 20,
    }
    for mode in module.CONVERSATION_MODES:
        rows = [row for row in plan if row["mode"] == mode]
        assert {row["snr_offset_db"] for row in rows} == {-3.0, 0.0, 3.0}
        assert {row["analysis_start_s"] for row in rows} == {
            module.analysis_start_for_mode(mode)
        }
        if mode in module.SER_MODES:
            assert {row["ser_db"] for row in rows} == {-10.0, -5.0, 0.0}
        else:
            assert {row["ser_db"] for row in rows} == {None}
        if mode in module.ECHO_MODES:
            assert len({row["echo_delay_ms"] for row in rows}) == 5
        else:
            assert {row["echo_delay_ms"] for row in rows} == {None}
    assert {
        row["analysis_start_s"]
        for row in plan
        if row["mode"] == "noise_only"
    } == {4.0}


@pytest.mark.parametrize("scene", ("sce1", "sce2", "sce3", "sce4"))
def test_each_scene_configures_and_builds_its_own_contract_plan(scene: str):
    module = _load_generator()
    module.configure_scene(scene)
    expected_ids = {
        row["id"]
        for row in module.background_noise_scene_items("WB-wired", scene)
    }
    assert module.SCENE == scene
    assert module.SCENE_PROFILE.scene == scene
    assert set(module.SCENE_PROFILE.contract_item_ids) == expected_ids
    assert module.DATASET_ID == module.SCENE_PROFILE.dataset_id

    plan = module.case_plan()
    assert len(plan) == 100
    assert Counter(row["mode"] for row in plan) == {
        "near_single": 20,
        "far_single": 20,
        "double_talk": 20,
        "transition": 20,
        "noise_only": 20,
    }
    speech_item_ids = {
        row["id"]
        for row in module.background_noise_scene_items("WB-wired", scene)
        if row["family"] in {"g_mos", "n_mos", "s_mos"}
    }
    cng_item_ids = {
        row["id"]
        for row in module.background_noise_scene_items("WB-wired", scene)
        if row["family"] in {"cng_level", "cng_spectrum"}
    }
    for row in plan:
        expected = (
            cng_item_ids
            if row["mode"] == "noise_only"
            else speech_item_ids
        )
        assert set(row["contract_item_ids"]) == expected
    assert all(
        row["case_id"].startswith(f"{scene}-wb-wired-")
        for row in plan
    )


def test_speaker_pools_validate_full_bank_then_split_disjointly():
    module = _load_generator()
    speakers: dict[str, list[dict]] = {}
    for index in range(40):
        speaker = f"speaker-{index:02d}"
        speakers[speaker] = [
            {
                "speaker": speaker,
                "path": f"clean/{speaker}-{file_index}.wav",
                "sha256": hashlib.sha256(
                    f"{speaker}:{file_index}".encode()
                ).hexdigest(),
                "seconds": 1.5,
            }
            for file_index in range(5)
        ]
    bank = SimpleNamespace(speakers=speakers)
    near, far, metadata = module._speaker_pools(bank)
    assert len(near) + len(far) == 200
    assert len(near) >= 5
    assert len(far) >= 5
    assert metadata["available_source_file_count"] == 200
    assert metadata["available_speaker_count"] == 40
    assert metadata["available_source_duration_s"] == 300.0
    assert metadata["speaker_pools_disjoint"] is True
    assert {
        row["speaker"] for row in near
    }.isdisjoint({
        row["speaker"] for row in far
    })


def test_source_duration_uses_real_seconds_key_and_fails_closed():
    module = _load_generator()
    assert module._record_duration_s({
        "path": "speech/a.wav",
        "seconds": 4.25,
    }) == 4.25
    with pytest.raises(ValueError, match="lacks required seconds"):
        module._record_duration_s({
            "path": "speech/b.wav",
            "duration_s": 4.25,
        })
    with pytest.raises(ValueError, match="non-positive seconds"):
        module._record_duration_s({
            "path": "speech/c.wav",
            "seconds": 0,
        })


def test_runtime_binding_v2_and_manifest_keep_legacy_top_level_fields():
    module = _load_generator()
    manifest = module._build_manifest([{} for _ in range(100)])
    assert set(manifest) == set(module.legacy_contract._MANIFEST_FIELDS)
    assert manifest["schema_version"] == "scene-condition-preview-v2"
    assert manifest["dataset_kind"] == (
        "background_noise_conversation_preview"
    )
    assert manifest["preview_condition_group_count"] == 100

    artifact_sha256 = {
        "manifest.json": "a" * 64,
        "assets.json": "b" * 64,
        "summary.json": "c" * 64,
        "quality.json": "d" * 64,
    }
    binding = module._runtime_binding(
        artifact_sha256=artifact_sha256,
        source_sha256=module._source_hash_snapshot(),
    )
    assert binding["schema_version"] == (
        "scene-condition-runtime-binding-v2"
    )
    assert binding["status"] == "reconciled"
    assert binding["dataset_id"] == module.DATASET_ID
    assert binding["artifact_sha256"] == artifact_sha256
    assert {
        "server/acoustic/scene_conversation_profiles.py",
        "scripts/acceptance_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
    }.issubset(binding["source_sha256"])


def test_source_reuse_policy_distinguishes_combinations_from_recordings():
    module = _load_generator()
    policy = module._source_reuse_policy({
        "available_source_file_count": 193,
        "available_speaker_count": 40,
    })
    assert policy["dataset_definition"] == "100_engineering_combinations"
    assert policy["independent_source_recordings_claimed"] is False
    assert policy["source_audio_may_be_reused_across_cases"] is True
    assert policy["within_case_source_looping"] is False
    assert policy["speaker_pools_disjoint_for_near_and_far"] is True
    assert policy["source_inventory"]["available_source_file_count"] == 193


def test_generator_post_pcm_activity_checks_double_talk_and_noise_only():
    module = _load_generator()
    for mode in ("double_talk", "noise_only"):
        signals, design = _render(mode, seed=101)
        decoded: dict[str, np.ndarray] = {}
        for role in module.ROLES:
            _, decoded[role] = module.encode(signals[role], module.RATE)
        mixture = module._verify_mixture_with_pcm_bound(
            signals,
            decoded,
            design,
        )
        activity = module._assert_pcm_activity(mode, decoded, design)
        measurements = module._post_pcm_measurements(mode, decoded, design)
        assert mixture["source_mix_residual_max_abs"] <= 1e-12
        assert max(mixture["pre_pcm_residuals_max_abs"]) <= 1e-12
        for residual, bound in zip(
            mixture["pcm_component_residuals_max_abs"],
            mixture["theoretical_bounds"],
            strict=True,
        ):
            assert residual <= bound["theoretical_max_abs_residual"]
        if mode == "double_talk":
            assert activity["double_talk_overlap_seconds"] >= 9.01
            assert measurements["ser_db_measured_post_pcm"] == pytest.approx(
                -5.0,
                abs=module.MEASUREMENT_DB_TOLERANCE,
            )
        else:
            assert np.count_nonzero(decoded["reference"][4 * module.RATE:]) == 0
            assert np.count_nonzero(decoded["far_end"][4 * module.RATE:]) == 0
            assert activity["noise_only_windows"][
                "observation_4_12_noise_rms"
            ] > 0.0


def test_pcm_mixture_bound_rejects_wrong_microphone_without_relaxation():
    module = _load_generator()
    signals, design = _render("double_talk", seed=707)
    decoded: dict[str, np.ndarray] = {}
    for role in module.ROLES:
        _, decoded[role] = module.encode(signals[role], module.RATE)
    accepted = module._verify_mixture_with_pcm_bound(
        signals,
        decoded,
        design,
    )
    channel_zero_bound = accepted["theoretical_bounds"][0][
        "theoretical_max_abs_residual"
    ]
    wrong = {
        role: values.copy()
        for role, values in decoded.items()
    }
    wrong["microphone"][0, 0] += channel_zero_bound + (
        2.0 * module.PCM16_MAX_QUANTIZATION_ERROR
    )
    with pytest.raises(
        ValueError,
        match="exceeds theoretical quantization bound",
    ):
        module._verify_mixture_with_pcm_bound(
            signals,
            wrong,
            design,
        )


def test_pre_pcm_mixture_limit_rejects_wrong_source_identity():
    module = _load_generator()
    signals, design = _render("double_talk", seed=808)
    decoded: dict[str, np.ndarray] = {}
    for role in module.ROLES:
        _, decoded[role] = module.encode(signals[role], module.RATE)
    wrong_design = deepcopy(design)
    wrong_design["source_mix_check"]["residual_max_abs"] = 1.01e-12
    with pytest.raises(ValueError, match="exceeds 1e-12"):
        module._verify_mixture_with_pcm_bound(
            signals,
            decoded,
            wrong_design,
        )
