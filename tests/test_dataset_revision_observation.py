"""Tests for real near-source selection in short-overlap revision datasets."""
from __future__ import annotations

import errno
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "scripts" / "dataset_revision_observation.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "itu_dataset_revision_observation_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stress:
    def __init__(self):
        self.calls = []

    def band_limit(self, waveform, rate, bandwidth):
        self.calls.append((rate, bandwidth))
        return np.asarray(waveform, dtype=np.float64).copy(), {
            "bandwidth": bandwidth,
            "sample_rate_hz": rate,
            "filter": "fake_passthrough",
        }


class _Generator:
    SCOPE = "double_talk"
    RATE = 1_000
    BANDWIDTH = "WB"

    def __init__(self, waveforms=None, plan=None):
        self.waveforms = waveforms or {}
        self.plan = plan or [
            {"group_id": 1, "category": "balanced"},
            {"group_id": 20, "category": "short_overlap"},
            {"group_id": 21, "category": "short_overlap"},
        ]
        self.crop_calls = []
        self.plan_calls = 0
        self.STRESS = _Stress()
        self._crop_16k = self.crop

    def case_plan(self):
        self.plan_calls += 1
        return self.plan

    def crop(self, bank, records, *, group_id, role):
        self.crop_calls.append((bank, records, group_id, role))
        factory = self.waveforms.get((group_id, role))
        if factory is None:
            factory = self.waveforms.get(group_id)
        if factory is None:
            raise AssertionError(f"unexpected crop {(group_id, role)}")
        waveform = factory() if callable(factory) else factory
        waveform = np.asarray(waveform, dtype=np.float64)
        info = {
            "group_id": group_id,
            "role": role,
            "repeated": False,
            "within_case_source_repetition": False,
            "segments": [
                {
                    "path": f"/source/{group_id}-{role}.wav",
                    "sha256": f"{group_id:064x}"[-64:],
                }
            ],
            "crop_recipe_sha256": f"recipe-{group_id}",
        }
        return waveform, info


def _tone(rate=1_000, amplitude=0.02):
    clock = np.arange(rate * 12, dtype=np.float64) / rate
    return amplitude * np.sin(2.0 * np.pi * 173.0 * clock)


def _quiet(rate=1_000):
    return np.zeros(rate * 12, dtype=np.float64)


def _one_loud_frame(rate=1_000):
    waveform = np.zeros(rate * 12, dtype=np.float64)
    start = int(round(6.5 * rate))
    frame = int(round(0.01 * rate))
    clock = np.arange(frame, dtype=np.float64) / rate
    waveform[start : start + frame] = 0.12 * np.sin(
        2.0 * np.pi * 173.0 * clock
    )
    return waveform


def _far_tail_only(rate=1_000):
    waveform = np.zeros(rate * 12, dtype=np.float64)
    start = int(round(6.5 * rate))
    delayed_end = int(round(7.0 * rate))
    delayed_source_end = delayed_end - int(round(0.02 * rate))
    waveform[start:delayed_source_end] = 0.005
    waveform[delayed_source_end:delayed_end] = 0.2
    return waveform


def test_quiet_crop_is_rejected_and_active_crop_is_reselected_untouched():
    module = _module()
    active = _tone()
    bank = object()
    records = [{"path": "/source/a.wav"}]
    generator = _Generator({20: _quiet(), 1_020: active})
    original_crop = generator._crop_16k

    module.install_observation_source_selection(generator)
    waveform, info = generator._crop_16k(
        bank,
        records,
        group_id=20,
        role="near",
    )

    assert generator._crop_16k is not original_crop
    assert [(call[2], call[3]) for call in generator.crop_calls] == [
        (20, "near"),
        (1_020, "near"),
    ]
    assert all(call[0] is bank and call[1] is records for call in generator.crop_calls)
    assert np.array_equal(waveform, active)
    assert info["group_id"] == 1_020
    selection = info["observation_source_selection"]
    assert selection["original_role"] == "near"
    assert selection["selected_role"] == "near"
    assert selection["original_group_id"] == 20
    assert selection["real_group_id"] == 20
    assert selection["crop_group_id"] == 1_020
    assert selection["attempt"] == 1
    assert selection["post_band_rms"] == pytest.approx(
        np.sqrt(np.mean(active[6_500:7_000] ** 2))
    )
    assert selection["activity"]["active_frame_ratio"] >= 0.98
    assert selection["source_untouched"] is True
    assert selection["source_looped"] is False
    assert selection["waveform_gain_applied"] is False
    assert selection["local_gating_applied"] is False
    assert selection["delayed_far_validation"] is None
    assert generator.STRESS.calls == [(1_000, "WB"), (1_000, "WB")]


def test_quiet_far_crop_is_rejected_and_traced_with_original_role():
    module = _module()
    generator = _Generator({20: _quiet(), 1_020: _tone()})
    module.install_observation_source_selection(generator)

    waveform, info = generator._crop_16k(
        object(),
        [],
        group_id=20,
        role="far_input",
    )

    assert [(call[2], call[3]) for call in generator.crop_calls] == [
        (20, "far_input"),
        (1_020, "far_input"),
    ]
    assert np.array_equal(waveform, _tone())
    selection = info["observation_source_selection"]
    assert selection["original_role"] == "far_input"
    assert selection["selected_role"] == "far_input"
    assert selection["original_group_id"] == 20
    assert selection["crop_group_id"] == 1_020
    assert selection["delayed_far_validation"]["passed"] is True
    assert selection["delayed_far_validation"]["echo_delay_s"] == 0.02


def test_far_delayed_20ms_window_rejects_tail_only_echo_source():
    module = _module()
    tail_only = _far_tail_only()
    base = tail_only[6_500:7_000]
    delayed_source = tail_only[6_500:6_980]
    assert np.sqrt(np.mean(base**2)) >= 0.01
    assert np.sqrt(np.mean(delayed_source**2)) < 0.01

    generator = _Generator({20: tail_only, 1_020: _tone()})
    module.install_observation_source_selection(generator)
    waveform, info = generator._crop_16k(
        object(),
        [],
        group_id=20,
        role="far_input",
    )

    assert [(call[2], call[3]) for call in generator.crop_calls] == [
        (20, "far_input"),
        (1_020, "far_input"),
    ]
    assert np.array_equal(waveform, _tone())
    selection = info["observation_source_selection"]
    assert selection["crop_group_id"] == 1_020
    delayed = selection["delayed_far_validation"]
    assert delayed["delayed_window"] == {
        "start_s": 6.52,
        "end_s": 7.0,
        "start_sample": 6_520,
        "end_sample": 7_000,
    }
    assert delayed["source_window"] == {
        "start_sample": 6_500,
        "end_sample": 6_980,
    }


def test_global_rms_cannot_mask_a_single_loud_frame():
    module = _module()
    global_pass = _one_loud_frame()
    observation = global_pass[6_500:7_000]
    assert np.sqrt(np.mean(observation**2)) >= 0.01

    generator = _Generator({20: global_pass, 1_020: _tone()})
    module.install_observation_source_selection(generator)
    waveform, info = generator._crop_16k(
        object(),
        [],
        group_id=20,
        role="near",
    )

    assert [(call[2], call[3]) for call in generator.crop_calls] == [
        (20, "near"),
        (1_020, "near"),
    ]
    assert info["observation_source_selection"]["crop_group_id"] == 1_020
    assert np.array_equal(waveform, _tone())


def test_other_scope_category_and_role_pass_through_without_retry():
    module = _module()
    active = _tone()

    generator = _Generator({1: active, 20: active})
    module.install_observation_source_selection(generator)
    bank = object()
    records = [{"path": "/source/a.wav"}]

    near_other_category, _ = generator._crop_16k(
        bank,
        records,
        group_id=1,
        role="near",
    )
    far_short_overlap, _ = generator._crop_16k(
        bank,
        records,
        group_id=1,
        role="far_input",
    )
    other_role_short_overlap, _ = generator._crop_16k(
        bank,
        records,
        group_id=20,
        role="microphone",
    )
    assert np.array_equal(near_other_category, active)
    assert np.array_equal(far_short_overlap, active)
    assert np.array_equal(other_role_short_overlap, active)
    assert [(call[2], call[3]) for call in generator.crop_calls] == [
        (1, "near"),
        (1, "far_input"),
        (20, "microphone"),
    ]
    assert generator.STRESS.calls == []

    echo = _Generator({20: active})
    echo.SCOPE = "echo"
    original = echo._crop_16k
    module.install_observation_source_selection(echo)
    assert echo._crop_16k is original
    assert echo.plan_calls == 0


@pytest.mark.parametrize("role", ("near", "far_input"))
def test_selection_is_deterministic_for_same_generator_and_crop(role):
    module = _module()
    results = []
    for _ in range(2):
        generator = _Generator({20: _quiet(), 1_020: _tone()})
        module.install_observation_source_selection(generator)
        waveform, info = generator._crop_16k(
            object(),
            [],
            group_id=20,
            role=role,
        )
        results.append(
            (
                waveform.copy(),
                info["observation_source_selection"],
                [(call[2], call[3]) for call in generator.crop_calls],
            )
        )

    assert np.array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]


@pytest.mark.parametrize("role", ("near", "far_input"))
def test_all_quiet_attempts_raise_after_exactly_128_distinct_crops(role):
    module = _module()
    generator = _Generator({})
    generator.waveforms = SimpleNamespace()
    generator._crop_16k = lambda bank, records, *, group_id, role: (
        generator.crop_calls.append((bank, records, group_id, role)) or (
            _quiet(),
            {
                "repeated": False,
                "within_case_source_repetition": False,
                "segments": [
                    {
                        "path": f"/source/{group_id}.wav",
                        "sha256": "0" * 64,
                    }
                ],
            },
        )
    )

    module.install_observation_source_selection(generator)
    with pytest.raises(
        module.ObservationSourceSelectionError,
        match="exhausted 128 distinct crop attempts",
    ):
        generator._crop_16k(
            object(),
            [],
            group_id=20,
            role=role,
        )

    assert len(generator.crop_calls) == 128
    assert [call[2] for call in generator.crop_calls] == [
        20 + attempt * 1000 for attempt in range(128)
    ]


@pytest.mark.parametrize(
    "error",
    (
        OSError(errno.ENOSPC, "No space left on device"),
        MemoryError("allocation failed"),
        RuntimeError("resource safety stop"),
    ),
    ids=("enospc", "memory-error", "resource-safety-stop"),
)
def test_resource_and_unexpected_errors_are_not_retried(error):
    module = _module()
    generator = _Generator({})

    def failing_crop(bank, records, *, group_id, role):
        generator.crop_calls.append((bank, records, group_id, role))
        raise error

    generator._crop_16k = failing_crop
    module.install_observation_source_selection(generator)
    with pytest.raises(type(error)) as caught:
        generator._crop_16k(
            object(),
            [],
            group_id=20,
            role="far_input",
        )

    assert caught.value is error
    assert len(generator.crop_calls) == 1


@pytest.mark.parametrize("error_type", ("value", "selection"))
def test_only_crop_selection_errors_are_retried(error_type):
    module = _module()
    generator = _Generator({1_020: _tone()})
    original = generator._crop_16k
    calls = 0

    def retrying_crop(bank, records, *, group_id, role):
        nonlocal calls
        calls += 1
        if calls == 1:
            generator.crop_calls.append((bank, records, group_id, role))
            if error_type == "value":
                raise ValueError("candidate crop cannot satisfy source contract")
            raise module.ObservationSourceSelectionError("nested selector failed")
        return original(bank, records, group_id=group_id, role=role)

    generator._crop_16k = retrying_crop
    module.install_observation_source_selection(generator)
    waveform, info = generator._crop_16k(
        object(),
        [],
        group_id=20,
        role="far_input",
    )

    assert calls == 2
    assert [call[2:] for call in generator.crop_calls] == [
        (20, "far_input"),
        (1_020, "far_input"),
    ]
    assert np.array_equal(waveform, _tone())
    assert info["observation_source_selection"]["crop_group_id"] == 1_020


def test_case_plan_fallback_supports_current_real_stress_generator():
    module = _module()
    generator = _Generator({1: _tone()})
    generator.case_plan = None
    generator._case_plan = lambda: [
        {"group_id": 1, "category": "short_overlap"},
    ]

    module.install_observation_source_selection(generator)
    waveform, _ = generator._crop_16k(
        object(),
        [],
        group_id=1,
        role="near",
    )

    assert np.array_equal(waveform, _tone())
    assert [(call[2], call[3]) for call in generator.crop_calls] == [
        (1, "near"),
    ]
