from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import acceptance_models
from server.acoustic.acceptance_contract import required_conditions
from server.acoustic.receive_simulation import (
    evaluate_synthetic_receive,
)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    rate: int = 16_000,
    subtype: str = "FLOAT",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(samples, dtype=np.float64), rate, subtype=subtype)
    return path


def _tone(
    seconds: float = 1.0,
    rate: int = 16_000,
    frequency: float = 440.0,
    amplitude: float = 0.05,
) -> np.ndarray:
    time = np.arange(round(seconds * rate), dtype=np.float64) / rate
    return amplitude * np.sin(2 * np.pi * frequency * time)


def _send_paths(tmp_path: Path, *, seconds: float = 1.0, rate: int = 16_000):
    rng = np.random.default_rng(20261103)
    frames = round(seconds * rate)
    reference = _tone(seconds, rate, 220.0, 0.08)
    noise = rng.normal(0.0, 0.002, frames)
    microphone = reference + noise
    far_end = _tone(seconds, rate, 310.0, 0.02)
    tx = reference + 0.1 * noise
    input_paths = {
        "reference": _write_wav(tmp_path / "inputs/reference.wav", reference, rate),
        "noise": _write_wav(tmp_path / "inputs/noise.wav", noise, rate),
        "microphone": _write_wav(tmp_path / "inputs/microphone.wav", microphone, rate),
        "far_end": _write_wav(tmp_path / "inputs/far_end.wav", far_end, rate),
    }
    outputs = {"tx": _write_wav(tmp_path / "outputs/after.wav", tx, rate)}
    return input_paths, outputs


def _conditions(item_id: str, **updates) -> dict:
    from server.acoustic.acceptance_models import _contract_lookup

    conditions = required_conditions(_contract_lookup()[item_id])
    conditions.update(updates)
    return conditions


def _execution(outputs: dict[str, Path], **updates) -> dict:
    values = {
        "batch_id": "batch-1",
        "dataset_id": "dataset-1",
        "cli_sha256": "a" * 64,
        "output_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in outputs.items()
        },
    }
    values.update(updates)
    return values


def _receive_frequency_report(
    mode: str,
    rate: int,
    bands: dict[str, float],
) -> dict:
    return {
        "mode": mode,
        "simulation": {"analysis_sample_rate_hz": rate},
        "metrics": {
            "third_octave_band_levels_dbfs": {
                "binaural": bands,
            }
        },
    }


def test_output_hash_map_must_identify_every_output():
    mismatch = acceptance_models._declared_output_mismatch
    assert mismatch({}, {"tx": Path("tx.wav")}, {"tx": "a" * 64}) is not None
    assert mismatch({"output_sha256": {}}, {"tx": Path("tx.wav")}, {"tx": "a" * 64}) is not None


@pytest.mark.parametrize("mode,rate,item_id", [
    ("NB-wired", 16000, "NB-wired:B13"),
    ("WB-wired", 24000, "WB-wired:B13"),
    ("SWB-wired", 48000, "SWB-wired:B12"),
])
def test_stationary_send_response_uses_known_excitation_without_speech_vad(
    tmp_path, monkeypatch, mode, rate, item_id,
):
    frequencies = [f for f in acceptance_models.FREQUENCY_RESPONSE_CENTERS_HZ
                   if f <= acceptance_models.MODE_UPPER_HZ[mode.split("-")[0]]]
    time = np.arange(3 * rate) / rate
    components = [.008 * np.sin(2 * np.pi * f * time + index * .31)
                  for index, f in enumerate(frequencies)]
    reference = np.sum(components, axis=0)
    # A known 6 dB rise at the last required band must survive the dispatch.
    output = .5 * reference + .5 * components[-1]
    inputs = {role: _write_wav(tmp_path / f"{role}.wav", values, rate)
              for role, values in {"reference": reference, "microphone": reference,
                                   "noise": np.zeros_like(time), "far_end": np.zeros_like(time)}.items()}
    outputs = {"tx": _write_wav(tmp_path / "tx.wav", output, rate)}
    def reject_speech_gate(**kwargs):
        pytest.fail("stationary excitation must not enter speech VAD")
    monkeypatch.setattr(acceptance_models, "_send_report", reject_speech_gate)
    row = acceptance_models.calculate_case_items(
        {"id": "stationary", "conditions": _conditions(item_id), "item_ids": [item_id]},
        inputs, outputs, _execution(outputs), report_id="stationary-h1",
    )[0]
    assert row["status"] == "review_required"
    assert row["value"] == pytest.approx(20 * np.log10(2), abs=.02)
    assert row["standard_compliance"] is False
    scope = row["model_details"]["send_frequency_response"]
    assert scope["speech_activity_gate"] is False
    assert list(map(int, scope["bands"])) == frequencies
    assert row["evidence_binding"]["report_id"] == "stationary-h1"


@pytest.mark.parametrize("failure", ["missing_band", "silent_output", "unrelated_output", "nyquist", "rate_mismatch",
                                     "noise_missing", "noise_rate", "noise_short", "short_input", "low_snr"])
def test_send_response_rejects_missing_or_unobservable_evidence(failure):
    rate = 16000
    mode = "WB-wired" if failure == "nyquist" else "NB-wired"
    time = np.arange(3 * rate) / rate
    frequencies = [f for f in acceptance_models.FREQUENCY_RESPONSE_CENTERS_HZ
                   if f <= 4000 and not (failure == "missing_band" and f == 4000)]
    x = np.sum([.008 * np.sin(2 * np.pi * f * time) for f in frequencies], axis=0)
    y = .5 * x
    if failure == "silent_output":
        y = np.zeros_like(x)
    if failure == "unrelated_output":
        y = np.random.default_rng(18).normal(0, .05, len(x))
    noise = {"samples": np.zeros_like(x), "rate": rate}
    if failure == "noise_missing":
        noise = None
    elif failure == "noise_rate":
        noise["rate"] = 24000
    elif failure == "noise_short":
        noise["samples"] = noise["samples"][:-1]
    elif failure == "short_input":
        x, y = x[:rate], y[:rate]
        noise["samples"] = noise["samples"][:rate]
    elif failure == "low_snr":
        noise["samples"] = x.copy()
    result = {}
    acceptance_models._apply_send_frequency_response(
        result, mode=mode, reference={"samples": x, "rate": rate}, noise=noise,
        tx={"samples": y, "rate": rate + (8000 if failure == "rate_mismatch" else 0)},
    )
    assert result["value"] is None
    assert result["status"] == "not_measured"
    assert result["standard_compliance"] is False


def test_numeric_floor_is_retained_as_bound_without_exact_measurement():
    row = {}
    acceptance_models._set_engineering_value(row, metric="output_rms_dbfs", value=-120., reason="digital floor")
    assert row["value"] is None
    assert row["status"] == "not_measured"
    assert row["bound"] == {"relation": "<=", "value": -120.0}


def test_oob_at_nyquist_cannot_be_a_floor_measurement(tmp_path, monkeypatch):
    inputs, outputs = _send_paths(tmp_path)
    item_id = "WB-wired:B17"
    monkeypatch.setattr(acceptance_models, "_send_report", lambda **kwargs: {})
    row = acceptance_models.calculate_case_items(
        {"id": "nyquist", "conditions": _conditions(item_id), "item_ids": [item_id]},
        inputs, outputs, _execution(outputs), report_id="nyquist-report",
    )[0]
    assert row["value"] is None
    assert row["status"] == "not_measured"
    assert "不可观测" in row["reason"]


def _fake_send_report(
    *,
    uploads: dict[str, tuple[str, bytes]],
    item_values: dict[str, float | None],
    expected_frames: int | None = None,
    analysis_values: dict[str, float] | None = None,
) -> dict:
    if expected_frames is not None:
        for name, blob in uploads.values():
            samples, rate = sf.read(BytesIO(blob), dtype="float64")
            assert Path(name).suffix == ".wav"
            assert len(samples) == expected_frames
            assert rate == 16_000
    return {
        "analysis": {
            "values": analysis_values or {},
            "model_a": {
                "model": "A",
                "compliance_status": "diagnostic_only",
                "values": {"SIG": 99.0, "BAK": 98.0, "OVRL": 97.0},
            }
        },
        "items": [
            {
                "item_id": item_id,
                "metric": f"raw_{item_id}",
                "value": value,
                "status": "pass" if value is not None else "not_measured",
                "reason": "fake evaluator deliberately returned pass",
            }
            for item_id, value in item_values.items()
        ],
    }


def test_send_values_are_review_required_and_model_a_stays_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths, outputs = _send_paths(tmp_path)
    item_ids = ["NB-wired:B19", "NB-wired:B21", "NB-wired:B23"]
    values = {item_ids[0]: 100.0, item_ids[1]: 200.0, item_ids[2]: 200.0}
    calls = 0

    def fake_evaluate_five_signal(**kwargs):
        nonlocal calls
        calls += 1
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values=values,
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        fake_evaluate_five_signal,
    )
    execution = _execution(outputs)
    rows = acceptance_models.calculate_case_items(
        {
            "id": "case-send",
            "conditions": _conditions("NB-wired:B19"),
            "item_ids": item_ids,
        },
        input_paths,
        outputs,
        execution,
        report_id="report-send",
    )

    assert calls == 1
    assert [row["item_id"] for row in rows] == item_ids
    assert all(row["status"] == "review_required" for row in rows)
    assert all(row["evidence_grade"] == "engineering_proxy" for row in rows)
    assert all(row["standard_compliance"] is False for row in rows)
    assert all(row["diagnostics"]["model_a"]["values"]["SIG"] == 99.0 for row in rows)
    assert all(row["model_details"]["model_a"]["values"]["SIG"] == 99.0 for row in rows)
    assert rows[0]["test_name"].startswith("U - 3GPP")
    assert rows[0]["evidence_binding"]["case_id"] == "case-send"
    assert rows[0]["evidence_binding"]["report_id"] == "report-send"
    assert rows[0]["evidence_binding"]["batch_id"] == "batch-1"
    assert rows[0]["evidence_binding"]["dataset_id"] == "dataset-1"
    assert rows[0]["evidence_binding"]["output_sha256"] == execution["output_sha256"]
    assert set(rows[0]["evidence_binding"]["input_sha256"]) == {
        "reference",
        "noise",
        "microphone",
        "far_end",
    }


def test_noise_window_is_applied_without_truncating_five_signal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths, outputs = _send_paths(tmp_path)
    item_id = "NB-wired:B18"
    conditions = _conditions(item_id)

    def fake_evaluate_five_signal(**kwargs):
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values={item_id: 35.0},
            expected_frames=16_000,
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        fake_evaluate_five_signal,
    )
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-window",
            "analysis_start_s": 0.25,
            "conditions": conditions,
            "item_ids": [item_id],
        },
        input_paths,
        outputs,
        _execution(outputs),
        report_id="report-window",
    )[0]

    assert conditions["window_ms"] == 100
    assert row["status"] == "review_required"
    assert row["metric"] == "selected_window_out_of_band_ratio_db"
    assert row["window"]["window_ms"] == 100
    assert row["window"]["analysis_start_s"] == 0.25
    assert row["window"]["selected_frames"] == 1_600
    assert row["window"]["cropped"] is True
    assert row["evidence_binding"]["window_ms"] == 100


def test_five_signal_full_context_ignores_analysis_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths, outputs = _send_paths(tmp_path)
    item_id = "NB-wired:B19"

    def fake_evaluate_five_signal(**kwargs):
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values={item_id: 42.0},
            expected_frames=16_000,
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        fake_evaluate_five_signal,
    )
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-full-five-signal",
            "analysis_start_s": 0.25,
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        input_paths,
        outputs,
        _execution(outputs),
        report_id="report-full-five-signal",
    )[0]

    assert row["status"] == "review_required"
    assert row["window"]["analysis_start_s"] == 0.0
    assert row["window"]["selected_frames"] == 16_000
    assert row["model_details"]["analysis_scope"] == {
        "model": "five_signal",
        "scope": "full_clip",
        "analysis_start_s_applied": False,
        "window_ms_applied": False,
        "note": (
            "analysis_start_s/window_ms 仅用于明确 selected-window 家族；"
            "五路信号与 DNSMOS 使用未截短的完整片段"
        ),
    }


def test_send_dnsmos_full_context_ignores_analysis_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate = 16_000
    seconds = 9.1
    item_id = "NB-wired:B24"
    input_paths, _ = _send_paths(tmp_path / "inputs")
    outputs = {
        "tx": _write_wav(
            tmp_path / "full/after.wav",
            _tone(seconds=seconds, rate=rate),
            rate,
        )
    }
    calls = []

    class FakeDNSMOS:
        rate = 16_000
        minimum_seconds = 9.01

        def score(self, samples, rate):
            calls.append((len(samples), rate))
            return {
                "available": True,
                "values": {"SIG": 3.2, "BAK": 3.8, "OVRL": 3.0},
            }

    def fake_evaluate_five_signal(**kwargs):
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values={},
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        fake_evaluate_five_signal,
    )
    monkeypatch.setattr(
        acceptance_models,
        "_dnsmos_evaluator",
        lambda: FakeDNSMOS(),
    )
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-full-dnsmos",
            "analysis_start_s": 0.5,
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        input_paths,
        outputs,
        _execution(outputs),
        report_id="report-full-dnsmos",
    )[0]

    assert row["status"] == "review_required"
    assert calls == [(round(seconds * rate), rate)]
    assert row["window"]["analysis_start_s"] == 0.0
    assert row["model_details"]["analysis_scope"]["model"] == "dnsmos_send"
    assert row["model_details"]["analysis_scope"]["scope"] == "full_clip"


def test_receive_frequency_response_filters_to_report_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate = 16_000
    item_id = "NB-wired:B7"
    bands = {
        str(frequency): -40.0
        for frequency in acceptance_models.FREQUENCY_RESPONSE_CENTERS_HZ
        if frequency <= 4_000
    }
    bands["160"] = -35.0
    bands["8000"] = -120.0
    report = _receive_frequency_report("NB-wired", rate, bands)
    outputs = {
        "rx": _write_wav(
            tmp_path / "rx.wav",
            _tone(seconds=1.0, rate=rate),
            rate,
        )
    }
    monkeypatch.setattr(
        acceptance_models,
        "_receive_report",
        lambda **kwargs: report,
    )

    row = acceptance_models.calculate_case_items(
        {
            "id": "case-nb-frequency",
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        {},
        outputs,
        _execution(outputs),
        report_id="report-nb-frequency",
    )[0]

    # Output-only band levels remain useful diagnostics, but are not a
    # transfer function and must not generate a finite acceptance value.
    assert row["status"] == "not_measured"
    assert row["value"] is None
    assert "传递函数" in row["reason"]
    filtered = row["model_details"]["receive_frequency_bands_dbfs"]
    assert "8000" not in filtered
    assert list(filtered) == [
        str(frequency)
        for frequency in acceptance_models.FREQUENCY_RESPONSE_CENTERS_HZ
        if frequency <= 4_000
    ]
    scope = row["model_details"]["receive_frequency_response_scope"]
    assert scope["report_mode"] == "NB-wired"
    assert scope["upper_hz"] == 4_000.0
    assert scope["missing_bands_hz"] == []
    assert row["model_details"]["analysis_scope"]["scope"] == "full_clip"


@pytest.mark.parametrize(
    ("mode", "rate", "item_id", "available_upper_hz", "missing_hz"),
    [
        ("WB-wired", 16_000, "WB-wired:B7", 4_000, [5000.0, 6300.0, 8000.0]),
        ("SWB-wired", 48_000, "SWB-wired:B5", 8_000, [10000.0, 12500.0]),
    ],
)
def test_receive_frequency_response_requires_all_mode_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    rate: int,
    item_id: str,
    available_upper_hz: int,
    missing_hz: list[float],
) -> None:
    bands = {
        str(frequency): -40.0
        for frequency in acceptance_models.FREQUENCY_RESPONSE_CENTERS_HZ
        if frequency <= available_upper_hz
    }
    report = _receive_frequency_report(mode, rate, bands)
    outputs = {
        "rx": _write_wav(
            tmp_path / f"{mode}.wav",
            _tone(seconds=1.0, rate=rate),
            rate,
        )
    }
    monkeypatch.setattr(
        acceptance_models,
        "_receive_report",
        lambda **kwargs: report,
    )

    row = acceptance_models.calculate_case_items(
        {
            "id": f"case-{mode}",
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        {},
        outputs,
        _execution(outputs),
        report_id=f"report-{mode}",
    )[0]

    assert row["value"] is None
    assert row["status"] == "not_measured"
    assert "缺少必需可观测第三倍频程" in row["reason"]
    scope = row["model_details"]["receive_frequency_response_scope"]
    assert scope["missing_bands_hz"] == missing_hz
    assert row["model_details"]["receive_frequency_bands_dbfs"] is None


def test_condition_mismatch_and_empty_value_never_borrow_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths, outputs = _send_paths(tmp_path)
    item_id = "NB-wired:B18"
    mismatched = _conditions(item_id)
    mismatched["window_ms"] = None
    called = False

    def must_not_run(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("mismatched condition must not be evaluated")

    monkeypatch.setattr(acceptance_models, "evaluate_five_signal", must_not_run)
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-mismatch",
            "conditions": mismatched,
            "item_ids": [item_id, "unknown:B99"],
        },
        input_paths,
        outputs,
        _execution(outputs),
        report_id="report-mismatch",
    )
    assert called is False
    assert row[0]["status"] == "not_measured"
    assert row[0]["value"] is None
    assert "do not match" in row[0]["reason"]
    assert row[1]["status"] == "not_measured"
    assert "unknown contract item" in row[1]["reason"]

    valid = _conditions("NB-wired:B21")

    def empty_evaluator(**kwargs):
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values={"NB-wired:B21": None},
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        empty_evaluator,
    )
    empty = acceptance_models.calculate_case_items(
        {
            "id": "case-empty",
            "conditions": valid,
            "item_ids": ["NB-wired:B21"],
        },
        input_paths,
        outputs,
        _execution(outputs),
        report_id="report-empty",
    )[0]
    assert empty["status"] == "not_measured"
    assert empty["value"] is None


def test_cng_spectrum_uses_normalized_shape_not_total_power(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate = 16_000
    rng = np.random.default_rng(20261104)
    noise = rng.normal(0.0, 0.02, rate)
    scaled = noise * 25.0
    shifted = np.sin(2 * np.pi * 3_000 * np.arange(rate) / rate)

    same, same_meta = acceptance_models._normalized_psd_shape_distance(
        noise,
        rate,
        scaled,
        rate,
        upper_hz=4_000.0,
    )
    different, _ = acceptance_models._normalized_psd_shape_distance(
        noise,
        rate,
        shifted,
        rate,
        upper_hz=4_000.0,
    )
    assert same == pytest.approx(0.0, abs=1e-7)
    assert different > 0.25
    assert same_meta["normalized_psd"] is True
    assert same_meta["uses_total_power_difference"] is False

    item_id = "NB-wired:B28"
    input_paths = {
        "reference": _write_wav(tmp_path / "reference.wav", _tone(rate=rate)),
        "noise": _write_wav(tmp_path / "noise.wav", noise, rate),
        "microphone": _write_wav(tmp_path / "microphone.wav", noise, rate),
        "far_end": _write_wav(tmp_path / "far.wav", np.zeros(rate), rate),
    }
    outputs = {
        "tx": _write_wav(tmp_path / "after.wav", scaled, rate),
    }

    def fake_evaluate_five_signal(**kwargs):
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values={item_id: 999.0},
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        fake_evaluate_five_signal,
    )
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-cng",
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        input_paths,
        outputs,
        _execution(outputs),
        report_id="report-cng",
    )[0]
    assert row["metric"] == "normalized_psd_shape_distance_bits"
    assert row["value"] == pytest.approx(0.0, abs=1e-7)
    assert row["status"] == "review_required"
    assert "not use total power" not in row["reason"].lower()
    assert "未使用总功率差" in row["reason"]
    assert row["diagnostics"]["cng_spectrum"]["uses_total_power_difference"] is False


def test_receive_noise_uses_the_actual_selected_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.acoustic import receive_simulation

    monkeypatch.setattr(receive_simulation, "STORE", tmp_path / "receive-store")
    rate = 16_000
    rng = np.random.default_rng(20261105)
    rx = rng.normal(0.0, 0.01, rate)
    rx_path = _write_wav(tmp_path / "rx-spk.wav", rx, rate)
    outputs = {"rx": rx_path}
    item_id = "NB-wired:B8"
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-rx-noise",
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        {},
        outputs,
        _execution(outputs),
        report_id="report-rx-noise",
    )[0]

    assert row["status"] == "review_required"
    assert row["metric"] == "selected_window_binaural_rms_dbfs"
    assert isinstance(row["value"], float)
    assert row["window"]["selected_frames"] == 1_600
    synthetic = row["diagnostics"]["synthetic_receive"]
    assert synthetic["evidence_kind"] == "synthetic_head"
    assert synthetic["formal_hats_evidence"] is False
    assert synthetic["metrics"]["noise"]["start_s"] == 0.0
    assert synthetic["metrics"]["noise"]["end_s"] == pytest.approx(0.1)


def test_short_receive_audio_does_not_run_or_repeat_dnsmos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.acoustic import receive_simulation

    monkeypatch.setattr(receive_simulation, "STORE", tmp_path / "receive-store")
    rate = 16_000
    rx = _tone(seconds=1.0, rate=rate)
    outputs = {"rx": _write_wav(tmp_path / "rx-spk.wav", rx, rate)}
    item_id = "NB-wired:B10"

    class NeverCalled:
        rate = 16_000
        minimum_seconds = 9.01

        def score(self, samples, rate):
            raise AssertionError("short receive audio must not reach DNSMOS")

    monkeypatch.setattr(
        acceptance_models,
        "_dnsmos_evaluator",
        lambda: NeverCalled(),
    )
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-rx-short",
            "conditions": _conditions(item_id),
            "item_ids": [item_id],
        },
        {},
        outputs,
        _execution(outputs),
        report_id="report-rx-short",
    )[0]

    assert row["status"] == "not_measured"
    assert row["value"] is None
    assert "too short" in row["reason"]
    assert "not repeated" in row["reason"]


def test_dnsmos_resampling_metadata_and_scene_identity_are_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths, first_outputs = _send_paths(tmp_path / "scene1")
    rate_48k = 48_000
    second_outputs = {
        "tx": _write_wav(
            tmp_path / "scene2/after.wav",
            _tone(seconds=9.1, rate=rate_48k, amplitude=0.08),
            rate_48k,
        )
    }
    first_outputs = {
        "tx": _write_wav(
            tmp_path / "scene1/after.wav",
            _tone(seconds=9.1, rate=16_000, amplitude=0.02),
            16_000,
        )
    }
    calls = []

    class FakeDNSMOS:
        rate = 16_000
        minimum_seconds = 9.01
        name = "fake-dnsmos"

        def score(self, samples, rate):
            assert rate == 16_000
            assert len(samples) >= round(9.01 * 16_000)
            rms = float(np.sqrt(np.mean(np.square(samples))))
            calls.append((len(samples), rms))
            return {
                "available": True,
                "values": {
                    "SIG": min(5.0, 2.0 + 20 * rms),
                    "BAK": 4.0,
                    "OVRL": min(5.0, 2.5 + 20 * rms),
                },
            }

    def fake_evaluate_five_signal(**kwargs):
        return _fake_send_report(
            uploads=kwargs["uploads"],
            item_values={},
        )

    monkeypatch.setattr(
        acceptance_models,
        "evaluate_five_signal",
        fake_evaluate_five_signal,
    )
    monkeypatch.setattr(
        acceptance_models,
        "_dnsmos_evaluator",
        lambda: FakeDNSMOS(),
    )

    first_item = "NB-wired:B24"
    second_item = "NB-wired:B29"
    first = acceptance_models.calculate_case_items(
        {
            "id": "case-scene-1",
            "conditions": _conditions(first_item),
            "item_ids": [first_item],
        },
        input_paths,
        first_outputs,
        _execution(first_outputs),
        report_id="report-scene-1",
    )[0]
    second = acceptance_models.calculate_case_items(
        {
            "id": "case-scene-2",
            "conditions": _conditions(second_item),
            "item_ids": [second_item],
        },
        input_paths,
        second_outputs,
        _execution(second_outputs),
        report_id="report-scene-2",
    )[0]

    assert first["status"] == second["status"] == "review_required"
    assert first["value"] != second["value"]
    assert first["evidence_binding"]["scene"] == "sce1"
    assert second["evidence_binding"]["scene"] == "sce2"
    assert first["evidence_binding"]["case_id"] != second["evidence_binding"]["case_id"]
    assert first["evidence_binding"]["output_sha256"] != second["evidence_binding"]["output_sha256"]
    assert first["diagnostics"]["dnsmos"]["resampling"]["resampled"] is False
    assert second["diagnostics"]["dnsmos"]["resampling"]["resampled"] is True
    assert second["diagnostics"]["dnsmos"]["resampling"]["source_sample_rate_hz"] == 48_000
    assert second["diagnostics"]["dnsmos"]["resampling"]["repeated"] is False
    assert len(calls) == 2


def test_output_hash_mismatch_invalidates_without_computing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths, outputs = _send_paths(tmp_path)
    called = False

    def must_not_run(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("hash mismatch must stop evaluation")

    monkeypatch.setattr(acceptance_models, "evaluate_five_signal", must_not_run)
    row = acceptance_models.calculate_case_items(
        {
            "id": "case-hash",
            "conditions": _conditions("NB-wired:B21"),
            "item_ids": ["NB-wired:B21"],
        },
        input_paths,
        outputs,
        _execution(outputs, output_sha256={"tx": "0" * 64}),
        report_id="report-hash",
    )[0]
    assert called is False
    assert row["status"] == "invalid"
    assert row["value"] is None
    assert "output_sha256" in row["reason"]
