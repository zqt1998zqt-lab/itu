from __future__ import annotations

from pathlib import Path
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import five_signal, service
from server.quality import api as quality_api
from server.quality.ts103281 import (
    MODEL_DIR,
    ModelAError,
    RandomForestModel,
    _align,
    _aggregate_basic_bands,
    evaluate_model_a,
    load_models,
    load_official_validation,
)


def test_published_model_files_load_with_expected_contract():
    models = load_models()
    assert set(models) == {"smos", "nmos", "gmos"}
    assert [len(models[name].trees) for name in ("smos", "nmos", "gmos")] == [
        50,
        50,
        50,
    ]
    assert len(models["smos"].feature_names) == 9
    assert len(models["nmos"].feature_names) == 4
    assert len(models["gmos"].feature_names) == 3


def test_random_forest_file_format_and_split_direction(tmp_path: Path):
    model_path = tmp_path / "model.model"
    model_path.write_text(
        "\n".join(
            [
                "1",
                "3",
                "0",
                "0.5",
                "1 2",
                "3.0 0",
                "0",
                "0",
                "0 0",
                "2.0 0",
                "0",
                "0",
                "0 0",
                "4.0 0",
                "PreProcessing.SNR-A",
                "S-MOS",
            ]
        )
    )
    model = RandomForestModel.load(model_path)
    assert model.predict({"PreProcessing.SNR-A": 0.25})["value"] == 2.0
    assert model.predict({"PreProcessing.SNR-A": 0.75})["value"] == 4.0


def test_random_forest_rejects_invalid_children(tmp_path: Path):
    model_path = tmp_path / "bad.model"
    model_path.write_text(
        "\n".join(
            [
                "1",
                "1",
                "0",
                "0.5",
                "2 3",
                "2.0 0",
                "PreProcessing.SNR-A",
                "S-MOS",
            ]
        )
    )
    with pytest.raises(ModelAError, match="child outside tree"):
        RandomForestModel.load(model_path)


def test_alignment_prepends_zeros_for_positive_degraded_delay():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=48_000)
    delay = 2_400
    degraded = np.r_[np.zeros(delay), reference[:-delay]]
    aligned, diagnostics = _align(reference, degraded)
    assert diagnostics["lag_samples"] == delay
    np.testing.assert_allclose(aligned, degraded, atol=1e-12)


def test_auditory_bands_aggregate_from_99_to_33_by_quadratic_mean():
    centres = np.arange(99, dtype=np.float64)
    values = np.arange(99 * 2, dtype=np.float64).reshape(99, 2)
    aggregated, basic_centres = _aggregate_basic_bands(values, centres)
    assert aggregated.shape == (33, 2)
    assert basic_centres.tolist() == list(range(2, 99, 3))
    expected = np.sqrt(np.mean(values.reshape(33, 3, 2) ** 2, axis=1))
    np.testing.assert_allclose(aggregated, expected)


def test_model_a_evaluate_is_finite_and_keeps_formal_boundary():
    rate = 48_000
    seconds = 1.0
    time = np.arange(round(rate * seconds)) / rate
    envelope = (time >= 0.1) & (time < 0.85)
    reference = np.zeros_like(time)
    reference[envelope] = 0.02 * np.sin(2 * np.pi * 440 * time[envelope])
    rng = np.random.default_rng(103281)
    degraded = 0.8 * reference + rng.normal(0, 0.0002, len(reference))
    result = evaluate_model_a(reference, degraded, rate)
    assert result["compliance_status"] == "experimental_nonconformant"
    assert result["formal_compliance_claim_allowed"] is False
    assert set(result["scores"]) == {"S-MOS", "N-MOS", "G-MOS"}
    assert all(
        np.isfinite(score["value"])
        for score in result["scores"].values()
    )
    assert all(
        1.0 <= score["value"] <= 5.0
        for score in result["scores"].values()
    )
    assert "diagnostic_scores" not in result
    assert result["verification"]["status"] == "failed_tolerance_gate"
    assert result["verification"]["condition_count"] == 48
    assert result["verification"]["passed_conditions"] == 0
    assert result["diagnostics"]["activity"]["activity_frame_ms"] == 10.0
    assert result["diagnostics"]["band_count"] == 33
    diagnostic = evaluate_model_a(
        reference,
        degraded,
        rate,
        include_diagnostic_calibration=True,
    )
    assert diagnostic["diagnostic_calibration"]["independent_validation"] is False
    assert all(
        np.isfinite(score["value"])
        for score in diagnostic["diagnostic_scores"].values()
    )


def test_model_a_resamples_reference_and_degraded_from_their_native_rates():
    reference_rate = 48_000
    degraded_rate = 24_000
    time = np.arange(reference_rate) / reference_rate
    active = (time >= 0.1) & (time < 0.85)
    reference = np.zeros_like(time)
    reference[active] = 0.02 * np.sin(2 * np.pi * 440 * time[active])
    degraded_48k = 0.8 * reference
    degraded = degraded_48k[::2]
    result = evaluate_model_a(
        reference,
        degraded,
        reference_rate,
        degraded_rate=degraded_rate,
    )
    assert all(
        np.isfinite(score["value"])
        for score in result["scores"].values()
    )


def test_asset_manifest_is_present():
    assert (MODEL_DIR / "SHA256SUMS").is_file()
    assert (MODEL_DIR / "annex-c-model-a.csv").is_file()
    assert (MODEL_DIR / "annex-c-validation.json").is_file()
    validation = load_official_validation()
    assert validation["stale"] is False
    assert validation["status"] == "failed_tolerance_gate"


def test_http_endpoint_returns_raw_and_diagnostic_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_api, "RUNS", tmp_path)
    rate = 48_000
    time = np.arange(rate) / rate
    active = (time >= 0.1) & (time < 0.85)
    reference = np.zeros_like(time)
    reference[active] = 0.02 * np.sin(2 * np.pi * 440 * time[active])
    degraded = 0.8 * reference

    def wav(samples: np.ndarray) -> bytes:
        buffer = BytesIO()
        sf.write(buffer, samples.astype(np.float32), rate, format="WAV")
        return buffer.getvalue()

    response = TestClient(app).post(
        "/api/quality/ts103281/model-a",
        files={
            "reference": ("reference.wav", wav(reference), "audio/wav"),
            "degraded": ("degraded.wav", wav(degraded), "audio/wav"),
        },
    )
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert set(report["scores"]) == {"S-MOS", "N-MOS", "G-MOS"}
    assert "diagnostic_scores" not in report
    assert report["compliance_status"] == "experimental_nonconformant"
    assert report["verification"]["passed_conditions"] == 0


def test_five_signal_report_embeds_model_a_result(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    calls = []

    def fake_model_a(reference, degraded, rate):
        calls.append((len(reference), len(degraded), rate))
        return {
            "standard": "ETSI TS 103 281 V1.3.2",
            "model": "A",
            "compliance_status": "experimental_nonconformant",
            "scores": {
                "S-MOS": {"value": 3.0},
                "N-MOS": {"value": 3.0},
                "G-MOS": {"value": 3.0},
            },
            "verification": {"passed_conditions": 0, "condition_count": 48},
        }

    monkeypatch.setattr(five_signal, "evaluate_model_a", fake_model_a)
    rate = 16_000
    time = np.arange(rate * 4) / rate
    near = np.zeros_like(time)
    far = np.zeros_like(time)
    near_mask = (time >= 0.25) & (time < 1.4)
    double_talk = (time >= 1.55) & (time < 2.5)
    far_mask = (time >= 2.65) & (time < 3.6)
    near[near_mask | double_talk] = (
        0.18
        * np.sin(2 * np.pi * 220 * time[near_mask | double_talk])
        * (0.65 + 0.35 * np.sin(2 * np.pi * 3.3 * time[near_mask | double_talk]))
    )
    far[double_talk | far_mask] = 0.12 * np.sin(
        2 * np.pi * 310 * time[double_talk | far_mask]
    )
    noise = np.random.default_rng(103281).normal(0, 0.003, len(time))
    echo = np.r_[np.zeros(160), far[:-160]] * 0.35
    microphone = near + echo + noise
    near_end = near + echo * 0.05 + noise * 0.2

    def encoded(samples: np.ndarray) -> bytes:
        buffer = BytesIO()
        sf.write(buffer, samples.astype(np.float32), rate, format="WAV")
        return buffer.getvalue()

    uploads = {
        "near_end": ("near.wav", encoded(near_end)),
        "noise": ("noise.wav", encoded(noise)),
        "microphone": ("mic.wav", encoded(microphone)),
        "far_end": ("far.wav", encoded(far)),
        "reference": ("reference.wav", encoded(near)),
    }
    report = five_signal.evaluate_five_signal(
        mode="WB-wired",
        uploads=uploads,
    )
    assert calls == [(rate * 4, rate * 4, rate)]
    assert (
        report["analysis"]["model_a"]["compliance_status"]
        == "experimental_nonconformant"
    )
