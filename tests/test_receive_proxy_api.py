from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.acoustic import receive_simulation as simulation
from server.acoustic import api as acoustic_api
from server.app import app


def wav(samples: np.ndarray, rate: int = 16_000) -> bytes:
    stream = BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="PCM_24")
    return stream.getvalue()


def tone(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    rate = 16_000
    time = np.arange(round(seconds * rate), dtype=np.float64) / rate
    return amplitude * np.sin(2 * np.pi * 440 * time)


@pytest.fixture(autouse=True)
def isolated_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    store = tmp_path / "synthetic-receive"
    monkeypatch.setattr(simulation, "STORE", store)
    return store


def create_receive(amplitude: float, seconds: float = 0.5) -> dict:
    return simulation.evaluate_synthetic_receive(
        "WB-wireless",
        "speaker.wav",
        wav(tone(seconds, amplitude)),
        azimuth_deg=30,
    )


def create_windowed_receive(amplitude: float) -> dict:
    samples = tone(0.5, amplitude)
    start = round(0.05 * 16_000)
    end = round(0.45 * 16_000)
    samples[:start] = 0.0
    samples[end:] = 0.0
    return simulation.evaluate_synthetic_receive(
        "WB-wireless",
        "speaker.wav",
        wav(samples),
        azimuth_deg=30,
    )


def test_rlr_linearity_api_groups_three_receive_runs() -> None:
    nominal = create_receive(0.1)
    minus = create_receive(0.1 * 10 ** (-5 / 20))
    plus = create_receive(0.1 * 10 ** (5 / 20))
    client = TestClient(app)

    response = client.post(
        "/api/acoustic/receive-simulation/rlr-linearity/evaluate",
        json={
            "evaluation_ids": {
                "nominal": nominal["evaluation_id"],
                "offset_-5": minus["evaluation_id"],
                "offset_+5": plus["evaluation_id"],
            },
            "conditions": {
                "nominal_volume": "virtual-volume-10",
                "fixture": {
                    "fixture_id": "software-drp",
                    "position": "driver",
                },
                "active_speech_window_s": [0.05, 0.45],
            },
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "pass"
    assert result["group_valid"] is True
    assert result["window_applied"] is True
    assert result["window_source"] == "provided_cropped_band_levels"
    assert result["analysis_window_s"] == [0.05, 0.45]
    assert abs(result["items"]["B5"]["residual_db"]) < 0.01
    assert abs(result["items"]["B6"]["residual_db"]) < 0.01
    assert result["evidence_level"] == "engineering_proxy"
    assert result["formal_compliance_claim_allowed"] is False
    assert len(result["canonical_sha256"]) == 64


def test_rlr_linearity_api_recomputes_bands_from_cropped_verified_wavs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {
        "nominal": create_windowed_receive(0.1),
        "offset_-5": create_windowed_receive(0.1 * 10 ** (-5 / 20)),
        "offset_+5": create_windowed_receive(0.1 * 10 ** (5 / 20)),
    }
    captured: dict[str, list[dict]] = {}
    original = acoustic_api.evaluate_rlr_linearity_group

    def capture(runs, conditions=None, *, include_canonical_hash=False):
        captured["runs"] = runs
        return original(
            runs,
            conditions,
            include_canonical_hash=include_canonical_hash,
        )

    monkeypatch.setattr(acoustic_api, "evaluate_rlr_linearity_group", capture)
    client = TestClient(app)
    response = client.post(
        "/api/acoustic/receive-simulation/rlr-linearity/evaluate",
        json={
            "evaluation_ids": {
                level: report["evaluation_id"]
                for level, report in reports.items()
            },
            "conditions": {
                "nominal_volume": "virtual-volume-10",
                "fixture": {
                    "fixture_id": "software-drp",
                    "position": "driver",
                },
                "active_speech_window_s": [0.05, 0.45],
            },
        },
    )

    assert response.status_code == 200, response.text
    assert len(captured["runs"]) == 3
    for run in captured["runs"]:
        level_id = run["level_id"]
        report = reports[level_id]
        left_path = simulation.get_synthetic_receive_file(
            report["evaluation_id"],
            "virtual_left.wav",
        )
        right_path = simulation.get_synthetic_receive_file(
            report["evaluation_id"],
            "virtual_right.wav",
        )
        assert left_path is not None and right_path is not None
        left, rate = sf.read(left_path, dtype="float64")
        right, right_rate = sf.read(right_path, dtype="float64")
        assert rate == right_rate
        start = round(0.05 * rate)
        end = round(0.45 * rate)
        expected = simulation._third_octave_levels(
            left[start:end],
            right[start:end],
            rate,
        )
        assert run["metrics"]["third_octave_band_levels_db"] == expected
        assert run["metrics"]["third_octave_band_levels_dbfs"] == expected
        assert (
            run["metrics"]["third_octave_band_levels_db"]
            != report["metrics"]["third_octave_band_levels_db"]
        )
        assert run["metrics"]["window_applied"] is True
        assert run["metrics"]["analysis_window_s"] == [0.05, 0.45]
        assert run["metrics"]["analysis_sample_bounds"] == {
            "left": [start, end],
            "right": [start, end],
        }
        assert run["metrics"]["analysis_artifact_sha256"] == {
            "left": report["artifacts"]["left"]["sha256"],
            "right": report["artifacts"]["right"]["sha256"],
        }


def test_rlr_linearity_api_requires_three_distinct_runs() -> None:
    report = create_receive(0.1)
    client = TestClient(app)
    response = client.post(
        "/api/acoustic/receive-simulation/rlr-linearity/evaluate",
        json={
            "evaluation_ids": {
                "nominal": report["evaluation_id"],
                "offset_-5": report["evaluation_id"],
                "offset_+5": report["evaluation_id"],
            },
            "conditions": {},
        },
    )
    assert response.status_code == 422
    assert "distinct" in response.json()["detail"]


def test_rlr_linearity_api_rejects_tampered_persisted_report(
    isolated_store: Path,
) -> None:
    nominal = create_receive(0.1)
    minus = create_receive(0.1 * 10 ** (-5 / 20))
    plus = create_receive(0.1 * 10 ** (5 / 20))
    report_path = isolated_store / nominal["evaluation_id"] / "report.json"
    report = json.loads(report_path.read_text())
    report["mode"] = "NB-wireless"
    report_path.write_text(json.dumps(report))

    client = TestClient(app)
    response = client.post(
        "/api/acoustic/receive-simulation/rlr-linearity/evaluate",
        json={
            "evaluation_ids": {
                "nominal": nominal["evaluation_id"],
                "offset_-5": minus["evaluation_id"],
                "offset_+5": plus["evaluation_id"],
            },
            "conditions": {},
        },
    )
    assert response.status_code == 409
    assert "hash mismatch" in response.json()["detail"]


def test_rlr_linearity_api_rejects_tampered_receive_wav() -> None:
    nominal = create_receive(0.1)
    minus = create_receive(0.1 * 10 ** (-5 / 20))
    plus = create_receive(0.1 * 10 ** (5 / 20))
    left_path = simulation.get_synthetic_receive_file(
        nominal["evaluation_id"],
        "virtual_left.wav",
    )
    assert left_path is not None
    left, rate = sf.read(left_path, dtype="float64")
    sf.write(left_path, left * 0.5, rate, format="WAV", subtype="PCM_24")

    client = TestClient(app)
    response = client.post(
        "/api/acoustic/receive-simulation/rlr-linearity/evaluate",
        json={
            "evaluation_ids": {
                "nominal": nominal["evaluation_id"],
                "offset_-5": minus["evaluation_id"],
                "offset_+5": plus["evaluation_id"],
            },
            "conditions": {
                "nominal_volume": "virtual-volume-10",
                "fixture": {
                    "fixture_id": "software-drp",
                    "position": "driver",
                },
                "active_speech_window_s": [0.05, 0.45],
            },
        },
    )
    assert response.status_code == 409
    assert "left artifact hash mismatch" in response.json()["detail"]


def test_b10_quality_proxy_uses_worst_ear_and_never_claims_tmos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = create_receive(0.1, seconds=9.1)
    calls = 0

    class FakeDNSMOS:
        rate = 16_000
        minimum_seconds = 9.01

        def score(self, samples: np.ndarray, rate: int) -> dict:
            nonlocal calls
            calls += 1
            values = (
                {"SIG": 3.8, "BAK": 4.2, "OVRL": 3.6}
                if calls == 1
                else {"SIG": 3.1, "BAK": 4.4, "OVRL": 2.9}
            )
            return {"available": True, "values": values}

    monkeypatch.setattr(acoustic_api, "_dnsmos_evaluator", lambda: FakeDNSMOS())
    client = TestClient(app)
    response = client.post(
        f"/api/acoustic/receive-simulation/evaluations/"
        f"{report['evaluation_id']}/quality-proxy"
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert calls == 2
    assert result["per_ear"]["left"]["OVRL"] == 3.6
    assert result["per_ear"]["right"]["OVRL"] == 2.9
    assert result["SIG_proxy"] == 3.1
    assert result["BAK_proxy"] == 4.2
    assert result["OVRL_proxy"] == 2.9
    assert result["TMOS_proxy"] == 2.9
    assert result["formal_tmos_available"] is False
    assert result["formal_polqa_available"] is False
    assert result["formal_compliance_claim_allowed"] is False


def test_b10_quality_proxy_rejects_short_receive_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = create_receive(0.1, seconds=1.0)

    class NeverCalled:
        rate = 16_000
        minimum_seconds = 9.01

        def score(self, samples: np.ndarray, rate: int) -> dict:
            raise AssertionError("short audio must fail before model execution")

    monkeypatch.setattr(acoustic_api, "_dnsmos_evaluator", lambda: NeverCalled())
    client = TestClient(app)
    response = client.post(
        f"/api/acoustic/receive-simulation/evaluations/"
        f"{report['evaluation_id']}/quality-proxy"
    )
    assert response.status_code == 422
    assert "too short" in response.json()["detail"]
