from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import service
from server.acoustic.five_signal import evaluate_five_signal


MODES = (
    "NB-wired", "NB-wireless", "NB-bt",
    "WB-wired", "WB-wireless", "WB-bt",
    "SWB-wired", "SWB-wireless",
)


def _wav(x: np.ndarray, rate: int) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, np.asarray(x, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _signals(rate: int = 16000, seconds: float = 4.0) -> dict[str, tuple[str, bytes]]:
    rng = np.random.default_rng(20261028)
    n = round(rate * seconds)
    t = np.arange(n) / rate
    near = np.zeros(n)
    far = np.zeros(n)
    active_near = (t >= 0.25) & (t < 1.4)
    active_dt = (t >= 1.55) & (t < 2.5)
    active_far = (t >= 2.65) & (t < 3.6)
    near[active_near | active_dt] = (
        0.18 * np.sin(2 * np.pi * 220 * t[active_near | active_dt])
        * (0.65 + 0.35 * np.sin(2 * np.pi * 3.3 * t[active_near | active_dt]))
    )
    far[active_dt | active_far] = (
        0.12 * np.sin(2 * np.pi * 310 * t[active_dt | active_far])
        * (0.7 + 0.3 * np.cos(2 * np.pi * 2.1 * t[active_dt | active_far]))
    )
    noise = rng.normal(0, 0.003, n)
    echo = np.r_[np.zeros(160), far[:-160]] * 0.35
    microphone = near + echo + noise
    near_end = near + echo * 0.05 + noise * 0.2
    reference = near
    return {
        "near_end": ("near-end.wav", _wav(near_end, rate)),
        "noise": ("noise.wav", _wav(noise, rate)),
        "microphone": ("microphone.wav", _wav(microphone, rate)),
        "far_end": ("far-end.wav", _wav(far, rate)),
        "reference": ("reference.wav", _wav(reference, rate)),
    }


def test_five_signal_maps_one_mode_to_42_rows_and_keeps_formal_boundary(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    report = evaluate_five_signal(mode="WB-wired", uploads=_signals())
    assert report["mode"] == "WB-wired"
    assert report["standard_compliance"] is False
    assert report["formal_test_status"] == "not_measured"
    assert report["summary"]["total"] == 42
    assert len({item["item_id"] for item in report["items"]}) == 42
    assert report["summary"]["measured"] > 0
    assert report["summary"]["not_measured"] > 0
    assert any(item["status"] == "pass" for item in report["items"])
    assert all(item["standard_compliance"] is False for item in report["items"])
    assert report["analysis"]["values"]["output_sisdr_db"] is not None
    assert report["analysis"]["segments"]["noise_only_seconds"] > 0
    result = build / report["result"]["path"]
    assert result.is_file()
    assert json.loads(result.read_text())["evaluation_id"] == report["evaluation_id"]


def test_all_eight_modes_produce_336_unique_contract_rows(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    rows = []
    for mode in MODES:
        report = evaluate_five_signal(mode=mode, uploads=_signals())
        assert report["summary"]["total"] == 42
        rows.extend(report["items"])
    assert len(rows) == 336
    assert len({row["item_id"] for row in rows}) == 336
    assert all(row["standard_compliance"] is False for row in rows)


def test_same_five_signals_are_not_treated_as_eight_formal_modes(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    report = evaluate_five_signal(mode="NB-wired", uploads=_signals(rate=8000))
    assert report["mode"] == "NB-wired"
    assert all("NB-wired:" in item["item_id"] for item in report["items"])
    assert report["limitations"][2].startswith("NB/WB/SWB 是业务配置")


def test_http_endpoint_accepts_five_files_and_persists_result(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    files = _signals()
    response = TestClient(app).post(
        "/api/acoustic/five-signal/evaluate",
        data={"mode": "WB-wired", "channel": "0", "max_delay_ms": "2000"},
        files={role: (name, blob, "audio/wav") for role, (name, blob) in files.items()},
    )
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["summary"]["total"] == 42
    replay = TestClient(app).get(
        f"/api/acoustic/five-signal/evaluations/{report['evaluation_id']}"
    )
    assert replay.status_code == 200
    assert replay.json()["items"] == report["items"]


def test_silent_reference_is_rejected(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    signals = _signals()
    signals["reference"] = ("silent.wav", _wav(np.zeros(16000 * 2), 16000))
    with pytest.raises(ValueError, match="参考音没有足够的有效语音活动"):
        evaluate_five_signal(mode="WB-wired", uploads=signals)
