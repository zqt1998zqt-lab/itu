"""Independent checks of actual audio and A/B evidence admission failures."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from server.app import app
from server import localvqe_runner
from server.localvqe_ab_results import summarize_ab

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_mdf_localvqe_mask_ab_v2.py"
RESULT = localvqe_runner.AB_RESULT

spec = importlib.util.spec_from_file_location("mdf_localvqe_ab", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def payload():
    data = json.loads(RESULT.read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == localvqe_runner.AB_RESULT_SHA256
    return data


def test_full_duration_and_four_groups_include_the_actual_fir_event(payload):
    summary = summarize_ab(payload)
    assert summary["case_count"] == 20
    assert summary["assessment"]["status"] == "not_recommended"
    assert summary["assessment"]["near_target_gain_db"] < -30
    assert set(summary["categories"]) == {"near_single", "far_single", "double_talk", "path_change"}
    for group in summary["categories"].values():
        assert group["case_count"] == 5
    for row in payload["cases"]:
        assert row["samples"] == 192000
        if row["category"] == "path_change":
            paths = row["scene"]["echo_path_segments"]
            assert module.verify_path_coverage(row["samples"], paths) == [6.0]
            with pytest.raises(ValueError, match="outside"):
                module.verify_path_coverage(4 * 16000, paths)


@pytest.mark.parametrize("defect", ["truncated", "same_fir", "rx_reference", "wrong_mdf", "extra_reset", "no_flush", "fake_green_controls", "zero_runtime", "controls_flag"])
def test_evidence_validator_rejects_broken_experiments(payload, defect):
    data = deepcopy(payload)
    path = next(row for row in data["cases"] if row["category"] == "path_change")
    if defect == "truncated":
        path["samples"] = 64000
    elif defect == "same_fir":
        path["scene"]["echo_path_segments"][1]["paths"] = deepcopy(path["scene"]["echo_path_segments"][0]["paths"])
    elif defect == "rx_reference":
        path["scene"]["tx_echo_pcm_sha256"] = "0" * 64
    elif defect == "wrong_mdf":
        path["mdf"]["report"]["speex_echo_tail_ms"] = 200
    elif defect == "extra_reset":
        path["routes"]["mdf_residual_mask"]["timing"]["reset_sample_offsets"] = [0, 96000]
    elif defect == "no_flush":
        data["alignment"]["flush_output"] = False
    elif defect == "fake_green_controls":
        path["controls"]["daf_replay_max_abs_error"] = 1.0
    elif defect == "controls_flag":
        data["controls_passed"] = False
    else:
        path["routes"]["mdf_residual_mask"]["timing"]["process_ms"] = 0
    with pytest.raises(ValueError):
        summarize_ab(data)


def test_actual_pcm_matches_rx_reference_and_all_output_records(payload):
    for row in payload["cases"]:
        folder = RESULT.parent / row["category"] / row["case_id"]
        playback, rate = sf.read(folder / "playback.wav", dtype="float32")
        rx, rx_rate = sf.read(folder / "rx.rx-spk.wav", dtype="float32")
        assert rate == rx_rate == 16000 and np.array_equal(playback, rx)
        capture, _ = sf.read(folder / "mdf.aec-input.wav", dtype="float32")
        residual, _ = sf.read(folder / "mdf-e.wav", dtype="float32")
        yhat, _ = sf.read(folder / "mdf-yhat.wav", dtype="float32")
        assert sf.info(folder / "mdf.aec-input.wav").subtype == "PCM_16"
        assert np.array_equal(capture, residual + yhat)
        for route in row["routes"].values():
            record = route["artifact"]
            file = folder / record["path"]
            output, output_rate = sf.read(file, dtype="float32")
            assert output_rate == 16000 and output.shape == (192000,) and np.isfinite(output).all()
            assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"]
            assert module.pcm_digest(output) == record["pcm_sha256"]


def test_scorer_does_not_accept_muting_or_remove_a_gain_penalty():
    time = np.arange(4000, dtype=np.float64) / 16000
    target = np.sin(2 * np.pi * 437 * time) * .1
    mask = np.ones(len(target), dtype=bool)
    muted = module.measure(np.zeros_like(target), target, target, mask, near_present=True)
    assert muted["near_si_sdr_db"] is None and muted["near_retention_probe"] is False
    scaled = module.measure(target * .01, target, target, mask, near_present=True)
    assert scaled["near_si_sdr_db"] > 100
    assert scaled["near_target_gain_db"] == pytest.approx(-40)
    assert scaled["near_retention_probe"] is False
    far = module.measure(target, target, np.zeros_like(target), mask, near_present=False)
    assert far["near_si_sdr_db"] is None and far["output_attenuation_db"] == pytest.approx(0)


def test_nonzero_mask_partial_hop_is_flushed_and_prefix_matches_zero_padding(payload):
    library = Path(payload["identity"]["library"]["path"])
    engine = module.Engine(library)
    rng = np.random.default_rng(9813)
    residual = rng.normal(0, .03, 8193).astype("float32")
    echo = rng.normal(0, .02, 8193).astype("float32")
    try:
        output, raw, timing = engine.run(residual, echo, mode="mask")
        long, long_raw, _ = engine.run(np.pad(residual, (0, 255)), np.pad(echo, (0, 255)), mode="mask")
    finally:
        engine.close()
    assert output.shape == (8193,) and np.isfinite(output).all()
    assert np.array_equal(output, long[:8193]) and np.array_equal(raw, long_raw)
    assert np.array_equal(output, raw[256:256 + 8193])
    assert timing["padding_samples"] == 511
    assert np.max(np.abs(output[-129:])) > 1e-7 and output[-1] != 0


def test_double_talk_invalid_stimulus_is_excluded_without_becoming_zero(payload):
    summary = summarize_ab(payload)
    dt = summary["categories"]["double_talk"]
    assert dt["measured_case_count"] == 4 and len(dt["invalid_cases"]) == 1
    invalid = next(row for row in payload["cases"] if row["measurement_status"] == "invalid_stimulus")
    assert invalid["case_id"] == "double_talk-05"
    assert invalid["activity"]["joint_s"] < .05
    assert all(route["metrics"]["near_si_sdr_db"] is None for route in invalid["routes"].values())


def test_report_api_and_audio_download_match_verified_release():
    client = TestClient(app)
    response = client.get("/api/localvqe/mdf-mask-ab")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["case_count"] == 20 and payload["negative_controls_passed"] is True
    report = client.get(payload["report_url"])
    assert report.status_code == 200 and report.json()["schema"] == module.SCHEMA
    for group in payload["categories"].values():
        audio = client.get(group["cases"][0]["audio"]["mdf_residual_mask"])
        assert audio.status_code == 200 and audio.headers["content-type"].startswith("audio/wav")
    assert client.get("/api/localvqe/mdf-mask-ab/cases/unknown/input_mic.wav").status_code == 404


def test_changed_report_is_rejected_before_publishing(monkeypatch, tmp_path):
    file = tmp_path / "corrupt-report.json"
    file.write_text(RESULT.read_text() + " ")
    monkeypatch.setattr(localvqe_runner, "AB_RESULT", file)
    response = TestClient(app).get("/api/localvqe/mdf-mask-ab")
    assert response.status_code == 409
