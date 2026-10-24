from datetime import datetime, timezone
import json
import numpy as np
import pytest
import soundfile as sf

from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import service
from server.acoustic.models import Artifact, sha256_file


def payload(*, formal: bool = False) -> dict:
    return {
        "mode": "formal" if formal else "precompliance",
        "confirm_formal": formal,
        "bandwidth": "SWB",
        "transport": "wireless",
        "fixture": {
            "fixture_id": "hats-lab-a",
            "fixture_type": "HATS",
            "room": "simulated_vehicle",
            "background_noise_dba": -70,
            "reverberation_rt60_ms": 150,
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "calibration_points": [],
        },
        "dut": {
            "name": "SU7 head unit",
            "firmware": "candidate-1",
            "interface": "acoustic",
            "network_profiles": ["VoLTE / EVS-WB / 48 kHz"],
            "volume_setting": "nominal",
            "audio_path": "HATS -> DUT -> network simulator -> POI",
        },
        "hardware": {
            "backend": "labCORE-adapter",
            "capabilities": [
                "playback", "capture", "timestamps",
                "level_calibration", "background_noise_control",
            ],
            "calibration_id": "calibration-2026-10",
        },
        "method": {
            "package_id": "P1120-2025-method-pack",
            "p835_version": "2026-07",
            "signal_pack_id": "P501-accessible-pack",
        },
    }


def test_formal_request_requires_attestation() -> None:
    body = payload(formal=True)
    body["confirm_formal"] = False
    assert TestClient(app).post("/api/acoustic/runs", json=body).status_code == 422


def test_missing_evidence_blocks_formal_and_downgrades() -> None:
    response = TestClient(app).post("/api/acoustic/runs", json=payload(formal=True))
    assert response.status_code == 201
    report = response.json()
    assert report["status"] == "blocked"
    assert report["requested_mode"] == "formal"
    assert report["effective_mode"] == "precompliance"
    assert report["standard"]["recommendation"] == "P.1120"
    assert "fixture_calibration_points" in report["blockers"]
    assert report["standard_compliance"] is False
    assert report["not_measured_items"] > 0
    replay = TestClient(app).get(f"/api/acoustic/runs/{report['run_id']}")
    assert replay.status_code == 200
    assert replay.json() == report


def test_simulation_is_not_silently_promoted_to_formal() -> None:
    body = payload()
    body["mode"] = "simulation"
    response = TestClient(app).post("/api/acoustic/runs", json=body)
    assert response.status_code == 201
    assert response.json()["effective_mode"] == "simulation"


def test_api_rejects_unknown_paths_and_run_ids() -> None:
    client = TestClient(app)
    assert client.get("/api/acoustic/runs/not-a-run").status_code == 404
    body = payload()
    body["fixture"]["evidence"] = [
        {"path": "../secret", "sha256": "0" * 64, "kind": "calibration"}
    ]
    assert client.post("/api/acoustic/runs", json=body).status_code == 422


def test_simulation_execution_is_diagnostic_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(service, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(service, "EXECUTIONS", tmp_path / "executions")
    body = payload()
    body["mode"] = "simulation"
    client = TestClient(app)
    plan = client.post("/api/acoustic/runs", json=body)
    assert plan.status_code == 201
    run_id = plan.json()["run_id"]
    assert client.post(
        f"/api/acoustic/runs/{run_id}/execute",
        json={"confirm_execute": False},
    ).status_code == 422
    response = client.post(
        f"/api/acoustic/runs/{run_id}/execute",
        json={"confirm_execute": True},
    )
    assert response.status_code == 201
    report = response.json()
    assert report["standard_compliance"] is False
    assert report["summary"]["not_measured"] == 42
    assert all(item["status"] == "not_measured" for item in report["items"])
    replay = client.get(f"/api/acoustic/runs/{run_id}").json()
    assert replay["status"] == "completed"
    assert replay["gate_status"] == "blocked"
    assert (tmp_path / "executions" / report["execution_id"] / "results.json").is_file()
    assert client.get(
        f"/api/acoustic/executions/{report['execution_id']}"
    ).status_code == 200


def test_evidence_hash_tamper_and_measurement_units_are_rejected(
    tmp_path, monkeypatch
) -> None:
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(service, "EVIDENCE", evidence_root)
    monkeypatch.setattr(service, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(service, "EXECUTIONS", tmp_path / "executions")
    client = TestClient(app)
    uploaded = client.post(
        "/api/acoustic/evidence",
        files={"file": ("trace.txt", b"trace", "text/plain")},
        data={"kind": "measurement_trace", "namespace": "rp-delay-1"},
    )
    assert uploaded.status_code == 201
    record = uploaded.json()
    artifact = record["artifact"]
    assert artifact["path"].startswith("rp-delay-1/")
    download = client.get(f"/api/acoustic/evidence/{artifact['path']}")
    assert download.status_code == 200
    assert download.content == b"trace"
    body = payload()
    body["mode"] = "simulation"
    body["fixture"]["evidence"] = [artifact]
    assert client.post("/api/acoustic/runs", json=body).status_code == 201
    target = evidence_root / artifact["path"]
    target.chmod(0o640)
    target.write_bytes(b"tampered")
    assert client.get(f"/api/acoustic/evidence/{artifact['path']}").status_code == 404
    assert client.post("/api/acoustic/runs", json=body).status_code == 422


def test_invalid_noise_scenario_and_pilot_plan_compatibility() -> None:
    client = TestClient(app)
    body = payload()
    body["mode"] = "simulation"
    body["fixture"]["noise_scenarios"] = [{
        "id": "C2", "vehicle_speed_kmh": 120, "fan_setting": "medium",
        "airflow": "toward windows", "windows": "up", "wipers": "off",
        "turn_signal": "off", "background_talkers": "none",
        "road_surface": "dry rough road; no impulse bumps",
        "wind_speed_mps": 1, "precipitation": "none", "temperature_c": 20,
    }]
    assert client.post("/api/acoustic/runs", json=body).status_code == 422


def test_pcm_evidence_compiler_and_tamper_detection_in_bundle(
    tmp_path, monkeypatch
) -> None:
    build = tmp_path / "build"
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    monkeypatch.setattr(service, "RUNS", build / "runs")
    monkeypatch.setattr(service, "EXECUTIONS", build / "executions")
    traces = build / "traces"
    traces.mkdir(parents=True)
    rate = 16000
    rng = np.random.default_rng(12)
    reference = rng.normal(0, .02, rate * 2)
    capture = np.r_[np.zeros(83), reference[:-83]]
    sf.write(traces / "reference.wav", reference, rate, subtype="PCM_16")
    sf.write(traces / "capture.wav", capture, rate, subtype="PCM_16")
    response = TestClient(app).post("/api/acoustic/measurements/compile", json={
        "item_id": "SWB-wireless:B2",
        "unit": "Calculated Value [ms]",
        "kind": "loopback",
        "reference_path": str(traces / "reference.wav"),
        "capture_path": str(traces / "capture.wav"),
        "max_delay_ms": 50,
    })
    assert response.status_code == 201, response.text
    compiled = response.json()
    assert 5 <= compiled["measurement"]["value"] <= 6
    assert compiled["bundle"]["artifact"]["kind"] == "measurement_bundle"
    assert compiled["traces"]["capture"]["artifact"]["kind"] == "measurement_trace"
    stored_bundle = json.loads(
        (build / "evidence" / compiled["bundle"]["artifact"]["path"]).read_text()
    )
    assert stored_bundle["backend"]["identity_sha256"] == compiled["bundle_sha256"]


def test_evidence_compiler_rejects_paths_outside_build(tmp_path, monkeypatch) -> None:
    build = tmp_path / "build"
    build.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not PCM")
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    response = TestClient(app).post("/api/acoustic/measurements/compile", json={
        "item_id": "SWB-wireless:B2",
        "unit": "Calculated Value [ms]",
        "kind": "loopback",
        "reference_path": str(outside),
        "capture_path": str(outside),
    })
    assert response.status_code == 422


def _compiler_paths(tmp_path, monkeypatch):
    build = tmp_path / "build"
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "BUILD", build)
    monkeypatch.setattr(service, "EVIDENCE", build / "evidence")
    traces = build / "traces"
    traces.mkdir(parents=True)
    return build, traces


def test_noise_compiler_measures_explicit_interval(tmp_path, monkeypatch) -> None:
    _, traces = _compiler_paths(tmp_path, monkeypatch)
    rate = 16000
    quiet = np.full(rate, .002)
    noise = np.full(rate, .02)
    trace = np.r_[quiet, noise]
    sf.write(traces / "reference.wav", quiet, rate, subtype="PCM_16")
    sf.write(traces / "capture.wav", trace, rate, subtype="PCM_16")
    response = TestClient(app).post("/api/acoustic/measurements/compile", json={
        "item_id": "SWB-wired:B7",
        "unit": "Level [dBFS]",
        "kind": "noise",
        "reference_path": str(traces / "reference.wav"),
        "capture_path": str(traces / "capture.wav"),
        "start_s": 1,
        "end_s": 2,
    })
    assert response.status_code == 201, response.text
    assert response.json()["measurement"]["value"] == pytest.approx(-33.98, abs=.02)


def test_transfer_compiler_normalizes_before_applying_mask(
    tmp_path, monkeypatch
) -> None:
    _, traces = _compiler_paths(tmp_path, monkeypatch)
    rate = 16000
    duration = 8
    rng = np.random.default_rng(19)
    reference = rng.normal(0, .02, rate * duration)
    # A fixed 6 dB gain changes absolute sensitivity but not normalized shape.
    capture = reference * 2
    sf.write(traces / "reference.wav", reference, rate, subtype="PCM_16")
    sf.write(traces / "capture.wav", capture, rate, subtype="PCM_16")
    mask = {
        "clause": "test-only wide open mask",
        "frequency_hz": [80, 1000, 2000],
        "lower_db": [-20, -20, -20],
        "upper_db": [20, 20, 20],
        "normalized_at_hz": 1000,
    }
    response = TestClient(app).post("/api/acoustic/measurements/compile", json={
        "item_id": "SWB-wireless:B5",
        "unit": "Min. dist. to tolerance scheme [dB]",
        "kind": "transfer",
        "reference_path": str(traces / "reference.wav"),
        "capture_path": str(traces / "capture.wav"),
        "mask": mask,
    })
    assert response.status_code == 201, response.text
    assert response.json()["measurement"]["value"] >= 19.9


def test_transfer_compiler_rejects_missing_normalization(
    tmp_path, monkeypatch
) -> None:
    _, traces = _compiler_paths(tmp_path, monkeypatch)
    rate = 16000
    reference = np.zeros(rate * 2)
    sf.write(traces / "reference.wav", reference, rate, subtype="PCM_16")
    sf.write(traces / "capture.wav", reference, rate, subtype="PCM_16")
    mask = {
        "frequency_hz": [80, 1000, 2000],
        "lower_db": [-20, -20, -20],
        "upper_db": [20, 20, 20],
    }
    response = TestClient(app).post("/api/acoustic/measurements/compile", json={
        "item_id": "SWB-wireless:B5",
        "unit": "Min. dist. to tolerance scheme [dB]",
        "kind": "transfer",
        "reference_path": str(traces / "reference.wav"),
        "capture_path": str(traces / "capture.wav"),
        "mask": mask,
    })
    assert response.status_code == 422
    assert "normalized_at_hz" in response.text


def test_transfer_compiler_negative_control_rejects_shape_violation(
    tmp_path, monkeypatch
) -> None:
    _, traces = _compiler_paths(tmp_path, monkeypatch)
    rate = 16000
    duration = 8
    rng = np.random.default_rng(23)
    reference = rng.normal(0, .02, rate * duration)
    spectrum = np.fft.rfft(reference)
    frequencies = np.fft.rfftfreq(len(reference), 1 / rate)
    high_band_gain = np.interp(frequencies, [1500, 2500], [1, 8])
    capture = np.fft.irfft(spectrum * high_band_gain, len(reference))
    sf.write(traces / "reference.wav", reference, rate, subtype="PCM_16")
    sf.write(traces / "capture.wav", capture, rate, subtype="PCM_16")
    mask = {
        "clause": "test-only negative control",
        "frequency_hz": [500, 1000, 2000],
        "lower_db": [-2, -2, -2],
        "upper_db": [2, 2, 2],
        "normalized_at_hz": 1000,
    }
    response = TestClient(app).post("/api/acoustic/measurements/compile", json={
        "item_id": "SWB-wireless:B5",
        "unit": "Min. dist. to tolerance scheme [dB]",
        "kind": "transfer",
        "reference_path": str(traces / "reference.wav"),
        "capture_path": str(traces / "capture.wav"),
        "mask": mask,
    })
    assert response.status_code == 201, response.text
    assert response.json()["measurement"]["value"] < -10


def _reconciliation_payload(contract_sha256: str, item_ids: list[str]) -> bytes:
    return json.dumps({
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "standard": "P.1120",
        "standard_edition": "2025-10",
        "standard_sha256":
            "3f28c4bab97084b506ab35187f29542ace88c7f42172134c90026c60d29b5a94",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": "independent-reviewer",
        "mappings": [{
            "item_id": item_id,
            "clause": "11.4",
            "table_or_figure": "Table 11-1",
            "official_limit": "mask",
            "customer_limit": "curve-or-mask",
            "disposition": "review",
            "note": "test fixture",
        } for item_id in item_ids],
    }).encode()


def test_limit_reconciliation_requires_exact_bandwidth_and_contract_identity(
    tmp_path, monkeypatch
) -> None:
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(service, "EVIDENCE", evidence_root)
    contract = json.loads(service.CONTRACT.read_text())
    ids = [
        item["id"] for item in contract["items"]
        if item["bandwidth"] == "SWB"
    ]
    assert len(ids) == 84
    client = TestClient(app)

    def upload(data: bytes) -> dict:
        response = client.post(
            "/api/acoustic/evidence",
            files={"file": ("reconciliation.json", data, "application/json")},
            data={"kind": "limit_reconciliation"},
        )
        assert response.status_code == 201, response.text
        return Artifact.model_validate(response.json()["artifact"])

    valid = upload(_reconciliation_payload(sha256_file(service.CONTRACT), ids))
    assert service._verify_reconciliation(
        upload(_reconciliation_payload(sha256_file(service.CONTRACT), ids)),
        bandwidth="SWB",
        standard=service._standard_for("SWB"),
    ) is not None
    with pytest.raises(service.EvidenceError, match="cover selected bandwidth exactly"):
        service._verify_reconciliation(
            upload(_reconciliation_payload(sha256_file(service.CONTRACT), ids[:-1])),
            bandwidth="SWB",
            standard=service._standard_for("SWB"),
        )
    with pytest.raises(service.EvidenceError, match="different contract"):
        service._verify_reconciliation(
            upload(_reconciliation_payload("0" * 64, ids)),
            bandwidth="SWB",
            standard=service._standard_for("SWB"),
        )
    assert valid.kind == "limit_reconciliation"
