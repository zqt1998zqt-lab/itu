"""Batch downloads must return the exact VCP/assessment evidence, not arbitrary paths."""
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from server.acoustic import acceptance_api, service
from server.app import app


@pytest.fixture
def artifact_case(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "BUILD", tmp_path)
    batch_id, case_id = "a" * 32, "case-0001"
    directory = tmp_path / "acceptance-batches" / batch_id / "cases" / case_id
    directory.mkdir(parents=True)
    output = directory / "tx.after.wav"
    output.write_bytes(b"RIFF-exact-vcp-output")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    execution = {"output_sha256": {"tx": digest}}
    assessment = directory / "assessment.json"
    assessment.write_text(json.dumps({"execution": execution, "items": []}))
    snapshot = {"cases": [{"case_id": case_id, "status": "completed", "execution": execution,
                            "assessment_sha256": hashlib.sha256(assessment.read_bytes()).hexdigest()}]}
    monkeypatch.setattr(acceptance_api, "get_batch", lambda identifier: snapshot if identifier == batch_id else None)
    return TestClient(app), f"/api/alignment/acceptance/batches/{batch_id}/cases/{case_id}", output, assessment


def test_downloads_exact_outputs_and_assessment(artifact_case):
    client, url, output, assessment = artifact_case
    response = client.get(f"{url}/outputs/tx")
    assert response.status_code == 200
    assert response.content == output.read_bytes()
    assert response.headers["content-type"] == "audio/wav"
    assert client.get(f"{url}/assessment").content == assessment.read_bytes()
    assert client.get(f"{url}/outputs/rx").status_code == 404
    assert client.get(f"{url}/outputs/unknown").status_code == 404


def test_download_rejects_changed_vcp_output(artifact_case):
    client, url, output, _ = artifact_case
    output.write_bytes(b"changed-vcp-output")
    assert client.get(f"{url}/outputs/tx").status_code == 409


def test_download_rejects_changed_assessment(artifact_case):
    client, url, _, assessment = artifact_case
    assessment.write_text("{}")
    assert client.get(f"{url}/assessment").status_code == 409
    assert client.get(f"{url}/outputs/tx").status_code == 409
