from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.acoustic import acceptance_batch


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/alignment/acceptance/batches", {
        "dataset_id": "8cbd70bebb164b2d9de11ea268096b47", "algorithm_cluster_id": "vcp_unified",
    }),
    ("/api/alignment/acceptance/scene-condition-batches", {
        "dataset_id": "cf29475e343052eaffa73dfebad1b0b0", "algorithm_cluster_id": "vcp_unified",
        "mode": "WB-wired", "scene": "sce1",
    }),
])
def test_valid_large_retired_json_is_http410_without_starting_a_job(monkeypatch, endpoint, payload):
    def forbidden(*args, **kwargs):
        pytest.fail("retired execution reached the worker queue")
    monkeypatch.setattr(acceptance_batch._EXECUTOR, "submit", forbidden)
    content = json.dumps(payload).encode() + b" " * (70 * 1024)
    response = TestClient(app).post(endpoint, content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 410, response.text
    body = response.json()
    assert body["dataset_id"] == payload["dataset_id"]
    assert body["replacement_dataset_id"] != payload["dataset_id"]
    assert "退役" in body["detail"]


def test_unrelated_http_errors_keep_the_original_handler():
    response = TestClient(app).get("/api/endpoint-that-does-not-exist")
    assert response.status_code == 404
