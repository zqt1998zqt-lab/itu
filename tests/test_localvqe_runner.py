"""Contract tests for the independent LocalVQE CPU integration."""
from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from server.app import app
from server.localvqe_runner import MODELS, sha256_file


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _wav(samples: np.ndarray, rate: int = 16000) -> bytes:
    stream = BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def test_pinned_models_are_present_and_hashed(client):
    response = client.get("/api/localvqe/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["library_available"] is True
    assert len(payload["models"]) == len(MODELS) == 4
    for row in payload["models"]:
        assert row["available"] is True
        assert sha256_file((__import__("pathlib").Path(__file__).parents[1] / "build/deps/localvqe-models" / row["file"])) == row["sha256"]


def test_form_model_id_selects_distinct_real_inference(client):
    rate = 16000
    time = np.arange(2048, dtype=np.float32) / rate
    microphone = np.ascontiguousarray(0.08 * np.sin(2 * np.pi * 440 * time))
    playback = np.ascontiguousarray(0.06 * np.sin(2 * np.pi * 330 * time))
    outputs = {}
    for identifier in ("aec200k", "aec2k", "joint", "compact"):
        response = client.post(
            "/api/localvqe/process",
            files={
                "microphone": ("mic.wav", _wav(microphone), "audio/wav"),
                "playback": ("ref.wav", _wav(playback), "audio/wav"),
            },
            data={"model_id": identifier, "threads": "2"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["model"]["id"] == identifier
        assert payload["metrics"]["samples"] == len(microphone)
        assert payload["metrics"]["process_ms"] >= 0
        assert payload["formal_compliance"] is False
        outputs[identifier] = payload["metrics"]["output_sha256"]
        assert client.get(payload["output"]["download_url"]).status_code == 200
    assert len(set(outputs.values())) >= 3


def test_invalid_model_rate_channels_are_rejected(client):
    microphone = np.zeros(2048, dtype=np.float32)
    response = client.post(
        "/api/localvqe/process",
        files={"microphone": ("m.wav", _wav(microphone), "audio/wav"),
               "playback": ("r.wav", _wav(microphone), "audio/wav")},
        data={"model_id": "not-a-model"},
    )
    assert response.status_code == 422
    response = client.post(
        "/api/localvqe/process",
        files={"microphone": ("m.wav", _wav(microphone, 48000), "audio/wav"),
               "playback": ("r.wav", _wav(microphone), "audio/wav")},
        data={"model_id": "aec200k"},
    )
    assert response.status_code == 422 and "16 kHz" in response.json()["detail"]
    response = client.post(
        "/api/localvqe/process",
        files={"microphone": ("m.wav", _wav(np.column_stack([microphone, microphone])), "audio/wav"),
               "playback": ("r.wav", _wav(microphone), "audio/wav")},
        data={"model_id": "aec200k"},
    )
    assert response.status_code == 422 and "单声道" in response.json()["detail"]
