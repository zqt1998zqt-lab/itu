from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def _wav(rate: int = 8_000, channels: int = 1) -> bytes:
    samples = 800
    time = np.arange(samples, dtype=np.float32) / rate
    mono = 0.08 * np.sin(2 * np.pi * 280 * time)
    values = np.repeat(mono[:, None], channels, axis=1)
    buffer = BytesIO()
    sf.write(buffer, values, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_rate_path_discovery_lists_three_primary_and_legacy_24k() -> None:
    response = client.get("/api/rate-paths")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["primary_sample_rates"] == [8_000, 16_000, 48_000]
    assert payload["legacy_sample_rates"] == [24_000]
    assert [row["sample_rate_hz"] for row in payload["paths"]] == [
        8_000,
        16_000,
        48_000,
        24_000,
    ]
    assert [row["cluster_id"] for row in payload["paths"][:3]] == [
        "speex_dpdfnet2_8khz",
        "localvqe_cofilite_native16",
        "aec3_serial",
    ]
    clusters = payload["clusters_by_sample_rate"]
    assert clusters["8000"] == ["speex_dpdfnet2_8khz"]
    assert "localvqe_cofilite_native16" in clusters["16000"]
    assert "vcp_aec3_dpdfnet2_48k_hr" not in clusters["16000"]
    assert "vcp_aec3_dpdfnet2_48k_hr" in clusters["48000"]
    assert "localvqe_cofilite_native16" not in clusters["48000"]


def test_nb8_endpoint_rejects_non_8k_before_processing() -> None:
    response = client.post(
        "/api/nb8/process",
        files={
            "microphone": ("microphone.wav", _wav(16_000), "audio/wav"),
            "reference": ("reference.wav", _wav(8_000), "audio/wav"),
        },
        data={"tail_ms": "500"},
    )
    assert response.status_code == 422
    assert "8 kHz" in response.text


def test_nb8_endpoint_rejects_stereo_and_invalid_tail() -> None:
    stereo = client.post(
        "/api/nb8/process",
        files={
            "microphone": ("microphone.wav", _wav(channels=2), "audio/wav"),
            "reference": ("reference.wav", _wav(), "audio/wav"),
        },
        data={"tail_ms": "500"},
    )
    assert stereo.status_code == 422
    assert "mono" in stereo.text

    invalid_tail = client.post(
        "/api/nb8/process",
        files={
            "microphone": ("microphone.wav", _wav(), "audio/wav"),
            "reference": ("reference.wav", _wav(), "audio/wav"),
        },
        data={"tail_ms": "9"},
    )
    assert invalid_tail.status_code == 422
    assert "tail_ms" in invalid_tail.text


@pytest.mark.skipif(
    not __import__("server.nb8_route", fromlist=["DEFAULT_SPEEX_CLI"]).DEFAULT_SPEEX_CLI.is_file(),
    reason="native 8 kHz Speex CLI is not built",
)
def test_nb8_endpoint_returns_downloadable_output_and_report() -> None:
    response = client.post(
        "/api/nb8/process",
        files={
            "microphone": ("microphone.wav", _wav(), "audio/wav"),
            "reference": ("reference.wav", _wav(), "audio/wav"),
        },
        data={"tail_ms": "250"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["result"]["cluster_id"] == "speex_dpdfnet2_8khz"
    assert payload["result"]["sample_rate"] == 8_000
    assert payload["result"]["fallback_frames"] == 0
    output = client.get(payload["output_url"])
    assert output.status_code == 200
    assert output.headers["content-type"].startswith("audio/wav")
    report = client.get(payload["report_url"])
    assert report.status_code == 200
    assert report.json()["output"]["sha256"] == payload["result"]["output"]["sha256"]
