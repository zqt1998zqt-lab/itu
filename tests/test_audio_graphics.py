from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ROOT = Path(__file__).resolve().parents[1]
CHAIN_CASE_ID = "chain-super_gaussian_optimal-pass"
CHAIN_CASE_MANIFEST = ROOT / "public/evaluation/pipeline-cases/manifest.json"


def wav(values, rate=16000):
    stream = io.BytesIO()
    sf.write(stream, values, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def request(before=None, after=None, options=None):
    files = {}
    if before is not None:
        files["before"] = ("before.wav", before, "audio/wav")
    if after is not None:
        files["after"] = ("after.wav", after, "audio/wav")
    return client.post(
        "/api/audio-graphics",
        files=files,
        data={"options": json.dumps(options or {})},
    )


@pytest.fixture
def paired_audio():
    rng = np.random.default_rng(123)
    rate = 16000
    time = np.arange(rate * 2) / rate
    source = (
        0.18 * np.sin(2 * np.pi * 440 * time)
        + 0.08 * np.sin(2 * np.pi * 1730 * time)
        + rng.normal(0, 0.025, len(time))
    )
    delayed = np.r_[np.zeros(640), source[:-640]]
    processed = delayed * 0.55
    return wav(source, rate), wav(processed, rate)


def test_paired_graphics_are_aligned_complete_and_downloadable(paired_audio):
    before, after = paired_audio
    response = request(
        before,
        after,
        {"sample_rate": 16000, "align": True, "max_delay_ms": 1000},
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["schema_version"] == 1
    assert report["standard_compliance"] is False
    assert report["analysis_rate"] == 16000
    assert report["alignment"]["applied"] is True
    assert report["alignment"]["delay_ms"] == pytest.approx(40, abs=0.01)
    assert len(report["source"]) == 2
    assert {figure["id"] for figure in report["figures"]} == {
        "waveform-energy",
        "log-stft-before",
        "log-stft-after",
        "spectral-delta",
        "ltas",
        "erb",
        "high-time",
        "triple",
    }
    assert {metric["id"] for metric in report["metrics"]} >= {
        "rms_before",
        "rms_after",
        "rms_delta",
        "spectral_median",
        "spectral_reduced",
    }
    assert [band["id"] for band in report["bands"]] == [
        "low_20_300",
        "mid_300_3000",
        "high_3000_nyquist",
    ]
    for figure in report["figures"]:
        image = client.get(figure["url"])
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"
        assert image.content.startswith(PNG_SIGNATURE)
        assert len(image.content) > 10_000


def test_single_input_generates_single_file_figures_without_fake_delta(paired_audio):
    _, after = paired_audio
    response = request(None, after)
    assert response.status_code == 200, response.text
    report = response.json()
    assert len(report["source"]) == 1
    assert report["alignment"]["reliable"] is None
    assert report["bands"] == []
    assert {figure["id"] for figure in report["figures"]} == {
        "waveform-energy",
        "log-stft-after",
        "ltas",
        "erb",
        "high-time",
    }
    assert all(not figure["id"].startswith("spectral") for figure in report["figures"])


def test_graphics_reject_invalid_options_missing_audio_and_expired_result(paired_audio):
    before, after = paired_audio
    assert request(before, after, {"sample_rate": 12345}).status_code == 422
    assert request(None, None).status_code == 422
    assert client.get("/api/audio-graphics/missing/figures/log-stft-after.png").status_code == 404


def test_graphics_select_requested_channel(paired_audio):
    before, _ = paired_audio
    left = np.frombuffer(before, dtype=np.uint8)
    assert left.size
    rate = 16000
    time = np.arange(rate) / rate
    stereo = np.column_stack(
        [
            np.zeros_like(time),
            0.1 * np.sin(2 * np.pi * 600 * time),
        ]
    )
    response = request(None, wav(stereo, rate), {"after_channel": 1})
    assert response.status_code == 200, response.text
    source = response.json()["source"][0]
    assert source["original_channels"] == 2
    assert source["selected_channel"] == 2


@pytest.mark.skipif(
    not CHAIN_CASE_MANIFEST.exists(),
    reason="published enhancement-chain case manifest is unavailable",
)
def test_frozen_enhancement_case_uses_fixed_window_and_delay():
    response = client.post(f"/api/audio-graphics/evaluation-cases/{CHAIN_CASE_ID}")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["scope"] == "frozen evaluation-case visualization and engineering diagnostics"
    assert report["evaluation_case"]["asset_id"] == CHAIN_CASE_ID
    assert report["evaluation_case"]["window_seconds"] == [13, 21]
    assert report["evaluation_case"]["delay_samples"] == 165
    assert report["duration_seconds"] == pytest.approx(8)
    assert report["figure_scope"]["time_offset_seconds"] == pytest.approx(13)
    assert report["figure_scope"]["fixed_delay_samples"] == 165
    assert report["alignment"] == {
        "requested": False,
        "applied": True,
        "delay_ms": pytest.approx(165 * 1000 / 16000),
        "correlation": None,
        "reliable": True,
        "note": "按冻结评测清单声明的固定因果延迟裁剪；未执行跨相关或内容匹配。",
        "overlap_seconds": pytest.approx(8),
        "search_ms": None,
    }
    assert len(report["figures"]) == 8
    for figure in report["figures"]:
        image = client.get(figure["url"])
        assert image.status_code == 200
        assert image.content.startswith(PNG_SIGNATURE)


def test_unknown_frozen_enhancement_case_is_rejected():
    response = client.post("/api/audio-graphics/evaluation-cases/not-a-chain-case")
    assert response.status_code == 404


@pytest.mark.skipif(
    not CHAIN_CASE_MANIFEST.exists(),
    reason="published pipeline case manifest is unavailable",
)
def test_module_case_uses_the_same_frozen_graphics_contract():
    response = client.post("/api/audio-graphics/evaluation-cases/ns-webrtc")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["evaluation_case"]["asset_id"] == "ns-webrtc"
    assert report["duration_seconds"] == pytest.approx(8)
    assert report["figure_scope"]["fixed_delay_samples"] == 416
    assert len(report["figures"]) == 8


@pytest.mark.skipif(
    not CHAIN_CASE_MANIFEST.exists(),
    reason="published enhancement-chain case manifest is unavailable",
)
def test_concurrent_pass_fail_case_requests_serialize_figure_generation():
    asset_ids = ["chain-all_serial-pass", "chain-all_serial-fail"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda asset_id: client.post(
                    f"/api/audio-graphics/evaluation-cases/{asset_id}"
                ),
                asset_ids,
            )
        )
    assert [response.status_code for response in responses] == [200, 200]
    for response, asset_id in zip(responses, asset_ids):
        report = response.json()
        assert report["evaluation_case"]["asset_id"] == asset_id
        assert len(report["figures"]) == 8
