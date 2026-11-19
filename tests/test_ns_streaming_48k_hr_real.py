"""End-to-end real model + native CLI callback test for 48 kHz HR."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from server import ns_routes


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build/itu-48k-online-20261119-v1/cpp/itu_audio_cli"
MODEL = ROOT / "build/ns-dpdfnet2-48k-hr-20261118/assets/dpdfnet2_48khz_hr.onnx"
RATE = 48_000
FRAME = 480


def _config() -> dict:
    return {
        "profile": "vcp",
        "selection": {
            "bf": "das",
            "aec": "webrtc",
            "ns": "dpdfnet2_48k_hr",
            "eq": "three_band",
            "agc": "guarded5",
            "drc": "default",
            "limiter": "on",
        },
    }


@pytest.mark.skipif(
    not CLI.is_file() or not MODEL.is_file(),
    reason="48 kHz online CLI or pinned HR model is unavailable",
)
def test_real_48k_online_cli_and_worker_report(tmp_path):
    count = 12
    time = np.arange(count * FRAME, dtype=np.float32) / np.float32(RATE)
    signal = (
        0.08 * np.sin(2 * np.pi * 440 * time)
        + 0.01 * np.cos(2 * np.pi * 231 * time)
    ).astype(np.float32)
    mic = tmp_path / "mic.wav"
    echo = tmp_path / "echo.wav"
    sf.write(mic, signal, RATE, subtype="PCM_16")
    sf.write(echo, signal, RATE, subtype="PCM_16")
    prefix = tmp_path / "online"
    command = [
        str(CLI),
        "--profile", "vcp",
        "--aec-backend", "webrtc",
        "--ns-backend", "webrtc",
        "--mic", str(mic),
        "--echo", str(echo),
        "--no-bf",
        "--no-low-cut",
        "--no-tx-drc",
        "--no-limiter",
        "--agc-fixed-gain", "0",
        "--diagnostic-stages",
        "--out", str(prefix),
    ]
    completed = ns_routes.execute_ns_command(
        command,
        configuration=_config(),
        timeout=90,
        capture_output=True,
        text=True,
    )
    assert isinstance(completed, subprocess.CompletedProcess)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    receipt = json.loads(Path(str(prefix) + ".ns-stream.json").read_text(encoding="utf-8"))

    assert report["ns_backend"] == "dpdfnet2_48k_hr"
    assert report["ns_execution_mode"] == "online_callback"
    assert report["ns_stateful"] is True
    assert report["ns_precomputed_replay"] is False
    assert report["ns_external_stream_enabled"] is True
    assert report["ns_external_pcm_enabled"] is False
    assert report["ns_external_fallback_frames"] == 0
    assert report["ns_external_delay_samples"] == 2_400
    assert report["ns_stream_chunk_samples"] == FRAME
    assert report["ns_stream_frame_samples"] == FRAME
    assert report["ns_stream_callback_frames"] == receipt["callback_frames"]
    assert report["ns_stream_callback_frames"] > count
    assert receipt["model_inferences"] == receipt["callback_frames"] - 1
    assert receipt["live_model_inferences"] == receipt["callback_frames"] - 1
    assert receipt["fallback_frames"] == 0
    assert receipt["stateful"] is True
    assert receipt["precomputed_replay"] is False
    assert report["aec_backend_effective"] == "webrtc_aec3"
    assert report["sample_rate"] == RATE
    assert report["frame_size"] == FRAME
