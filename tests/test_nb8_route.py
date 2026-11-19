from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server import dpdfnet8
from server import nb8_route


SPEEX_CLI = nb8_route.DEFAULT_SPEEX_CLI

pytestmark = pytest.mark.skipif(
    not SPEEX_CLI.is_file() or not dpdfnet8.DEFAULT_MODEL_PATH.is_file(),
    reason="native 8 kHz Speex CLI or DPDFNet-2 8 kHz model is missing",
)


def _scratch_dir(prefix: str) -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix=f"itu-nb8-route-{prefix}-"))


def _pair(samples: int = 1201) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(samples, dtype=np.float64)
    far = 0.12 * np.sin(2 * np.pi * 230 * index / 8_000)
    near = 0.035 * np.sin(2 * np.pi * 510 * index / 8_000)
    reference = np.asarray(
        np.clip(np.rint(far * 32768), -32768, 32767).astype(np.int16),
        dtype=np.float32,
    ) / np.float32(32768)
    microphone = np.asarray(
        np.clip(np.rint((0.62 * far + near) * 32768), -32768, 32767).astype(np.int16),
        dtype=np.float32,
    ) / np.float32(32768)
    return microphone, reference


def test_native_8k_route_preserves_timeline_and_has_no_fallback() -> None:
    scratch = _scratch_dir("success")
    microphone, reference = _pair()
    microphone_path = scratch / "microphone.wav"
    reference_path = scratch / "reference.wav"
    sf.write(microphone_path, microphone, 8_000, subtype="PCM_16")
    sf.write(reference_path, reference, 8_000, subtype="PCM_16")

    result = nb8_route.run(
        microphone_path,
        reference_path,
        scratch / "result",
        speex_cli=SPEEX_CLI,
        tail_ms=250,
    )

    assert result["schema"] == nb8_route.SCHEMA
    assert result["cluster_id"] == nb8_route.CLUSTER_ID
    assert result["sample_rate"] == 8_000
    assert result["frame_samples"] == 80
    assert result["input_samples"] == result["output_samples"] == microphone.size
    assert result["no_resample"] is True
    assert result["fallback_frames"] == 0
    assert result["algorithm_delay_samples"] == 320
    assert result["stages"]["speex_mdf"]["backend"] == "speex_mdf_nb8"
    assert result["stages"]["dpdfnet2_8khz"]["model_sha256"] == dpdfnet8.MODEL_SHA256

    output = Path(result["output_path"])
    info = sf.info(output)
    assert info.format == "WAV"
    assert info.samplerate == 8_000
    assert info.channels == 1
    assert info.frames == microphone.size
    values, rate = sf.read(output, dtype="float32", always_2d=True)
    assert rate == 8_000
    assert values.shape == (microphone.size, 1)
    assert np.isfinite(values).all()
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["output"]["sha256"] == result["output"]["sha256"]


@pytest.mark.parametrize(
    ("role", "rate", "channels", "message"),
    [
        ("microphone", 16_000, 1, "8 kHz"),
        ("microphone", 8_000, 2, "mono"),
        ("reference", 16_000, 1, "8 kHz"),
        ("reference", 8_000, 2, "mono"),
    ],
)
def test_route_rejects_non_native_inputs(
    role: str, rate: int, channels: int, message: str
) -> None:
    scratch = _scratch_dir("reject")
    mono, reference = _pair(400)
    values = np.repeat(mono[:, None], channels, axis=1)
    microphone_path = scratch / "microphone.wav"
    reference_path = scratch / "reference.wav"
    sf.write(microphone_path, mono, 8_000, subtype="PCM_16")
    sf.write(reference_path, reference, 8_000, subtype="PCM_16")
    target = microphone_path if role == "microphone" else reference_path
    sf.write(target, values, rate, subtype="PCM_16")
    with pytest.raises(nb8_route.NB8RouteError, match=message):
        nb8_route.run(
            microphone_path,
            reference_path,
            scratch / "result",
            speex_cli=SPEEX_CLI,
        )


def test_route_rejects_length_mismatch_and_invalid_tail() -> None:
    scratch = _scratch_dir("mismatch")
    microphone, reference = _pair(800)
    microphone_path = scratch / "microphone.wav"
    reference_path = scratch / "reference.wav"
    sf.write(microphone_path, microphone, 8_000, subtype="PCM_16")
    sf.write(reference_path, reference[:-80], 8_000, subtype="PCM_16")
    with pytest.raises(nb8_route.NB8RouteError, match="相同样本数"):
        nb8_route.run(
            microphone_path,
            reference_path,
            scratch / "result",
            speex_cli=SPEEX_CLI,
        )
    sf.write(reference_path, reference, 8_000, subtype="PCM_16")
    for tail in (True, 9, 10_001, 12.5):
        with pytest.raises(nb8_route.NB8RouteError, match="tail_ms"):
            nb8_route.run(
                microphone_path,
                reference_path,
                scratch / "result",
                speex_cli=SPEEX_CLI,
                tail_ms=tail,  # type: ignore[arg-type]
            )


def test_receive_only_uses_dpdfnet_and_preserves_native_8k() -> None:
    scratch = _scratch_dir("receive")
    _, reference = _pair(1201)
    reference_path = scratch / "far-end.wav"
    sf.write(reference_path, reference, 8_000, subtype="PCM_16")

    result = nb8_route.run_receive(
        reference_path,
        scratch / "rx",
    )

    assert result["schema"] == nb8_route.RX_SCHEMA
    assert result["cluster_id"] == nb8_route.CLUSTER_ID
    assert result["direction"] == "receive"
    assert result["sample_rate"] == 8_000
    assert result["aec_applicability"] == "not_applicable_receive_only"
    assert result["no_resample"] is True
    assert result["fallback_frames"] == 0
    assert result["algorithm_delay_samples"] == 320
    output = Path(result["output_path"])
    assert output.name == "rx.rx-spk.wav"
    info = sf.info(output)
    assert info.samplerate == 8_000
    assert info.channels == 1
    assert info.frames == reference.size
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["output"]["sha256"] == result["output"]["sha256"]
