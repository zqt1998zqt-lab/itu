"""Integration checks for offline external post-NS PCM replay.

Set ITU_TEST_CLI to the real candidate binary.  The tests deliberately use
the public CLI/WAV contract; LocalVQE cases are skipped when the candidate
does not expose the pinned model assets.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

import pytest


@pytest.fixture()
def replay_tmp() -> Path:
    # Keep every generated WAV and report for postmortem inspection. Project
    # policy forbids deleting test artifacts, including temporary evidence.
    return Path(tempfile.mkdtemp(prefix="itu-external-ns-replay-"))


def _cli() -> Path:
    value = os.environ.get("ITU_TEST_CLI")
    if not value:
        pytest.skip("ITU_TEST_CLI is not set")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.fail(f"ITU_TEST_CLI does not exist: {path}")
    return path


def _run(
    cli: Path, args: list[str], prefix: Path, expect: int = 0
) -> subprocess.CompletedProcess[str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [str(cli), *args, "--out", str(prefix)]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    (Path(str(prefix) + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode != expect:
        raise AssertionError(
            f"unexpected exit {result.returncode} (expected {expect}): "
            f"{' '.join(command)}\n{result.stdout}{result.stderr[-12000:]}"
        )
    return result


def _write_pcm16(path: Path, samples: list[int], rate: int = 16000,
                 channels: int = 1) -> None:
    assert samples
    assert len(samples) % channels == 0
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _read_pcm16(path: Path) -> tuple[int, int, list[int]]:
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        channels = stream.getnchannels()
        payload = stream.readframes(stream.getnframes())
    return rate, channels, list(struct.unpack(f"<{len(payload)//2}h", payload))


def _write_float32(path: Path, samples: list[float], rate: int = 16000) -> None:
    assert samples
    payload = b"".join(struct.pack("<f", float(v)) for v in samples)
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _read_float32(path: Path) -> tuple[int, int, list[float]]:
    blob = path.read_bytes()
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"
    end = 8 + struct.unpack_from("<I", blob, 4)[0]
    fmt = None
    data = None
    offset = 12
    while offset + 8 <= end:
        tag = blob[offset:offset + 4]
        size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        payload = blob[start:start + size]
        if tag == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", payload)
        elif tag == b"data":
            data = payload
        offset = start + size + (size & 1)
    assert fmt is not None and data is not None
    tag, channels, rate, _byte_rate, _block, bits = fmt
    assert tag == 3 and channels == 1 and bits == 32 and len(data) % 4 == 0
    return rate, channels, list(struct.unpack(f"<{len(data)//4}f", data))


def _tone(
    count: int,
    frequency: float = 440.0,
    amplitude: float = 0.18,
    rate: int = 16000,
) -> list[int]:
    scale = 32768.0
    return [
        int(round(scale * amplitude * math.sin(2.0 * math.pi * frequency * i / rate)))
        for i in range(count)
    ]


def _post_pcm_for_delay(
    samples: list[int], delay: int, frame_size: int = 160
) -> list[float]:
    values = [v / 32768.0 for v in samples]
    assert delay % frame_size == 0
    return [0.0] * delay + values


def _report(prefix: Path) -> dict:
    return json.loads(Path(str(prefix) + ".json").read_text())


def _stage_files(prefix: Path) -> dict[str, dict]:
    metadata = json.loads(Path(str(prefix) + ".stages.json").read_text())
    stages = metadata.get("pipeline_stages", metadata.get("stages", []))
    return {stage["id"]: stage for stage in stages if "id" in stage}


@pytest.mark.parametrize("delay", [0, 160])
def test_aec_off_identity_uses_external_stream_at_both_delays(
    replay_tmp: Path, delay: int
) -> None:
    cli = _cli()
    mic = _tone(480)
    mic_path = replay_tmp / f"mic-{delay}.wav"
    post_path = replay_tmp / f"post-{delay}.float.wav"
    prefix = replay_tmp / f"identity-{delay}"
    _write_pcm16(mic_path, mic)
    external = _post_pcm_for_delay(mic, delay)
    _write_float32(post_path, external)
    _run(cli, [
        "--mic", str(mic_path),
        "--no-aec", "--no-agc", "--no-tx-drc", "--no-limiter", "--no-low-cut",
        "--diagnostic-stages",
        "--post-ns-pcm", str(post_path),
        "--post-ns-delay-samples", str(delay),
    ], prefix)
    rate, channels, actual = _read_pcm16(Path(str(prefix) + ".after.wav"))
    assert (rate, channels) == (16000, 1)
    expected = [0] * delay + mic
    assert actual == expected
    report = _report(prefix)
    assert report["ns_backend"] == "external_pcm"
    assert report["ns_external_pcm_enabled"] is True
    assert report["ns_external_delay_samples"] == delay
    assert report["ns_external_applied_frames"] == len(expected) // 160
    assert report["ns_external_fallback_frames"] == 0
    stages = _stage_files(prefix)
    _, _, selected = _read_float32(Path(stages["tx_ns_selected"]["file"]))
    _, _, protected = _read_float32(Path(stages["tx_ns"]["file"]))
    _, _, ns_input = _read_float32(Path(stages["tx_ns_input"]["file"]))
    assert selected == pytest.approx(external, abs=1e-7)
    assert protected == pytest.approx(external, abs=1e-7)
    assert ns_input == pytest.approx([v / 32768.0 for v in mic], abs=1e-7)


def test_external_rejects_short_nonfinite_wrong_rate_and_unsupported_aec(
    replay_tmp: Path,
) -> None:
    cli = _cli()
    mic_path = replay_tmp / "mic.wav"
    _write_pcm16(mic_path, _tone(320))

    short = replay_tmp / "short.float.wav"
    _write_float32(short, [0.0] * 160)
    _run(cli, ["--mic", str(mic_path), "--no-aec", "--no-agc",
               "--no-tx-drc", "--no-limiter", "--no-low-cut",
               "--post-ns-pcm", str(short)], replay_tmp / "short", expect=1)

    bad = replay_tmp / "nan.float.wav"
    _write_float32(bad, [float("nan")] * 320)
    _run(cli, ["--mic", str(mic_path), "--no-aec", "--no-agc",
               "--no-tx-drc", "--no-limiter", "--no-low-cut",
               "--post-ns-pcm", str(bad)], replay_tmp / "nan", expect=1)

    wrong_rate = replay_tmp / "rate.float.wav"
    _write_float32(wrong_rate, [0.0] * 320, rate=8000)
    _run(cli, ["--mic", str(mic_path), "--no-aec", "--no-agc",
               "--no-tx-drc", "--no-limiter", "--no-low-cut",
               "--post-ns-pcm", str(wrong_rate)], replay_tmp / "rate", expect=1)

    ref_path = replay_tmp / "ref.wav"
    _write_pcm16(ref_path, _tone(320, 220.0))
    valid = replay_tmp / "valid.float.wav"
    _write_float32(valid, [0.0] * 320)
    unsupported = _run(
        cli,
        ["--mic", str(mic_path), "--echo", str(ref_path),
         "--aec-backend", "speex", "--post-ns-pcm", str(valid)],
        replay_tmp / "speex",
        expect=1,
    )
    assert "requires AEC-off mono, AEC3, or LocalVQE AEC" in unsupported.stderr

    _run(cli, ["--mic", str(mic_path), "--no-aec",
               "--post-ns-delay-samples", "100"], replay_tmp / "bad-delay", expect=1)
    _run(cli, ["--mic", str(mic_path), "--no-aec",
               "--post-ns-delay-samples", "160"], replay_tmp / "delay-without-pcm", expect=1)


def test_old_default_aec_off_no_ns_is_still_identity(replay_tmp: Path) -> None:
    cli = _cli()
    mic = _tone(320)
    mic_path = replay_tmp / "mic.wav"
    _write_pcm16(mic_path, mic)
    prefix = replay_tmp / "legacy"
    _run(cli, ["--mic", str(mic_path), "--no-aec", "--no-ns",
               "--no-agc", "--no-tx-drc", "--no-limiter", "--no-low-cut"], prefix)
    _, _, actual = _read_pcm16(Path(str(prefix) + ".after.wav"))
    assert actual == mic


def _aec3_48k_common(
    mic_path: Path,
    echo_path: Path | None,
    post_path: Path,
    delay: int = 1920,
) -> list[str]:
    args = ["--mic", str(mic_path)]
    if echo_path is not None:
        args.extend(["--echo", str(echo_path)])
    args.extend([
        "--aec-backend", "webrtc",
        "--no-agc", "--no-tx-drc", "--no-limiter", "--no-low-cut",
        "--post-ns-pcm", str(post_path),
        "--post-ns-delay-samples", str(delay),
    ])
    return args


def _aec3_48k_post(mic: list[int], delay: int = 1920) -> list[float]:
    frame_size = 480
    # Cover the physical post-NS delay, every input callback, and the
    # callback-owned AEC3/NS drain tail.
    return (
        _post_pcm_for_delay(mic, delay, frame_size=frame_size)
        + [0.0] * (12 * frame_size)
    )


def test_aec3_48k_post_ns_replay_accepts_480_frame_callbacks(
    replay_tmp: Path,
) -> None:
    cli = _cli()
    rate = 48000
    frame_size = 480
    mic = _tone(12 * frame_size, rate=rate)
    echo = _tone(12 * frame_size, frequency=220.0, amplitude=0.08, rate=rate)
    mic_path = replay_tmp / "mic-48k.wav"
    echo_path = replay_tmp / "echo-48k.wav"
    post_path = replay_tmp / "post-48k.float.wav"
    prefix = replay_tmp / "aec3-48k-replay"
    _write_pcm16(mic_path, mic, rate=rate)
    _write_pcm16(echo_path, echo, rate=rate)
    _write_float32(post_path, _aec3_48k_post(mic), rate=rate)

    _run(cli, _aec3_48k_common(mic_path, echo_path, post_path), prefix)

    output_rate, output_channels, _ = _read_pcm16(Path(str(prefix) + ".after.wav"))
    report = _report(prefix)
    assert (output_rate, output_channels) == (48000, 1)
    assert report["sample_rate"] == 48000
    assert report["frame_size"] == 480
    assert report["ns_backend"] == "external_pcm"
    assert report["ns_external_pcm_enabled"] is True
    assert report["ns_external_delay_samples"] == 1920
    assert report["ns_external_fallback_frames"] == 0
    expected_callbacks = (
        len(mic) // frame_size
        + math.ceil(report["dsp_delay_samples"] / frame_size)
    )
    assert report["ns_external_applied_frames"] == expected_callbacks
    assert report["aec_backend"] == "webrtc_aec3"
    assert report["aec_backend_requested"] == "webrtc"
    assert report["aec_backend_effective"] == "webrtc_aec3"
    assert report["internal_sample_rate"] == 48000
    assert report["aec_internal_sample_rate"] == 48000
    assert report["ns_internal_sample_rate"] == 48000
    assert report["rate_converter"] == "none"


@pytest.mark.parametrize("post_rate", [16000, 44100])
def test_aec3_48k_rejects_non_48k_post(
    replay_tmp: Path, post_rate: int
) -> None:
    cli = _cli()
    rate = 48000
    frame_size = 480
    mic = _tone(6 * frame_size, rate=rate)
    echo = _tone(6 * frame_size, frequency=220.0, amplitude=0.08, rate=rate)
    mic_path = replay_tmp / f"mic-{post_rate}-post.wav"
    echo_path = replay_tmp / f"echo-{post_rate}-post.wav"
    post_path = replay_tmp / f"post-{post_rate}.float.wav"
    _write_pcm16(mic_path, mic, rate=rate)
    _write_pcm16(echo_path, echo, rate=rate)
    _write_float32(post_path, _aec3_48k_post(mic), rate=post_rate)

    result = _run(
        cli,
        _aec3_48k_common(mic_path, echo_path, post_path),
        replay_tmp / f"reject-post-{post_rate}",
        expect=1,
    )
    assert "sample rate must match" in result.stderr


@pytest.mark.parametrize("delay", [1919, 1921])
def test_aec3_48k_rejects_non_480_aligned_delay(
    replay_tmp: Path, delay: int
) -> None:
    cli = _cli()
    rate = 48000
    frame_size = 480
    mic = _tone(6 * frame_size, rate=rate)
    echo = _tone(6 * frame_size, frequency=220.0, amplitude=0.08, rate=rate)
    mic_path = replay_tmp / f"mic-delay-{delay}.wav"
    echo_path = replay_tmp / f"echo-delay-{delay}.wav"
    post_path = replay_tmp / f"post-delay-{delay}.float.wav"
    _write_pcm16(mic_path, mic, rate=rate)
    _write_pcm16(echo_path, echo, rate=rate)
    _write_float32(post_path, _aec3_48k_post(mic), rate=rate)

    result = _run(
        cli,
        _aec3_48k_common(mic_path, echo_path, post_path, delay=delay),
        replay_tmp / f"reject-delay-{delay}",
        expect=1,
    )
    assert "10 ms frame multiple" in result.stderr


def test_aec3_48k_rejects_non_480_aligned_post_file(replay_tmp: Path) -> None:
    cli = _cli()
    rate = 48000
    frame_size = 480
    mic = _tone(6 * frame_size, rate=rate)
    echo = _tone(6 * frame_size, frequency=220.0, amplitude=0.08, rate=rate)
    mic_path = replay_tmp / "mic-nonaligned-post.wav"
    echo_path = replay_tmp / "echo-nonaligned-post.wav"
    post_path = replay_tmp / "post-nonaligned.float.wav"
    _write_pcm16(mic_path, mic, rate=rate)
    _write_pcm16(echo_path, echo, rate=rate)
    _write_float32(post_path, _aec3_48k_post(mic)[:-1], rate=rate)

    result = _run(
        cli,
        _aec3_48k_common(mic_path, echo_path, post_path),
        replay_tmp / "reject-nonaligned-post",
        expect=1,
    )
    assert "complete 10 ms callbacks" in result.stderr


def test_aec3_48k_rejects_missing_playback_reference(replay_tmp: Path) -> None:
    cli = _cli()
    rate = 48000
    frame_size = 480
    mic = _tone(6 * frame_size, rate=rate)
    mic_path = replay_tmp / "mic-missing-echo.wav"
    post_path = replay_tmp / "post-missing-echo.float.wav"
    _write_pcm16(mic_path, mic, rate=rate)
    _write_float32(post_path, _aec3_48k_post(mic), rate=rate)

    result = _run(
        cli,
        _aec3_48k_common(mic_path, None, post_path),
        replay_tmp / "reject-missing-echo",
        expect=1,
    )
    assert "needs actual playback reference" in result.stderr


def _localvqe_available(cli: Path) -> bool:
    result = subprocess.run([str(cli), "--aec-capabilities"], text=True,
                            capture_output=True, timeout=60, check=False)
    if result.returncode != 0:
        return False
    try:
        caps = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return any(item.get("id") == "localvqe_aec200k" and item.get("available")
               for item in caps.get("backends", []))


def test_localvqe_replays_the_actual_post_ns_timeline(replay_tmp: Path) -> None:
    cli = _cli()
    if not _localvqe_available(cli):
        pytest.skip("LocalVQE assets are unavailable in this candidate")
    # 608 + LocalVQE 512 + Speex NS 160 is a callback-aligned 1280 samples.
    mic = _tone(608)
    ref = _tone(608, 220.0, 0.08)
    mic_path = replay_tmp / "mic.wav"
    ref_path = replay_tmp / "ref.wav"
    _write_pcm16(mic_path, mic)
    _write_pcm16(ref_path, ref)
    common = ["--mic", str(mic_path), "--echo", str(ref_path),
              "--aec-backend", "localvqe_aec200k", "--ns-backend", "speex",
              "--speex-suppression-db", "0", "--no-agc", "--no-tx-drc",
              "--no-limiter", "--no-low-cut"]
    baseline = replay_tmp / "localvqe-baseline"
    _run(cli, [*common, "--diagnostic-stages"], baseline)
    stages = _stage_files(baseline)
    assert "tx_ns_input" in stages
    assert stages["tx_ns_input"]["fixed_delay_samples"] >= 512
    tx_output = Path(stages["tx_output"]["file"])
    _, _, post_ns = _read_float32(tx_output)
    # Pad only the callback-owned drain tail; never pad the input itself.
    # The replay declares one additional 160-sample physical NS delay, so its
    # drain contract includes one extra callback beyond the baseline tail.
    callback_samples = math.ceil(len(post_ns) / 160) * 160 + 160
    post_ns.extend([0.0] * (callback_samples - len(post_ns)))
    post_path = replay_tmp / "localvqe-post.float.wav"
    _write_float32(post_path, post_ns)
    replay = replay_tmp / "localvqe-replay"
    _run(cli, [*common, "--post-ns-pcm", str(post_path),
               "--post-ns-delay-samples", "160"], replay)
    _, _, baseline_audio = _read_pcm16(Path(str(baseline) + ".after.wav"))
    _, _, replay_audio = _read_pcm16(Path(str(replay) + ".after.wav"))
    assert replay_audio == baseline_audio
    report = _report(replay)
    assert report["ns_backend"] == "external_pcm"
    assert report["ns_external_delay_samples"] == 160
    assert report["ns_external_fallback_frames"] == 0


def test_localvqe_bf_exports_real_ns_input_tap(replay_tmp: Path) -> None:
    cli = _cli()
    if not _localvqe_available(cli):
        pytest.skip("LocalVQE assets are unavailable in this candidate")
    mono = _tone(606)
    mic = [v for pair in zip(mono, mono) for v in pair]
    mic_path = replay_tmp / "mic-2ch.wav"
    ref_path = replay_tmp / "ref.wav"
    _write_pcm16(mic_path, mic, channels=2)
    _write_pcm16(ref_path, _tone(606, 220.0, 0.08))
    common = ["--mic", str(mic_path), "--beam-delays", "2,0",
              "--echo", str(ref_path), "--aec-backend", "localvqe_aec200k",
              "--ns-backend", "speex", "--speex-suppression-db", "0",
              "--no-agc", "--no-tx-drc", "--no-limiter", "--no-low-cut"]
    prefix = replay_tmp / "localvqe-bf"
    _run(cli, [*common, "--diagnostic-stages"], prefix)
    stages = _stage_files(prefix)
    assert "tx_ns_input" in stages
    assert stages["tx_ns_input"]["fixed_delay_samples"] == 514
    assert stages["tx_ns_input"]["samples_per_channel"] == 606 + 514
    _, _, post_ns = _read_float32(Path(stages["tx_output"]["file"]))
    post_ns.extend([0.0] * (math.ceil(len(post_ns) / 160) * 160 + 160 - len(post_ns)))
    post_path = replay_tmp / "localvqe-bf-post.float.wav"
    _write_float32(post_path, post_ns)
    replay = replay_tmp / "localvqe-bf-replay"
    _run(cli, [*common, "--post-ns-pcm", str(post_path),
               "--post-ns-delay-samples", "160"], replay)
    _, _, baseline_audio = _read_pcm16(Path(str(prefix) + ".after.wav"))
    _, _, replay_audio = _read_pcm16(Path(str(replay) + ".after.wav"))
    assert replay_audio == baseline_audio
    assert _report(replay)["ns_backend"] == "external_pcm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
