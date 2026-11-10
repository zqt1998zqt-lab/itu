"""Exercise the real VCP binary across supported business recording rates."""
import hashlib

import numpy as np
import pytest
import soundfile as sf

from server.acoustic.acceptance_batch import _run_cli
from server.alignment_runner import resolve_cli


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
def test_real_cli_adapts_nb_and_preserves_declared_latency(tmp_path, rate):
    if resolve_cli() is None:
        pytest.skip("VCP CLI not built")
    t = np.arange(rate // 2) / rate
    source = 0.03 * np.sin(2 * np.pi * 440 * t)
    paths = {}
    for role in ("microphone", "far_end", "noise", "reference"):
        path = tmp_path / f"{role}.wav"
        sf.write(path, source, rate, subtype="PCM_16")
        paths[role] = path
    destination = tmp_path / "case"
    destination.mkdir()
    outputs, execution = _run_cli({"conditions": {"direction": "round_trip", "mode": "NB-wired" if rate == 8000 else "WB-wired"}}, paths, destination, "vcp_unified")
    for role, output in outputs.items():
        data, output_rate = sf.read(output)
        assert output_rate == (16000 if rate == 8000 else rate)
        assert len(data) == output_rate // 2
        assert np.isfinite(data).all()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == execution["output_sha256"][role]
    assert execution["pipeline_reports"]["rx"]["data"]["rx_processing_frames"] > 0
    if rate == 8000:
        assert execution["input_adaptation"]["microphone"]["source_rate"] == 8000
    alignment = execution["output_alignment"]["tx"]
    assert alignment["trimmed_leading_samples"] > 0
    assert (destination / alignment["raw_filename"]).is_file()
    assert sf.info(destination / alignment["raw_filename"]).frames > sf.info(outputs["tx"]).frames


@pytest.mark.parametrize("rate", [32000, 44100])
def test_cli_does_not_silently_downsample_swb(tmp_path, rate):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(rate // 10), rate)
    with pytest.raises(ValueError, match="不受 VCP 支持"):
        _run_cli({"conditions": {"mode": "SWB-wired", "direction": "receive"}},
                 {"far_end": path}, tmp_path, "vcp_unified")
