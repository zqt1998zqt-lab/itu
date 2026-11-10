"""Receive acceptance uses input/output evidence, not output spectrum shape."""
from pathlib import Path
import hashlib

import numpy as np
import pytest
import soundfile as sf

from server.acoustic import acceptance_models as models
from server.acoustic import receive_simulation as simulation
from server.acoustic.acceptance_contract import required_conditions


def _calculate(tmp_path, monkeypatch, excitation, *, config=None, ir_gain=None,
               item_id="WB-wired:B7", rate=24000):
    monkeypatch.setattr(simulation, "STORE", tmp_path / "virtual")
    far = tmp_path / "far.wav"
    rx = tmp_path / "rx.wav"
    sf.write(far, excitation, rate, subtype="FLOAT")
    sf.write(rx, excitation, rate, subtype="FLOAT")
    ir_paths = None
    if ir_gain is not None:
        ir_paths = {}
        for role in ("left_ir", "right_ir"):
            ir_paths[role] = tmp_path / f"{role}.wav"
            sf.write(ir_paths[role], np.array([ir_gain, 0., 0.]), rate, subtype="FLOAT")
        config = {"model": "paired_ir", "left_ir": "left_ir.wav",
                  "right_ir": "right_ir.wav", "measurement_point": "unspecified",
                  "provenance": "unit-test FIR, not measured acoustics", "ir_kind": "unspecified"}
    contract = models._contract_lookup()[item_id]
    result = models.calculate_case_items(
        {"id": "transfer-test", "conditions": required_conditions(contract), "item_ids": [item_id]},
        {"far_end": far}, {"rx": rx},
        {"output_sha256": {"rx": hashlib.sha256(rx.read_bytes()).hexdigest()}},
        report_id="transfer-assessment", receive_config=config or {"model": "head", "azimuth_deg": 0},
        receive_ir_paths=ir_paths,
    )[0]
    return result


@pytest.mark.parametrize("colour", ["white", "pink"])
def test_identity_transfer_ignores_excitation_colour(tmp_path, monkeypatch, colour):
    rate = 24000
    x = np.random.default_rng(124).normal(size=rate * 4)
    if colour == "pink":
        f = np.fft.rfftfreq(len(x), 1 / rate)
        x = np.fft.irfft(np.fft.rfft(x) / np.sqrt(np.maximum(f, 1)), n=len(x))
    x *= .02 / np.sqrt(np.mean(x * x))
    row = _calculate(tmp_path, monkeypatch, x)
    assert row["status"] == "review_required"
    assert row["metric"] == "receive_h1_frequency_response_span_db"
    assert row["value"] < .01
    assert row["standard_compliance"] is False
    transfer = row["model_details"]["receive_transfer_response"]
    assert transfer["available"] is True
    assert transfer["reference_sha256"] == row["evidence_binding"]["input_sha256"]["far_end"]
    assert "VCP RX" in transfer["signal_path"]


def test_paired_ir_enters_actual_computation(tmp_path, monkeypatch):
    x = .02 * np.random.default_rng(222).normal(size=24000 * 4)
    row = _calculate(tmp_path, monkeypatch, x, ir_gain=.5)
    assert row["status"] == "review_required"
    assert row["value"] < .01
    transfer = row["model_details"]["receive_transfer_response"]
    assert transfer["available"]
    assert transfer["bands"]["1000"]["binaural_gain_db"] == pytest.approx(-6.0206, abs=.01)
    diagnostics = row["diagnostics"]["synthetic_receive"]
    assert diagnostics["evidence_kind"] == "synthetic_ir"
    assert diagnostics["simulation"]["head_model"] is None
    assert diagnostics["acoustic_configuration"]["model"] == "paired_ir"


def test_single_tone_cannot_fill_missing_frequency_bands(tmp_path, monkeypatch):
    x = .05 * np.sin(2 * np.pi * 1000 * np.arange(24000 * 4) / 24000)
    row = _calculate(tmp_path, monkeypatch, x)
    assert row["status"] == "not_measured"
    assert row["value"] is None
    assert row["model_details"]["receive_transfer_response"]["available"] is False


def test_normalised_output_does_not_claim_original_receive_level(tmp_path, monkeypatch):
    x = .1 * np.sin(2 * np.pi * 1000 * np.arange(24000) / 24000)
    row = _calculate(tmp_path, monkeypatch, x, ir_gain=20, item_id="WB-wired:B3")
    assert row["status"] == "not_measured"
    assert row["value"] is None
    assert "归一化" in row["reason"]


def test_silent_noise_peak_is_unobservable_instead_of_zero():
    mapped = models._receive_family_value(
        "idle_noise_peaks", {"metrics": {"noise": {
            "binaural_digital_rms_dbfs": -120., "peak_prominence_db": 0.,
        }}}, mode="NB-wired",
    )
    assert mapped[1] is None
    assert "下限" in mapped[2]


def test_extreme_ir_sample_rate_is_rejected_before_polyphase_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported rate reached polyphase allocation")
    monkeypatch.setattr(simulation.signal, "resample_poly", forbidden)
    with pytest.raises(simulation.SyntheticReceiveError, match="sample rate"):
        simulation._resample_ir({"samples": np.array([1.]), "sample_rate_hz": 1_000_000_007}, 48000)
