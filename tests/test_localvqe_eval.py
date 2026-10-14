"""Independent known-waveform checks for alternative-backend scoring."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
import run_localvqe_ab as ab
import recompute_vehicle_native as vehicle_native


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_native_attenuation_and_declared_delay_negative_control(tmp_path, rate):
    source = tmp_path / "source"
    source.mkdir()
    run = tmp_path / "io" / "candidate"
    run.mkdir(parents=True)
    rng = np.random.default_rng(724)
    t = np.arange(32000) / 16000
    clean = .07 * (np.sin(2*np.pi*711*t) + .4*np.sin(2*np.pi*1873*t))
    x = clean.copy()
    x[16000:] = .02*rng.normal(size=16000)
    clean[16000:] = 0
    sf.write(source / "before.wav", x, 16000, subtype="FLOAT")
    sf.write(source / "clean.wav", clean, 16000, subtype="FLOAT")
    up = resample_poly(x, rate // 16000, 1) if rate != 24000 else resample_poly(x, 3, 2)
    sf.write(run.parent / "before.wav", up, rate, subtype="FLOAT")
    # Exactly delay the speech; noise output is digitally silent. This
    # exercises the conservative quantization bound after rate adaptation.
    delay = rate // 100
    output = np.zeros(len(up)+delay)
    output[delay:delay+rate] = up[:rate]
    prefix = run / "result"
    sf.write(prefix.with_suffix(".after.wav"), output, rate, subtype="PCM_16")
    engine = dict(dsp_delay_samples=delay, process_p95_ms=.1, process_max_ms=.1,
                  process_rtf=.01, linear_aec_available=False)
    prefix.with_suffix(".json").write_text(json.dumps(engine))
    with prefix.with_suffix(".frames.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "process_ms"])
        writer.writerows((i, .1) for i in range(200))
    scene = dict(windows={"near_only": [.1, .8], "noise_only": [1.2, 1.9]}, noise="white")
    _, result = ab.measure(scene, source, prefix, rate)
    # Known native PCM16 bound: half-LSB RMS upper bound for silent output.
    native, _ = sf.read(run.parent / "before.wav")
    noise = native[round(1.2*rate):round(1.9*rate)]
    expected = 10*np.log10(np.mean(noise**2)/(.5/32768)**2)
    assert result["segments"]["noise_only"]["attenuation_lower_bound_db"] == pytest.approx(expected, abs=1e-9)
    assert result["segments"]["near_only"]["after"]["sisdr_db"] > 50
    corrected = vehicle_native.update(dict(segments=dict(noise_only=dict(linear_erle_engineering_db=17))),
        native, output[delay:], rate, dict(noise_only=[1.2, 1.9]))
    assert corrected["segments"]["noise_only"]["attenuation_lower_bound_db"] == pytest.approx(expected, abs=1e-9)
    assert corrected["segments"]["noise_only"]["linear_erle_engineering_db"] == 17
    clipped = np.zeros(rate)
    clipped[round(.2*rate)] = 1
    assert vehicle_native.levels(clipped)["clipping_percent"] == pytest.approx(100/rate)
    # Same total length, wrong delay: it must damage SI-SDR, not be fitted away.
    engine["dsp_delay_samples"] = delay + 1
    prefix.with_suffix(".json").write_text(json.dumps(engine))
    sf.write(prefix.with_suffix(".after.wav"), np.append(output, 0), rate, subtype="PCM_16")
    _, wrong = ab.measure(scene, source, prefix, rate)
    assert wrong["segments"]["near_only"]["after"]["sisdr_db"] < 35


def test_negative_control_requires_actual_suppression_failure():
    row = dict(negative_control=True, acceptance=dict(gates=[dict(
        status="pass", optional=False, metric="total_attenuation_db")]))
    with pytest.raises(ValueError, match="negative control"):
        ab.negative_control_summary([row])
    row["acceptance"]["gates"][0]["status"] = "fail"
    assert ab.negative_control_summary([row])["verified"] is True
    assert ab.negative_control_summary([])["verified"] is False


def test_recovered_nonfinite_model_hop_cannot_pass_acceptance():
    rules = json.loads(ab.ac.RULES.read_text())
    scene = dict(id="integrity", family="ns", stress="none", windows={})
    measured = dict(mode="full", segments={}, engine=dict(
        process_p95_ms=.1, process_max_ms=.2, process_rtf=.01,
        model_nonfinite_output_hops=0), process_frames_over_10ms=0,
        output_levels=dict(peak_dbfs=-2, clipping_percent=0))
    assert ab.ac.check_scene(scene, measured, rules)["passed"]
    measured["engine"]["model_nonfinite_output_hops"] = 1
    result = ab.ac.check_scene(scene, measured, rules)
    assert not result["passed"]
    failed = [g for g in result["gates"] if g["status"] == "fail"]
    assert [(g["segment"], g["metric"], g["threshold"]) for g in failed] == [
        ("execution", "model_nonfinite_output_hops", 0)]
