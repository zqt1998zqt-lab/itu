"""Actual WAV/CLI duplex replay, including partial frames and pre-AEC BF taps."""
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import wave


def write_pcm(path, rate, channels, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def read_pcm(path):
    with wave.open(str(path), "rb") as stream:
        shape = (stream.getframerate(), stream.getnchannels(), stream.getnframes())
        payload = stream.readframes(shape[2])
    return shape, struct.unpack(f"<{len(payload)//2}h", payload)


def run(command, log, success=True):
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    log.write_text(result.stdout + result.stderr)
    if (result.returncode == 0) != success:
        raise AssertionError(f"exit={result.returncode}: {command}\n{result.stdout}{result.stderr}")


def check_joint_cli(cli, root):
    """Check the real sample path, bounds, opt-in/rollback and duplex contract."""
    for rate in (16000, 24000, 48000):
        case = root / f"{rate}-2mic"
        base = [cli, "--mic", str(case/"mic.wav"), "--beam-delays", "0,0",
                "--rx-in", str(case/"rx.wav")]
        for cap in (0.0, 0.25, 1.0):
            prefix = case/f"joint-cap-{cap}"
            run([cli, "--profile", "vcp_joint", *base[1:],
                 "--vehicle-ns-dry-max", str(cap), "--out", str(prefix)],
                Path(str(prefix)+".log"))
            report = json.loads(Path(str(prefix)+".json").read_text())
            assert report["vehicle_ns_unified_raw_budget"] is True
            assert report["speex_multirender_context"] is True
            assert report["ns_backend"] == "super_gaussian"
            assert report["tx_drc_config"]["threshold_dbfs"] == -6.0
            assert report["tx_drc_config"]["ratio"] == 2.0
            with Path(str(prefix)+".frames.csv").open() as stream:
                frames = list(csv.DictReader(stream))
            assert frames
            actual = []
            for frame in frames:
                budget = float(frame["vehicle_ns_raw_mix_budget"])
                lo, hi = (float(frame[k]) for k in
                          ("vehicle_ns_raw_mix_min", "vehicle_ns_raw_mix_max"))
                assert all(math.isfinite(v) for v in (budget, lo, hi))
                assert -1e-6 <= lo <= hi+1e-6 <= budget+2e-6 <= cap+3e-6
                actual.append(hi)
                assert int(frame["post_aec_near_state"]) in (0, 1, 2)
            if cap > 0:
                assert max(actual) > 0, "startup never exercised raw budget"
            shape, audio = read_pcm(Path(str(prefix)+".after.wav"))
            input_shape, _ = read_pcm(case/"mic.wav")
            assert shape == (rate, 1, input_shape[2]+report["dsp_delay_samples"])
            assert any(audio), "joint path unexpectedly silent"
            assert read_pcm(Path(str(prefix)+".rx-spk.wav")) == read_pcm(
                Path(str(prefix)+".rx-ref.wav"))
        # Explicitly disabling new controls must preserve legacy audio exactly.
        disabled = case/"joint-disabled"
        run([cli, "--profile", "vcp", *base[1:], "--eq-preamp-db", "-6",
             "--vehicle-ns-legacy-budget", "--no-speex-multirender-context",
             "--out", str(disabled)], case/"joint-disabled.log")
        assert read_pcm(Path(str(disabled)+".after.wav")) == read_pcm(case/"duplex.after.wav")
        print(f"CLI unified raw bounds/duplex/rollback {rate} Hz: PASS")
    first = root/"16000-2mic"
    args = ["--mic", str(first/"mic.wav"), "--beam-delays", "0,0",
            "--rx-in", str(first/"rx.wav")]
    alias = first/"joint-alias"
    run([cli, "--profile", "vcp", "--vcp-joint-control", *args,
         "--out", str(alias)], first/"joint-alias.log")
    assert read_pcm(Path(str(alias)+".after.wav")) == read_pcm(first/"joint-cap-0.25.after.wav")
    for override in ("--no-aec", "--aec-backend"):
        label = "off" if override == "--no-aec" else "webrtc"
        prefix = first/f"joint-aec-{label}"
        options = [override] if label == "off" else [override, "webrtc"]
        run([cli, "--profile", "vcp_joint", *args, *options, "--out", str(prefix)],
            first/f"joint-aec-{label}.log")
        report = json.loads(Path(str(prefix)+".json").read_text())
        assert report["speex_multirender_context"] is False


def check_unified_cli(cli, root):
    first = root / "16000-2mic"
    prefix = first / "unified-ns"
    run([cli, "--profile", "vcp_unified", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--no-aec", "--out", str(prefix)],
        first / "unified-ns.log")
    report = json.loads(Path(str(prefix) + ".json").read_text())
    assert report["ns_backend"] == "unified"
    assert len(report["unified_ns_weights"]) == 4
    assert len(report["unified_ns_target_weights"]) == 4
    assert report["unified_ns_noise_updates"] >= 0
    with Path(str(prefix) + ".frames.csv").open() as stream:
        fields = next(csv.reader(stream))
    required = {"unified_ns_w0", "unified_ns_target_w3", "unified_ns_weight_entropy",
                "unified_ns_probability", "unified_ns_noise_updates"}
    assert required.issubset(fields)
    print("CLI unified NS JSON/CSV diagnostics: PASS")


def main():
    cli = str(Path(sys.argv[1]).resolve())
    root = Path(sys.argv[2]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    # Run a multichannel case first: the old AEC-before-BF executable must
    # disagree with the raw-capture beam, even if it prints plausible metadata.
    for rate in (16000, 24000, 48000):
        for m in (2, 1, 4):
            case = root / f"{rate}-{m}mic"
            case.mkdir(exist_ok=True)
            h, n, refs = rate//100, rate//5+7, 2 if m > 1 else 1
            mic = [round(4000*math.sin(i*.19+ch*.7)) for i in range(n) for ch in range(m)]
            rx = [round(14000*math.sin(i*.23+ch*.9)) for i in range(n) for ch in range(refs)]
            write_pcm(case/"mic.wav", rate, m, mic)
            write_pcm(case/"rx.wav", rate, refs, rx)
            command = [cli, "--profile", "vcp", "--mic", str(case/"mic.wav"),
                       "--rx-in", str(case/"rx.wav"), "--out", str(case/"duplex"),
                       "--eq-preamp-db", "-6", "--diagnostic-stages"]
            if m > 1:
                command += ["--beam-delays", ",".join(["0"]*m)]
            run(command, case/"duplex.log")
            report = json.loads((case/"duplex.json").read_text())
            shape, beam = read_pcm(case/"duplex.bf.wav")
            if m > 1:
                assert shape == (rate, 1, n), f"BF tap must precede AEC: {shape}"
                error = max(abs(beam[i]-sum(mic[i*m:(i+1)*m])/m) for i in range(n))
                assert error <= 1, f"BF tap contains AEC-processed audio: max PCM error {error}"
            speaker_shape, speaker = read_pcm(case/"duplex.rx-spk.wav")
            ref_shape, reference = read_pcm(case/"duplex.rx-ref.wav")
            assert speaker_shape == ref_shape == (rate, refs, n)
            assert speaker == reference, "AEC reference is not the actual playback tap"
            assert speaker != tuple(rx), "active RX processing was bypassed"
            assert report["rx_processing_frames"] == (n+h-1)//h
            assert report["reference_source"] == "post_rx_processing"
            assert report["aec_capture_channels"] == 1
            assert report["beamforming_before_aec"] == (m > 1)
            assert report["aec_backend"] == "speex_mdf"
            assert report["speex_coherence_residual_suppression"] is (refs == 1)
            shape, _ = read_pcm(case/"duplex.after.wav")
            assert shape == (rate, 1, n+report["dsp_delay_samples"])
            stages = json.loads((case/"duplex.stages.json").read_text())["stages"]
            for stage in stages:
                assert stage["sample_rate"] == (48000 if rate == 24000 else rate)
                assert Path(stage["file"]).is_file()
            # RX-only must accept the actual file rate and preserve every channel.
            run([cli, "--profile", "vcp", "--rx-in", str(case/"rx.wav"),
                 "--out", str(case/"rx-only")], case/"rx-only.log")
            assert read_pcm(case/"rx-only.rx-spk.wav") == read_pcm(case/"duplex.rx-spk.wav")
            rows.append({"sample_rate": rate, "microphones": m, "references": refs,
                         "input_samples": n, "rx_frames": report["rx_processing_frames"],
                         "dsp_delay_samples": report["dsp_delay_samples"], "passed": True})
            print(f"CLI duplex {rate} Hz, {m} mic, {refs} refs: PASS")
    check_joint_cli(cli, root)
    check_unified_cli(cli, root)
    first = root/"16000-2mic"
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--rx-in", str(first/"rx.wav"),
         "--repair-stage", "4",
         "--stage4-noise-release", "--no-tx-drc",
         "--out", str(first/"stage4-config")],
        first/"stage4-config.log")
    stage4 = json.loads((first/"stage4-config.json").read_text())
    assert stage4["repair_stage"] == 4
    assert stage4["stage4_noise_release"] is True
    assert stage4["tx_drc_enabled"] is False
    assert stage4["agc_fault"] == "none"
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--echo", str(first/"rx.wav"),
         "--aec-backend", "webrtc", "--out", str(first/"aec3-override")],
        first/"aec3-override.log")
    aec3_override = json.loads((first/"aec3-override.json").read_text())
    assert aec3_override["aec_backend"] == "webrtc_aec3"
    assert aec3_override["speex_coherence_residual_suppression"] is False
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--echo", str(first/"rx.wav"),
         "--no-speex-coherence-residual", "--out", str(first/"speex-fallback")],
        first/"speex-fallback.log")
    speex_fallback = json.loads((first/"speex-fallback.json").read_text())
    assert speex_fallback["aec_backend"] == "speex_mdf"
    assert speex_fallback["speex_coherence_residual_suppression"] is False
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--no-aec", "--out", str(first/"aec-off")],
        first/"aec-off.log")
    aec_off = json.loads((first/"aec-off.json").read_text())
    assert aec_off["aec_enabled"] is False
    assert aec_off["speex_coherence_residual_suppression"] is False
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--no-aec", "--tx-drc",
         "--tx-drc-threshold", "-18", "--tx-drc-ratio", "8",
         "--tx-drc-knee", "4", "--tx-drc-attack-ms", "3",
         "--tx-drc-release-ms", "80", "--tx-drc-makeup-db", "0",
         "--tx-drc-ceiling-dbfs", "-0.5", "--tx-drc-lookahead-ms", "2",
         "--out", str(first/"tx-drc-config")],
        first/"tx-drc-config.log")
    tx_drc_report = json.loads((first/"tx-drc-config.json").read_text())
    tx_drc = tx_drc_report["tx_drc_config"]
    assert tx_drc == {
        "threshold_dbfs": -18.0, "ratio": 8.0, "knee_width_db": 4.0,
        "attack_ms": 3.0, "release_ms": 80.0, "makeup_gain_db": 0.0,
        "ceiling_dbfs": -0.5, "lookahead_ms": 2.0,
    }
    assert tx_drc_report["tx_drc_delay_samples"] == 32
    assert tx_drc_report["dsp_delay_samples"] >= 32
    assert stage4["tx_drc_delay_samples"] == 0
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--no-aec", "--eq-phase", "linear",
         "--eq-preamp-db", "-6", "--eq-band", "peak,1000,-3,1",
         "--out", str(first/"linear-eq")],
        first/"linear-eq.log")
    linear = json.loads((first/"linear-eq.json").read_text())
    assert linear["output_eq_phase"] == "linear"
    assert linear["output_eq_linear_taps"] == 321
    assert linear["output_eq_delay_samples"] == 160
    assert linear["dsp_delay_samples"] >= 160
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--no-aec", "--eq-phase", "linear",
         "--eq-linear-taps", "34", "--eq-band", "peak,1000,-3,1",
         "--out", str(first/"invalid-linear-eq")],
        first/"invalid-linear-eq.log", success=False)
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--no-aec", "--tx-drc-ratio", "21",
         "--out", str(first/"invalid-tx-drc")],
        first/"invalid-tx-drc.log", success=False)
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--rx-in", str(first/"rx.wav"),
         "--agc-fault", "inverted",
         "--out", str(first/"agc-fault")],
        first/"agc-fault.log")
    fault = json.loads((first/"agc-fault.json").read_text())
    assert fault["agc_fault"] == "inverted"
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--rx-in", str(first/"rx.wav"),
         "--agc-fault", "unknown",
         "--out", str(first/"invalid-fault")],
        first/"invalid-fault.log", success=False)
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--repair-stage", "5",
         "--out", str(first/"invalid-stage")],
        first/"invalid-stage.log", success=False)
    run([cli, "--profile", "vcp", "--mic", str(first/"mic.wav"),
         "--beam-delays", "0,0", "--rx-in", str(first/"rx.wav"),
         "--echo", str(first/"rx.wav"), "--out", str(first/"invalid")],
        first/"ambiguous-reference.log", success=False)
    (root/"summary.json").write_text(json.dumps({"cases": rows, "passed": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
