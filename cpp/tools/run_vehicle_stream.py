#!/usr/bin/env python3
"""One continuous vehicle stream; labels are used only for offline measurements."""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import run_audio_eval as ev
import check_audio_acceptance as ac
from evaluation_targets import calibrated_beam_target

VEHICLE_TIMELINE = (
    ("cold_start_speech", dict(family="ns", noise="tcar", snr_db=30), 2, 9),
    ("engine_change", dict(family="ns", noise="car_engine_ramp", snr_db=5), 0, 24),
    ("window_wind", dict(family="ns", noise="car_window_wind", snr_db=0), 0, 24),
    ("radio_and_road", dict(family="combined", playback_source="radio", noise="tcar", snr_db=0), 0, 30),
    ("long_radio_path", dict(family="aec", playback_source="radio", echo_delay_ms=180, ser_db=-10), 0, 30),
    ("reference_clock_drift", dict(family="stress", stress="clock_300ppm"), 0, 30),
    ("reference_mismatch", dict(family="stress", stress="wrong_reference"), 0, 30),
    ("path_change", dict(family="stress", stress="path_change"), 0, 30),
    ("radio_restart", dict(family="combined", playback_source="radio", noise="tcar", snr_db=10), 0, 30),
)


def select_timeline(scenes, pair, microphones=1):
    if microphones not in (1, 2, 4):
        raise ValueError("vehicle microphones must be 1, 2 or 4")
    candidates = [scene for scene in scenes if not scene.get("catalog_excluded", False)
                  and scene["pair"] == pair and scene["mics"] == microphones]

    def pick(**conditions):
        matches = [scene for scene in candidates
                   if all(scene.get(key) == value for key, value in conditions.items())]
        if not matches:
            raise ValueError(f"vehicle condition missing: {conditions}")
        if conditions.get("family") == "ns":
            corrected = [scene for scene in matches if scene.get("profile") == "curated-v4"]
            if corrected:
                return corrected[0]
        return matches[0]

    return [(label, pick(**conditions), begin, end)
            for label, conditions, begin, end in VEHICLE_TIMELINE]


def sequence_beam_delays(timeline, microphones):
    if microphones == 1:
        return None
    delays = [scene["beam_delays_samples"] for _, scene, _, _ in timeline]
    if any(len(row) != microphones or any(not math.isfinite(v) or v < 0 or v > 4096 for v in row)
           for row in delays) or any(row != delays[0] for row in delays[1:]):
        raise ValueError("continuous array requires fixed calibrated delays for every microphone")
    return delays[0]


def validate_stream_frames(frames, frame_count, delay_samples, sample_rate=16000):
    hop = sample_rate // 100
    expected = frame_count + (delay_samples + hop - 1) // hop
    if len(frames) != expected:
        raise ValueError("missing or extra stream timing frames")
    for index, frame in enumerate(frames):
        if int(frame["frame"]) != index or not math.isclose(float(frame["time_s"]), index / 100, abs_tol=1e-7):
            raise ValueError("unordered or mistimed stream frames")
    ev.frame_timing(frames, frame_count)


def segment_timing(frames, start_s, end_s, paced):
    begin, end = round(start_s * 100), round(end_s * 100)
    selected = frames[begin:end]
    if len(selected) != end - begin or any(int(row["frame"]) != begin + i for i, row in enumerate(selected)):
        raise ValueError("missing or unordered segment frames")
    timing, over = ev.frame_timing([dict(row, frame=i) for i, row in enumerate(selected)], end - begin)
    timing["process_frames_over_10ms"] = over
    if paced:
        lateness = [float(row["release_lateness_ms"]) for row in selected]
        misses = [int(row["completion_late"]) for row in selected]
        if any(not math.isfinite(value) or value < 0 for value in lateness) or any(value not in (0, 1) for value in misses):
            raise ValueError("invalid paced timing measurements")
        timing.update(completion_deadline_misses=sum(misses), release_lateness_max_ms=max(lateness))
    return selected, timing, over


def stream_verdict(details):
    return dict(all_available_passed=all(scene["passed"] for scene in details),
                required_measurements_complete=all(g["status"] != "unavailable" or g["optional"]
                                                  for scene in details for g in scene["gates"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--radio-dataset", type=Path, help="Optional frozen radio scenes supplement")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--microphones", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--sample-rate", type=int, choices=(16000, 24000, 48000), default=16000)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--simulate-clock", action="store_true", help="Simulated timestamps and asynchronous playback PCM")
    parser.add_argument("--disable-clock-correction", action="store_true", help="Paired negative control")
    parser.add_argument("--profile", choices=("referenced_radio", "vehicle_stream"), default="vehicle_stream")
    parser.add_argument("--cli-option", action="append", default=[], help="Paired diagnostic overrides; never final acceptance")
    args = parser.parse_args()
    dataset, root, cli = args.dataset.resolve(), args.root.resolve(), args.cli.resolve()
    manifest = json.loads((dataset / "inputs-manifest.json").read_text())
    radio_manifest_sha256 = None
    if args.radio_dataset:
        radio_path = args.radio_dataset.resolve()
        radio_manifest = json.loads((radio_path / "inputs-manifest.json").read_text())
        radio_manifest_sha256 = ev.digest(radio_path / "inputs-manifest.json")
        manifest["scenes"] += [dict(scene, source_dataset=str(radio_path))
                               for scene in radio_manifest["scenes"] if scene.get("playback_source") == "radio"]
    if args.disable_clock_correction and not args.simulate_clock:
        raise ValueError("negative clock control requires simulation")
    timeline = select_timeline(manifest["scenes"], args.pair, args.microphones)
    beam_delays = sequence_beam_delays(timeline, args.microphones)
    root.mkdir(parents=True, exist_ok=False)
    signals = {key: [] for key in ("before.wav", "ref.wav", "clean.wav", "clean-channel0.wav")}
    if beam_delays:
        signals["clean-array.wav"] = []
    segments, cursor = [], 0
    for label, scene, begin, end in timeline:
        folder = Path(scene.get("source_dataset", dataset)) / "cases" / scene.get("source_case_id", scene["id"])
        length = round((end - begin) * ev.RATE)
        for name in signals:
            actual = ("clean-mics.wav" if name == "clean-array.wav" else
                      "clean-beam.wav" if name == "clean.wav" and beam_delays else
                      "clean.wav" if name == "clean-channel0.wav" else name)
            if ev.digest(folder / actual) != scene["files"][actual]["sha256"]:
                raise ValueError(f"input changed: {scene['id']} {actual}")
            signal = ev.read(folder / actual)
            array_signal = name in ("before.wav", "clean-array.wav")
            channels = args.microphones if array_signal else 1
            if signal.shape[1] != channels or len(signal) < round(end * ev.RATE):
                raise ValueError("vehicle input channels or duration do not match selection")
            piece = signal[round(begin * ev.RATE):round(end * ev.RATE)]
            signals[name].append(piece if array_signal else piece[:, 0])
        windows = {}
        for name, (lo, hi) in scene["windows"].items():
            lo, hi = max(begin, lo), min(end, hi)
            if hi > lo:
                windows[name] = [cursor / ev.RATE + lo - begin, cursor / ev.RATE + hi - begin]
        segments.append(dict(label=label, source_id=scene["id"], start_s=cursor / ev.RATE,
                             end_s=(cursor + length) / ev.RATE, windows=windows, scene=scene))
        cursor += length
    target_alignment = None
    if beam_delays:
        # Build one continuous offline target so BF history is retained across
        # scene joins. Its origin follows the declared processing-rate latency;
        # neither measured output nor a fitted lag enters this calculation.
        backend = "webrtc"
        for i, option in enumerate(args.cli_option[:-1]):
            if option == "--ns-backend":
                backend = args.cli_option[i + 1]
        core_rate = 48000 if args.sample_rate == 24000 or args.sample_rate == 16000 and backend == "rnnoise" else args.sample_rate
        target, target_alignment = calibrated_beam_target(
            np.concatenate(signals["clean-array.wav"]), beam_delays, args.sample_rate, core_rate=core_rate)
        signals["clean.wav"] = [target]
    common = math.gcd(ev.RATE, args.sample_rate)
    for name, pieces in signals.items():
        source = np.concatenate(pieces)
        pcm = resample_poly(source, args.sample_rate // common, ev.RATE // common, axis=0) if args.sample_rate != ev.RATE else source
        sf.write(root / name, pcm, args.sample_rate, subtype="PCM_16")
    identity = dict(scope="continuous vehicle replay; not Android acoustic certification",
                    engine_resets=0, initial_noise_learning_required=False,
                    source_manifest_sha256=ev.digest(dataset / "inputs-manifest.json"),
                    radio_source_manifest_sha256=radio_manifest_sha256,
                    rules_sha256=ev.digest(ac.RULES), cli_sha256=ev.digest(cli),
                    sample_rate=args.sample_rate, source_sample_rate=ev.RATE,
                    highband_source_verified=False, frames=cursor // 160, seconds=cursor / ev.RATE,
                    microphones=args.microphones, beam_delays_samples=beam_delays,
                    target_scope="continuous fixed-geometry clean array at actual core rate and declared latency" if beam_delays else "channel 0 clean target",
                    target_alignment=target_alignment,
                    target_runner_sha256=ev.digest(Path(__file__).with_name("evaluation_targets.py")),
                    baseline_target_scope="channel 0 clean target; input baseline never uses the beam target",
                    processing_profile=args.profile, diagnostic_overrides=args.cli_option,
                    segments=segments, files={name: ev.digest(root / name) for name in signals})
    (root / "timeline.json").write_text(json.dumps(identity, indent=2) + "\n")
    command = [str(cli), "--profile", args.profile, "--mic", str(root / "before.wav"),
               "--echo", str(root / "ref.wav"), "--out", str(root / "stream"),
               "--microphones", str(args.microphones)]
    if beam_delays:
        command.extend(["--beam-delays", ",".join(f"{value * args.sample_rate / ev.RATE:.9f}" for value in beam_delays)])
    if args.pace:
        command.append("--pace-10ms")
    if args.simulate_clock:
        events = [(round(segment["start_s"]*100),
                   300 if segment["scene"].get("stress") == "clock_300ppm" else 0)
                  for segment in segments]
        event_path = root / "simulated-clock-events.csv"
        event_path.write_text("".join(f"{frame},{ppm}\n" for frame, ppm in events))
        identity["simulated_clock"] = dict(events=events, correction=not args.disable_clock_correction,
                                          events_sha256=ev.digest(event_path))
        (root / "timeline.json").write_text(json.dumps(identity, indent=2)+"\n")
        command.extend(["--clock-events", str(event_path)])
        if args.disable_clock_correction:
            command.append("--disable-clock-correction")
    command.extend(args.cli_option)
    run = subprocess.run(command, capture_output=True, text=True,
                         timeout=identity["seconds"] + 180 if args.pace else 180)
    (root / "invocation.json").write_text(json.dumps(dict(argv=command, cli_sha256=ev.digest(cli),
        exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr), indent=2) + "\n")
    run.check_returncode()
    identity["output_files"]={path.name:ev.digest(path) for path in sorted(root.glob("stream.*")) if path.is_file()}
    engine = json.loads((root / "stream.json").read_text())
    output, actual_rate = sf.read(root / "stream.after.wav", always_2d=True)
    output = output[:, 0]
    delay = engine["dsp_delay_samples"]
    io_samples = cursor * args.sample_rate // ev.RATE
    if actual_rate != args.sample_rate or len(output) != io_samples + delay:
        raise ValueError("stream length or DSP delay mismatch")
    after_io = output[delay:delay + io_samples]
    before_io, before_rate = sf.read(root / "before.wav", always_2d=True)
    if before_rate != args.sample_rate or len(before_io) != io_samples:
        raise ValueError("native suppression input contract mismatch")
    before_io = before_io[:, 0]
    after = after_io
    if args.sample_rate != ev.RATE:
        after = resample_poly(after, ev.RATE // common, args.sample_rate // common)
    if len(after) != cursor:
        raise ValueError("quality adapter length mismatch")
    before, clean = np.concatenate(signals["before.wav"])[:, 0], np.concatenate(signals["clean.wav"])
    baseline_clean = np.concatenate(signals["clean-channel0.wav"])
    linear = conditioned = None
    if engine["linear_aec_available"]:
        linear_delay, input_delay = engine["linear_aec_delay_samples"], engine["aec_input_delay_samples"]
        linear = ev.read(root / "stream.linear-aec.wav")[linear_delay:linear_delay + cursor, 0]
        conditioned, conditioned_rate = sf.read(root / "stream.aec-input.wav", always_2d=True)
        if conditioned_rate != args.sample_rate:
            raise ValueError("AEC input tap rate mismatch")
        conditioned = conditioned[input_delay:input_delay + io_samples, 0]
        if args.sample_rate != ev.RATE:
            conditioned = resample_poly(conditioned, ev.RATE // common, args.sample_rate // common)
    frames = list(csv.DictReader((root / "stream.frames.csv").open()))
    validate_stream_frames(frames, identity["frames"], delay, args.sample_rate)
    if engine["stream_resets"] != 0 or engine["input_samples_per_channel"] != io_samples or engine["microphones"] != args.microphones:
        raise ValueError("engine reset or input length mismatch")
    measured, accepted = [], []
    rules = json.loads(ac.RULES.read_text())
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("frozen engineering thresholds changed")
    for segment in segments:
        row = dict(id=segment["source_id"], family=segment["scene"]["family"], mode="full",
                   segments={}, engine=dict(engine))
        for name, limits in segment["windows"].items():
            b, a, target = (ev.window(signal, limits) for signal in (before, after, clean))
            if name.startswith("near") or name == "double_talk":
                qb, qa = ev.quality(ev.window(baseline_clean, limits), b), ev.quality(target, a)
                row["segments"][name] = dict(before=qb, after=qa,
                    stoi_improvement=None if qa["stoi"] is None or qb["stoi"] is None else qa["stoi"] - qb["stoi"])
            else:
                window_before = ev.window(before_io, limits, args.sample_rate)
                window_after = ev.window(after_io, limits, args.sample_rate)
                row["segments"][name] = ev.attenuation_measurement(
                    window_before, window_after, pcm16=True)
                # Diagnostic provenance for this window; no gate consumes it.
                row.setdefault("applicability", {})[name] = ev.measurement_applicability(
                    window_before, window_after, args.sample_rate)
                if name.startswith("far") and linear is not None and segment["scene"]["noise"] == "none":
                    p, q = ev.power(ev.window(conditioned, limits)), ev.power(ev.window(linear, limits))
                    row["segments"][name]["linear_erle_engineering_db"] = ev.db(p / q) if p > 1e-12 and q > 1e-12 else None
        selected, timing, over = segment_timing(frames, segment["start_s"], segment["end_s"], engine["paced_10ms"])
        row["engine"].update(timing)
        row["engine"]["reference_unreliable_frames"] = sum(int(r["reference_state"]) == 2 for r in selected)
        row["process_frames_over_10ms"] = over
        block = ev.window(after_io, [segment["start_s"], segment["end_s"]], args.sample_rate)
        row["output_levels"] = dict(peak_dbfs=ev.db(float(np.max(np.abs(block))) ** 2),
                                    clipping_percent=float(np.mean(np.abs(block) >= .999) * 100))
        condition = dict(segment["scene"], windows=segment["windows"])
        verdict = ac.check_scene(condition, row, rules)
        verdict["label"] = segment["label"]
        accepted.append(verdict)
        measured.append(dict(label=segment["label"], row=row))
    report = dict(identity=identity, engine=engine, measurements=measured, details=accepted,
                  **stream_verdict(accepted),
                  final_acceptance_eligible=engine["paced_10ms"] and not args.cli_option and args.sample_rate == ev.RATE
                                            and not args.simulate_clock,
                  unavailable_gates=[dict(label=s["label"], segment=g["segment"], metric=g["metric"], reason=g["reason"])
                                     for s in accepted for g in s["gates"] if g["status"] == "unavailable"],
                  failures=[dict(label=s["label"], **g) for s in accepted for g in s["gates"] if g["status"] == "fail"])
    (root / "stream-evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(seconds=identity["seconds"], frames=identity["frames"], engine_resets=0,
                         failures=len(report["failures"]), process_p95_ms=engine["process_p95_ms"],
                         process_max_ms=engine["process_max_ms"],
                         completion_deadline_misses=engine["completion_deadline_misses"]), indent=2))


if __name__ == "__main__":
    main()
