#!/usr/bin/env python3
"""Reproducible real-source mixtures evaluated through the actual C++ CLI."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import rir_generator as rir
import soundfile as sf
from pystoi import stoi
from scipy import signal

RATE = 16000
SEED = 20261008
EPS = 1e-20
PROCESSING_PROFILES = ("general", "referenced_radio", "vehicle_stream")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power(x):
    return float(np.mean(np.square(x)))


def db(p):
    return float(10 * np.log10(max(p, EPS)))


def read(path):
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != RATE:
        raise ValueError(f"source must be 16 kHz: {path}")
    return data


def write(path, data, subtype="PCM_16"):
    if not np.isfinite(data).all() or np.max(np.abs(data)) >= 1:
        raise ValueError(f"nonfinite or clipped generated PCM: {path}")
    sf.write(path, data, RATE, subtype=subtype)


def level(data, target_dbfs, active=None):
    p = power(data if active is None else data[active])
    if p < 1e-15:
        raise ValueError("source unexpectedly silent")
    return data * np.sqrt(10 ** (target_dbfs / 10) / p)


def shift(data, samples):
    # Matches the integration layer's linear fractional delay, with zero padding.
    return np.interp(np.arange(len(data)) - samples, np.arange(len(data)), data, left=0, right=0)


def mask_windows(seconds, windows):
    mask = np.zeros(round(seconds * RATE), dtype=bool)
    for start, end in windows:
        mask[round(start * RATE):round(end * RATE)] = True
    return mask


def schedule(source, seconds, windows):
    data = np.zeros(round(seconds * RATE))
    cursor = 0
    for start, end in windows:
        count = round((end - start) * RATE)
        indices = np.arange(cursor, cursor + count) % len(source)
        piece = source[indices].copy()
        fade = min(320, len(piece) // 10)
        piece[:fade] *= np.linspace(0, 1, fade)
        piece[-fade:] *= np.linspace(1, 0, fade)
        data[round(start * RATE):round(end * RATE)] = piece
        cursor += count
    return data


def sources(root):
    manifest = json.loads((root / "sources/manifest.json").read_text())
    speakers = defaultdict(list)
    noises = {}
    for entry in manifest["sources"]:
        path = root / "sources" / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        if entry["kind"] == "clean_speech":
            speakers[entry["speaker"]].append(read(path)[:, 0])
        elif entry["kind"] == "noise":
            key = entry["scene"].lower()
            if key in noises:
                raise ValueError(f"duplicate noise source key: {key}")
            noises[key] = read(path)[:, 0]
        elif entry["kind"] != "speech_candidate":
            raise ValueError(f"unknown source kind: {entry['kind']}")
    return {key: np.concatenate(value) for key, value in speakers.items()}, noises


def matrix(speaker_ids):
    cases = []
    for pair in range(3):
        near, far = speaker_ids[2 * pair:2 * pair + 2]
        base = {"pair": pair, "near_speaker": near, "far_speaker": far,
                "near_dbfs": -26, "mics": 1, "noise": "none", "snr_db": None,
                "ser_db": 0, "rt60_s": 0.15, "echo_delay_ms": 80, "stress": "none"}
        for noise in ("white", "ooffice", "dwashing", "straffic"):
            for snr in (-5, 5, 15):
                cases.append(dict(base, family="ns", noise=noise, snr_db=snr))
        for rt60 in (0.15, 0.45):
            for delay in (20, 80, 180):
                for ser in (-10, 0, 10):
                    cases.append(dict(base, family="aec", rt60_s=rt60, echo_delay_ms=delay, ser_db=ser))
        for noise in ("ooffice", "straffic"):
            for snr in (0, 10):
                cases.append(dict(base, family="combined", noise=noise, snr_db=snr, ser_db=-5, rt60_s=0.35))
        for stress in ("path_change", "clock_300ppm", "wrong_reference", "nonlinear_speaker"):
            cases.append(dict(base, family="stress", stress=stress, noise="ooffice", snr_db=15,
                              rt60_s=0.3, ser_db=-5))
        for near_dbfs in (-45, -30, -15):
            cases.append(dict(base, family="agc", near_dbfs=near_dbfs, noise="ooffice", snr_db=20))
        for mics in (2, 4):
            cases.append(dict(base, family="bf", mics=mics, noise="straffic", snr_db=0,
                              ser_db=0, rt60_s=0.25))
    for index, case in enumerate(cases):
        case["id"] = f"{index:03d}-{case['family']}-p{case['pair']}"
        case["seed"] = SEED + index
    return cases


def noise_background(case, n, speakers, noises, rng):
    kind = case["noise"]
    if kind == "car_radio":
        raise ValueError("radio belongs in playback_source='radio' with an AEC reference, not background noise")
    if kind in noises:
        original = noises[kind]
        if len(original) < n:
            raise ValueError(f"noise source shorter than scene: {kind}")
        start = int(rng.integers(0, len(original) - n + (1 if case.get("profile") else 0)))
        return original[start:start + n].copy(), start, []
    if kind.startswith("car_"):
        original = noises["tcar"]
        start = int(rng.integers(0, len(original) - n + 1))
        road = level(original[start:start + n].copy(), -26)
        t = np.arange(n) / RATE
        contributors = []
        if kind == "car_hvac":
            source = noises["hvac"]
            offset = int(rng.integers(0, len(source) - n + 1))
            interference = source[offset:offset + n]
        elif kind == "car_window_wind":
            interference = signal.sosfilt(signal.butter(2, 250, fs=RATE, output="sos"), rng.normal(size=n))
            interference *= .3 + .7 * np.sin(2 * np.pi * .4 * t) ** 2
        elif kind == "car_rain":
            drops = rng.exponential(size=n) * (rng.random(n) < .02)
            interference = .2 * rng.normal(size=n) + signal.lfilter([1, .6, .3], [1], drops - np.mean(drops))
        elif kind == "car_horn":
            gate = ((t % 4) > 1) & ((t % 4) < 1.7)
            interference = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 550 * t)) * gate
        elif kind == "car_indicator":
            elapsed = t % .5
            interference = np.exp(-elapsed * 120) * np.sin(2 * np.pi * 1800 * t)
        elif kind == "car_passenger":
            available = sorted(set(speakers) - {case["near_speaker"], case["far_speaker"]})
            contributors = [str(rng.choice(available))]
            source = speakers[contributors[0]]
            offset = int(rng.integers(0, len(source)))
            interference = source[(np.arange(n) + offset) % len(source)]
        elif kind == "car_engine_ramp":
            phase = 2 * np.pi * (40 * t + 25 * t ** 2 / (n / RATE))
            interference = sum(np.sin(phase * harmonic) / harmonic for harmonic in (1, 2, 3, 4))
        else:
            raise ValueError(f"unknown vehicle mixture: {kind}")
        return road + level(interference, -26), start, contributors
    if kind == "white":
        return rng.normal(size=n), None, []
    if kind in ("pink", "brown"):
        frequencies = np.fft.rfftfreq(n, 1 / RATE)
        spectrum = np.fft.rfft(rng.normal(size=n))
        spectrum[0] = 0
        spectrum[1:] /= np.maximum(frequencies[1:], 5) ** (.5 if kind == "pink" else 1)
        return np.fft.irfft(spectrum, n=n), None, []
    if kind == "hum":
        t = np.arange(n) / RATE
        return (sum(np.sin(2 * np.pi * 50 * harmonic * t + rng.uniform(0, 2 * np.pi)) / harmonic
                    for harmonic in (1, 2, 3, 4)) + .1 * rng.normal(size=n)), None, []
    if kind == "burst":
        envelope = .05 + .95 * ((np.arange(n) / RATE % 2) < .5)
        return rng.normal(size=n) * envelope, None, []
    if kind == "babble":
        available = sorted(set(speakers) - {case["near_speaker"], case["far_speaker"]})
        if len(available) < 4:
            raise ValueError("babble needs four independent interfering speakers")
        contributors = list(rng.choice(available, 4, replace=False))
        signals = []
        for speaker in contributors:
            original = speakers[speaker]
            offset = int(rng.integers(len(original)))
            signals.append(level(original[(np.arange(n) + offset) % len(original)], -26))
        return sum(signals), None, contributors
    raise ValueError(f"unknown noise source: {kind}")


def render_source(case, speakers, seconds, windows):
    kind = case.get("playback_source", "far_speech")
    if kind == "none":
        return np.zeros(round(seconds * RATE))
    source = speakers[case["far_speaker"]]
    if kind == "radio":
        source = signal.sosfilt(signal.butter(3, [300, 3400], btype="bandpass", fs=RATE, output="sos"), source)
    elif kind != "far_speech":
        raise ValueError(f"unknown playback source: {kind}")
    return schedule(source, seconds, windows)


def make_case(case, speakers, noises, destination, rir_cache):
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(case["seed"])
    acoustic = case["family"] not in ("ns", "agc")
    if not acoustic and case.get("playback_source") == "radio":
        raise ValueError("radio playback requires an AEC-capable scene")
    seconds = 30 if acoustic else 24
    near_windows = case.get("near_windows", [(11, 24)] if acoustic else [(1, 10), (12, 22)])
    far_windows = [(1, 10), (16, 27)] if acoustic else []
    near_mask = mask_windows(seconds, near_windows)
    far_mask = mask_windows(seconds, far_windows)
    near = level(schedule(speakers[case["near_speaker"]], seconds, near_windows), case["near_dbfs"], near_mask)
    far = render_source(case, speakers, seconds, far_windows)
    if acoustic:
        far = level(far, -23, far_mask)
    n = len(near)
    render_channels=case.get("reference_channels",1)
    if render_channels>1 and (not acoustic or case["stress"]!="none"):
        raise ValueError("multi-render generator requires acoustic stress=none")
    if render_channels>1:
        extra=level(schedule(speakers[case["second_far_speaker"]],seconds,far_windows),-23,far_mask)
        far=np.column_stack((far,extra))
    count = case["mics"]
    microphones = np.array([[2 + (ch - (count - 1) / 2) * 0.035, 2, 1.2] for ch in range(count)])
    near_position = np.array([3.3, 3.2, 1.5])
    room=case.get("room_dimensions_m",[5,4,2.8])
    if case.get("microphone_positions_m"):
        microphones=np.asarray(case["microphone_positions_m"],float)
    near_position=np.asarray(case.get("near_position_m",near_position),float)
    render_positions=case.get("render_positions_m",[[1,3,1.1],[4,1,1.4]])
    distance = np.linalg.norm(microphones - near_position, axis=1)
    microphones = microphones[np.argsort(distance)]
    distance = np.sort(distance)
    arrivals = (distance - distance.min()) * RATE / 343
    compensation = arrivals.max() - arrivals
    clean_channels = np.column_stack([shift(near, delay) for delay in arrivals])
    if case.get("near_rt60_s"):
        hn=rir.generate(c=343,fs=RATE,r=microphones,s=near_position,L=room,
            reverberation_time=case["near_rt60_s"],nsample=round(case["near_rt60_s"]*RATE),hp_filter=False)
        # Absolute acoustic propagation is a source property, not algorithm delay.
        clean_channels=np.column_stack([signal.fftconvolve(near,hn[:,ch])[:n] for ch in range(count)])
        clean_channels *= np.sqrt(power(near[near_mask])/max(power(clean_channels[near_mask,0]),EPS))
        np.save(destination/"near-rir.npy",hn)
    echo = np.zeros((n, count))
    h = None
    if acoustic:
        key = (count, case["rt60_s"],tuple(np.ravel(microphones)),tuple(room),tuple(render_positions[0]))
        if key not in rir_cache:
            rir_cache[key] = rir.generate(c=343, fs=RATE, r=microphones, s=render_positions[0],
                L=room, reverberation_time=case["rt60_s"],
                nsample=round(case["rt60_s"] * RATE), hp_filter=False)
        h = rir_cache[key]
        playback = far.copy()
        if case["stress"] == "clock_300ppm":
            playback = np.interp(np.arange(n) * (1 + 300e-6), np.arange(n), playback, left=0, right=0)
        if case["stress"] == "nonlinear_speaker":
            playback = np.tanh(playback * 6) / 6
        delay = round(case["echo_delay_ms"] * RATE / 1000)
        first_playback=playback if render_channels==1 else playback[:,0]
        echo = np.column_stack([shift(signal.fftconvolve(first_playback, h[:, ch])[:n], delay) for ch in range(count)])
        if render_channels>1:
            h_second=rir.generate(c=343,fs=RATE,r=microphones,s=render_positions[1],L=room,
                reverberation_time=case["rt60_s"],nsample=len(h),hp_filter=False)
            echo += np.column_stack([shift(signal.fftconvolve(playback[:,1],h_second[:,ch])[:n],delay) for ch in range(count)])
            np.save(destination/"echo-rir-second.npy",h_second)
        if case["stress"] == "path_change":
            h2 = rir.generate(c=343, fs=RATE, r=microphones, s=[4, 1, 1.4],
                L=[5, 4, 2.8], reverberation_time=case["rt60_s"], nsample=len(h), hp_filter=False)
            other = np.column_stack([shift(signal.fftconvolve(playback, h2[:, ch])[:n], delay + 400) for ch in range(count)])
            echo[16 * RATE:] = other[16 * RATE:]
            np.save(destination / "echo-path-after-change.npy", h2)
        echo *= np.sqrt(power(near[near_mask]) / (10 ** (case["ser_db"] / 10) * power(echo[far_mask, 0])))
        np.save(destination / "echo-rir.npy", h)
    noise = np.zeros((n, count))
    noise_start = None
    noise_speakers = []
    spatial_audit = None
    if case["noise"] != "none":
        background, noise_start, noise_speakers = noise_background(case, n, speakers, noises, rng)
        background = level(background, case["near_dbfs"] - case["snr_db"], near_mask)
        if count>1 and case.get("spatial_noise_model"):
            from spatial_noise import field, audit
            model=case["spatial_noise_model"]
            noise=field(background,microphones,RATE,rng,model)
            # Normalize channel 0 to the requested SNR without changing coherence.
            noise *= np.sqrt(power(background[near_mask])/max(power(noise[near_mask,0]),EPS))
            spatial_audit=audit(noise,microphones,RATE,model)
        elif count == 1:
            noise[:, 0] = background
        else:
            # Directional real background plus independent sensor noise (10 dB lower).
            noise_position = np.array([1.0, 1.5, 1.2])
            d = np.linalg.norm(microphones - noise_position, axis=1)
            noise_arrivals = (d - d.min()) * RATE / 343
            for ch in range(count):
                noise[:, ch] = shift(background, noise_arrivals[ch]) + level(rng.normal(size=n), case["near_dbfs"] - case["snr_db"] - 10)
        for event in case.get("noise_steps",[]):
            noise[round(event["time_s"]*RATE):] *= 10**(event["gain_db"]/20)
    before = clean_channels + echo + noise
    reference = far.copy()
    if case["stress"] == "wrong_reference":
        reference = level(schedule(speakers[case["near_speaker"]], seconds, far_windows), -23, far_mask)
    peak = max(np.max(np.abs(before)), np.max(np.abs(far)), np.max(np.abs(clean_channels)))
    if case.get("profile"):
        peak = max(peak, np.max(np.abs(reference)), np.max(np.abs(echo)), np.max(np.abs(noise)))
    common_gain = min(1., 0.90 / max(peak, EPS))
    clean_channels *= common_gain
    far *= common_gain
    echo *= common_gain
    noise *= common_gain
    before *= common_gain
    reference *= common_gain
    for name, values in {"clean.wav": clean_channels[:, 0], "ref.wav": reference,
                         "playback.wav": far, "before.wav": before,
                         "echo-component.wav": echo, "noise-component.wav": noise,
                         "clean-mics.wav": clean_channels}.items():
        write(destination / name, values,case.get("pcm_subtype","PCM_16"))
    # Evaluate from quantized targets and input actually passed to the C++ reader.
    clean_channels = read(destination / "clean-mics.wav")
    beam_delay = int(np.ceil(compensation.max()))
    beam_target = np.mean([shift(np.pad(clean_channels[:, ch], (0, beam_delay)), compensation[ch])
                           for ch in range(count)], axis=0)[beam_delay:beam_delay + n]
    write(destination / "clean-beam.wav", beam_target,case.get("pcm_subtype","PCM_16"))
    windows = ({"near_only": [12, 15], "double_talk": [17, 23], "far_only": [6, 9],
                "far_startup": [2, 4], "far_recovery": [25.5, 26.5], "noise_only": [28, 30]}
               if acoustic else {"near_only": [2, 9], "near_late": [13, 21], "noise_only": [22, 24]})
    metadata = dict(case, seconds=seconds, sample_rate=RATE, common_gain=common_gain,
                    effective_near_dbfs=case["near_dbfs"] + db(common_gain ** 2),
                    noise_source_start_sample=noise_start, near_schedule=near_windows,
                    far_schedule=far_windows, windows=windows, microphone_positions_m=microphones.tolist(),
                    near_position_m=near_position.tolist(), beam_delays_samples=compensation.tolist(),
                    near_arrival_samples=arrivals.tolist(), rir_scope=("simulated near and echo room paths" if case.get("near_rt60_s") else "echo path only; near is dry with geometric relative delays"),
        echo_path_delay_scope="additional transport delay plus RIR propagation; synchronous buffers, stream_delay_ms=0",
                    measured_input_snr_db=(db(power(clean_channels[near_mask, 0]) / max(power(noise[near_mask, 0]), EPS))
                                           if case["noise"] != "none" else None),
                    measured_ser_db=(db(power(clean_channels[near_mask, 0]) / power(echo[far_mask, 0])) if acoustic else None))
    if spatial_audit:
        metadata["spatial_field_audit"]=spatial_audit
    metadata["playback_source"] = case.get("playback_source", "far_speech") if acoustic else "none"
    metadata["playback_reference_available"] = acoustic and case["stress"] != "wrong_reference"
    metadata["playback_origin"] = ("band-limited recorded speech simulating radio content; not a radio recording"
                                   if metadata["playback_source"] == "radio" else "recorded far-end speech" if acoustic else "none")
    if case.get("profile") in ("expanded-v2", "expanded-v3"):
        quantized_noise = read(destination / "noise-component.wav")[:, 0]
        metadata["measured_input_snr_db"] = (db(power(clean_channels[near_mask, 0]) / max(power(quantized_noise[near_mask]), EPS))
                                              if case["noise"] != "none" else None)
        metadata["noise_speakers"] = noise_speakers
        metadata["noise_origin"] = ("real recording" if case["noise"] in noises else
                                    "DEMAND TCAR cabin recording plus controlled interference; not a measured speed/HVAC setting"
                                    if case["noise"].startswith("car_")
                                    else "four independent recorded speakers" if case["noise"] == "babble"
                                    else "none" if case["noise"] == "none" else "generated signal")
    metadata["files"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(destination.iterdir()) if p.is_file()}
    (destination / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def modes(case):
    family = case["family"]
    no_aec = ["--no-aec"] if family in ("ns", "agc") else []
    result = {"full": no_aec, "no_agc": no_aec + ["--no-agc"],
              "bypass": ["--no-aec", "--no-ns", "--no-agc", "--no-limiter"]}
    if family == "ns":
        result["no_ns"] = ["--no-aec", "--no-ns"]
    elif family == "aec":
        result["aec_only"] = ["--no-ns", "--no-agc", "--no-limiter"]
        result.pop("no_agc")
        result["no_aec"] = ["--no-aec"]
    elif family == "agc":
        result["agc_only"] = ["--no-aec", "--no-ns"]
    else:
        result["no_ns"] = ["--no-ns"]
        result["no_aec"] = ["--no-aec"]
    if family == "bf":
        result["no_bf"] = []
        result["beam_only"] = ["--no-aec", "--no-ns", "--no-agc", "--no-limiter"]
    if case["stress"] == "wrong_reference":
        result["correct_reference"] = []
    return result


def invocation(scene, mode, cli, directory, processing_profile="general"):
    if processing_profile not in PROCESSING_PROFILES:
        raise ValueError("unknown processing profile")
    prefix = directory / "runs" / mode / "result"
    command = [str(cli)]
    if processing_profile != "general":
        command += ["--profile", processing_profile]
    command += ["--mic", str(directory / "before.wav"), "--out", str(prefix)] + modes(scene)[mode]
    if scene["family"] not in ("ns", "agc"):
        reference = "playback.wav" if mode == "correct_reference" else "ref.wav"
        command += ["--echo", str(directory / reference)]
    if scene["family"] == "bf" and mode not in ("no_bf", "bypass"):
        command += ["--beam-delays", ",".join(f"{d:.9f}" for d in scene["beam_delays_samples"])]
    elif scene.get("mics", 1) > 1:
        command += ["--no-bf"]
    return command


def verify_processing_profile(engine, processing_profile):
    if processing_profile not in PROCESSING_PROFILES:
        raise ValueError("unknown processing profile")
    if engine.get("processing_profile") != processing_profile:
        raise ValueError("engine processing profile differs from requested profile")
    if engine.get("continuous_vehicle_policy") != (processing_profile == "vehicle_stream"):
        raise ValueError("engine vehicle policy differs from requested profile")


def frame_timing(frame_rows, frame_count):
    if len(frame_rows) < frame_count or any(int(row["frame"]) != i for i, row in enumerate(frame_rows)):
        raise ValueError("missing or unordered timing frames")
    durations = np.array([float(row["process_ms"]) for row in frame_rows[:frame_count]])
    if not np.isfinite(durations).all() or np.min(durations) < 0:
        raise ValueError("invalid processing durations")
    ordered = np.sort(durations)
    return dict(process_p95_ms=float(ordered[int(np.ceil(.95 * frame_count)) - 1]),
                process_max_ms=float(ordered[-1]), process_rtf=float(np.mean(durations) / 10)), int(np.sum(durations > 10))


def attenuation_measurement(before, after, pcm16=False):
    p, q = power(before), power(after)
    exact = db(p) - db(q) if p > 1e-12 and q > 1e-12 else None
    lower_bound = None
    upper_level = None
    if exact is None and p > 1e-12 and pcm16 and np.max(np.abs(after)) < 32767 / 32768:
        # lround PCM16 quantization: each decoded sample differs from the
        # float output by at most half an LSB. RMS obeys the triangle bound.
        upper_power = (np.sqrt(q) + .5 / 32768) ** 2
        upper_level = db(upper_power)
        lower_bound = db(p) - upper_level
    return dict(before_dbfs=db(p), after_dbfs=db(q), total_attenuation_db=exact,
                attenuation_lower_bound_db=lower_bound,
                output_quantization_upper_bound_dbfs=upper_level)


def measurement_applicability(before, after, rate=RATE, edge_ms=1.0):
    """Report whether a window actually excites its measurement.

    Diagnostics only: no threshold, denominator or gate reads these fields.
    `before_edge_*` / `after_edge_*` count non-zero samples inside the first and
    last millisecond of the window. A silent window whose edge count is not
    zero is contaminated by the adjacent scene: the harness resamples the whole
    continuous stream in one polyphase pass, so a scene onset leaks backwards
    into the preceding samples and a far-end tail leaks forwards.
    """
    edge = max(1, int(round(rate * edge_ms / 1000.0)))

    def nz(x):
        return int(np.count_nonzero(x))

    before_dbfs, after_dbfs = db(power(before)), db(power(after))
    before_nz, before_edge_nz = nz(before), nz(before[-edge:])
    return dict(sample_rate=rate, samples=int(len(before)), edge_samples=edge,
                before_nonzero_samples=before_nz, after_nonzero_samples=nz(after),
                before_head_nonzero_samples=nz(before[:edge]),
                before_edge_nonzero_samples=before_edge_nz,
                after_head_nonzero_samples=nz(after[:edge]),
                after_edge_nonzero_samples=nz(after[-edge:]),
                before_peak=float(np.max(np.abs(before))) if len(before) else 0.0,
                after_peak=float(np.max(np.abs(after))) if len(after) else 0.0,
                before_dbfs=before_dbfs, after_dbfs=after_dbfs,
                # A window whose entire non-zero content fits inside one
                # millisecond at its trailing edge is not excited: those samples
                # are backward leakage from the next scene's onset through the
                # harness polyphase resampler, not input excitation. Such a
                # window cannot support an attenuation claim in either
                # direction. Diagnostic only; no gate consumes it.
                edge_leakage_only=bool(0 < before_nz <= edge and before_edge_nz == before_nz),
                measurable_excitation=bool(before_dbfs > -120.0))


def quality(target, observed):
    target = target - np.mean(target)
    observed = observed - np.mean(observed)
    alpha = float(np.dot(target, observed) / max(np.dot(target, target), EPS))
    projected = alpha * target
    error = observed - projected
    if power(projected) < 1e-15:
        return {"sisdr_db": None, "stoi": None, "projection_gain_db": None, "correlation": None}
    censored = power(error) <= power(projected) * 1e-12
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intelligibility = float(stoi(target, observed, RATE, extended=False))
    return {"sisdr_db": min(120., db(power(projected) / max(power(error), EPS))),
            "sisdr_censored": bool(censored), "stoi": intelligibility if not caught and np.isfinite(intelligibility) else None,
            "projection_gain_db": db(alpha ** 2), "correlation": float(np.corrcoef(target, observed)[0, 1]),
            "rms_dbfs": db(power(observed)), "peak_dbfs": db(float(np.max(np.abs(observed))) ** 2),
            "clipping_percent": float(np.mean(np.abs(observed) >= .999) * 100)}


def window(x, limits, rate=RATE):
    return x[round(limits[0] * rate):round(limits[1] * rate)]


def hpf_coefficients():
    path = Path(__file__).resolve().parents[2] / "build/webrtc-src/modules/audio_processing/high_pass_filter.cc"
    source = path.read_text().split("kHighPassFilterCoefficients16kHz = {{", 1)[1].split("}};", 1)[0]
    rows = []
    for b, a in re.findall(r"\.b = \{([^}]+)\},\s*\.a = \{([^}]+)\}", source):
        coefficients = lambda text: [float(np.float32(float(value.strip().removesuffix("f")))) for value in text.split(",")]
        rows.append(coefficients(b) + [1.] + coefficients(a))
    if len(rows) != 3:
        raise AssertionError("pinned WebRTC HPF coefficients changed")
    return np.array(rows), str(path), digest(path)


def measure(scene, mode, directory, write_metrics=True, quality_cache=None):
    prefix = directory / "runs" / mode / "result"
    engine = json.loads(prefix.with_suffix(".json").read_text())
    before = read(directory / "before.wav")[:, 0]
    use_beam = scene["family"] == "bf" and mode not in ("no_bf", "bypass")
    clean = read(directory / ("clean-beam.wav" if use_beam else "clean.wav"))[:, 0]
    output = read(prefix.with_suffix(".after.wav"))[:, 0]
    pcm16_output = sf.info(prefix.with_suffix(".after.wav")).subtype == "PCM_16"
    delay = engine["dsp_delay_samples"]
    if len(output) != len(before) + delay:
        raise AssertionError(f"output delay/length mismatch {scene['id']} {mode}")
    after = output[delay:delay + len(before)]
    # Fixed, causal reference filtering. This is a secondary diagnostic; raw
    # metrics remain unchanged. AEC and NS each require HPF in this split APM.
    hpf_stages = engine.get("hpf_stages", int(engine["aec_enabled"]) + int(engine["ns_enabled"]))
    filtered_clean, filtered_before = clean.copy(), before.copy()
    coefficients, hpf_path, hpf_sha = hpf_coefficients()
    for _ in range(hpf_stages):
        filtered_clean = signal.sosfilt(coefficients, filtered_clean)
        filtered_before = signal.sosfilt(coefficients, filtered_before)
    segments = {}
    def paired_quality(target, observed):
        if quality_cache is None:
            return quality(target, observed)
        key = (hashlib.sha256(target.tobytes()).digest(), hashlib.sha256(observed.tobytes()).digest())
        if key not in quality_cache:
            quality_cache[key] = quality(target, observed)
        return quality_cache[key]
    for name, limits in scene["windows"].items():
        b, a, target = window(before, limits), window(after, limits), window(clean, limits)
        if name.startswith("near") or name == "double_talk":
            qb, qa = paired_quality(target, b), paired_quality(target, a)
            segments[name] = {"before": qb, "after": qa,
                "sisdr_improvement_db": (qa["sisdr_db"] - qb["sisdr_db"]
                    if qa["sisdr_db"] is not None and qb["sisdr_db"] is not None and not qb.get("sisdr_censored") and not qa.get("sisdr_censored") else None),
                "stoi_improvement": qa["stoi"] - qb["stoi"] if qa["stoi"] is not None and qb["stoi"] is not None else None}
            matched_before = qb if hpf_stages == 0 else paired_quality(window(filtered_clean, limits), window(filtered_before, limits))
            matched_after = qa if hpf_stages == 0 else paired_quality(window(filtered_clean, limits), a)
            segments[name]["hpf_matched_after"] = matched_after
            segments[name]["hpf_matched_before"] = matched_before
            segments[name]["hpf_matched_sisdr_improvement_db"] = (
                matched_after["sisdr_db"] - matched_before["sisdr_db"]
                if matched_after["sisdr_db"] is not None and matched_before["sisdr_db"] is not None
                and not matched_before.get("sisdr_censored") and not matched_after.get("sisdr_censored") else None)
        else:
            # These are total residual levels, not inferred separated noise/echo.
            segments[name] = attenuation_measurement(b, a, pcm16=pcm16_output)
            if name.startswith("far") and engine["linear_aec_available"]:
                linear = read(prefix.with_suffix(".linear-aec.wav"))[:, 0]
                ld = engine["linear_aec_delay_samples"]
                residual = window(linear[ld:ld + len(before)], limits)
                value = db(power(b)) - db(power(residual)) if power(residual) > 1e-12 else None
                segments[name]["linear_residual_attenuation_db"] = value
                linear_input = b
                if engine.get("linear_erle_uses_conditioned_input"):
                    conditioned = read(prefix.with_suffix(".aec-input.wav"))[:, 0]
                    cd = engine["aec_input_delay_samples"]
                    linear_input = window(conditioned[cd:cd + len(before)], limits)
                erle = db(power(linear_input)) - db(power(residual)) if power(linear_input) > 1e-12 and power(residual) > 1e-12 else None
                segments[name]["linear_erle_engineering_db"] = erle if scene["noise"] == "none" else None
    with prefix.with_suffix(".frames.csv").open() as f:
        frame_rows = list(csv.DictReader(f))
    timing, frames_over_10ms = frame_timing(frame_rows, round(scene["seconds"] * 100))
    for key, value in timing.items():
        # The legacy CSV has six significant digits; report uses nine.
        if not np.isclose(engine[key], value, rtol=2e-5, atol=1e-8):
            raise ValueError(f"engine timing disagrees with frame measurements: {scene['id']} {key}")
        engine[key] = value
    result = {"id": scene["id"], "family": scene["family"], "mode": mode, "segments": segments,
              "engine": engine, "output_sha256": digest(prefix.with_suffix(".after.wav")),
              "output_levels": {"peak_dbfs": db(float(np.max(np.abs(output))) ** 2),
                                "clipping_percent": float(np.mean(np.abs(output) >= .999) * 100)},
              "process_frames_over_10ms": frames_over_10ms,
              "hpf_reference_diagnostic": {"stages": hpf_stages, "source": hpf_path, "source_sha256": hpf_sha,
                   "sos": coefficients.tolist(), "scope": "secondary same fixed causal HPF on clean target and input baseline; no lag optimization; raw metrics retained"},
              "bf_reference_scope": "BF on uses the exact geometric delay-and-sum clean target; input baseline is channel 0" if use_beam else None,
              "timing_scope": "single process, C++ process only; per-mode repeated recordings, host scheduling affects timing"}
    result["files"] = {str(p.relative_to(directory)): digest(p) for p in sorted(prefix.parent.iterdir()) if p.is_file() and p.name != "metrics.json"}
    if write_metrics:
        (prefix.parent / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def aggregate(results):
    groups = defaultdict(list)
    for result in results:
        groups[(result["family"], result["mode"])].append(result)
    table = []
    for (family, mode), rows in groups.items():
        values = defaultdict(list)
        for row in rows:
            values["process_p95_ms"].append(row["engine"]["process_p95_ms"])
            values["rtf"].append(row["engine"]["process_rtf"])
            values["process_max_ms"].append(row["engine"]["process_max_ms"])
            for segment, metrics in row["segments"].items():
                for key in ("sisdr_improvement_db", "hpf_matched_sisdr_improvement_db", "stoi_improvement", "total_attenuation_db", "linear_erle_engineering_db"):
                    value = metrics.get(key)
                    if value is not None:
                        values[f"{segment}.{key}"].append(value)
                for key in ("sisdr_db", "stoi", "projection_gain_db", "rms_dbfs", "correlation"):
                    value = metrics.get("after", {}).get(key)
                    if value is not None:
                        values[f"{segment}.after.{key}"].append(value)
                for key in ("sisdr_db", "projection_gain_db"):
                    value = metrics.get("hpf_matched_after", {}).get(key)
                    if value is not None:
                        values[f"{segment}.hpf_matched_after.{key}"].append(value)
        table.append({"family": family, "mode": mode, "runs": len(rows), "metrics": {
            key: {"count": len(v), "median": float(np.median(v)), "min": min(v), "max": max(v)} for key, v in values.items()}})
    return table


def summarize(root, manifest, results):
    groups = aggregate(results)
    summary = {"schema_version": 1, "scenes": len(manifest), "runs": len(results), "groups": groups,
        "negative_controls": {}, "worst_full_runs": []}
    by_id = defaultdict(dict)
    for row in results:
        by_id[row["id"]][row["mode"]] = row
    errors = []
    for scene in manifest:
        if "bypass" not in by_id[scene["id"]]:
            continue
        row = by_id[scene["id"]]["bypass"]
        before = read(root / "cases" / scene["id"] / "before.wav")[:, 0]
        after = read(root / "cases" / scene["id"] / "runs/bypass/result.after.wav")[:, 0]
        errors.append(float(np.max(np.abs(before - after))))
    if errors and max(errors) > 1 / 32768:
        raise AssertionError("bypass changes microphone channel 0")
    summary["negative_controls"]["bypass"] = {"cases": len(errors), "max_pcm16_lsb_error": max(errors) * 32768 if errors else None, "passed": bool(errors)}
    full = [r for r in results if r["mode"] == "full"]
    for row in sorted(full, key=lambda x: x["segments"]["near_only"].get("sisdr_improvement_db") if x["segments"]["near_only"].get("sisdr_improvement_db") is not None else 120)[:12]:
        summary["worst_full_runs"].append({"id": row["id"], "family": row["family"], "near_only": row["segments"]["near_only"], "double_talk": row["segments"].get("double_talk")})
    for control, family, segment, key, better in (("ns", "ns", "noise_only", "total_attenuation_db", "no_ns"),
                                               ("aec", "aec", "far_only", "total_attenuation_db", "no_aec")):
        differences = []
        for scene in manifest:
            if scene["family"] != family:
                continue
            if better not in by_id[scene["id"]]:
                continue
            a = by_id[scene["id"]]["full"]["segments"][segment][key]
            b = by_id[scene["id"]][better]["segments"][segment][key]
            if a is not None and b is not None:
                differences.append(a - b)
        summary["negative_controls"][control] = {"cases": len(differences), "median_full_vs_disabled_db": float(np.median(differences)) if differences else None,
                                                  "positive_cases": sum(d > 0 for d in differences)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (root / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    flat = []
    for row in results:
        for segment, metrics in row["segments"].items():
            item = {"id": row["id"], "family": row["family"], "mode": row["mode"], "segment": segment,
                    "process_p95_ms": row["engine"]["process_p95_ms"], "process_rtf": row["engine"]["process_rtf"]}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    item.update({f"{key}_{k}": v for k, v in value.items()})
                else:
                    item[key] = value
            flat.append(item)
    with (root / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# ITU WebRTC offline simulation", "", f"Scenes: {len(manifest)}; C++ runs: {len(results)}.", "",
        "Real LibriSpeech speech and DEMAND recordings; mixtures, echo paths and array geometry are simulated.",
        "Fixed DSP delays are removed before comparisons; no per-output optimal-lag search.",
        "SI-SDR and STOI are engineering measures, not MOS or ITU compliance.",
        "Raw SI-SDR includes WebRTC HPF phase differences. Secondary HPF-matched metrics apply the pinned fixed causal HPF to the target and baseline (AEC one stage, NS one stage). Both remain visible; this does not undo nonlinear distortion or optimize lag.",
        "Linear ERLE is reported only for noise-free far-only scenes, before nonlinear suppression, BF, NS and AGC2.",
        "Noisy far-only scenes report total linear residual attenuation, not echo-only ERLE.", "",
        "| Family | Mode | Runs | Raw near SI-SDR delta median (dB) | HPF-matched near SI-SDR delta median (dB) | Near STOI delta median | HPF-matched double-talk SI-SDR delta median (dB) | Far total attenuation median (dB) | C++ P95 median (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def show(metrics, key):
        return f"{metrics[key]['median']:.3f}" if key in metrics else "n/a"
    for group in groups:
        m = group["metrics"]
        lines.append(f"| {group['family']} | {group['mode']} | {group['runs']} | {show(m, 'near_only.sisdr_improvement_db')} | {show(m, 'near_only.hpf_matched_sisdr_improvement_db')} | {show(m, 'near_only.stoi_improvement')} | {show(m, 'double_talk.hpf_matched_sisdr_improvement_db')} | {show(m, 'far_only.total_attenuation_db')} | {show(m, 'process_p95_ms')} |")
    lines += ["", "## Worst full-chain near-only cases", "", "Detailed conditions are in cases/<id>/scene.json; every output has a hash and per-frame timing CSV.", ""]
    for row in summary["worst_full_runs"]:
        q = row["near_only"]
        lines.append(f"- {row['id']}: SI-SDR delta {q['sisdr_improvement_db']}; STOI delta {q['stoi_improvement']}")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/itu-eval-v1"))
    parser.add_argument("--cli", type=Path, default=Path("build/webrtc-host/cpp/itu_audio_cli"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inputs-from", type=Path, help="replay frozen scene inputs in a separate output dataset")
    parser.add_argument("--full-only", action="store_true", help="diagnostic repair sweep; not the final control matrix")
    parser.add_argument("--profile", choices=("v1", "expanded-v2"), default="v1")
    parser.add_argument("--processing-profile", choices=PROCESSING_PROFILES, default="general")
    parser.add_argument("--prepare-only", action="store_true", help="freeze all inputs before processing")
    args = parser.parse_args()
    root, cli = args.root.resolve(), args.cli.resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "artifacts/itu_audio_cli"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists() and digest(artifact) != digest(cli):
        raise ValueError("dataset binary already frozen; choose a new output directory")
    if not artifact.exists():
        shutil.copy2(cli, artifact)
    cli = artifact
    frozen = args.inputs_from.resolve() if args.inputs_from else None
    if frozen:
        if frozen == root:
            raise ValueError("frozen baseline and output dataset must differ")
        cases = json.loads((frozen / "manifest.json").read_text())["scenes"]
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "sources").exists():
            (root / "sources").symlink_to(frozen / "sources", target_is_directory=True)
        if (frozen / "ATTRIBUTION.md").exists():
            (root / "ATTRIBUTION.md").write_bytes((frozen / "ATTRIBUTION.md").read_bytes())
    else:
        speakers, noises = sources(root)
        if args.profile == "expanded-v2":
            from eval_profiles import expanded_matrix
            source_manifest = json.loads((root / "sources/manifest.json").read_text())
            cases = expanded_matrix(source_manifest["speaker_metadata"])
        else:
            cases = matrix(list(speakers))
    if args.limit:
        cases = cases[:args.limit]
    manifest, results, cache = [], [], {}
    started = time.time()
    root.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        directory = root / "cases" / case["id"]
        if frozen:
            scene = json.loads((frozen / "cases" / case["id"] / "scene.json").read_text())
            directory.mkdir(parents=True, exist_ok=True)
            for name, record in scene["files"].items():
                source = frozen / "cases" / case["id"] / name
                if digest(source) != record["sha256"]:
                    raise AssertionError(f"frozen input hash changed: {case['id']} {name}")
                target = directory / name
                if not target.exists():
                    target.symlink_to(source)
                if digest(target) != record["sha256"]:
                    raise AssertionError(f"replay input hash changed: {case['id']} {name}")
            (directory / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
        elif args.resume and (directory / "scene.json").exists():
            scene = json.loads((directory / "scene.json").read_text())
            for name, record in scene["files"].items():
                if digest(directory / name) != record["sha256"]:
                    raise AssertionError(f"scene hash changed: {case['id']} {name}")
        else:
            scene = make_case(case, speakers, noises, directory, cache)
        manifest.append(scene)
        quality_cache = {}
        for mode, flags in modes(case).items():
            if args.prepare_only:
                break
            if args.full_only and mode != "full":
                continue
            prefix = directory / "runs" / mode / "result"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            command = invocation(scene, mode, cli, directory, args.processing_profile)
            if args.resume and prefix.with_suffix(".json").exists():
                log = json.loads((prefix.parent / "invocation.json").read_text())
                if log["cli_sha256"] != digest(cli) or log["argv"] != command:
                    raise AssertionError("resume engine/arguments changed")
            else:
                run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                (prefix.parent / "invocation.json").write_text(json.dumps({"argv": command, "cli_sha256": digest(cli),
                    "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}, indent=2) + "\n")
                run.check_returncode()
            result = measure(scene, mode, directory, quality_cache=quality_cache)
            verify_processing_profile(result["engine"], args.processing_profile)
            if mode == "full":
                # Four-role files immediately usable by the existing upload workflow.
                after, _ = sf.read(prefix.with_suffix(".after.wav"), dtype="int16")
                sf.write(directory / "denoise.wav", after, RATE, subtype="PCM_16")
                (directory / "website-options.json").write_text(json.dumps({"sample_rate": RATE,
                    "speech": {"start": scene["windows"]["near_only"][0], "end": scene["windows"]["near_only"][1]},
                    "noise": {"start": scene["windows"]["noise_only"][0], "end": scene["windows"]["noise_only"][1]},
                    "after_offset_ms": result["engine"]["dsp_delay_samples"] * 1000 / RATE,
                    "common_clock": case["stress"] != "clock_300ppm", "speech_confirmed": True,
                    "far_end_only": False, "aec_only": False, "stationary_noise": case["noise"] == "white"}, indent=2) + "\n")
                scene["denoise_sha256"] = digest(directory / "denoise.wav")
            results.append(result)
        if not case.get("profile") or (index + 1) % 50 == 0 or index + 1 == len(cases):
            (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "seed": SEED, "scenes": manifest,
                "processing_profile": args.processing_profile,
                "environment": {"machine": platform.machine(), "system": platform.platform(), "cli_sha256": digest(cli),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile", "pystoi", "rir-generator")}},
                "source_manifest_sha256": digest(root / "sources/manifest.json"),
                "generator_sha256": digest(__file__), "elapsed_seconds": time.time() - started}, indent=2) + "\n")
        print(f"{index + 1}/{len(cases)} {case['id']} ({len(results)} runs, {time.time() - started:.1f}s)", flush=True)
    if args.prepare_only:
        sha = digest(root / "manifest.json")
        (root / "manifest.sha256").write_text(sha + "\n")
        print(f"frozen {len(manifest)} scenes; manifest SHA256 {sha}", flush=True)
        return
    summary = summarize(root, manifest, results)
    print(json.dumps({"scenes": summary["scenes"], "runs": summary["runs"], "negative_controls": summary["negative_controls"]}, indent=2))


if __name__ == "__main__":
    main()
