#!/usr/bin/env python3
"""Offline window diagnostics. No acceptance threshold or algorithm input changes.

Clean targets are used only for offline projection measurements. Energy islands
are acoustic proxies, not linguistic word boundaries. All values are diagnostics.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

NEAR_ROLES = frozenset(("near_only", "near_late", "double_talk"))
PARAMETERS = {"version": "ecnr-window-diagnostics-v2", "envelope_ms": 20,
              "boundary_ms": 80, "tail_ms": 200, "energy_island_relative_power": .01,
              "energy_island_gap_ms": 40, "short_island_ms": 400,
              "near_floor_power_multiple": 10, "gain_jump_diagnostic_db": 1,
              "dropout_output_power_multiple": 10, "linguistic_word_detection": False,
              "post_playback_tail": {"reference_block_ms": 10, "minimum_activity_ms": 100,
                                     "minimum_quiet_ms": 200, "absolute_activity_dbfs": -75,
                                     "relative_activity_db": -40, "floor_power_multiple": 10,
                                     "scan_seconds": [.5, 1., 2.]}}
SCOPE = "Offline diagnostics only; original quality gates unchanged; not ITU acoustic compliance."


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def quantization_power(subtype: str = "PCM_16") -> float | None:
    """Ideal uniform quantization variance, not a measured physical noise floor."""
    bits = {"PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(subtype)
    return (2.0**(-(bits-1)))**2/12 if bits else None


def levels(samples: np.ndarray, quantization: float | None = None) -> dict:
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
        return {"available": False, "reason": "empty_nonfinite_or_nonmono"}
    power = float(np.mean(y*y))
    peak = float(np.max(np.abs(y)))
    floor = quantization if quantization is not None else 1e-20
    return {"available": True, "samples": len(y), "rms": float(np.sqrt(power)),
            "rms_dbfs": 10*np.log10(power) if power > 0 else None,
            "peak": peak, "peak_dbfs": 20*np.log10(peak) if peak > 0 else None,
            "zero_samples": int(np.count_nonzero(y == 0)),
            "quantization_power": quantization,
            "quantization_dbfs": 10*np.log10(quantization) if quantization else None,
            "near_measurement_floor": power <= 10*floor}


def attenuation(before: np.ndarray, after: np.ndarray, before_floor: float | None,
                after_floor: float | None) -> dict:
    a, b = levels(before, before_floor), levels(after, after_floor)
    if not a["available"] or not b["available"]:
        return {"available": False, "reason": "invalid_audio", "db": None}
    if a["near_measurement_floor"]:
        return {"available": False, "reason": "near_zero_denominator", "db": None,
                "before": a, "after": b}
    floor = max(after_floor or 1e-20, 1e-20)
    output_power = b["rms"]**2
    if output_power <= floor:
        return {"available": True, "db": None, "lower_bound_db":
                float(10*np.log10(a["rms"]**2/floor)), "censored": True,
                "reason": "output_at_or_below_quantization_floor"}
    return {"available": True, "db": float(10*np.log10(a["rms"]**2/output_power)),
            "censored": False, "interpretation": "level change; speech windows are not net NR"}


def projection(clean: np.ndarray, observed: np.ndarray, floor: float | None) -> dict:
    x, y = np.asarray(clean, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        return {"available": False, "reason": "invalid_projection_shape"}
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"available": False, "reason": "nonfinite_projection"}
    clean_power = float(np.mean(x*x))
    if clean_power <= 10*(floor or 1e-20):
        return {"available": False, "reason": "clean_at_measurement_floor"}
    gain = float(np.dot(x, y)/np.dot(x, x))
    residual = y-gain*x
    norm = float(np.linalg.norm(x)*np.linalg.norm(y))
    return {"available": True, "signed_gain": gain,
            "absolute_gain_db": float(20*np.log10(abs(gain))) if gain != 0 else None,
            "inverted": gain < 0, "zero_gain": gain == 0,
            "correlation": float(np.dot(x, y)/norm) if norm else None,
            "residual_rms_dbfs": levels(residual)["rms_dbfs"],
            "clean_rms_dbfs": float(10*np.log10(clean_power)),
            "interpretation": "signed clean projection at fixed declared delay; no best-lag fitting"}


def envelope(samples: np.ndarray, rate: int) -> np.ndarray:
    width = max(1, int(round(PARAMETERS["envelope_ms"]*rate/1000)))
    y = np.asarray(samples, dtype=np.float64)
    starts = np.arange(0, len(y), width, dtype=np.int64)
    return np.array([np.mean(y[a:min(a+width, len(y))]**2) for a in starts])


def speech_diagnostics(clean: np.ndarray, observed: np.ndarray, rate: int,
                       floor: float | None = None) -> dict:
    x, y = np.asarray(clean, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    summary = projection(x, y, floor)
    if not summary["available"]:
        return {"projection": summary, "energy_islands": [], "short_islands": [],
                "dropout_energy_blocks": [], "reason": summary["reason"]}
    width = max(1, int(round(PARAMETERS["envelope_ms"]*rate/1000)))
    clean_env, observed_env = envelope(x, rate), envelope(y, rate)
    threshold = max(float(np.max(clean_env))*PARAMETERS["energy_island_relative_power"],
                    (floor or 1e-20)*PARAMETERS["near_floor_power_multiple"])
    active = clean_env > threshold
    gap_blocks = int(PARAMETERS["energy_island_gap_ms"]/PARAMETERS["envelope_ms"])
    # Merge only bounded internal gaps; boundary decisions remain tied to clean.
    merged = active.copy()
    indexes = np.flatnonzero(active)
    for left, right in zip(indexes[:-1], indexes[1:]):
        if 1 < right-left <= gap_blocks+1:
            merged[left:right+1] = True
    padded = np.r_[False, merged, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    boundary = max(1, int(round(PARAMETERS["boundary_ms"]*rate/1000)))
    islands = []
    for first, last in zip(starts, ends):
        a, b = int(first*width), min(int(last*width), len(x))
        islands.append({"start_s": a/rate, "end_s": b/rate, "duration_s": (b-a)/rate,
                        "projection": projection(x[a:b], y[a:b], floor),
                        "onset_projection": projection(x[a:min(a+boundary, b)],
                                                       y[a:min(a+boundary, b)], floor),
                        "offset_projection": projection(x[max(a, b-boundary):b],
                                                        y[max(a, b-boundary):b], floor)})
    short = [i for i in islands if i["duration_s"] <= PARAMETERS["short_island_ms"]/1000+1e-9]
    drop_indexes = np.flatnonzero(active & (observed_env <=
                                (floor or 1e-20)*PARAMETERS["dropout_output_power_multiple"]))
    tail = max(1, int(round(PARAMETERS["tail_ms"]*rate/1000)))
    db_clean = [float(10*np.log10(p)) if p > 0 else None for p in clean_env]
    db_observed = [float(10*np.log10(p)) if p > 0 else None for p in observed_env]
    return {"projection": summary, "energy_island_threshold_dbfs": float(10*np.log10(threshold)),
            "energy_islands": islands, "short_islands": short,
            "dropout_energy_blocks": [{"start_s": int(i*width)/rate,
                                       "end_s": min(int((i+1)*width), len(x))/rate}
                                      for i in drop_indexes],
            "tail": {"duration_s": min(tail, len(x))/rate,
                     "clean": levels(x[-tail:], floor), "observed": levels(y[-tail:], floor),
                     "projection": projection(x[-tail:], y[-tail:], floor)},
            "envelope": {"block_ms": PARAMETERS["envelope_ms"],
                         "clean_dbfs": db_clean, "observed_dbfs": db_observed},
            "limitations": "Energy islands are not words; reverberation and EQ phase can lower fixed-delay projection."}


def gain_diagnostics(times: np.ndarray, gains: np.ndarray,
                     start_s: float, end_s: float) -> dict:
    """CSV applied_gain_db is commanded gain, not waveform-estimated gain."""
    t, g = np.asarray(times, dtype=np.float64), np.asarray(gains, dtype=np.float64)
    if t.shape != g.shape or t.ndim != 1:
        raise ValueError("gain vectors mismatch")
    selection = (t >= start_s) & (t < end_s) & np.isfinite(t) & np.isfinite(g)
    ts, gs = t[selection], g[selection]
    if not len(gs):
        return {"available": False, "reason": "no_finite_commanded_gain_frames"}
    # Include the entering transition at the window boundary if a preceding frame exists.
    selected_indexes = np.flatnonzero(selection)
    jumps = []
    for index in selected_indexes:
        if index > 0 and np.isfinite(g[index-1]) and np.isfinite(t[index-1]):
            step = float(g[index]-g[index-1])
            jumps.append((float(t[index]), step))
    return {"available": True, "scope": "commanded gain from CSV, not measured gain",
            "frames": len(gs), "min_db": float(np.min(gs)), "max_db": float(np.max(gs)),
            "max_step_db": max((abs(step) for _, step in jumps), default=0.0),
            "jump_events": [{"time_s": time, "step_db": step} for time, step in jumps
                            if abs(step) >= PARAMETERS["gain_jump_diagnostic_db"]]}


def post_playback_tail_audit(reference: np.ndarray, before: np.ndarray,
                             after: np.ndarray, rate: int, *,
                             reference_floor: float | None = None,
                             before_floor: float | None = None,
                             after_floor: float | None = None) -> dict:
    """Locate digital reference stops; scan mixture tails, not isolated echo.

    Reference energy does not establish acoustic reference reliability, including
    during double talk. All channels participate without downmix cancellation.
    """
    r, pre, post = (np.asarray(x, dtype=np.float64) for x in (reference, before, after))
    if r.ndim == 1:
        r = r[:, np.newaxis]
    base = {"scope": "digital playback stop and mixed waveform tail diagnostics",
            "reference_reliability_evaluated": False, "uses_clean_target": False,
            "time_origin": "declared common input timeline; output DSP delay removed",
            "parameters": PARAMETERS["post_playback_tail"], "events": []}
    if r.ndim != 2 or not r.shape[1] or pre.ndim != 1 or post.ndim != 1:
        raise ValueError("invalid reference/tail channel shape")
    if len(r) != len(pre) or len(r) != len(post) or rate <= 0:
        raise ValueError("reference/tail common timeline length or sample rate mismatch")
    if not len(r):
        return {**base, "available": False, "reason": "empty_reference"}
    if not all(np.all(np.isfinite(x)) for x in (r, pre, post)):
        return {**base, "available": False, "reason": "nonfinite_reference_or_tail"}
    config = PARAMETERS["post_playback_tail"]
    width = max(1, int(round(config["reference_block_ms"]*rate/1000)))
    full = len(r)//width
    powers = (np.max(np.mean(r[:full*width].reshape(full, width, r.shape[1])**2, axis=1), axis=1)
              if full else np.array([], dtype=np.float64))
    if full*width < len(r):
        powers = np.r_[powers, np.max(np.mean(r[full*width:]**2, axis=0))]
    threshold = max(10**(config["absolute_activity_dbfs"]/10),
                    (reference_floor or 1e-20)*config["floor_power_multiple"],
                    float(np.max(powers))*10**(config["relative_activity_db"]/10))
    active = powers > threshold
    base.update({"reference_channels": r.shape[1], "reference_activity_threshold_dbfs":
                 float(10*np.log10(threshold)),
                 "reference_channel_policy": "maximum channel block power, never downmix",
                 "reference_active_blocks": int(np.count_nonzero(active))})
    if not np.any(active):
        return {**base, "available": False, "reason": "reference_below_activity_threshold"}
    # Fill sub-guard internal gaps, so syllabic pauses do not become stops.
    merged = active.copy()
    indexes = np.flatnonzero(active)
    minimum_quiet_samples = int(round(config["minimum_quiet_ms"]*rate/1000))
    for left, right in zip(indexes[:-1], indexes[1:]):
        if (right-left-1)*width < minimum_quiet_samples:
            merged[left:right+1] = True
    padded = np.r_[False, merged, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    minimum_active_samples = int(round(config["minimum_activity_ms"]*rate/1000))
    events = []
    for index, (first, last) in enumerate(zip(starts, ends)):
        active_start, stop = int(first*width), min(int(last*width), len(r))
        next_start = int(starts[index+1]*width) if index+1 < len(starts) else len(r)
        quiet_samples = next_start-stop
        effective_active_samples = int(np.count_nonzero(active[first:last]))*width
        if effective_active_samples < minimum_active_samples or quiet_samples < minimum_quiet_samples:
            continue
        scans = []
        for horizon in config["scan_seconds"]:
            wanted = int(round(horizon*rate))
            finish = min(stop+wanted, next_start, len(r))
            pre_part, post_part = pre[stop:finish], post[stop:finish]
            complete = finish-stop == wanted
            pre_env, post_env = envelope(pre_part, rate), envelope(post_part, rate)
            scans.append({"horizon_s": horizon, "observed_duration_s": (finish-stop)/rate,
                          "complete": complete,
                          "truncation_reason": None if complete else
                              ("reference_restarted" if next_start < len(r) else "end_of_stream"),
                          "before": levels(pre_part, before_floor),
                          "after": levels(post_part, after_floor),
                          "envelope": {"block_ms": PARAMETERS["envelope_ms"],
                                       "time_origin_s": stop/rate,
                                       "before_dbfs": [float(10*np.log10(x)) if x > 0 else None
                                                        for x in pre_env],
                                       "after_dbfs": [float(10*np.log10(x)) if x > 0 else None
                                                       for x in post_env]}})
        events.append({"reference_active_start_s": active_start/rate, "stop_s": stop/rate,
                       "preceding_activity_duration_s": (stop-active_start)/rate,
                       "preceding_active_energy_duration_s": effective_active_samples/rate,
                       "quiet_reference_duration_s": quiet_samples/rate,
                       "stop_resolution_ms": width/rate*1000, "scans": scans})
    return {**base, "available": True, "events": events,
            "reason": None if events else "no_stop_with_required_quiet_guard",
            "limitations": ["Digital activity is not proof of acoustic reference reliability.",
                            "Tail contains near speech, background noise and possible echo.",
                            "No residual echo, ERLE, physical playback-stop or acceptance claim.",
                            "Activity boundaries have reference-block resolution."]}


def load_stream_audio(folder: Path, *, include_clean: bool = True,
                      include_reference: bool = False) -> dict:
    """Read each audio file once; do not expose clean to perceptual model scoring."""
    directory = Path(folder)
    identities = {}
    def load_json(name):
        payload = (directory/name).read_bytes()
        identities[name] = {"sha256": digest(payload), "bytes": len(payload)}
        return json.loads(payload)
    identity, engine = load_json("timeline.json"), load_json("stream.json")
    rate = int(engine["sample_rate"])
    if rate != int(identity["sample_rate"]):
        raise ValueError("timeline/engine sample rate mismatch")
    expected = int(engine["input_samples_per_channel"])
    delay = int(engine["dsp_delay_samples"])
    if delay < 0 or expected <= 0:
        raise ValueError("invalid declared DSP delay or input length")
    subtypes = {}
    def load_wave(name, channel):
        payload = (directory/name).read_bytes()
        stream = io.BytesIO(payload)
        info = sf.info(stream)
        stream.seek(0)
        audio, sr = sf.read(stream, dtype="float64", always_2d=True)
        if sr != rate or (channel is not None and (channel < 0 or channel >= audio.shape[1])):
            raise ValueError("audio sample rate/channel mismatch: "+name)
        identities[name] = {"sha256": digest(payload), "bytes": len(payload),
                            "sample_rate": sr, "samples": len(audio),
                            "channels": audio.shape[1], "selected_channel": channel}
        subtypes[name] = info.subtype
        return audio if channel is None else audio[:, channel]
    before, after_raw = load_wave("before.wav", 0), load_wave("stream.after.wav", 0)
    if len(before) != expected or len(after_raw) != expected+delay:
        raise ValueError("audio length does not match declared delay/input length")
    after = after_raw[delay:delay+expected]
    result = {"identity": identity, "engine": engine, "sample_rate": rate,
              "before": before, "after": after, "subtypes": subtypes,
              "source_identity": {"files": identities,
                                  "alignment": {"policy": "declared DSP delay; no lag optimization",
                                                "after_trim_samples": delay,
                                                "aligned_samples": expected,
                                                "sample_rate": rate},
                                  "cli_sha256": identity.get("cli_sha256"),
                                  "source_manifest_sha256": identity.get("source_manifest_sha256")}}
    if include_reference:
        reference = load_wave("ref.wav", None)
        if len(reference) != expected:
            raise ValueError("digital reference common timeline length mismatch")
        result["reference"] = reference
    if include_clean:
        clean_before, clean_after = load_wave("clean-channel0.wav", 0), load_wave("clean.wav", 0)
        if len(clean_before) != expected or len(clean_after) != expected:
            raise ValueError("offline clean target length mismatch")
        result.update({"clean_before": clean_before, "clean_after": clean_after,
                       "clean_scope": "channel 0 before; runner calibrated target after; offline measurement only"})
    return result


def audit_stream(folder: Path) -> dict:
    loaded = load_stream_audio(folder, include_clean=True, include_reference=True)
    directory, rate = Path(folder), loaded["sample_rate"]
    gain_times, gain_values = [], []
    csv_path = directory/"stream.frames.csv"
    csv_sha = hashlib.sha256()
    with csv_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            csv_sha.update(chunk)
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            gain_times.append(float(row["time_s"]))
            gain_values.append(float(row["applied_gain_db"]))
    times, gains = np.asarray(gain_times), np.asarray(gain_values)
    before_floor = quantization_power(loaded["subtypes"]["before.wav"])
    after_floor = quantization_power(loaded["subtypes"]["stream.after.wav"])
    clean_before_floor = quantization_power(loaded["subtypes"]["clean-channel0.wav"])
    clean_after_floor = quantization_power(loaded["subtypes"]["clean.wav"])
    windows = []
    for phase in loaded["identity"]["segments"]:
        for kind, limits in phase["windows"].items():
            start, end = map(float, limits)
            if not 0 <= start < end <= len(loaded["before"])/rate+1e-9:
                raise ValueError("window is outside aligned audio duration")
            a, b = int(round(start*rate)), int(round(end*rate))
            pre, post = loaded["before"][a:b], loaded["after"][a:b]
            row = {"label": phase["label"], "kind": kind, "window_s": limits,
                   "before": levels(pre, before_floor), "after": levels(post, after_floor),
                   "level_change": attenuation(pre, post, before_floor, after_floor),
                   "commanded_gain": gain_diagnostics(times, gains, start, end)}
            if kind in NEAR_ROLES:
                row.update({"before_speech": speech_diagnostics(
                                loaded["clean_before"][a:b], pre, rate, clean_before_floor),
                            "after_speech": speech_diagnostics(
                                loaded["clean_after"][a:b], post, rate, clean_after_floor)})
            windows.append(row)
    tails = post_playback_tail_audit(
        loaded["reference"], loaded["before"], loaded["after"], rate,
        reference_floor=quantization_power(loaded["subtypes"]["ref.wav"]),
        before_floor=before_floor, after_floor=after_floor)
    return {"schema_version": 2, "scope": SCOPE, "diagnostics_only": True,
            "quality_gates_changed": False, "folder": str(directory.resolve()),
            "algorithm_sha256": digest(Path(__file__).read_bytes()), "parameters": PARAMETERS,
            "source_identity": loaded["source_identity"], "csv_sha256": csv_sha.hexdigest(),
            "clean_scope": loaded["clean_scope"], "windows": windows,
            "sample_rate": rate, "post_playback_tails": tails,
            "limitations": ["Quantization floor is theoretical variance, not measured acoustic floor.",
                            "Speech-window level change is not net noise attenuation.",
                            "Projection is fixed declared delay and is sensitive to EQ/reverberation.",
                            "No transcript-based onset/offset/word detection or human MOS."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_stream(args.folder)
    save_json(args.output, result)
    print(json.dumps({"output": str(args.output), "windows": len(result["windows"]),
                      "diagnostics_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
