"""Compare executed R2 trials and align the GCC phase timestamps to real callbacks."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import statistics

from build_r2 import need, new_dir, read, save, sha
from verify_r2 import HOP_NS, frames


def abba(path: Path) -> dict:
    runs = read(path)["trials"]
    need([r["mode"] for r in runs] == ["norm", "norm_align", "norm_align", "norm"], "Expected ABBA run order")
    need(all(r["wav_sha256"] == runs[0]["wav_sha256"] for r in runs), "ABBA PCM changed")

    def entry(r):
        w = r["windows"]
        return {"trial": r["trial"], "mode": r["mode"], "all_wall_p99_ms": w["all"]["wall"]["p99_ms"],
                "steady_wall_p99_ms": w["steady"]["wall"]["p99_ms"],
                "steady_cpu_mean_ms": w["steady"]["cpu"]["mean_ms"],
                "over16": w["all"]["over16"], "steady_over16": w["steady"]["over16"],
                "startup_max_ms": w["startup"]["callback_to_complete"]["max_ms"],
                "drain_over16": w["drain"]["over16"], "xrun": r["xrun"]}

    rec = [entry(r) for r in runs]
    pairs = [{"control": rec[a]["trial"], "candidate": rec[b]["trial"],
              **{key + "_delta": rec[b][key] - rec[a][key]
                 for key in ("steady_wall_p99_ms", "steady_cpu_mean_ms", "steady_over16", "xrun")}}
             for a, b in ((0, 1), (3, 2))]
    return {"runs": rec, "pairs": pairs, "state_bit_exact": read(path)["state_bit_exact"],
            "steady_gain_repeated": all(p["steady_wall_p99_ms_delta"] < 0 and p["steady_over16_delta"] <= 0
                                        for p in pairs)}


def gcc(path: Path, *, table: bool) -> dict:
    data = read(path)
    need(data["gcc_profile"] and len(data["trials"]) == 1, "Expected one diagnosed norm run")
    trial = data["trials"][0]
    folder = Path(trial["directory"]) / "collected"
    log = (folder / "stderr.log").read_text()
    trace = frames(folder / "out")
    need(len(trace) == 3753, "Unexpected 60-second trace length")
    events = []
    ends = []
    for line in log.splitlines():
        if line.startswith("ITU_GCC_DIAG "):
            _, kind, *tokens = line.split()
            record = dict(token.split("=", 1) for token in tokens)
            if kind == "event":
                events.append({k: float(v) if k == "conf" else int(v) for k, v in record.items()})
            elif kind == "end" and int(record["attempted"]) > 0:
                ends.append({k: int(v) for k, v in record.items()})
    need(len(ends) == 1 and ends[0] == {"schema": 1, "attempted": len(events),
         "emitted": len(events), "dropped": 0, "clock_errors": 0} and len(events) == 6,
         "Missing or incomplete measured GCC events")
    hits = [(int(a), int(b)) for a, b in re.findall(r"^ITU_GCC_FFT_TABLE enabled=([01]) hits=(\d+)$", log, re.M)]
    if table:
        need([v for v in hits if v[1] != 0] == [(1, 18)], "FFT table not executed exactly 18 times")
    else:
        need(not hits, "Original diagnostic contains the FFT table")
    for i, e in enumerate(events):
        # The original debug label is off by one on the first 128 samples of a hop.
        # Use the consumed sample count and independently match the monotonic range.
        n = e["n_seen"]
        frame = (n - 1) // 256
        row = trace[frame]
        need(e["seq"] == i and n >= 16384 and (n - 16384) % 8000 < 128
             and n % 128 == 0 and row["frame"] == frame and row["clocks_valid"] == 1,
             "GCC sample/trace identity mismatch")
        need(row["input_ready_ns"] <= e["wall_enter_ns"] <= e["wall_exit_ns"] <= row["completed_ns"]
             and e["wall_ns"] == e["wall_exit_ns"] - e["wall_enter_ns"]
             and e["cpu_ns"] == e["cpu_exit_ns"] - e["cpu_enter_ns"], "GCC clock alignment mismatch")
        need(all(e[key] == 1 for key in ("clock_ok", "keep", "ref_ok", "mic_ok", "ifft_ok"))
             and e["locked"] == int(i == 5), "GCC decision/phase record incomplete")
        e["frame"] = frame
        e["over16"] = row["completed_ns"] - row["callback_start_ns"] > HOP_NS
        e["callback_ms"] = (row["completed_ns"] - row["callback_start_ns"]) / 1e6
        e["gcc_wall_ms"] = e["wall_ns"] / 1e6
        e["fft_wall_ms"] = sum(e[key] for key in ("ref_wall_ns", "mic_wall_ns", "ifft_wall_ns")) / 1e6
        e["fft_cpu_ms"] = sum(e[key] for key in ("ref_cpu_ns", "mic_cpu_ns", "ifft_cpu_ns")) / 1e6
        need(0 < e["fft_wall_ms"] <= e["gcc_wall_ms"], "Unexpected FFT timing extent")
    need([e["frame"] for e in events] == [63, 95, 126, 157, 188, 220], "Wrong observed GCC window")
    return {"evidence": str(path), "events": events, "log_sha256": sha(folder / "stderr.log"),
            "fft_hits": hits, "wave_sha256": trial["wav_sha256"], "xrun": trial["xrun"],
            "startup": trial["windows"]["startup"],
            "median_fft_cpu_ms": statistics.median(e["fft_cpu_ms"] for e in events),
            "median_gcc_wall_ms": statistics.median(e["gcc_wall_ms"] for e in events),
            "gcc_frames_over16": sum(e["over16"] for e in events)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--align", required=True, type=Path)
    parser.add_argument("--gcc-original", required=True, type=Path)
    parser.add_argument("--gcc-table", type=Path)
    a = parser.parse_args()
    original = gcc(a.gcc_original, table=False)
    result = {"align": abba(a.align), "original_gcc": original}
    if a.gcc_table:
        table = gcc(a.gcc_table, table=True)
        keys = ("n_seen", "frame", "keep", "conf", "shift", "locked")
        need(table["wave_sha256"] == original["wave_sha256"], "Table FFT changed PCM")
        need(all(all(old[k] == new[k] for k in keys) for old, new in zip(
            original["events"], table["events"], strict=True)), "Table FFT changed a GCC decision")
        table["all_six_fft_cpu_faster"] = all(new["fft_cpu_ms"] < old["fft_cpu_ms"] for old, new in zip(
            original["events"], table["events"], strict=True))
        table["median_fft_cpu_delta_pct"] = 100 * (table["median_fft_cpu_ms"] / original["median_fft_cpu_ms"] - 1)
        result["table_gcc"] = table
    out = new_dir("final-evidence") / "RESULT.json"
    save(out, result)
    print("RESULT", out)


if __name__ == "__main__":
    main()
