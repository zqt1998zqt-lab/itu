"""Independent graph, PCM, and timing checks for R2."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import wave

from build_r2 import BASE_MODES, BUILD, R1, ROOT, guard, need, new_dir, read, save, sha

NODES = {"norm": (29, 0, 4, 32, 37, 23), "norm_align": (29, 1, 3, 32, 37, 23)}
INVENTORY = re.compile(
    r"^itu_fusion_r2 mode=(\S+) norm=(\d+) align=(\d+) sum_rows=(\d+) "
    r"im2col=(\d+) mul_mat=(\d+) silu=(\d+) accepted=(\d+)$", re.M)
TAPS = (".bf.wav", ".aec-input.wav", ".aec.wav", ".after.wav")
HOP_NS = 16_000_000
FROZEN_WAVS = R1 / "host-accept-20261118T044539984688Z/RESULT.json"
FROZEN_WAVS_SHA = "47e95f28d5f34c8415fedf3636a8e372b0781a43423bc016364dbc8e39ec771f"
MIC = R1 / "state-host.GZNmgZ/mic-prefix.wav"
REF = BUILD / "localvqe-opt-20261117/probe-double_talk-20261117T172221778317/collected/ref.wav"
MIC_SHA = "06c31b6f62d7ddec1dd64098251a859dc14a5435c8f4f2ecd0f62d8eabf6ebbd"
REF_SHA = "84c5f7df978f0d85c8c809f61db91c694f402639562ed234ed178a4d81a313d1"


def graph_gate(log: str, mode: str) -> list[tuple[str, ...]]:
    graphs = INVENTORY.findall(log)
    need(bool(graphs) and all(g[0] == mode and tuple(map(int, g[1:])) == (*NODES[mode], 1)
                             for g in graphs), f"Graph identity mismatch for {mode}")
    return graphs


def manifest(path: Path, platform: str) -> dict:
    m = read(path)
    need(m["schema"] == "itu-lvq-fusion-r2-build" and m["platform"] == platform
         and m["modes"] == (["norm"] if m["gcc_profile"] else list(BASE_MODES)),
         "Unexpected R2 artifact")
    for key in ("runtime", "binary", "state_runner", "diagnostic"):
        if m[key]:
            need(sha(Path(m[key]["path"])) == m[key]["sha256"], f"Build identity drift: {key}")
    for model, location in m["model_paths"].items():
        need(sha(Path(location)) == m["model_sha256"][model], f"Model identity drift: {model}")
    return m


def stats(ns: list[int]) -> dict:
    a = sorted(ns)
    return {**{f"p{k}_ms": a[math.ceil(f * len(a)) - 1] / 1e6
               for k, f in (("95", .95), ("99", .99), ("99_9", .999))},
            "mean_ms": statistics.fmean(a) / 1e6, "max_ms": a[-1] / 1e6}


def frames(prefix: Path) -> list[dict]:
    with prefix.with_suffix(".audio-clock.frames.csv").open(newline="") as f:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]


def qualify(prefix: Path, build: dict, mode: str, stderr: str, input_frames: int,
            *, phone: bool) -> dict:
    graph = graph_gate(stderr, mode)
    samples = input_frames * 256
    report = read(prefix.with_suffix(".json"))
    expected = {"input_samples_per_channel": samples, "dsp_delay_samples": 513, "frame_size": 256,
                "aec_provider_sha256": build["runtime"]["sha256"],
                "aec_model_sha256": build["model_sha256"]["localvqe"],
                "aec_localvqe_inferences": input_frames + 2, "ns_inference_calls": input_frames + 3,
                "aec_localvqe_fallback_frames": 0, "ns_external_fallback_frames": 0,
                "native16_in_process_ort": True}
    need(all(report.get(k) == v for k, v in expected.items()), f"DSP contract: {prefix}")
    with wave.open(str(prefix.with_suffix(".after.wav")), "rb") as w:
        need((w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()) ==
             (16000, samples + 513, 1, 2), f"PCM extent: {prefix}")

    rows = frames(prefix)
    side = read(prefix.with_suffix(".audio-clock.json"))
    need(len(rows) == input_frames + 3 == side["callbacks"] and side["clock_errors"] == 0
         and side["hardware_audio_clock_executed"] is phone and not side["mock_clock"]
         and not side["performance_hint"]["requested"], f"Clock contract: {prefix}")
    elapsed = []
    for i, row in enumerate(rows):
        start, ready, done = (row[k] for k in ("callback_start_ns", "input_ready_ns", "completed_ns"))
        need(row["frame"] == i and row["clocks_valid"] == 1 and start <= ready <= done <= row["collected_ns"]
             and row["process_wall_ns"] == done - ready and row["process_thread_cpu_ns"] >= 0,
             f"Invalid frame {i}: {prefix}")
        elapsed.append(done - start)

    stream = side["audio_stream"] if phone else None
    if phone:
        need(stream.get("error") == "none" and all(stream.get(k) is True for k in
             ("completed", "closed", "stop_clean", "callback_quiesced", "requested_modes_met"))
             and (stream["actual_sample_rate"], stream["actual_channel_count"], stream["actual_format"],
                  stream["actual_frames_per_callback"], stream["actual_frames_per_burst"],
                  stream["actual_buffer_size_frames"], stream["actual_sharing_mode_name"],
                  stream["actual_performance_mode_name"]) ==
                 (48000, 1, 2, 768, 192, 1536, "shared", "low_latency")
             and stream["xrun_delta"] == stream["xrun_count_after"] - stream["xrun_count_before"] >= 0,
             f"Unexpected AAudio stream: {prefix}")
        origin = rows[0]["callback_start_ns"]
        need(all(r["expected_ns"] == origin + i * HOP_NS and r["completion_late"] ==
                 int(r["completed_ns"] > r["expected_ns"] + HOP_NS) for i, r in enumerate(rows)),
             f"Cadence ledger: {prefix}")

    def window(start: int, stop: int) -> dict:
        section = rows[start:stop]
        return {"n": len(section), "wall": stats([r["process_wall_ns"] for r in section]),
                "cpu": stats([r["process_thread_cpu_ns"] for r in section]),
                "callback_to_complete": stats(elapsed[start:stop]),
                "over16": sum(t > HOP_NS for t in elapsed[start:stop]),
                "first_start_grid_late": sum(r["completion_late"] for r in section)}

    return {"mode": mode, "graphs": graph, "input_frames": input_frames, "dsp": expected,
            "wav_sha256": {s: sha(prefix.with_suffix(s)) for s in TAPS},
            "windows": {"all": window(0, input_frames), "startup": window(0, 250),
                        "steady": window(250, input_frames), "drain": window(input_frames, len(rows))},
            "xrun": stream["xrun_delta"] if phone else None}


def host(args) -> None:
    build = manifest(args.manifest, "host")
    need(not build["gcc_profile"] and sha(FROZEN_WAVS) == FROZEN_WAVS_SHA and
         sha(MIC) == MIC_SHA and sha(REF) == REF_SHA, "Frozen host reference drift")
    old = read(FROZEN_WAVS)["quality_runs"]
    out = new_dir("host-accept")
    print("EVIDENCE", out, flush=True)

    def run(label: str, argv: list[str], env: dict, exitcode: int = 0) -> tuple[Path, str, str]:
        guard(out)
        d = out / label
        d.mkdir()
        save(d / "COMMAND.json", {"argv": argv, "env": env})
        with (d / "stdout").open("xb") as stdout, (d / "stderr").open("xb") as stderr:
            proc = subprocess.run(argv, cwd=ROOT, env={**os.environ, **env},
                                  stdout=stdout, stderr=stderr, timeout=90)
        save(d / "EXITCODE", str(proc.returncode) + "\n")
        need(proc.returncode == exitcode, f"Unexpected exit: {d}")
        return d, (d / "stdout").read_text(), (d / "stderr").read_text()

    model = build["model_paths"]["localvqe"]
    lib = build["diagnostic"]["path"]
    for mode, check, status in [("norm", "exact", 0), ("norm_align", "exact", 0),
                                ("norm_align", "negative", 3)]:
        _, stdout, stderr = run(f"state-{mode}-{check}", [build["state_runner"]["path"], lib, lib,
            model, str(MIC), str(REF), mode, check], {}, status)
        graph_gate(stderr, "norm") if mode == "norm" else need(
            any(x[0] == mode and tuple(map(int, x[1:])) == (*NODES[mode], 1)
                for x in INVENTORY.findall(stderr)), "State probe did not load candidate")
        message = (f"STATE_PASS mode={mode} frames=750 tensors=77 bytes_per_frame=1353104 pcm_samples=192000"
                   if check == "exact" else "STATE_MISMATCH mode=norm_align frame=12 first_state_byte=128 pcm_exact=1")
        need(message in stdout + stderr, "State contract mismatch")
        print(message, flush=True)

    results = {}
    for scene in read(BUILD / "localvqe-opt-20261117/INPUTS.json")["cases"]:
        name = scene["condition"]
        previous, = (x for x in old[name] if x["mode"] == "norm")
        prior_command = read(Path(previous["directory"]) / "COMMAND.json")
        need({key: sha(Path(spec["path"])) for key, spec in scene["files"].items()}
             == previous["inputs"], f"Frozen scene drift: {name}")
        results[name] = []
        for mode in BASE_MODES:
            label = name + "-" + mode
            prefix = out / label / "out"
            argv = list(prior_command["argv"])
            argv[0] = build["binary"]["path"]
            argv[argv.index("--out") + 1] = str(prefix)
            _, _, stderr = run(label, argv, {**prior_command["env"], "ITU_LVQ_FUSION": mode,
                                            "ITU_LOCALVQE_LIBRARY": build["runtime"]["path"]})
            result = qualify(prefix, build, mode, stderr, 750, phone=False)
            need(result["wav_sha256"] == previous["wav_sha256"], f"WAV mismatch: {name}/{mode}")
            results[name].append(result)
            print("PCM_PASS", name, mode, flush=True)
    need(len(results) == 5, "Expected five distinct frozen scenes")
    guard(out)
    save(out / "RESULT.json", {"manifest": str(args.manifest.resolve()), "state_pass": True,
                               "state_negative_rejected": True, "wav_comparisons": 40, "runs": results})
    print("PASS", out / "RESULT.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    host(p.parse_args())
