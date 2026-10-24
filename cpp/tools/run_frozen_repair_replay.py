#!/usr/bin/env python3
"""Replay existing PCM with a separately frozen CLI and run-scoped case identity."""
import os

# One thread per worker: numpy/OpenBLAS defaults to one thread per core in every
# worker, so --jobs 8 on a 20-core host became 160 competing threads. Keep the
# per-worker BLAS pool pinned unless the caller overrides it explicitly.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import run_controlled_native_scenes as ctl
import run_audio_eval as evaluator

PROJECT = Path(__file__).resolve().parents[2]
SOURCES = (("controlled-native-full-20261017", "itu-controlled-quality-v1-20261017"),
           ("controlled-cavemove-20261017", "itu-cavemove-quality-v1-20261017"))


def select_cases(scope):
    requested = set()
    if scope == "representatives":
        for area in ("ablations", "noise-controls"):
            p = PROJECT / "build/failure-cause-analysis-20261017" / area / "selection.json"
            requested.update((Path(s["run"]).name, s["id"]) for s in json.loads(p.read_text()))
    selected = []
    for run, manifest in SOURCES:
        source = PROJECT / "build" / run
        core = {s["id"] for s in json.loads((PROJECT / "datasets" / manifest / "manifest.json").read_text())["scenes"]
                if s["track"] == "spectrum_screened_core"}
        for line in (source / "results.jsonl").open():
            row = json.loads(line)
            scene = row["scene"]
            if scene["id"] not in core:
                continue
            if scope == "representatives" and (run, scene["id"]) not in requested:
                continue
            if scope == "agc" and scene["family"] != "agc":
                continue
            if scope == "screen" and not (scene["family"] != "ns" or scene.get("snr_db") in (-10, 0, 10, 20, 30)):
                continue
            selected.append((run, row))
    if not selected or len({(run, row["scene"]["id"]) for run, row in selected}) != len(selected):
        raise ValueError("empty or duplicate run-scoped selection")
    if scope == "representatives" and len(selected) != len(requested):
        raise ValueError("representative selection not completely resolved")
    return selected


def replay(task):
    root, run, original, cli, flags, stages, paced = task
    scene = original["scene"]
    origin = PROJECT / "build" / run / "cases" / scene["id"]
    local = root / run
    folder = local / "cases" / scene["id"]
    folder.mkdir(parents=True)
    for name, meta in scene["files"].items():
        source = origin / name
        if ctl.ev.digest(source) != meta["sha256"]:
            raise ValueError(f"input identity differs: {source}")
        (folder / name).symlink_to(source.resolve())
    if (origin / "oracle").is_dir():
        (folder / "oracle").symlink_to((origin / "oracle").resolve(), target_is_directory=True)
    ctl.legacy.save(folder / "scene.json", scene)
    prefix = folder / "runs/full/result"
    command = ctl.command_for(scene, folder, prefix, str(cli), paced=paced) + flags
    if stages:
        command += ["--diagnostic-stages"]
    ctl.legacy.invoke(command, prefix, ctl.ev.digest(cli))
    result = ctl.legacy.measure(scene, local, "full")
    result["original_run"] = run
    result["original_status"] = original["acceptance"]["status"]
    result["baseline_output_sha256"] = original["metrics"]["output_sha256"]
    result["same_output_as_original"] = result["metrics"]["output_sha256"] == result["baseline_output_sha256"]
    return result


def admission_summary(
    results,
    *,
    identity_verified=True,
    changed_dependencies=(),
    diagnostic_stages=False,
    expected_gain_mode=None,
):
    transitions = Counter(
        row["original_status"] + "->" + row["acceptance"]["status"]
        for row in results
    )
    fixes = sum(
        row["original_status"] == "fail" and row["acceptance"]["status"] == "pass"
        for row in results
    )
    regressions = sum(
        row["original_status"] == "pass" and row["acceptance"]["status"] == "fail"
        for row in results
    )
    same_output = sum(row["same_output_as_original"] for row in results)
    config_errors = []
    for row in results:
        engine = row["metrics"]["engine"]
        if diagnostic_stages:
            if engine.get("diagnostic_stages") is not True:
                config_errors.append("AGC stage taps missing")
                break
        if expected_gain_mode == "speech":
            expected = (True, True, False)
            actual = (
                engine.get("agc2_enabled"),
                engine.get("agc2_speech_gain_policy"),
                engine.get("agc2_adaptive_enabled"),
            )
            if actual != expected:
                config_errors.append(f"AGC speech mode differs: {actual}")
                break
        elif expected_gain_mode == "adaptive":
            expected = (True, True, True)
            actual = (
                engine.get("agc2_enabled"),
                engine.get("agc2_speech_gain_policy"),
                engine.get("agc2_adaptive_enabled"),
            )
            if actual != expected:
                config_errors.append(f"AGC adaptive mode differs: {actual}")
                break
        elif expected_gain_mode == "fixed":
            expected = (True, False, False)
            actual = (
                engine.get("agc2_enabled"),
                engine.get("agc2_speech_gain_policy"),
                engine.get("agc2_adaptive_enabled"),
            )
            if actual != expected:
                config_errors.append(f"AGC fixed mode differs: {actual}")
                break
        elif expected_gain_mode == "off":
            if engine.get("agc2_enabled"):
                config_errors.append("AGC off mode enabled AGC2")
                break
    if not identity_verified:
        status = "blocked_identity"
    elif config_errors:
        status = "blocked_config"
    elif regressions:
        status = "blocked_regressions"
    elif same_output == len(results):
        status = "blocked_no_effect"
    else:
        status = "eligible"
    return dict(
        fixes=fixes,
        regressions=regressions,
        transitions=transitions,
        same_output_as_original=same_output,
        candidate_eligible=status == "eligible",
        admission_status=status,
        identity_verified=identity_verified,
        changed_dependencies=list(changed_dependencies),
        config_errors=config_errors,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--scope", choices=("representatives", "agc", "screen", "core"), default="representatives")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--expected-gain-mode", choices=("speech", "adaptive", "fixed", "off"))
    args, flags = parser.parse_known_args()
    if args.jobs < 1 or args.jobs > 32 or (args.paced and args.jobs != 1):
        raise ValueError("jobs must be 1..32, paced execution requires jobs=1")
    root = args.root.resolve()
    selected = select_cases(args.scope)
    root.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir()
    cli = root / "artifacts/itu_audio_cli"
    shutil.copy2(args.cli, cli)
    deps = [Path(__file__), Path(ctl.__file__), Path(ctl.legacy.__file__), Path(ctl.ev.__file__),
            Path(ctl.legacy.acceptance.__file__), ctl.legacy.acceptance.RULES]
    for run, manifest in SOURCES:
        deps += [PROJECT / "build" / run / "results.jsonl",
                 PROJECT / "datasets" / manifest / "manifest.json"]
    bindings = {str(p): ctl.ev.digest(p) for p in deps}
    for p in deps[:6]:
        shutil.copy2(p, root / "artifacts" / p.name)
    ctl.legacy.save(root / "identity.json", dict(cli_sha256=ctl.ev.digest(cli), dependencies=bindings,
        flags=flags, scope=args.scope, stages=args.stages, paced=args.paced))
    ctl.legacy.save(root / "selection.json", [dict(run=run, id=row["scene"]["id"]) for run, row in selected])
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (root / "results.jsonl").open("x") as stream:
        tasks = [(root, run, row, cli, flags, args.stages, args.paced) for run, row in selected]
        for future in as_completed([pool.submit(replay, task) for task in tasks]):
            row = future.result()
            results.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(results) % 50 == 0 or len(results) == len(selected):
                print(json.dumps(dict(completed=len(results), total=len(selected))), flush=True)
    changed = [p for p, expected in bindings.items() if ctl.ev.digest(Path(p)) != expected]
    counts = Counter(row["acceptance"]["status"] for row in results)
    admission = admission_summary(
        results,
        identity_verified=not changed,
        changed_dependencies=changed,
        diagnostic_stages=args.stages,
        expected_gain_mode=args.expected_gain_mode,
    )
    ctl.legacy.save(root / "report.json", dict(state="completed", cases=len(results), counts=counts,
        **admission,
        elapsed_seconds=time.monotonic() - started,
        scope=args.scope, flags=flags, paced=args.paced, standard_compliance=False))
    if changed:
        raise ValueError(f"dependencies changed: {changed}")
    print(json.dumps(dict(cases=len(results), counts=counts, **admission)), flush=True)


if __name__ == "__main__":
    main()
