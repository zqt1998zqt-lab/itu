#!/usr/bin/env python3
"""Run two bounded AEC configuration probes on the frozen 27-scene factorial."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import shutil
import time
import traceback
from pathlib import Path

import run_controlled_native_scenes as ctl


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "build/failure-cause-analysis-20261017/aec-factorial"
CLI_SOURCE = PROJECT / "build/repair-sequential-20261017/baseline/itu_audio_cli"
ROOT = PROJECT / "build/repair-sequential-20261017/aec-config-probe"
MODE = "aec_config_probe"
VARIANTS = (
    ("filter_blocks_128", ["--aec-filter-blocks", "128"]),
    ("filter_blocks_128_fast_adaptation", [
        "--aec-filter-blocks", "128", "--aec-fast-adaptation"
    ]),
)
FORBIDDEN_FLAGS = {
    "--no-reference-guard",
    "--no-linear-speech-protection",
    "--aec-delay",
    "--known-delay",
    "--delay",
}


def digest(path: Path) -> str:
    return ctl.ev.digest(path)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            rows.append(json.loads(line))
    return rows


def acceptance_map(acceptance: dict) -> dict[tuple[str, str], dict]:
    return {
        (gate["segment"], gate["metric"]): gate
        for gate in acceptance["gates"]
    }


def near_dt(segment: str) -> bool:
    return segment.startswith("near") or segment == "double_talk"


def metric_values(measured: dict) -> dict:
    segments = measured["metrics"]["segments"]
    far = segments.get("far_only", {})
    return {
        "far_only_total_attenuation_db": far.get("total_attenuation_db"),
        "far_only_linear_erle_engineering_db": far.get("linear_erle_engineering_db"),
        "acceptance_status": measured["acceptance"].get("status"),
    }


def new_near_dt_failures(baseline: dict, candidate: dict) -> list[dict]:
    old = acceptance_map(baseline)
    new = acceptance_map(candidate)
    failures = []
    for key, gate in new.items():
        segment, metric = key
        if not near_dt(segment) or gate.get("status") != "fail":
            continue
        prior = old.get(key)
        if prior is None or prior.get("status") != "fail":
            failures.append({
                "segment": segment,
                "metric": metric,
                "baseline_status": None if prior is None else prior.get("status"),
                "candidate_status": gate.get("status"),
                "value": gate.get("value"),
                "threshold": gate.get("threshold"),
                "operator": gate.get("operator"),
            })
    return failures


def prepare_case(source_case: Path, target_case: Path, scene: dict) -> None:
    target_case.mkdir(parents=True, exist_ok=False)
    for name in scene["files"]:
        source_file = source_case / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        (target_case / name).symlink_to(source_file.resolve())
    source_oracle = source_case / "oracle"
    if source_oracle.is_dir():
        (target_case / "oracle").symlink_to(source_oracle.resolve(), target_is_directory=True)


def run_one(args: tuple) -> dict:
    (variant, config_args, scene, baseline_acceptance, baseline_metrics,
     variant_root, cli, cli_sha) = args
    case_id = scene["id"]
    folder = variant_root / "cases" / case_id
    prefix = folder / "runs" / MODE / "result"
    command = ctl.command_for(scene, folder, prefix, str(cli))
    command.extend(config_args)
    prohibited = sorted(FORBIDDEN_FLAGS.intersection(command))
    if prohibited:
        return {
            "variant": variant,
            "scene_id": case_id,
            "command": command,
            "error": "prohibited command flags: " + ",".join(prohibited),
            "traceback": None,
        }
    try:
        ctl.legacy.invoke(command, prefix, cli_sha)
        measured = ctl.legacy.measure(scene, variant_root, MODE)
        return {
            "variant": variant,
            "config_args": config_args,
            "scene": scene,
            "command": command,
            "baseline_acceptance": baseline_acceptance,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": measured["metrics"],
            "candidate_acceptance": measured["acceptance"],
            "candidate_values": metric_values(measured),
            "new_near_dt_failures": new_near_dt_failures(
                baseline_acceptance, measured["acceptance"]
            ),
            "error": None,
        }
    except Exception as exc:  # Record every attempted scene and continue the bounded matrix.
        return {
            "variant": variant,
            "config_args": config_args,
            "scene_id": case_id,
            "scene": scene,
            "baseline_acceptance": baseline_acceptance,
            "baseline_metrics": baseline_metrics,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def audit_scene_inputs(rows: list[dict], *, phase: str) -> dict[str, dict]:
    audited = {}
    for row in rows:
        scene = row["scene"]
        case_id = scene["id"]
        source_case = SOURCE / "cases" / case_id
        scene_file = source_case / "scene.json"
        disk_scene = json.loads(scene_file.read_text())
        if disk_scene != scene:
            raise ValueError(f"scene.json differs from results.jsonl scene: {case_id}")
        files = {}
        for name, meta in sorted(scene["files"].items()):
            path = source_case / name
            actual = digest(path)
            if actual != meta["sha256"]:
                raise ValueError(f"input SHA-256 mismatch: {case_id}/{name}")
            if path.stat().st_size != meta["bytes"]:
                raise ValueError(f"input size mismatch: {case_id}/{name}")
            files[name] = {"expected_sha256": meta["sha256"], "actual_sha256": actual,
                           "bytes": meta["bytes"]}
        rx_input = Path(scene.get("rx_input", ""))
        if rx_input and rx_input.is_file():
            if digest(rx_input) != scene["files"]["rx-input.wav"]["sha256"]:
                raise ValueError(f"duplex RX input hash mismatch: {case_id}")
        audited[case_id] = {
            "phase": phase,
            "scene_json_sha256": digest(scene_file),
            "input_file_count": len(files),
            "input_files": files,
            "verified": True,
        }
    return audited


def summarize(records: list[dict], baseline_by_id: dict[str, dict]) -> dict:
    result = {"variants": {}, "baseline_same_scene": {}}
    for variant, _ in VARIANTS:
        rows = [r for r in records if r["variant"] == variant]
        errors = [r for r in rows if r.get("error")]
        okay = [r for r in rows if not r.get("error")]
        near_failures = [failure for row in okay for failure in row["new_near_dt_failures"]]
        near_fail_scenes = sorted({
            row["scene"]["id"] for row in okay if row["new_near_dt_failures"]
        })
        far_count = sum(
            row["candidate_values"]["far_only_total_attenuation_db"] is not None
            and row["candidate_values"]["far_only_total_attenuation_db"] >= 15.0
            for row in okay
        )
        erle_count = sum(
            row["candidate_values"]["far_only_linear_erle_engineering_db"] is not None
            and row["candidate_values"]["far_only_linear_erle_engineering_db"] >= 6.0
            for row in okay
        )
        groups = defaultdict(list)
        for row in rows:
            if "scene" in row:
                scene = row["scene"]
                groups[(scene["sample_rate"], scene["rt60_s"])].append(row)
        group_rows = []
        for (rate, rt60), group in sorted(groups.items()):
            valid = [r for r in group if not r.get("error")]
            group_rows.append({
                "sample_rate": rate,
                "rt60_s": rt60,
                "scenes": len(group),
                "completed": len(valid),
                "far_ge_15_count": sum(
                    r["candidate_values"]["far_only_total_attenuation_db"] is not None
                    and r["candidate_values"]["far_only_total_attenuation_db"] >= 15.0
                    for r in valid
                ),
                "linear_erle_ge_6_count": sum(
                    r["candidate_values"]["far_only_linear_erle_engineering_db"] is not None
                    and r["candidate_values"]["far_only_linear_erle_engineering_db"] >= 6.0
                    for r in valid
                ),
                "near_dt_new_failure_gate_count": sum(
                    len(r["new_near_dt_failures"]) for r in valid
                ),
                "near_dt_new_failure_scene_count": sum(
                    bool(r["new_near_dt_failures"]) for r in valid
                ),
            })
        result["variants"][variant] = {
            "scenes": len(rows),
            "completed": len(okay),
            "errors": len(errors),
            "far_ge_15_count": far_count,
            "far_ge_15_denominator": len(okay),
            "linear_erle_ge_6_count": erle_count,
            "linear_erle_ge_6_denominator": len(okay),
            "near_dt_new_failure_gate_count": len(near_failures),
            "near_dt_new_failure_scene_count": len(near_fail_scenes),
            "near_dt_new_failure_scenes": near_fail_scenes,
            "near_dt_new_failures": near_failures,
            "by_sample_rate_rt60": group_rows,
        }

    base_rows = list(baseline_by_id.values())
    baseline_near_failures = []
    baseline_far = baseline_erle = 0
    for row in base_rows:
        segments = row["metrics"]["segments"]
        far = segments.get("far_only", {})
        attenuation = far.get("total_attenuation_db")
        erle = far.get("linear_erle_engineering_db")
        baseline_far += attenuation is not None and attenuation >= 15.0
        baseline_erle += erle is not None and erle >= 6.0
        for gate in row["acceptance"]["gates"]:
            if near_dt(gate["segment"]) and gate.get("status") == "fail":
                baseline_near_failures.append({
                    "scene_id": row["scene"]["id"],
                    "segment": gate["segment"],
                    "metric": gate["metric"],
                    "value": gate.get("value"),
                    "threshold": gate.get("threshold"),
                })
    result["baseline_same_scene"] = {
        "scenes": len(base_rows),
        "far_ge_15_count": baseline_far,
        "linear_erle_ge_6_count": baseline_erle,
        "near_dt_failed_gate_count": len(baseline_near_failures),
        "near_dt_failed_gates": baseline_near_failures,
    }
    return result


def main() -> int:
    if ROOT.exists():
        raise FileExistsError(f"output already exists: {ROOT}")
    if not CLI_SOURCE.is_file():
        raise FileNotFoundError(CLI_SOURCE)
    source_rows = read_jsonl(SOURCE / "results.jsonl")
    if len(source_rows) != 27 or len({r["scene"]["id"] for r in source_rows}) != 27:
        raise ValueError(f"expected 27 unique fixed scenes, found {len(source_rows)}")

    source_identity_path = SOURCE / "identity.json"
    source_identity = json.loads(source_identity_path.read_text())
    cli_sha = digest(CLI_SOURCE)
    factorial_cli = SOURCE / "artifacts/itu_audio_cli"
    if cli_sha != source_identity["cli_sha256"] or cli_sha != digest(factorial_cli):
        raise ValueError("current baseline CLI SHA-256 differs from factorial CLI identity")
    rules = Path(ctl.legacy.acceptance.RULES).resolve()
    rules_sha = digest(rules)
    if rules_sha != ctl.legacy.RULES_SHA:
        raise ValueError("acceptance rules digest differs from legacy frozen digest")
    changed_dependencies = []
    for name, expected in source_identity["dependencies"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            changed_dependencies.append(name)
    if changed_dependencies:
        raise ValueError("factorial measurement dependencies changed: " + ",".join(changed_dependencies))

    baseline_by_id = {r["scene"]["id"]: r for r in source_rows}
    inputs_pre = audit_scene_inputs(source_rows, phase="before")
    rates = Counter(r["scene"]["sample_rate"] for r in source_rows)
    rt60s = Counter(r["scene"]["rt60_s"] for r in source_rows)
    if rates != {16000: 9, 24000: 9, 48000: 9} or rt60s != {0.15: 9, 0.45: 9, 0.6: 9}:
        raise ValueError(f"unexpected fixed-scene grid: rates={dict(rates)}, rt60={dict(rt60s)}")

    ROOT.mkdir(parents=True, exist_ok=False)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir()
    copied_cli = artifacts / "itu_audio_cli"
    shutil.copy2(CLI_SOURCE, copied_cli)
    if digest(copied_cli) != cli_sha:
        raise ValueError("copied CLI SHA-256 mismatch")
    copied_rules = artifacts / "acceptance-v1.json"
    shutil.copy2(rules, copied_rules)
    if digest(copied_rules) != rules_sha:
        raise ValueError("copied acceptance rules SHA-256 mismatch")

    tool_paths = [Path(ctl.__file__).resolve(), Path(ctl.legacy.__file__).resolve(),
                  Path(ctl.ev.__file__).resolve(), Path(ctl.legacy.acceptance.__file__).resolve(),
                  Path(__file__).resolve()]
    dependency_hashes = {str(p): digest(p) for p in tool_paths}
    dependency_hashes[str(rules)] = rules_sha
    dependency_hashes[str(CLI_SOURCE.resolve())] = cli_sha
    dependency_hashes[str(factorial_cli.resolve())] = digest(factorial_cli)
    dependency_hashes[str(source_identity_path.resolve())] = digest(source_identity_path)
    dependency_hashes[str((SOURCE / "results.jsonl").resolve())] = digest(SOURCE / "results.jsonl")
    identity = {
        "probe_script": str(Path(__file__).resolve()),
        "probe_script_sha256": digest(Path(__file__).resolve()),
        "cli_path": str(CLI_SOURCE.resolve()),
        "cli_sha256": cli_sha,
        "factorial_cli_path": str(factorial_cli.resolve()),
        "factorial_cli_sha256": digest(factorial_cli),
        "source_factorial_identity_path": str(source_identity_path.resolve()),
        "source_factorial_identity_sha256": digest(source_identity_path),
        "source_factorial_results_sha256": digest(SOURCE / "results.jsonl"),
        "rules_path": str(rules),
        "rules_sha256": rules_sha,
        "copied_rules_sha256": digest(copied_rules),
        "factorial_dependencies_verified": True,
        "factorial_dependency_count": len(source_identity["dependencies"]),
        "dependencies": dependency_hashes,
    }
    save_json(ROOT / "identity.json", identity)
    scene_plan = [{
        "id": row["scene"]["id"],
        "sample_rate": row["scene"]["sample_rate"],
        "rt60_s": row["scene"]["rt60_s"],
        "echo_delay_ms": row["scene"]["echo_delay_ms"],
        "input_file_count": len(row["scene"]["files"]),
        "baseline_acceptance_status": row["acceptance"]["status"],
    } for row in source_rows]
    plan = {
        "source": str(SOURCE.resolve()),
        "cli_sha256": cli_sha,
        "rules_sha256": rules_sha,
        "max_jobs": 2,
        "total_cli_invocations": 54,
        "variants": [{"name": name, "arguments": args} for name, args in VARIANTS],
        "reference_guard_disabled": False,
        "linear_speech_protection_disabled": False,
        "known_delay_argument_added": False,
        "scenes": scene_plan,
    }
    save_json(ROOT / "plan.json", plan)

    variant_roots = {}
    jobs = []
    for variant, config_args in VARIANTS:
        variant_root = ROOT / "variants" / variant
        variant_root.mkdir(parents=True)
        variant_roots[variant] = variant_root
        for row in source_rows:
            scene = row["scene"]
            source_case = SOURCE / "cases" / scene["id"]
            case_folder = variant_root / "cases" / scene["id"]
            prepare_case(source_case, case_folder, scene)
            jobs.append((variant, config_args, scene, row["acceptance"], row["metrics"],
                         variant_root, copied_cli, cli_sha))

    started_utc = datetime.now(timezone.utc).isoformat()
    started_mono = time.monotonic()
    save_json(ROOT / "report.json", {
        "state": "running",
        "started_utc": started_utc,
        "cases_expected": 54,
        "max_jobs": 2,
        "identity_sha256": digest(ROOT / "identity.json"),
    })
    records = []
    results_path = ROOT / "results.jsonl"
    with results_path.open("x") as stream:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    record = future.result()
                except Exception as exc:
                    job = jobs[futures.index(future)]
                    record = {
                        "variant": job[0], "config_args": job[1],
                        "scene_id": job[2]["id"], "command": None,
                        "error": f"worker {type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                print(json.dumps({
                    "completed": index,
                    "total": len(jobs),
                    "variant": record.get("variant"),
                    "scene": record.get("scene_id", record.get("scene", {}).get("id")),
                    "status": "error" if record.get("error") else "ok",
                }), flush=True)

    inputs_post = audit_scene_inputs(source_rows, phase="after")
    for case_id in inputs_pre:
        if inputs_pre[case_id]["input_files"] != inputs_post[case_id]["input_files"]:
            raise ValueError(f"input SHA-256 changed during probe: {case_id}")
        inputs_post[case_id]["verified_before_and_after"] = True
    save_json(ROOT / "input-hashes.json", inputs_post)

    summary = summarize(records, baseline_by_id)
    save_json(ROOT / "summary.json", summary)
    invocation_rows = []
    for variant, _ in VARIANTS:
        for row in source_rows:
            path = (variant_roots[variant] / "cases" / row["scene"]["id"] /
                    "runs" / MODE / "invocation.json")
            if path.is_file():
                invocation_rows.append(json.loads(path.read_text()))
    exit_counts = Counter(str(row.get("exit_code")) for row in invocation_rows)
    errors = [r for r in records if r.get("error")]
    changed_dependencies = [
        path for path, expected in identity["dependencies"].items()
        if not Path(path).is_file() or digest(Path(path)) != expected
    ]
    copied_artifact_mismatch = (
        digest(copied_cli) != cli_sha or digest(copied_rules) != rules_sha
    )
    finished_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "state": "completed" if len(records) == 54 and not errors and len(invocation_rows) == 54
                 and exit_counts == {"0": 54} and not changed_dependencies
                 and not copied_artifact_mismatch else "completed_with_errors",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": round(time.monotonic() - started_mono, 3),
        "records_written": len(records),
        "cli_invocation_records": len(invocation_rows),
        "cli_exit_code_counts": dict(exit_counts),
        "execution_errors": len(errors),
        "errors": [{"variant": r.get("variant"), "scene_id": r.get("scene_id"),
                    "error": r.get("error")} for r in errors],
        "changed_dependencies": changed_dependencies,
        "copied_artifact_mismatch": copied_artifact_mismatch,
        "input_scenes_verified_before_and_after": len(inputs_post),
        "input_files_verified_before_and_after": sum(x["input_file_count"] for x in inputs_post.values()),
        "max_jobs": 2,
        "identity_sha256": digest(ROOT / "identity.json"),
        "results_jsonl_sha256": digest(results_path),
        "summary": summary,
    }
    save_json(ROOT / "report.json", report)
    print(json.dumps({
        "state": report["state"],
        "records_written": report["records_written"],
        "cli_invocation_records": report["cli_invocation_records"],
        "cli_exit_code_counts": report["cli_exit_code_counts"],
        "execution_errors": report["execution_errors"],
        "elapsed_seconds": report["elapsed_seconds"],
    }), flush=True)
    return 0 if report["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
