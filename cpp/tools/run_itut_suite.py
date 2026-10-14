#!/usr/bin/env python3
"""One audited vehicle regression matrix; process success is separate from acceptance."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import run_audio_eval as ev
import check_audio_acceptance as ac
from run_vehicle_stream import select_timeline, sequence_beam_delays
from build_itut_ledger import write_ledger
from publish_itut_ledger import publish

TOOLS = Path(__file__).resolve().parent
RUNNERS = ("run_itut_suite.py", "run_vehicle_regression.py", "run_vehicle_stream.py",
           "run_simulated_clock.py", "run_audio_eval.py", "check_audio_acceptance.py", "build_itut_ledger.py", "publish_itut_ledger.py")


def runner_identity():
    return {name: ev.digest(TOOLS / name) for name in RUNNERS}


def build_jobs(dataset, cli, root, rates, microphones, pairs, radio=None, array=None, pace=False, eq=None):
    jobs = []
    for count in microphones:
        source = array if count > 1 else dataset
        if source is None:
            raise ValueError("2/4 microphones require a separate array dataset, not duplicated mono channels")
        manifest = json.loads((source / "inputs-manifest.json").read_text())
        scenes = list(manifest["scenes"])
        if radio and count == 1:
            scenes += json.loads((radio / "inputs-manifest.json").read_text())["scenes"]
        for pair in pairs:
            sequence_beam_delays(select_timeline(scenes, pair, count), count)
        for rate in rates:
            name = f"vehicle-{rate}-{count}ch"
            command = [sys.executable, str(TOOLS / "run_vehicle_regression.py"), "--dataset", str(source),
                       "--root", str(root / name), "--cli", str(cli), "--jobs", "1" if pace else "2",
                       "--microphones", str(count), "--sample-rate", str(rate),
                       "--pairs", ",".join(map(str, pairs)), "--simulate-clock"]
            if radio and count == 1:
                command += ["--radio-dataset", str(radio)]
            if pace:
                command += ["--pace"]
            jobs.append(dict(name=name, kind="vehicle", command=command, report=str(root / name / "regression-summary.json")))
            if eq:
                tuned = list(command)
                tuned[tuned.index("--root") + 1] = str(root / (name + "-eq"))
                tuned += [f"--cli-option={value}" for value in eq]
                jobs.append(dict(name=name + "-eq", kind="vehicle", command=tuned,
                                 report=str(root / (name + "-eq") / "regression-summary.json")))
    if radio:
        name = "clock-paired"
        jobs.append(dict(name=name, kind="clock", command=[sys.executable, str(TOOLS / "run_simulated_clock.py"),
                         "--dataset", str(radio), "--root", str(root / name), "--cli", str(cli),
                         "--pair", str(pairs[0]), "--workers", "1" if pace else "3"] + (["--pace"] if pace else []),
                         report=str(root / name / "report.json")))
    return jobs


def collect(jobs, outcomes):
    rows = []
    for job, result in zip(jobs, outcomes, strict=True):
        path = Path(job["report"])
        if not path.is_file():
            rows.append(dict(name=job["name"], process_exit_code=result, status="execution_error", acceptance_passed=False))
            continue
        try:
            report = json.loads(path.read_text())
            passed = report["all_available_passed"] if job["kind"] == "vehicle" else report["passed"]
            if not isinstance(passed, bool):
                raise ValueError("acceptance must be boolean")
            if job["kind"] == "vehicle":
                complete = report["required_measurements_complete"]
                phases, passed_phases = report["phases"], report["passed_phases"]
                if not isinstance(complete, bool) or type(phases) is not int or phases <= 0 or type(passed_phases) is not int or not 0 <= passed_phases <= phases:
                    raise ValueError("complete nonempty phase measurements required")
                passed = passed and complete and passed_phases == phases
        except (ValueError, KeyError, TypeError) as error:
            rows.append(dict(name=job["name"], process_exit_code=result, status="invalid_report",
                             acceptance_passed=False, report=str(path), error=str(error)))
            continue
        rows.append(dict(name=job["name"], process_exit_code=result,
                         status="engineering_pass" if passed and result == 0 else "engineering_fail",
                         acceptance_passed=passed and result == 0, report_sha256=ev.digest(path), report=str(path),
                         passed_phases=report.get("passed_phases"), phases=report.get("phases"),
                         failed_gate_counts=report.get("failed_gate_counts", {}), failures=report.get("failures", [])))
    return dict(jobs=rows, all_engineering_passed=bool(rows) and all(x["acceptance_passed"] for x in rows),
                standard_compliance=False, formal_test_status="not_measured",
                rule="missing reports, failed processes and failed gates all remain failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--radio-dataset", type=Path)
    parser.add_argument("--array-dataset", type=Path)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-rates", type=int, nargs="+", choices=(16000, 24000, 48000), default=[16000, 24000, 48000])
    parser.add_argument("--microphones", type=int, nargs="+", choices=(1, 2, 4), default=[1])
    parser.add_argument("--pairs", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--pace", action="store_true", help="serialize timing runs at real 10 ms cadence")
    parser.add_argument("--eq-option", action="append", default=[], help="optional paired EQ flags only")
    parser.add_argument("--execute", action="store_true", help="default writes a validated plan only")
    parser.add_argument("--publish-to", type=Path, help="publish audited reports to a local website directory after completion")
    args = parser.parse_args()
    for value in args.eq_option:
        if value.startswith("--") and value not in ("--eq-band", "--eq-preamp-db"):
            parser.error("only EQ options allowed in the paired tuning arm")
    for values in (args.sample_rates, args.microphones, args.pairs):
        if len(values) != len(set(values)):
            parser.error("duplicate matrix selections")
    source, cli, root = args.dataset.resolve(), args.cli.resolve(), args.root.resolve()
    if ev.digest(ac.RULES) != ac.RULES_SHA256:
        parser.error("frozen engineering rules changed")
    radio = args.radio_dataset.resolve() if args.radio_dataset else None
    array = args.array_dataset.resolve() if args.array_dataset else None
    # Validate every selection before creating a partial suite.
    jobs = build_jobs(source, cli, root, args.sample_rates, args.microphones, args.pairs, radio, array, args.pace, args.eq_option)
    root.mkdir(parents=True, exist_ok=False)
    artifact = root / "itu_audio_cli"
    shutil.copy2(cli, artifact)
    for job in jobs:
        job["command"][job["command"].index("--cli") + 1] = str(artifact)
    identity = dict(cli_sha256=ev.digest(artifact), rules_sha256=ev.digest(ac.RULES),
                    source_manifest_sha256=ev.digest(source / "inputs-manifest.json"),
                    radio_manifest_sha256=ev.digest(radio / "inputs-manifest.json") if radio else None,
                    array_manifest_sha256=ev.digest(array / "inputs-manifest.json") if array else None,
                    alignment_sha256=ev.digest(TOOLS.parent / "evaluation/itut-alignment-v1.json"),
                    contract_sha256=ev.digest(TOOLS.parent / "evaluation/nv200-test-contract.json"),
                    runners_sha256=runner_identity(),
                    runner_sha256=ev.digest(Path(__file__)), paced=args.pace, jobs=jobs,
                    scope="continuous vehicle + paired clock; host simulation, not acoustic certification")
    (root / "plan.json").write_text(json.dumps(identity, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(dict(plan=str(root / "plan.json"), jobs=len(jobs), executed=False), indent=2))
        return 0
    outcomes = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for job in jobs:
        if runner_identity() != identity["runners_sha256"] or ev.digest(ac.RULES) != identity["rules_sha256"]:
            raise ValueError("suite runner or rules changed during execution")
        for path, key in ((source, "source_manifest_sha256"), (radio, "radio_manifest_sha256"), (array, "array_manifest_sha256")):
            if path and ev.digest(path / "inputs-manifest.json") != identity[key]:
                raise ValueError(f"suite input manifest changed: {key}")
        if ev.digest(artifact) != identity["cli_sha256"]:
            raise ValueError("suite frozen executable changed")
        print(f"running {job['name']}", flush=True)
        try:
            with (root / (job["name"] + ".log")).open("x") as log:
                result = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3600)
            outcomes.append(result.returncode)
        except subprocess.TimeoutExpired:
            outcomes.append(124)
        except OSError:
            outcomes.append(127)
        summary = dict(identity=identity, **collect(jobs[:len(outcomes)], outcomes))
        summary["completed_jobs"] = len(outcomes)
        summary["planned_jobs"] = len(jobs)
        (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        write_ledger(root, summary)
    print(json.dumps(dict(all_engineering_passed=summary["all_engineering_passed"], standard_compliance=False,
                         report=str(root / "summary.json")), indent=2))
    if args.publish_to:
        publish(root, args.publish_to.resolve())
    return 0 if summary["all_engineering_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
