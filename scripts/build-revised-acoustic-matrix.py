#!/usr/bin/env python3
"""Build the 48-store revision in isolated processes, retaining every log."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True


def resource_guard():
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if shutil.disk_usage(PROJECT).free < 100 * 1024**3 or available < 5 * 1024**3:
        raise RuntimeError("resource floor crossed; stop without deleting anything")


def main():
    from server.acoustic.acceptance_scope import ACCEPTANCE_MODES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--kind", choices=("scene", "stress", "all"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    resource_guard()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    logs = args.logs or PROJECT / "build/dataset-revision-20261117" / f"generation-{stamp}"
    logs.mkdir(parents=True, exist_ok=False)
    tasks = []
    stopped = threading.Event()
    for mode in sorted(ACCEPTANCE_MODES):
        if args.kind in {"scene", "all"}:
            tasks.extend(("scene", scene, mode) for scene in ("sce1", "sce2", "sce3", "sce4"))
        if args.kind in {"stress", "all"}:
            tasks.extend(("stress", scope, mode) for scope in ("echo", "double_talk"))

    def run(task):
        kind, condition, mode = task
        if stopped.is_set():
            return {"kind": kind, "condition": condition, "mode": mode,
                    "exit_code": 75, "log": None}
        resource_guard()
        command = [sys.executable, str(PROJECT / "scripts/create-revised-acoustic-datasets-v2.py"),
                   "--kind", kind, "--scene" if kind == "scene" else "--scope", condition,
                   "--mode", mode]
        log = logs / f"{kind}-{condition}-{mode}.log"
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    resource_guard()
                except RuntimeError:
                    stopped.set()
                if stopped.is_set():
                    process.terminate()
                    process.wait()
                    return {"kind": kind, "condition": condition, "mode": mode,
                            "exit_code": 75, "log": str(log)}
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        if process.returncode == 75 or (process.returncode and any(
            token in log.read_text(encoding="utf-8") for token in ("ENOSPC", "No space left on device", "ResourceSafetyStop")
        )):
            stopped.set()
        return {"kind": kind, "condition": condition, "mode": mode,
                "exit_code": process.returncode, "log": str(log)}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(tasks)} exit={result['exit_code']} "
                  f"{result['kind']}/{result['condition']}/{result['mode']}", flush=True)
    report = {"tasks": len(tasks), "passed": sum(row["exit_code"] == 0 for row in results),
              "results": results}
    resource_guard()
    with (logs / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}) + f" logs={logs}")
    return 0 if report["passed"] == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
