#!/usr/bin/env python3
"""Publish only checked reports, never raw audio or historical acceptance replacements."""
import argparse
import json
import shutil
from pathlib import Path

import run_audio_eval as ev


def publish(suite, out):
    ledger = suite / "itut-ledger.json"
    report = json.loads(ledger.read_text())
    if report["standard_compliance"] is not False or report["formal_test_status"] != "not_measured":
        raise ValueError("this publisher supports simulation ledgers only")
    for job in report["engineering"]["results"]:
        if job.get("report_sha256") and ev.digest(Path(job["report"])) != job["report_sha256"]:
            raise ValueError(f"report changed: {job['name']}")
    if report.get("paced_diagnostic"):
        paced = report["paced_diagnostic"]
        if ev.digest(Path(paced["report"])) != paced["report_sha256"]:
            raise ValueError("paced report changed")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("itut-ledger.json", "itut-ledger.md", "summary.json", "plan.json"):
        shutil.copy2(suite / name, out / name)
    index = dict(source_suite=str(suite), standard_compliance=False,
                 files={name: ev.digest(out / name) for name in ("itut-ledger.json", "itut-ledger.md", "summary.json", "plan.json")})
    (out / "publication.json").write_text(json.dumps(index, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    publish(args.suite.resolve(), args.out.resolve())
    print(args.out)


if __name__ == "__main__":
    main()
