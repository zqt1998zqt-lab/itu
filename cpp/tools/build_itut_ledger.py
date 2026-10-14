#!/usr/bin/env python3
"""Join workbook tests, physical evidence gaps and measured simulation results."""
import argparse
import collections
import json
from pathlib import Path

import run_audio_eval as ev

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"


def build_ledger(contract, alignment, summary):
    items = []
    for test in contract["items"]:
        required = alignment["required_evidence"][test["family"]]
        items.append(dict(id=test["id"], sheet=test["sheet"], source_cell=test["source_cell"],
                          test_name=test["test_name"], family=test["family"], direction=test["direction"],
                          bandwidth=test["bandwidth"], transport=test["transport"], unit=test["unit"],
                          lower_limit=test["lower_limit"], upper_limit=test["upper_limit"],
                          limit_source=test["limit_source"], status="not_measured",
                          missing_evidence=required,
                          engineering_proxy=alignment["engineering_proxies"].get(test["family"]),
                          historical_result_is_current=False))
    jobs = []
    for source_job in summary.get("jobs", []):
        job = dict(source_job)
        path = Path(job["report"]) if job.get("report") else None
        if path and path.is_file() and job.get("report_sha256"):
            if ev.digest(path) != job["report_sha256"]:
                raise ValueError(f"simulation report changed: {job['name']}")
            evidence = json.loads(path.read_text())
            job["quality_passed_phases"] = evidence.get("quality_passed_phases")
            job["required_measurements_complete"] = evidence.get("required_measurements_complete")
        jobs.append(job)
    complete = bool(jobs) and summary.get("completed_jobs") == summary.get("planned_jobs") == len(jobs)
    return dict(schema_version=1, standard_compliance=False, formal_test_status="not_measured",
                workbook_sha256=contract["source_sha256"], standards=alignment["standards"],
                formal_counts=dict(collections.Counter(item["status"] for item in items)),
                family_counts=dict(collections.Counter(item["family"] for item in items)),
                items=items, simulation_identity=summary.get("identity", {}),
                engineering=dict(complete=complete, passed_jobs=sum(bool(j.get("acceptance_passed")) for j in jobs),
                                 jobs=len(jobs), all_passed=complete and all(j.get("acceptance_passed") for j in jobs),
                                 results=jobs),
                online_adaptation=alignment["online_adaptation"],
                forbidden_online_inputs=alignment["forbidden_online_inputs"],
                completion_rule=alignment["formal_completion_rule"])


def write_ledger(root, summary, paced_report=None):
    contract_path = EVALUATION / "nv200-test-contract.json"
    alignment_path = EVALUATION / "itut-alignment-v1.json"
    report = build_ledger(json.loads(contract_path.read_text()), json.loads(alignment_path.read_text()), summary)
    report.update(contract_sha256=ev.digest(contract_path), alignment_sha256=ev.digest(alignment_path))
    methods_path = EVALUATION / "p1120-2025-methods.json"
    report["official_method_evidence"] = json.loads(methods_path.read_text())
    report["official_method_sha256"] = ev.digest(methods_path)
    masks = report["official_method_evidence"]["frequency_response_masks"]
    report["sampling_coverage"] = [dict(sample_rate=rate, service=service,
        spectrum_covers_required_range=rate / 2 > mask["measurement_max_hz"],
        native_highband_verified=False, formal_status="not_measured")
        for rate in (16000, 24000, 48000) for service, mask in masks.items()]
    if paced_report:
        measured = json.loads(paced_report.read_text())
        if (not measured["engine"]["paced_10ms"] or
                measured["identity"]["cli_sha256"] != summary["identity"]["cli_sha256"] or
                measured["identity"]["rules_sha256"] != summary["identity"]["rules_sha256"]):
            raise ValueError("paced evidence identity differs or lacks real cadence")
        report["paced_diagnostic"] = dict(report=str(paced_report), report_sha256=ev.digest(paced_report),
            sample_rate=measured["identity"]["sample_rate"], microphones=measured["identity"]["microphones"],
            seconds=measured["identity"]["seconds"], engine_resets=measured["identity"]["engine_resets"],
            process_p95_ms=measured["engine"]["process_p95_ms"], process_max_ms=measured["engine"]["process_max_ms"],
            completion_deadline_misses=measured["engine"]["completion_deadline_misses"],
            passed_phases=sum(item["passed"] for item in measured["details"]), phases=len(measured["details"]),
            standard_compliance=False, simulated_clock="simulated_clock" in measured["identity"])
    path = root / "itut-ledger.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["# ITU-T alignment ledger", "", "Formal status: **not measured**. Simulation is not certification.", "",
             f"Workbook rows: {len(report['items'])}; engineering jobs: {report['engineering']['passed_jobs']}/{report['engineering']['jobs']} passed.", "",
             "| Family | Workbook rows | Missing physical/method evidence |", "| --- | ---: | --- |"]
    for family, count in sorted(report["family_counts"].items()):
        missing = next(item["missing_evidence"] for item in report["items"] if item["family"] == family)
        lines.append(f"| {family} | {count} | {'; '.join(missing)} |")
    lines += ["", "| Simulation job | Acceptance | All gates phases | Quality phases |", "| --- | --- | --- | --- |"]
    for job in report["engineering"]["results"]:
        lines.append(f"| {job['name']} | {job['status']} | {job.get('passed_phases')}/{job.get('phases')} | {job.get('quality_passed_phases')}/{job.get('phases')} |")
    lines += ["", "Detailed workbook cells, units, customer limits, artifact identities and failures: `itut-ledger.json`.", ""]
    if report.get("paced_diagnostic"):
        lines += ["Separate paced evidence:", "", json.dumps(report["paced_diagnostic"], indent=2), ""]
    (root / "itut-ledger.md").write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--paced-report", type=Path)
    args = parser.parse_args()
    print(write_ledger(args.suite.resolve(), json.loads((args.suite / "summary.json").read_text()), args.paced_report))


if __name__ == "__main__":
    main()
