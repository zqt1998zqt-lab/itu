#!/usr/bin/env python3
"""Publish an identified current input scope and fresh continuous measurements."""
import argparse
import json
import shutil
from pathlib import Path

import run_audio_eval as ev
import check_audio_acceptance as ac
from run_vehicle_stream import VEHICLE_TIMELINE
from run_itut_suite import collect
from build_itut_ledger import write_ledger
from publish_itut_ledger import publish


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "array", "matrix", "scope", "publish-to", "ledger-to"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--frontend-to", type=Path)
    args = parser.parse_args()
    dataset, array, root = args.dataset.resolve(), args.array.resolve(), args.matrix.resolve()
    scope = json.loads(args.scope.read_text())
    report = json.loads((root / "report.json").read_text())
    manifest = json.loads((dataset / "inputs-manifest.json").read_text())
    spatial = json.loads((array / "inputs-manifest.json").read_text())
    labels = [label for label, _, _, _ in VEHICLE_TIMELINE]
    seconds = sum(end - begin for _, _, begin, end in VEHICLE_TIMELINE)
    if (not report["complete"] or report["identity"]["phase_labels"] != labels or
            report["identity"]["seconds_per_sequence"] != seconds or not report["paced"] or
            scope["manifest_sha256"] != ev.digest(dataset / "inputs-manifest.json") or
            scope["array_manifest_sha256"] != ev.digest(array / "inputs-manifest.json") or
            report["identity"]["manifest_sha256"] != scope["array_manifest_sha256"]):
        raise ValueError("scope or continuous matrix incomplete or changed")
    if any(s["noise"] == "car_passenger" for s in manifest["scenes"] + spatial["scenes"]):
        raise ValueError("excluded passenger speech remains in current inputs")
    if ev.digest(ac.RULES) != report["identity"]["rules_sha256"] or ev.digest(ac.RULES) != ac.RULES_SHA256:
        raise ValueError("engineering thresholds changed")
    if ev.digest(root / "itu_audio_cli") != report["identity"]["cli_sha256"]:
        raise ValueError("matrix CLI changed")
    for filename, checksum in report["identity"]["runner_sha256"].items():
        if ev.digest(Path(__file__).with_name(filename)) != checksum:
            raise ValueError(f"stream runner changed: {filename}")
    expected = {(rate, mics, variant) for rate in (16000, 24000, 48000) for mics in (1, 2, 4)
                for variant in (["control"] if mics == 1 else ["control", "spatial"])}
    rows = report["rows"]
    if len(rows) != len(expected) or {(r["rate"], r["microphones"], r["variant"]) for r in rows} != expected:
        raise ValueError("missing or duplicate matrix arm")
    jobs = []
    for row in rows:
        path = root / row["report"]
        measured = json.loads(path.read_text())
        if (row["phases"] != len(labels) * len(report["identity"]["pairs"]) or
                measured["identity"]["phase_labels"] != labels):
            raise ValueError("matrix phase coverage differs")
        jobs.append(dict(name=path.parent.name, kind="vehicle", report=str(path)))
        for pair in report["identity"]["pairs"]:
            stream = json.loads((path.parent / f"pair-{pair:02d}/stream-evaluation.json").read_text())
            folder = path.parent / f"pair-{pair:02d}"
            invocation = json.loads((folder / "invocation.json").read_text())
            if (invocation["exit_code"] != 0 or stream["identity"]["cli_sha256"] != report["identity"]["cli_sha256"] or
                    stream["identity"]["source_manifest_sha256"] != scope["array_manifest_sha256"] or
                    stream["identity"]["seconds"] != seconds or stream["engine"]["stream_resets"] != 0 or
                    [d["label"] for d in stream["details"]] != labels):
                raise ValueError("continuous execution differs from publication")
            for filename, checksum in {**stream["identity"]["files"], **stream["identity"]["output_files"]}.items():
                if ev.digest(folder / filename) != checksum:
                    raise ValueError("continuous input or output changed after measurement")
    identity = dict(report["identity"], source_manifest_sha256=scope["manifest_sha256"],
                    array_manifest_sha256=scope["array_manifest_sha256"], paced=False)
    ledger_root = root / "current-ledger"
    ledger_root.mkdir(exist_ok=False)
    summary = dict(identity=identity, **collect(jobs, [0] * len(jobs)), completed_jobs=len(jobs), planned_jobs=len(jobs))
    write(ledger_root / "summary.json", summary)
    write(ledger_root / "plan.json", identity)
    write_ledger(ledger_root, summary, root / report["paced"]["report"])
    publish(ledger_root, args.ledger_to.resolve())
    out = args.publish_to.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for source, name in ((args.scope, "scope.json"), (dataset / "inputs-manifest.json", "inputs-manifest.json"),
                         (root / "report.json", "matrix-report.json"), (dataset / "operations.jsonl", "operations.jsonl")):
        temporary = out / (name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(out / name)
    write(out / "publication.json", dict(source_dataset=str(dataset), array_dataset=str(array),
          matrix_root=str(root), standard_compliance=False, full_main_corpus_replayed=False,
          files={name: ev.digest(out / name) for name in ("scope.json", "inputs-manifest.json", "matrix-report.json", "operations.jsonl")}))
    if args.frontend_to:
        frontend = args.frontend_to.resolve()
        summary_path = frontend / "summary.json"
        previous = json.loads(summary_path.read_text())
        previous["sequence"]["streams"] = dict(phases=report["quality_phases"],
            quality_passed=report["quality_passed_phases"], seconds_per_sequence=seconds,
            paced=report["paced"], identity=report["identity"])
        write(frontend / "streams-report.json", report)
        write(summary_path, previous)
        publication_path = frontend / "publication.json"
        publication = json.loads(publication_path.read_text())
        publication["stream_source_root"] = str(root)
        publication["stream_scope"] = "fresh no-passenger continuous batch; independent CLI identity from field/native experiments"
        publication["files"].update({name: ev.digest(frontend / name) for name in ("streams-report.json", "summary.json")})
        write(publication_path, publication)
    print(json.dumps(dict(scenes=scope["scenes"], quality_phases=report["quality_phases"],
                         quality_passed=report["quality_passed_phases"], published=str(out)), indent=2))


if __name__ == "__main__":
    main()
