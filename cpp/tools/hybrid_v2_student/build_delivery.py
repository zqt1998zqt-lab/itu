#!/usr/bin/env python3
"""Build and independently verify the self-contained Hybrid-v2 delivery."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DELIVERY = ROOT / "delivery"
ARCHIVE = DELIVERY / "hybrid-v2.tar.gz"
RECEIPT = DELIVERY / "hybrid-v2-receipt.json"
CHECKSUM = DELIVERY / "hybrid-v2.sha256"

CORE = (
    "summarize_v2.py", "build_delivery.py", "verify_scored_provenance.py", "compute_paired_ci.py", "native/hybrid_v2_probe-fixed", "bin/itu_audio_cli",
    "student/__init__.py", "student/model.py", "student/export.py", "student/verify.py", "student/README.md",
    "train_v2.py", "training-contract.md", "contract.json", "prepare_expansion.py",
    "run_candidate.py", "run_suite.py", "score_parallel.py", "evaluation/common.py", "evaluation/dedupe_store.py",
)
TRAIN_NAMES = ("oldloss256", "newloss256", "newloss320")
TRAIN_FILES = ("best.pt", "run.json", "training.csv", "export/weights.bin", "export/step.onnx", "export/model.json")
DATA_JSON = (
    "data/manifest.json", "data/manifest-at-feature-cache.json", "data/final-receipt.json",
    "data/train-dev-receipt.json", "data/newtest-receipt.json", "data/newtest-stage-receipt.json",
    "data/source-receipt.json", "data/source-quality.json", "data/source-quality-newtest.json",
    "data/exclude-receipt.json", "data/cases-newtest16.json",
)
REPORT_FILES = (
    "reports/final.md", "reports/score-summary.json", "reports/stage-scores.csv", "reports/paired-deltas.csv",
    "reports/source-integration.json", "reports/quality-decision.json", "reports/recipe-hashes.json", "reports/final-binaries.json",
    "reports/versioned-sources.json", "reports/scored-provenance.json", "reports/final-coverage.json", "reports/paired-ci.json",
    "reports/native-correction.md", "reports/eval-coverage.json", "reports/trainer-maintenance.json",
)
FORBIDDEN_SUFFIXES = {".npz", ".f32", ".wav", ".flac", ".zip", ".pyc"}
FORBIDDEN_PARTS = {"__pycache__", "hardlinks", "cache", "npz", "native-inputs", "audio", "raw-noise"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def relative_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        relative = path.relative_to(PROJECT)
        return Path("source") / relative if relative.parts and relative.parts[0] == "cpp" else relative


def archive_path_for(path: Path) -> Path:
    if path.name == "run_manifest.json" and path.parent.name == "scored" and "evaluation" in path.parts:
        return Path("evidence/scored-manifests") / path.parents[4].name / path.parents[2].name / path.parents[1].name / path.name
    if path.name == "receipt.json" and path.parent.name == "candidate" and "evaluation" in path.parts:
        return Path("evidence/candidate-receipts") / path.parents[4].name / path.parents[2].name / path.parents[1].name / path.name
    if path.name == "receipt.json" and path.parent.name == "raw" and "new48-dpdf" in str(path):
        return Path("evidence/dpdf/raw-receipt.json")
    return relative_path(path)


def evidence_paths() -> tuple[list[Path], list[Path], list[Path]]:
    summary_path = ROOT / "reports/score-summary.json"
    if not summary_path.is_file():
        return [], [], []
    coverage = (load_json(summary_path).get("coverage") or {})
    manifests, receipts, extras = [], [], []
    for suite in coverage.get("active_suites", []):
        if not str(suite.get("id", "")).startswith(("nativefixed-", "new48-")):
            continue
        for raw_path in suite.get("manifests", []):
            manifest = Path(raw_path)
            if manifest.is_file() and manifest.name == "run_manifest.json":
                manifests.append(manifest)
                candidate = manifest.parent.parent / "candidate/receipt.json"
                if candidate.is_file():
                    receipts.append(candidate)
                elif "new48-dpdf" in str(manifest):
                    raw_receipt = manifest.parent.parent / "raw/receipt.json"
                    if raw_receipt.is_file():
                        extras.append(raw_receipt)
    return sorted(set(manifests)), sorted(set(receipts)), sorted(set(extras))


def add(paths: list[Path], path: Path, required: bool, missing: list[str], strict: bool) -> None:
    if path.is_file() and not path.is_symlink():
        if path not in paths:
            paths.append(path)
    elif required or strict:
        missing.append(str(relative_path(path)))


def selected_files(strict: bool) -> tuple[list[Path], list[str]]:
    paths, missing = [], []
    for name in CORE:
        add(paths, ROOT / name, True, missing, strict)
    for name in TRAIN_NAMES:
        for relative in TRAIN_FILES:
            add(paths, ROOT / "train" / name / relative, True, missing, strict)
    for name in DATA_JSON:
        add(paths, ROOT / name, True, missing, strict)
    add(paths, ROOT / "features/manifest.json", True, missing, strict)
    add(paths, ROOT / "features/new-rows.json", False, missing, strict)
    for path in sorted((ROOT / "features/native").glob("*.probe.json")):
        add(paths, path, False, missing, strict)
    for path in sorted((ROOT / "verification").rglob("*")):
        relative_parts = path.relative_to(ROOT / "verification").parts
        if path.is_file() and path.suffix.lower() in {".json", ".md"} and not (relative_parts[:2] == ("runtime", "raw")):
            add(paths, path, False, missing, strict)
    for path in sorted((ROOT / "verification/runtime").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".sha256"} and "raw" not in path.relative_to(ROOT / "verification/runtime").parts:
            add(paths, path, False, missing, strict)
    for name in REPORT_FILES:
        add(paths, ROOT / name, name in {
            "reports/final.md", "reports/score-summary.json", "reports/stage-scores.csv",
            "reports/paired-deltas.csv", "reports/source-integration.json",
            "reports/quality-decision.json", "reports/recipe-hashes.json", "reports/final-binaries.json",
            "reports/versioned-sources.json", "reports/scored-provenance.json", "reports/final-coverage.json",
            "reports/paired-ci.json",
        }, missing, strict)
    for path in sorted((ROOT / "reports/figures").glob("*.png")):
        add(paths, path, False, missing, strict)
    hybrid_doc_candidates = (PROJECT / "cpp/tools/HYBRID_V2.md", ROOT / "source/cpp/tools/HYBRID_V2.md")
    hybrid_docs = [path for path in hybrid_doc_candidates if path.is_file()][:1]
    if hybrid_docs:
        for path in hybrid_docs:
            add(paths, path, False, missing, strict)
    elif strict:
        missing.append("cpp/tools/HYBRID_V2.md")
    versioned_root = PROJECT / "cpp/tools/hybrid_v2_student"
    for path in sorted(versioned_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".json"}:
            add(paths, path, False, missing, strict)
    if strict and not any(versioned_root in path.parents for path in paths):
        missing.append("cpp/tools/hybrid_v2_student")
    provenance_paths = sorted(ROOT.glob("evaluation/hybridv2-nativefixed-*/variants/*/*/provenance/run-provenance.json"))
    provenance_paths += sorted(ROOT.glob("evaluation/hybridv2-new48-dpdf-*/references/*/*/provenance/run-provenance.json"))
    provenance_paths += sorted(ROOT.glob("evaluation/hybridv2-nativefixed-newloss320-pitch05-*/provenance/pitch05-rerun-provenance.json"))
    for path in provenance_paths:
        add(paths, path, False, missing, strict)
    if strict and len(provenance_paths) < 34:
        missing.append("evaluation provenance sidecars (expected 33 groups + pitch rerun)")
    evidence_manifests, evidence_receipts, evidence_extras = evidence_paths()
    for path in [*evidence_manifests, *evidence_receipts, *evidence_extras]:
        add(paths, path, True, missing, strict)
    if strict and len(evidence_manifests) != 34:
        missing.append(f"scored manifest evidence count {len(evidence_manifests)} != 34")
    if strict and len(evidence_receipts) != 33:
        missing.append(f"candidate receipt evidence count {len(evidence_receipts)} != 33")
    if strict and not evidence_extras:
        missing.append("DPDF raw receipt evidence is missing")
    report_path = ROOT / "reports/source-integration.json"
    if report_path.is_file():
        source_report = load_json(report_path)
        for item in source_report.get("files", []):
            if isinstance(item, dict) and item.get("path"):
                add(paths, PROJECT / item["path"], True, missing, strict)
        patch = source_patch_path(source_report)
        if patch:
            add(paths, patch, True, missing, strict)
    clean = []
    for path in paths:
        relative = relative_path(path)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not (path.suffix.lower() == ".pt" and path.name == "best.pt"):
            raise RuntimeError(f"forbidden delivery artifact: {relative}")
        if FORBIDDEN_PARTS.intersection(relative.parts):
            raise RuntimeError(f"forbidden delivery path: {relative}")
        if relative.as_posix() in {ARCHIVE.name, RECEIPT.name, CHECKSUM.name}:
            continue
        clean.append(path)
    return sorted(clean, key=lambda path: relative_path(path).as_posix()), sorted(set(missing))


def source_patch_path(report: dict) -> Path | None:
    value = report.get("patch_path") or report.get("patch")
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def source_integration(strict: bool) -> dict:
    path = ROOT / "reports/source-integration.json"
    if not path.is_file():
        if strict:
            raise FileNotFoundError(f"strict delivery requires {path.relative_to(ROOT)}")
        return {"status": "pending", "required_for_final": True, "files_verified": 0, "expected_file_count": None}
    report = load_json(path)
    file_items = report.get("files")
    hashes = report.get("file_hashes") or {}
    pairs = []
    if isinstance(file_items, dict):
        pairs = [(str(key), value.get("sha256") if isinstance(value, dict) else str(value)) for key, value in file_items.items()]
    elif isinstance(file_items, list):
        for item in file_items:
            if isinstance(item, dict):
                pairs.append((str(item.get("path")), item.get("sha256")))
            else:
                pairs.append((str(item), hashes.get(str(item))))
    else:
        pairs = [(str(name), hashes.get(str(name))) for name in report.get("changed_files", [])]
    unique = {}
    for relative, expected in pairs:
        if relative and expected:
            unique[relative] = str(expected).lower()
    expected_count = report.get("expected_file_count")
    if expected_count is None:
        expected_count = len(unique)
    expected_count = int(expected_count)
    errors = []
    if not unique:
        errors.append("source file list is empty")
    if len(unique) != expected_count:
        errors.append(f"source file count {len(unique)} != {expected_count}")
    for relative, expected in sorted(unique.items()):
        disk = PROJECT / relative
        actual = sha256_file(disk) if disk.is_file() else None
        if actual != expected:
            errors.append(f"source hash mismatch: {relative}: report={expected} disk={actual}")
    patch = source_patch_path(report)
    patch_sha = sha256_file(patch) if patch and patch.is_file() else None
    expected_patch_sha = report.get("patch_sha256") or ((report.get("patch") or {}).get("sha256") if isinstance(report.get("patch"), dict) else None)
    if not patch_sha:
        errors.append("source patch is missing")
    elif expected_patch_sha and patch_sha != expected_patch_sha:
        errors.append(f"source patch hash mismatch: report={expected_patch_sha} disk={patch_sha}")
    if report.get("status") not in {None, "complete", "pass"}:
        errors.append(f"source integration status is {report.get('status')!r}")
    if report.get("default_changed") is True:
        errors.append("default_changed must be false")
    if strict and errors:
        raise RuntimeError("source integration verification failed: " + "; ".join(errors))
    return {
        "status": "complete" if not errors else "pending", "required_for_final": True,
        "files_verified": len(unique) - len([item for item in errors if item.startswith("source hash mismatch")]),
        "expected_file_count": expected_count, "patch": str(patch.relative_to(ROOT)) if patch and patch.is_file() else None,
        "patch_sha256": patch_sha, "default_changed": bool(report.get("default_changed", False)), "errors": errors,
    }


def training_receipt() -> dict:
    result = {}
    all_trained = True
    for name in TRAIN_NAMES:
        run = load_json(ROOT / "train" / name / "run.json")
        model = load_json(ROOT / "train" / name / "export/model.json")
        trained = run.get("trained") is True and model.get("trained") is True and model.get("fixture") is False
        all_trained &= trained
        result[name] = {
            "trained": trained, "run_id": run.get("run_id"), "steps": run.get("steps"),
            "step_budget": run.get("step_budget"), "epochs_completed": run.get("epochs_completed"),
            "epochs_requested": run.get("epochs_requested"), "parameter_count": model.get("parameter_count"),
            "mac_per_second_at_100fps": model.get("mac_per_second_at_100fps"),
            "weights_sha256": model.get("weights_sha256"), "onnx_sha256": model.get("onnx_sha256"),
            "best_checkpoint_sha256": sha256_file(ROOT / "train" / name / "best.pt"),
        }
    return {"trained": all_trained, "models": result}


def quality_gain():
    path = ROOT / "reports/quality-decision.json"
    if not path.is_file():
        return None, None
    document = load_json(path)
    verified = document.get("quality_gain_verified")
    if verified is None:
        verified = document.get("quality_pass")
    if verified is None and document.get("status") in {"pass", "verified", "quality_pass"}:
        verified = True
    if verified is None:
        verified = document.get("decision") in {"pass", "verified", "quality_pass"}
    return bool(verified), sha256_file(path)


def versioned_sources(strict: bool) -> dict:
    report_path = ROOT / "reports/versioned-sources.json"
    if not report_path.is_file():
        if strict:
            raise FileNotFoundError(f"strict delivery requires {report_path.relative_to(ROOT)}")
        return {"status": "pending", "file_count": 0, "sha256": None}
    report = load_json(report_path)
    errors = []
    files = report.get("files") or {}
    for relative, item in files.items():
        path = PROJECT / relative
        actual = sha256_file(path) if path.is_file() else None
        expected = item.get("sha256") if isinstance(item, dict) else item
        if actual != expected:
            errors.append(f"versioned source hash mismatch: {relative}")
    if report.get("status") != "complete":
        errors.append(f"versioned source status is {report.get('status')!r}")
    if not files:
        errors.append("versioned source file list is empty")
    if any("c64_student" in str(relative) for relative in files):
        errors.append("old c64_student source leaked into v2 snapshot")
    if strict and errors:
        raise RuntimeError("versioned source verification failed: " + "; ".join(errors))
    return {"status": "complete" if not errors else "pending", "file_count": len(files),
            "sha256": sha256_file(report_path), "errors": errors}


def provenance_check(strict: bool) -> dict:
    groups = sorted(ROOT.glob("evaluation/hybridv2-nativefixed-*/variants/*/*/provenance/run-provenance.json"))
    groups += sorted(ROOT.glob("evaluation/hybridv2-new48-dpdf-*/references/*/*/provenance/run-provenance.json"))
    pitch_paths = sorted(ROOT.glob("evaluation/hybridv2-nativefixed-newloss320-pitch05-*/provenance/pitch05-rerun-provenance.json"))
    errors = []
    run_count = 0
    for path in groups:
        document = load_json(path)
        for run in document.get("runs", []):
            run_count += 1
            if run.get("upstream_input_exact") is not True or run.get("clean_used_for_processing") is not False:
                errors.append(f"provenance invariant failed: {path.relative_to(ROOT)}:{run.get('case_id')}")
    pitch_records = 0
    for path in pitch_paths:
        document = load_json(path)
        for run in document.get("records", []):
            pitch_records += 1
            if run.get("new_upstream_input_exact") is not True or run.get("new_clean_used_for_processing") is not False or run.get("new_status") != "ok":
                errors.append(f"pitch rerun invariant failed: {path.relative_to(ROOT)}:{run.get('case_id')}")
    if len(groups) != 33:
        errors.append(f"group provenance count {len(groups)} != 33")
    if len(pitch_paths) != 1:
        errors.append(f"pitch rerun provenance count {len(pitch_paths)} != 1")
    consolidated_path = ROOT / "reports/scored-provenance.json"
    consolidated = load_json(consolidated_path) if consolidated_path.is_file() else {}
    consolidated_ok = (
        consolidated.get("status") == "verified"
        and consolidated.get("scored_manifest_count") == 32
        and consolidated.get("runs") == 704
        and consolidated.get("pure_noise_runs") == 32
        and consolidated.get("all_upstream_input_exact") is True
        and consolidated.get("all_clean_used_for_processing_false") is True
        and consolidated.get("all_pure_noise_gsn_none") is True
        and consolidated.get("scorer_outputs_modified") is False
        and consolidated.get("probe_sha256") == "cca8c9ab5955b2bdde7912203c1c61a1ffdcbd45b124c26c78d952ab7540c152"
    )
    if not consolidated_ok:
        errors.append("consolidated scored provenance failed its 32-manifest/704-run invariants")
    if strict and errors:
        raise RuntimeError("provenance verification failed: " + "; ".join(errors[:20]))
    return {"status": "complete" if not errors else "pending", "group_sidecars": len(groups),
            "group_runs": run_count, "pitch_rerun_sidecars": len(pitch_paths), "pitch_rerun_records": pitch_records,
            "consolidated": {"path": "reports/scored-provenance.json", "sha256": sha256_file(consolidated_path) if consolidated_path.is_file() else None,
                             "status": consolidated.get("status"), "scored_manifest_count": consolidated.get("scored_manifest_count"),
                             "runs": consolidated.get("runs"), "pure_noise_runs": consolidated.get("pure_noise_runs")},
            "upstream_input_exact": not errors, "clean_used_for_processing": False, "errors": errors}


def native_correction_receipt() -> dict:
    record = ROOT / "reports/native-correction.md"
    recipe = load_json(ROOT / "reports/recipe-hashes.json") if (ROOT / "reports/recipe-hashes.json").is_file() else {}
    final_binaries = load_json(ROOT / "reports/final-binaries.json") if (ROOT / "reports/final-binaries.json").is_file() else {}
    return {
        "record": "reports/native-correction.md", "record_sha256": sha256_file(record) if record.is_file() else None,
        "active_probe_path": "native/hybrid_v2_probe-fixed", "active_probe_sha256": final_binaries.get("probe_sha256"),
        "invalid_prior_probe_sha256": recipe.get("invalid_prior_binary"), "invalid_prior_packaged": False,
        "projection_offset_corrected": final_binaries.get("projection_offset_corrected"),
    }


def coverage_receipt():
    path = ROOT / "reports/score-summary.json"
    if not path.is_file():
        return {"status": "pending", "complete": False, "sha256": None}
    document = load_json(path)
    coverage = document.get("coverage") or {}
    return {
        "status": document.get("status") or coverage.get("status") or "pending",
        "complete": bool(coverage.get("complete")),
        "sha256": sha256_file(path), "invalid_suites": document.get("invalid_suites", []),
    }


def parity_check(strict: bool) -> dict:
    final_path = ROOT / "reports/final-binaries.json"
    final_binaries = load_json(final_path) if final_path.is_file() else {}
    probe_path = Path(final_binaries.get("probe_path", ROOT / "native/hybrid_v2_probe-fixed"))
    binary = probe_path if probe_path.is_absolute() else ROOT / probe_path
    binary_sha = sha256_file(binary) if binary.is_file() else None
    declared_probe_sha = final_binaries.get("probe_sha256")
    cli_path = Path(final_binaries.get("cli_path", ROOT / "bin/itu_audio_cli"))
    cli = cli_path if cli_path.is_absolute() else ROOT / cli_path
    cli_sha = sha256_file(cli) if cli.is_file() else None
    checks = {}
    errors = []
    if declared_probe_sha and binary_sha != declared_probe_sha:
        errors.append(f"probe binary hash != final-binaries.json: {binary_sha} != {declared_probe_sha}")
    if final_binaries.get("cli_sha256") and cli_sha != final_binaries.get("cli_sha256"):
        errors.append(f"CLI binary hash != final-binaries.json: {cli_sha} != {final_binaries.get('cli_sha256')}")
    for name in TRAIN_NAMES:
        path = ROOT / "verification/root-fixed-parity" / f"{name}.json"
        document = load_json(path) if path.is_file() else {}
        ok = document.get("status") == "pass" and document.get("binary") == binary_sha
        checks[name] = {"path": str(path.relative_to(ROOT)), "status": document.get("status"), "binary_match": document.get("binary") == binary_sha}
        if not ok:
            errors.append(f"parity not verified: {name}")
    if strict and errors:
        raise RuntimeError("native/PyTorch parity failed: " + "; ".join(errors))
    return {"probe_path": str(binary.relative_to(ROOT)) if binary.is_relative_to(ROOT) else str(binary), "binary_sha256": binary_sha,
            "declared_probe_sha256": declared_probe_sha, "cli_sha256": cli_sha, "checks": checks, "pass": not errors, "errors": errors}


def payloads(files: list[Path]):
    result = []
    for path in files:
        archive_name = archive_path_for(path).as_posix()
        try:
            disk_relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            disk_relative = path.relative_to(PROJECT).as_posix()
        data = path.read_bytes()
        encoding = "identity"
        if (archive_name in {"reports/stage-scores.csv", "reports/paired-deltas.csv", "reports/score-summary.json", "reports/scored-provenance.json", "features/manifest.json"}
                or archive_name.startswith("evidence/")):
            data = gzip.compress(data, compresslevel=9, mtime=0)
            archive_name += ".gz"
            encoding = "gzip"
        result.append({
            "source_path": disk_relative, "archive_path": archive_name, "source_sha256": sha256_bytes(path.read_bytes()),
            "source_bytes": path.stat().st_size, "payload_sha256": sha256_bytes(data), "payload_bytes": len(data),
            "encoding": encoding, "data": data,
        })
    return result


def tar_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def write_archive(items: list[dict], receipt: dict) -> None:
    receipt_data = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    DELIVERY.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                archive.addfile(tar_info("receipt.json", len(receipt_data)), io.BytesIO(receipt_data))
                for item in items:
                    mode = 0o755 if item["archive_path"] in {"native/hybrid_v2_probe-fixed", "bin/itu_audio_cli"} else 0o644
                    archive.addfile(tar_info(item["archive_path"], len(item["data"]), mode), io.BytesIO(item["data"]))


def verify_archive(items: list[dict], receipt: dict) -> dict:
    archive_sha = sha256_file(ARCHIVE)
    expected = {item["archive_path"]: item for item in items}
    with tarfile.open(ARCHIVE, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise AssertionError("duplicate archive members")
        if any(not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts for member in members):
            raise AssertionError("unsafe archive member")
        modes = {member.name: member.mode for member in members}
        for name in ("native/hybrid_v2_probe-fixed", "bin/itu_audio_cli"):
            if modes.get(name) != 0o755:
                raise AssertionError(f"binary mode mismatch: {name} is {oct(modes.get(name, 0))}")
        if any(mode != 0o644 for name, mode in modes.items() if name not in {"native/hybrid_v2_probe-fixed", "bin/itu_audio_cli"}):
            raise AssertionError("non-binary archive member mode is not 0644")
        for name, item in expected.items():
            data = archive.extractfile(name).read()
            if sha256_bytes(data) != item["payload_sha256"]:
                raise AssertionError(f"archive member hash mismatch: {name}")
        packaged_receipt = json.loads(archive.extractfile("receipt.json").read())
        if packaged_receipt.get("artifact_count") != len(items):
            raise AssertionError("embedded receipt artifact count mismatch")
    CHECKSUM.write_text(f"{archive_sha}  {ARCHIVE.name}\n", encoding="utf-8")
    return {"archive_sha256": archive_sha, "archive_bytes": ARCHIVE.stat().st_size, "member_count": len(items) + 1,
            "binary_modes": {"native/hybrid_v2_probe-fixed": "0755", "bin/itu_audio_cli": "0755"},
            "other_modes": "0644", "verified": True}


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    files, missing = selected_files(args.strict)
    integration = source_integration(args.strict)
    versioned = versioned_sources(args.strict)
    provenance = provenance_check(args.strict)
    training = training_receipt()
    coverage = coverage_receipt()
    quality, quality_sha = quality_gain()
    parity = parity_check(args.strict)
    native_correction = native_correction_receipt()
    if args.strict and not training["trained"]:
        raise RuntimeError("strict delivery requires trained=true for all three models")
    if args.strict and coverage.get("status") != "complete":
        raise RuntimeError(f"strict delivery requires complete score summary, got {coverage.get('status')!r}")
    ready = (not missing and integration["status"] == "complete" and versioned["status"] == "complete"
             and provenance["status"] == "complete" and coverage.get("complete") is True and parity["pass"])
    items = payloads(files)
    receipt = {
        "schema_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if ready else "pending", "allow_partial": args.allow_partial,
        "trained": training["trained"], "models": training["models"],
        "default_changed": False, "implementation_complete": ready,
        "quality_gain_verified": quality, "quality_decision_sha256": quality_sha,
        "source_error": None if not integration.get("errors") else integration.get("errors"),
        "coverage": coverage, "source_integration": integration, "versioned_sources": versioned,
        "provenance": provenance, "native_parity": parity, "native_correction": native_correction,
        "missing": missing, "artifact_count": len(items),
        "excludes": ["teacher checkpoints", "last.pt", "full stage audio", "raw-noise zip", "NPZ", "f32", "__pycache__", "cache"],
        "manifest_format": "archive_path -> {source_path, source_sha256, source_bytes, payload_sha256, payload_bytes, encoding}",
        "artifacts": {item["archive_path"]: {key: item[key] for key in (
            "source_path", "source_sha256", "source_bytes", "payload_sha256", "payload_bytes", "encoding")} for item in items},
    }
    write_archive(items, receipt)
    receipt["package"] = verify_archive(items, receipt)
    atomic_json(RECEIPT, receipt)
    print(json.dumps({
        "status": receipt["status"], "artifacts": len(items), "missing": len(missing),
        "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "sha256": receipt["package"]["archive_sha256"], "quality_gain_verified": quality,
        "coverage": coverage.get("status"), "source_integration": integration["status"],
    }, ensure_ascii=False))
    return 0 if receipt["status"] == "complete" or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
