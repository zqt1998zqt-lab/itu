#!/usr/bin/env python3
"""Analyze the frozen paired minima/history propagation probe outputs.

This tool reads the observer's stages.csv/stages.bin files only. It does not
invoke the candidate or replay any audio. The binary file is loaded once per
role and interpreted as little-endian float32; CSV offsets are byte offsets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_SIDS = (
    "field-p0-m2-cold-snr15",
    "field-p0-m4-cold-snr0",
)
EXPECTED_ROLES = ("flagoff", "flagon")
EXPECTED_HOPS = tuple(range(60))
EXPECTED_BINS = 257
EXPECTED_TRACE_BYTES = 1_537_280
EXPECTED_PROTOCOL_SHA256 = "f72a9620cd981e19cf3a8a6dbba9609ec8a0646cde9e254ecbbb73c003e1ec88"
EXPECTED_IDENTITY_LOCK_SHA256 = "7989da6265e59ce9594b63c0a7b49a598dee7c1d1bea86d3618f482727f19663"
ANALYSIS_PLAN_SHA256 = "bcf2e82a89edbb07e55b7ec65389c42e9a19d89d1f44f2f82beb439b83f02213"
FROZEN_DECISION = {
    "candidate_status": "rejected",
    "candidate_replayed_by_independent_review": False,
    "actual_executed_scenes": ["field-p0-m2-cold-snr15", "field-p0-m4-cold-snr0"],
    "planned_but_unexecuted_scene": "field-p0-m4-cold-snr15",
    "not_run_stages": ["controls12", "onset6", "staged24", "full72"],
    "terminal_failure": {
        "scene_id": "field-p0-m4-cold-snr0",
        "window": "cold_start",
        "references": ["joint_v2", "state_v1"],
        "sisdr_delta_db": -1.1208428809377153,
        "stoi_delta": -0.08039659380934419,
        "limits": {"sisdr_delta_db_min": -0.1, "stoi_delta_min": -0.02},
    },
    "rule": "This trace analysis cannot reverse the rejection, relax a quality gate, or imply that a measured internal effect repairs the failed full-chain window.",
}
REMOTE_Q = PurePosixPath(
    "/home/zhangqingtian/project/itu/build/vcp-joint-structure-20261024/"
    "luna-minima-propagation-20261024"
)
EXPECTED_WET_SHA256 = {
    ("field-p0-m2-cold-snr15", "flagoff"): "3c1cd9fe5dbc7e03761e3b00f454e5c84492dafa0343ab8a3a7be364c8810cf8",
    ("field-p0-m2-cold-snr15", "flagon"): "40a44059b01214c34e18cd82a69bcc27b0fc58b79da03e49565ffecf89d9549f",
    ("field-p0-m4-cold-snr0", "flagoff"): "93e192fbe63053520b25c8c6b206e10c6d404d2fb16a50728884ef401c851abc",
    ("field-p0-m4-cold-snr0", "flagon"): "b7590681866c770fba6fe6c926bf6f888b55f17ad13740244fd98f428df0a446",
}
INDEX_COLUMNS = ("hop", "stage", "field", "offset", "count")
METRIC_SOURCES = {
    "raw_psd_ratio_db": ("before_estimation", "noise_raw", 10.0),
    "gamma_ratio_db": ("posterior", "gamma", 10.0),
    "xi_ratio_db": ("posterior", "xi", 10.0),
    "gain_delta_db": ("gain", "gain", 20.0),
}


class AnalysisError(RuntimeError):
    """Raised when observer output violates the frozen schema or scope."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def validate_probe_provenance(root: Path) -> dict[str, Any]:
    """Gate analysis on the frozen protocol, source lock, and four wet checks."""
    q = root.parent
    protocol_path = q / "protocol.json"
    identity_path = q / "identity-lock.json"
    lock_path = q / "probe-lock.json"
    report_path = q / "probe-report.json"
    for path in (protocol_path, identity_path, lock_path, report_path):
        if not path.is_file():
            raise AnalysisError(f"missing provenance gate file: {path}")

    protocol_sha = sha256_file(protocol_path)
    identity_sha = sha256_file(identity_path)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise AnalysisError(f"frozen protocol hash mismatch: {protocol_sha}")
    if identity_sha != EXPECTED_IDENTITY_LOCK_SHA256:
        raise AnalysisError(f"identity-lock hash mismatch: {identity_sha}")

    protocol = _load_json(protocol_path)
    protocol_cases = protocol.get("cases")
    if not isinstance(protocol_cases, list) or len(protocol_cases) != len(EXPECTED_SIDS):
        raise AnalysisError("frozen protocol must contain exactly the two approved cases")
    protocol_by_sid: dict[str, dict[str, Any]] = {}
    external_expectations: dict[str, dict[str, Any]] = {}
    for case in protocol_cases:
        if not isinstance(case, dict):
            raise AnalysisError("frozen protocol contains a malformed case")
        sid = case.get("scene_id")
        if sid not in EXPECTED_SIDS or sid in protocol_by_sid:
            raise AnalysisError(f"frozen protocol has unexpected or duplicate case {sid!r}")
        if case.get("hops") != 2402 or case.get("sample_rate") != 16000 or case.get("hop_samples") != 160:
            raise AnalysisError(f"frozen protocol has unexpected stream dimensions for {sid}")
        input_sha = case.get("input_sha256")
        if not isinstance(input_sha, str) or len(input_sha) != 64:
            raise AnalysisError(f"frozen protocol input hash is malformed for {sid}")
        protocol_by_sid[sid] = case
        external_specs = (
            ("input", "input_sha256", "input"),
            ("flagoff_expected", "flagoff_sha256", "flagoff"),
            ("flagon_expected", "flagon_sha256", "flagon"),
        )
        for path_key, hash_key, role in external_specs:
            raw_path, expected_sha = case.get(path_key), case.get(hash_key)
            if not isinstance(raw_path, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise AnalysisError(f"frozen protocol has malformed {path_key}/{hash_key} for {sid}")
            external_path = PurePosixPath(raw_path)
            if not external_path.is_absolute() or ".." in external_path.parts:
                raise AnalysisError(f"frozen protocol has unsafe external path for {sid}/{role}: {raw_path}")
            if role in {"flagoff", "flagon"}:
                expected_wet = EXPECTED_WET_SHA256[(sid, role)]
                if expected_sha != expected_wet:
                    raise AnalysisError(f"frozen protocol wet hash mismatch for {sid}/{role}")
            if raw_path in external_expectations:
                raise AnalysisError(f"frozen protocol repeats external path {raw_path}")
            external_expectations[raw_path] = {
                "sha256": expected_sha,
                "bytes": EXPECTED_TRACE_BYTES,
                "scene_id": sid,
                "role": role,
            }
    if set(protocol_by_sid) != set(EXPECTED_SIDS):
        raise AnalysisError("frozen protocol does not contain the exact approved scene IDs")

    lock_sha = sha256_file(lock_path)
    lock = _load_json(lock_path)
    report = _load_json(report_path)
    if report.get("status") != "passed" or report.get("full_wet_exact_controls") != 4:
        raise AnalysisError(
            "probe-report gate failed: expected status=passed and full_wet_exact_controls=4"
        )
    if report.get("protocol_sha256") != protocol_sha:
        raise AnalysisError("probe-report protocol hash does not match frozen protocol.json")
    if report.get("probe_lock_sha256") != lock_sha:
        raise AnalysisError("probe-report probe_lock_sha256 does not match probe-lock.json")

    locked_files = lock.get("files")
    if not isinstance(locked_files, list) or not locked_files:
        raise AnalysisError("probe-lock.json has no locked files")
    # The observer lock must bind the protocol, identity, source review, actual
    # probe sources/patch, build report, and executable. These are checked again
    # by the capture runner immediately before each of the four exact replays.
    lock_relpaths: set[str] = set()
    locked_external_paths: set[str] = set()
    for entry in locked_files:
        if not isinstance(entry, dict) or not {"path", "sha256", "bytes"} <= set(entry):
            raise AnalysisError("probe-lock.json contains a malformed file entry")
        lock_sha_value = entry["sha256"]
        try:
            lock_bytes_value = int(entry["bytes"])
        except (TypeError, ValueError) as exc:
            raise AnalysisError("probe-lock.json contains an invalid byte count") from exc
        if not isinstance(lock_sha_value, str) or len(lock_sha_value) != 64 or lock_bytes_value < 0:
            raise AnalysisError("probe-lock.json contains an invalid hash/size")
        locked_path = PurePosixPath(str(entry["path"]))
        try:
            relative = locked_path.relative_to(REMOTE_Q).as_posix()
        except ValueError as exc:
            external = locked_path.as_posix()
            expected_external = external_expectations.get(external)
            if expected_external is None:
                raise AnalysisError(f"probe lock path is outside the frozen run root and protocol pins: {locked_path}") from exc
            if external in locked_external_paths:
                raise AnalysisError(f"probe-lock.json repeats external path {external}")
            if lock_sha_value != expected_external["sha256"] or lock_bytes_value != expected_external["bytes"]:
                raise AnalysisError(
                    f"probe lock external file hash/size disagrees with the frozen protocol: {external}"
                )
            locked_external_paths.add(external)
            continue
        if any(part in {".", ".."} for part in PurePosixPath(relative).parts):
            raise AnalysisError(f"probe lock path contains traversal components: {locked_path}")
        if relative in lock_relpaths:
            raise AnalysisError(f"probe-lock.json repeats path {relative}")
        lock_relpaths.add(relative)
    if locked_external_paths != set(external_expectations):
        missing = sorted(set(external_expectations) - locked_external_paths)
        extra = sorted(locked_external_paths - set(external_expectations))
        raise AnalysisError(f"probe lock external-file set mismatch: missing={missing}, extra={extra}")

    required_locked = {
        "protocol.json",
        "identity-lock.json",
        "independent/propagation_source_review.json",
        "probe/probe.cpp",
        "probe/cpp/src/omlsa_imcra.hpp",
        "probe/cpp/src/gated_bias_sg.hpp",
        "probe/build-report.json",
        "probe/observer.patch",
        "probe/propagation_probe",
    }
    if not required_locked <= lock_relpaths:
        raise AnalysisError(f"probe lock omits required sources: {sorted(required_locked - lock_relpaths)}")

    expected = {(sid, role) for sid in EXPECTED_SIDS for role in EXPECTED_ROLES}
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise AnalysisError("probe-report must contain exactly four role records")
    observed: set[tuple[str, str]] = set()
    wet_checks = []
    role_file_inventory: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise AnalysisError("probe-report has a malformed role record")
        sid, role = record.get("scene_id"), record.get("role")
        key = (sid, role)
        if key not in expected or key in observed:
            raise AnalysisError(f"probe-report has unexpected or duplicate case/role {key!r}")
        observed.add(key)
        protocol_case = protocol_by_sid[sid]
        if record.get("input_sha256") != protocol_case.get("input_sha256"):
            raise AnalysisError(f"probe-report input hash does not match frozen protocol for {sid}/{role}")
        expected_sha = EXPECTED_WET_SHA256[key]
        if (
            record.get("wet_exact") is not True
            or record.get("wet_bytes") != EXPECTED_TRACE_BYTES
            or record.get("wet_sha256") != expected_sha
            or record.get("expected_sha256") != expected_sha
        ):
            raise AnalysisError(f"probe-report wet-exact identity failed for {sid}/{role}")
        probe = record.get("probe")
        if not isinstance(probe, dict) or probe.get("status") != "passed" or probe.get("wet_exact") is not True:
            raise AnalysisError(f"probe executable did not pass wet-exact checks for {sid}/{role}")
        if probe.get("hops") != 2402:
            raise AnalysisError(f"probe report has wrong hop count for {sid}/{role}")
        expected_helper_calls = 1 if role == "flagon" else 0
        expected_counters = {
            "base_calls": 2402,
            "alt_calls": 2402,
            "helper_entries": expected_helper_calls,
            "helper_exits": expected_helper_calls,
            "alt_helper_entries": 0,
            "mix_nonzero_hops": 0,
        }
        for name, expected_value in expected_counters.items():
            if probe.get(name) != expected_value:
                raise AnalysisError(
                    f"probe-report counter {name} mismatch for {sid}/{role}: "
                    f"expected {expected_value}, got {probe.get(name)!r}"
                )
        helper_record = record.get("helper")
        if not isinstance(helper_record, dict):
            raise AnalysisError(f"probe-report helper write-set missing for {sid}/{role}")
        if helper_record.get("entries") != expected_helper_calls or helper_record.get("exits") != expected_helper_calls:
            raise AnalysisError(f"probe-report helper invocation counts mismatch for {sid}/{role}")
        if role == "flagon":
            if not set(helper_record.get("changed_fields", [])) <= {
                "minimum", "conditional_minimum", "sub_minimum", "conditional_sub_minimum",
                "history", "conditional_history", "head", "frames", "rotations",
            }:
                raise AnalysisError(f"probe-report helper write set is unexpected for {sid}/{role}")

        wet_path = root / "runs" / sid / role / "wet.f32"
        if not wet_path.is_file():
            raise AnalysisError(f"missing exact wet stream {wet_path}")
        wet_sha = sha256_file(wet_path)
        wet_bytes = wet_path.stat().st_size
        if wet_sha != expected_sha or wet_bytes != EXPECTED_TRACE_BYTES:
            raise AnalysisError(f"stored wet stream is not frozen-byte-exact for {sid}/{role}")

        files = record.get("files")
        if not isinstance(files, list):
            raise AnalysisError(f"probe-report file inventory missing for {sid}/{role}")
        reported_file_map = {PurePosixPath(str(item.get("path", ""))).name: item for item in files if isinstance(item, dict)}
        if len(reported_file_map) != len(files):
            raise AnalysisError(f"probe-report has malformed or duplicate file paths for {sid}/{role}")
        role_file_inventory.setdefault(sid, {})[role] = reported_file_map
        for name in ("stages.csv", "stages.bin", "hops.csv", "wet.f32"):
            item = reported_file_map.get(name)
            path = root / "runs" / sid / role / name
            if item is None or not path.is_file():
                raise AnalysisError(f"probe-report/file inventory missing {sid}/{role}/{name}")
            if path.stat().st_size != item.get("bytes"):
                raise AnalysisError(f"probe output changed after capture: {sid}/{role}/{name}")
            # stages.bin and stages.csv are hash-checked on the one read used
            # for analysis; avoid a provenance pre-read of the large float data.
            if name == "hops.csv" and sha256_file(path) != item.get("sha256"):
                raise AnalysisError(f"probe output changed after capture: {sid}/{role}/{name}")
            if name == "wet.f32" and wet_sha != item.get("sha256"):
                raise AnalysisError(f"probe output changed after capture: {sid}/{role}/{name}")
        wet_checks.append({"scene_id": sid, "role": role, "bytes": wet_bytes, "sha256": wet_sha})

    if observed != expected:
        raise AnalysisError(f"probe-report missing role records: {sorted(expected - observed)}")
    return {
        "protocol_sha256": protocol_sha,
        "identity_lock_sha256": identity_sha,
        "probe_lock_sha256": lock_sha,
        "probe_report_sha256": sha256_file(report_path),
        "locked_files": len(locked_files),
        "locked_external_files": len(locked_external_paths),
        "wet_exact_checks": wet_checks,
        "role_file_inventory": role_file_inventory,
    }


def _int_column(value: str | None, name: str, row_no: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError as exc:
        raise AnalysisError(f"stages.csv row {row_no}: {name} must be an integer") from exc
    return parsed


def load_stream(
    role_dir: Path,
    expected_files: dict[str, dict[str, Any]] | None = None,
) -> tuple[OrderedDict[tuple[int, str, str], np.ndarray], dict[str, Any]]:
    """Load one role's index and binary payload, validating all byte ranges."""
    csv_path = role_dir / "stages.csv"
    bin_path = role_dir / "stages.bin"
    if not csv_path.is_file() or not bin_path.is_file():
        raise AnalysisError(f"missing stages.csv or stages.bin under {role_dir}")

    # A single read keeps the offset interpretation and the recorded hash tied
    # to exactly the same immutable byte sequence.
    payload = bin_path.read_bytes()
    if len(payload) % 4:
        raise AnalysisError(f"{bin_path}: byte length {len(payload)} is not float32-aligned")
    flat = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(flat).all():
        bad = int(np.flatnonzero(~np.isfinite(flat))[0])
        raise AnalysisError(f"{bin_path}: non-finite float32 at element {bad}")

    records: OrderedDict[tuple[int, str, str], np.ndarray] = OrderedDict()
    csv_payload = csv_path.read_bytes()
    index_hash = hashlib.sha256(csv_payload)
    expected_offset = 0
    seen_hops: set[int] = set()
    try:
        decoded_index = csv_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalysisError(f"{csv_path}: index is not UTF-8") from exc
    with io.StringIO(decoded_index, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != INDEX_COLUMNS:
            raise AnalysisError(
                f"{csv_path}: expected header {','.join(INDEX_COLUMNS)}, got {reader.fieldnames}"
            )
        for row_no, row in enumerate(reader, start=2):
            hop = _int_column(row.get("hop"), "hop", row_no)
            stage = (row.get("stage") or "").strip()
            field = (row.get("field") or "").strip()
            offset = _int_column(row.get("offset"), "offset", row_no)
            count = _int_column(row.get("count"), "count", row_no)
            if hop not in EXPECTED_HOPS:
                raise AnalysisError(f"{csv_path} row {row_no}: hop {hop} is outside 0..59")
            if not stage or not field:
                raise AnalysisError(f"{csv_path} row {row_no}: stage and field must be nonempty")
            if count <= 0:
                raise AnalysisError(f"{csv_path} row {row_no}: count must be positive")
            if offset < 0 or offset % 4:
                raise AnalysisError(f"{csv_path} row {row_no}: byte offset must be nonnegative and float32-aligned")
            if offset != expected_offset:
                raise AnalysisError(
                    f"{csv_path} row {row_no}: expected contiguous byte offset {expected_offset}, got {offset}"
                )
            byte_end = offset + count * 4
            if byte_end > len(payload):
                raise AnalysisError(
                    f"{csv_path} row {row_no}: record ends at byte {byte_end}, beyond {len(payload)}-byte binary"
                )
            key = (hop, stage, field)
            if key in records:
                raise AnalysisError(f"{csv_path} row {row_no}: duplicate paired key {key!r}")
            begin_element = offset // 4
            records[key] = flat[begin_element : begin_element + count]
            expected_offset = byte_end
            seen_hops.add(hop)

    if expected_offset != len(payload):
        raise AnalysisError(
            f"{bin_path}: index covers {expected_offset} bytes but file contains {len(payload)} bytes"
        )
    if seen_hops != set(EXPECTED_HOPS):
        missing = sorted(set(EXPECTED_HOPS) - seen_hops)
        raise AnalysisError(f"{csv_path}: missing trace hops {missing}")

    metadata = {
        "stages_csv_sha256": index_hash.hexdigest(),
        "stages_bin_sha256": sha256_bytes(payload),
        "stages_csv_bytes": len(csv_payload),
        "stages_bin_bytes": len(payload),
        "records": len(records),
        "trace_hops": [0, 59],
        "binary_dtype": "little-endian float32",
        "offset_unit": "bytes",
    }
    if expected_files is not None:
        for name, actual_hash, actual_bytes in (
            ("stages.csv", metadata["stages_csv_sha256"], metadata["stages_csv_bytes"]),
            ("stages.bin", metadata["stages_bin_sha256"], metadata["stages_bin_bytes"]),
        ):
            expected = expected_files.get(name)
            if expected is None or actual_hash != expected.get("sha256") or actual_bytes != expected.get("bytes"):
                raise AnalysisError(f"probe output changed after capture: {role_dir / name}")
    return records, metadata


def _diff_record(
    key: tuple[int, str, str],
    left: np.ndarray,
    right: np.ndarray,
    first_index: int,
) -> dict[str, Any]:
    left_value = np.float32(left[first_index])
    right_value = np.float32(right[first_index])
    left_bits = int(left_value.view(np.uint32))
    right_bits = int(right_value.view(np.uint32))
    return {
        "hop": key[0],
        "stage": key[1],
        "field": key[2],
        "bin": first_index,
        "flagoff": float(left_value),
        "flagon": float(right_value),
        "signed_delta_flagon_minus_flagoff": float(np.float64(right_value) - np.float64(left_value)),
        "flagoff_bits_hex": f"0x{left_bits:08x}",
        "flagon_bits_hex": f"0x{right_bits:08x}",
        "ratio_flagon_over_flagoff": (
            float(np.float64(right_value) / np.float64(left_value)) if left_value != 0 else None
        ),
    }


def paired_differences(
    flagoff: OrderedDict[tuple[int, str, str], np.ndarray],
    flagon: OrderedDict[tuple[int, str, str], np.ndarray],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Find first bitwise differences, retaining the probe's CSV record order."""
    off_keys = set(flagoff)
    on_keys = set(flagon)
    common_keys = off_keys & on_keys
    off_only = off_keys - on_keys
    on_only = on_keys - off_keys

    # The helper entry/exit snapshots are expected only when the option is on.
    # Ring snapshots are keyed by each role's real rotation hop. The state
    # clocks can diverge, so a ring event is role-local rather than a pairable
    # checkpoint. The actual helper snapshots exist only on the enabled role.
    unexpected_off = sorted(key for key in off_only if key[1] != "ring")
    unexpected_on = sorted(
        key for key in on_only if key[1] not in {"helper_entry", "helper_exit", "ring"}
    )
    if unexpected_off or unexpected_on:
        raise AnalysisError(
            "paired record-key mismatch outside the enabled-only helper snapshots: "
            f"flagoff-only={unexpected_off[:5]!r}, flagon-only={unexpected_on[:5]!r}"
        )

    first_overall: dict[str, Any] | None = None
    by_variable: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, dict[str, Any]] = {}
    compared = 0
    differing_records = 0
    for key, left in flagoff.items():
        if key not in common_keys:
            continue
        right = flagon[key]
        if left.size != right.size:
            raise AnalysisError(f"paired key {key!r} has counts {left.size} and {right.size}")
        compared += 1
        diff = np.flatnonzero(left.view(np.uint32) != right.view(np.uint32))
        if diff.size:
            differing_records += 1
            record = _diff_record(key, left, right, int(diff[0]))
            if first_overall is None:
                first_overall = record
            by_variable.setdefault(key[2], record)
            by_stage.setdefault(key[1], record)

    # Keep first_differences easy to consume by state variable while retaining
    # a true global first event and a stage index for downstream reports.
    differences: dict[str, Any] = {"overall": first_overall}
    differences.update(by_variable)
    differences["by_stage"] = by_stage
    differences["unpaired_role_snapshots"] = {
        "flagoff_only": [
            {"hop": hop, "stage": stage, "field": field, "count": int(flagoff[(hop, stage, field)].size)}
            for hop, stage, field in sorted(off_only)
        ],
        "flagon_only": [
            {"hop": hop, "stage": stage, "field": field, "count": int(flagon[(hop, stage, field)].size)}
            for hop, stage, field in sorted(on_only)
        ],
    }
    return differences, {
        "paired_records": compared,
        "differing_records": differing_records,
        "flagoff_only_records": len(off_only),
        "flagon_only_records": len(on_only),
    }


def metric_ratio_db(
    sid: str,
    metric: str,
    source: tuple[str, str, float],
    flagoff: OrderedDict[tuple[int, str, str], np.ndarray],
    flagon: OrderedDict[tuple[int, str, str], np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    stage, field, multiplier = source
    out = np.empty((len(EXPECTED_HOPS), EXPECTED_BINS), dtype=np.float64)
    nonfinite_per_hop: list[dict[str, int]] = []
    for hop in EXPECTED_HOPS:
        key = (hop, stage, field)
        if key not in flagoff or key not in flagon:
            raise AnalysisError(f"{sid}: metric {metric} is missing paired key {key!r}")
        left = flagoff[key]
        right = flagon[key]
        if left.size != EXPECTED_BINS or right.size != EXPECTED_BINS:
            raise AnalysisError(
                f"{sid}: metric {metric} key {key!r} must have {EXPECTED_BINS} bins, got {left.size}/{right.size}"
            )
        if np.any(left < 0) or np.any(right < 0):
            raise AnalysisError(f"{sid}: metric {metric} has negative values at hop {hop}; ratio dB is undefined")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Keep IEEE +/-inf and nan when a measured denominator/numerator is
            # exactly zero. No implicit floor or regularization is introduced.
            ratio = right.astype(np.float64) / left.astype(np.float64)
            out[hop] = multiplier * np.log10(ratio)
        bad = ~np.isfinite(out[hop])
        if bad.any():
            nonfinite_per_hop.append({"hop": hop, "bins": int(bad.sum())})
    return out, {
        "source": f"{stage}/{field}",
        "formula": f"{multiplier:g}*log10(flag_on/flag_off)",
        "shape": list(out.shape),
        "nonfinite_bins": int(np.count_nonzero(~np.isfinite(out))),
        "nonfinite_by_hop": nonfinite_per_hop,
    }


def recurrence_checks(
    sid: str,
    role: str,
    records: OrderedDict[tuple[int, str, str], np.ndarray],
) -> dict[str, Any]:
    """Recompute captured gamma, xi, and PSD update from stored float32 values."""
    beta = np.float32(1.47)
    totals: dict[str, dict[str, Any]] = {}

    def compare(name: str, expected: np.ndarray, observed: np.ndarray, *, atol: float, rtol: float) -> None:
        if expected.shape != observed.shape:
            raise AnalysisError(f"{sid}/{role}: {name} shape mismatch {expected.shape}/{observed.shape}")
        ok = np.isclose(expected, observed, atol=atol, rtol=rtol, equal_nan=False)
        bad = np.flatnonzero(~ok)
        if bad.size:
            flat_index = int(bad[0])
            hop, bin_index = divmod(flat_index, EXPECTED_BINS)
            raise AnalysisError(
                f"{sid}/{role}: {name} recurrence mismatch at hop {hop}, bin {bin_index}: "
                f"expected={expected.flat[flat_index]!r}, observed={observed.flat[flat_index]!r}"
            )
        totals[name] = {
            "checked_values": int(expected.size),
            "max_abs_error": float(np.max(np.abs(expected.astype(np.float64) - observed.astype(np.float64)))),
            "atol": atol,
            "rtol": rtol,
            "result": "within_float32_tolerance",
        }

    def gather(stage: str, field: str) -> np.ndarray:
        chunks = []
        for hop in EXPECTED_HOPS:
            key = (hop, stage, field)
            if key not in records:
                raise AnalysisError(f"{sid}/{role}: recurrence input missing {key!r}")
            values = records[key]
            if values.size != EXPECTED_BINS:
                raise AnalysisError(f"{sid}/{role}: recurrence input {key!r} has {values.size} bins")
            chunks.append(values)
        return np.stack(chunks, axis=0)

    power = gather("after_power", "power")
    raw_psd = gather("before_estimation", "noise_raw")
    gamma = gather("posterior", "gamma")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gamma_expected = np.minimum(
            np.float32(1e8),
            power / np.maximum(np.float32(1e-16), beta * raw_psd),
        ).astype(np.float32)
    compare("gamma", gamma_expected, gamma, atol=1e-6, rtol=0.0)

    previous_dd = gather("posterior", "previous_h1_snr")
    xi = gather("posterior", "xi")
    xi_expected = np.clip(
        np.float32(.92) * previous_dd + np.float32(.08) * np.maximum(gamma - np.float32(1.0), np.float32(0.0)),
        np.float32(1e-3),
        np.float32(1e8),
    ).astype(np.float32)
    compare("xi", xi_expected, xi, atol=1e-6, rtol=0.0)

    update_power = gather("update2", "power")
    previous = gather("update2", "previous")
    coefficient = gather("update2", "a")
    next_noise = gather("update2", "next")
    stored_noise = gather("update2", "stored")
    recurrence_expected = (
        coefficient * previous + (np.float32(1.0) - coefficient) * update_power
    ).astype(np.float32)
    compare("noise_recursion_next", recurrence_expected, next_noise, atol=1e-12, rtol=1e-6)
    stored_expected = np.maximum(np.float32(1e-16), recurrence_expected).astype(np.float32)
    compare("noise_recursion_stored", stored_expected, stored_noise, atol=1e-12, rtol=1e-6)

    outer_probability = gather("outer", "p")
    outer_h1 = gather("outer", "h1")
    gain = gather("gain", "gain")
    floor_gain = np.float32(10.0 ** (-9.0 / 20.0))
    h1_bounded = np.maximum(floor_gain, outer_h1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gain_expected = np.exp(
            outer_probability * np.log(h1_bounded)
            + (np.float32(1.0) - outer_probability) * np.log(floor_gain)
        ).astype(np.float32)
    compare("spectral_gain", gain_expected, gain, atol=1e-6, rtol=1e-6)
    return {
        "formulae": {
            "gamma": "min(1e8, power/max(1e-16, 1.47*raw_psd))",
            "xi": "clamp(0.92*previous_h1_snr + 0.08*max(gamma-1,0), 1e-3, 1e8)",
            "noise_recursion_next": "a*previous + (1-a)*power",
            "noise_recursion_stored": "max(1e-16, next)",
            "spectral_gain": "exp(p*log(max(floor_gain,h1)) + (1-p)*log(floor_gain)); floor_gain=10^(-9/20)",
        },
        "checks": totals,
        "interpretation": "recurrence-consistent measurements; not an isolated causal estimate",
    }


def analyze_case(
    root: Path,
    sid: str,
    role_files: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    role_data: dict[str, OrderedDict[tuple[int, str, str], np.ndarray]] = {}
    role_meta: dict[str, dict[str, Any]] = {}
    for role in EXPECTED_ROLES:
        records, metadata = load_stream(root / "runs" / sid / role, role_files[role])
        role_data[role] = records
        role_meta[role] = metadata

    first_differences, counts = paired_differences(role_data["flagoff"], role_data["flagon"])
    expected_helper_counts = {"flagoff": 0, "flagon": 1}
    helper_reports = {}
    for role in EXPECTED_ROLES:
        entry_count = len({hop for hop, stage, _field in role_data[role] if stage == "helper_entry"})
        exit_count = len({hop for hop, stage, _field in role_data[role] if stage == "helper_exit"})
        if entry_count != expected_helper_counts[role] or exit_count != expected_helper_counts[role]:
            raise AnalysisError(f"{sid}/{role}: expected helper entry/exit counts 0/0 or 1/1; got {entry_count}/{exit_count}")
        entry_hops = {hop for hop, stage, _field in role_data[role] if stage == "helper_entry"}
        exit_hops = {hop for hop, stage, _field in role_data[role] if stage == "helper_exit"}
        if entry_hops != exit_hops or (entry_hops and entry_hops != {1}):
            raise AnalysisError(f"{sid}/{role}: helper snapshots must pair at hop 1")
        helper_reports[role] = {"entries": entry_count, "exits": exit_count}

    # Validate the immediate helper write-set: only minima/history/ring-clock
    # fields may change, while all other captured state must remain bit-exact.
    if helper_reports["flagon"]["entries"]:
        entry = {field: value for (hop, stage, field), value in role_data["flagon"].items()
                 if hop == 1 and stage == "helper_entry"}
        exit_ = {field: value for (hop, stage, field), value in role_data["flagon"].items()
                 if hop == 1 and stage == "helper_exit"}
        if set(entry) != set(exit_):
            raise AnalysisError(f"{sid}/flagon: helper entry/exit fields differ")
        allowed = {"minimum", "conditional_minimum", "sub_minimum", "conditional_sub_minimum",
                   "history", "conditional_history", "head", "frames", "rotations"}
        changed, preserved = [], []
        for field in entry:
            differs = entry[field].view(np.uint32).tobytes() != exit_[field].view(np.uint32).tobytes()
            if differs:
                if field not in allowed:
                    raise AnalysisError(f"{sid}: helper unexpectedly changed {field}")
                changed.append(field)
            else:
                preserved.append(field)
        for clock in ("head", "frames", "rotations"):
            if exit_[clock].size != 1 or float(exit_[clock][0]) != 0.0:
                raise AnalysisError(f"{sid}: helper exit clock field {clock} did not reset to zero")
        helper_reports["flagon"].update({"changed_fields": sorted(changed), "preserved_fields": sorted(preserved)})

    recurrence = {role: recurrence_checks(sid, role, role_data[role]) for role in EXPECTED_ROLES}
    gamma_off = np.stack([role_data["flagoff"][(hop, "posterior", "gamma")] for hop in EXPECTED_HOPS])
    gamma_on = np.stack([role_data["flagon"][(hop, "posterior", "gamma")] for hop in EXPECTED_HOPS])
    gain_off = np.stack([role_data["flagoff"][(hop, "gain", "gain")] for hop in EXPECTED_HOPS])
    gain_on = np.stack([role_data["flagon"][(hop, "gain", "gain")] for hop in EXPECTED_HOPS])
    gamma_down_gain_up = (gamma_on < gamma_off) & (gain_on > gain_off)
    example_indices = np.argwhere(gamma_down_gain_up)
    nonmonotonic_examples = []
    for hop, bin_index in example_indices[:20]:
        nonmonotonic_examples.append({
            "hop": int(hop),
            "bin": int(bin_index),
            "frequency_hz": float(bin_index * 16000.0 / 512.0),
            "gamma_flagoff": float(gamma_off[hop, bin_index]),
            "gamma_flagon": float(gamma_on[hop, bin_index]),
            "gain_flagoff": float(gain_off[hop, bin_index]),
            "gain_flagon": float(gain_on[hop, bin_index]),
        })
    nonmonotonicity = {
        "gamma_down_gain_up_count": int(np.count_nonzero(gamma_down_gain_up)),
        "gamma_down_gain_up_examples_first_20_hop_bin_order": nonmonotonic_examples,
        "interpretation": "SG posterior/H1 gain is nonlinear; lower gamma does not imply lower final gain at every bin",
    }
    arrays: dict[str, np.ndarray] = {}
    metric_meta = {}
    for metric, source in METRIC_SOURCES.items():
        values, details = metric_ratio_db(sid, metric, source, role_data["flagoff"], role_data["flagon"])
        arrays[f"{sid}__{metric}"] = values
        metric_meta[metric] = details

    return {
        "scene_id": sid,
        "roles": role_meta,
        "paired_record_counts": counts,
        "helper": helper_reports,
        "recurrence_checks": recurrence,
        "gamma_gain_nonmonotonicity": nonmonotonicity,
        "first_differences": first_differences,
        "metrics": metric_meta,
    }, arrays


def write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends .npz to string paths; writing to an exclusive file handle
    # preserves the requested path and prevents accidental replacement.
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-root",
        type=Path,
        required=True,
        help="Q directory containing runs/<scene_id>/{flagoff,flagon}/stages.{csv,bin}",
    )
    parser.add_argument("--output-json", type=Path, help="default: <probe-root>/propagation-analysis.json")
    parser.add_argument("--output-npz", type=Path, help="default: <probe-root>/plot-data.npz")
    args = parser.parse_args()

    root = args.probe_root.resolve()
    expected_runs = root / "runs"
    if not expected_runs.is_dir():
        raise AnalysisError(f"probe root has no runs directory: {expected_runs}")
    provenance = validate_probe_provenance(root)
    role_file_inventory = provenance.pop("role_file_inventory")
    actual_sids = sorted(path.name for path in expected_runs.iterdir() if path.is_dir())
    if tuple(actual_sids) != tuple(sorted(EXPECTED_SIDS)):
        raise AnalysisError(f"expected exactly frozen scenes {sorted(EXPECTED_SIDS)}, found {actual_sids}")

    cases = []
    plot_arrays: dict[str, np.ndarray] = {}
    for sid in EXPECTED_SIDS:
        case, arrays = analyze_case(root, sid, role_file_inventory[sid])
        cases.append(case)
        plot_arrays.update(arrays)

    output_json = args.output_json or (root / "propagation-analysis.json")
    output_npz = args.output_npz or (root / "plot-data.npz")
    report = {
        "schema_id": "vcp.minima_history_propagation_analysis.v1",
        "status": "paired_observer_trace_analyzed",
        "scope": {
            "scene_ids": list(EXPECTED_SIDS),
            "roles": list(EXPECTED_ROLES),
            "trace_hops": [0, 59],
            "bins": EXPECTED_BINS,
            "comparison": "exact float32 bit comparison; no replay or intervention",
            "audio_bridge": (
                "conditional only; rejected full-chain runs lack direct same-run SG-boundary float capture, "
                "so component traces do not establish the audio outcome bridge"
            ),
        },
        "frozen_decision": FROZEN_DECISION,
        "analysis_plan_sha256": ANALYSIS_PLAN_SHA256,
        "provenance": provenance,
        "metric_definitions": {
            "raw_psd_ratio_db": "10*log10(flag_on/flag_off) at before_estimation/noise_raw",
            "gamma_ratio_db": "10*log10(flag_on/flag_off) at posterior/gamma",
            "xi_ratio_db": "10*log10(flag_on/flag_off) at posterior/xi",
            "gain_delta_db": "20*log10(flag_on/flag_off) at gain/gain",
            "zero_policy": "preserve IEEE inf/nan; no ratio floor is applied",
        },
        "plot_data_npz": str(output_npz),
        "cases": cases,
    }
    write_npz_exclusive(output_npz, plot_arrays)
    write_json_exclusive(output_json, report)
    print(json.dumps({
        "status": report["status"],
        "cases": len(cases),
        "json": str(output_json),
        "npz": str(output_npz),
        "arrays": {name: list(value.shape) for name, value in plot_arrays.items()},
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        raise SystemExit(f"propagation_analyze: {exc}")
