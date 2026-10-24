"""Evidence-gated P.1100/P.1110/P.1120 acoustic qualification orchestration."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock

from pydantic import ValidationError

from .bundle import build_measurement_bundle
from .models import (
    Artifact, ContractItem, CustomerLimitReconciliation, EvidenceRecord,
    ExecutionRecord, GateResult, ItemResult, MeasurementBundle, RunCreate,
    RunMode, RunRecord, RunStatus, canonical_sha256, new_id, sha256_file,
    utc_now,
)


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "acoustic-qualification"
RUNS = BUILD / "runs"
EVIDENCE = BUILD / "evidence"
EXECUTIONS = BUILD / "executions"
CONTRACT = ROOT / "cpp" / "evaluation" / "nv200-test-contract.json"
CATALOG = Path(__file__).with_name("catalog.json")
LOCAL_STANDARDS = {
    "P.1120": (
        ROOT / "build" / "itut-standards-20261010" / "P.1120.pdf",
        ROOT / "build" / "itut-standards-20261010" / "P.1120.txt",
        "d1ca3e5054a859ee84a635ce4c438e35c0f663f5819320f8e2f54eafa4f5ef7d",
        "3f28c4bab97084b506ab35187f29542ace88c7f42172134c90026c60d29b5a94",
    )
}
REQUIRED_POINTS = {"HATS_MRP", "HATS_DRP", "HFRP", "POI_SEND", "POI_RECEIVE"}
REQUIRED_SCENARIOS = {"C1", "C2", "C3", "C4"}
_LOCK = Lock()
_EVIDENCE_LOCK = Lock()
_EXECUTION_LOCK = Lock()
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


class EvidenceError(ValueError):
    """A requested evidence reference is absent, changed or unsafe."""


def _verify_reconciliation(
    artifact: Artifact | None,
    *,
    bandwidth: str,
    standard: dict[str, object],
) -> CustomerLimitReconciliation | None:
    if artifact is None:
        return None
    path = _evidence_path(artifact)
    if artifact.kind != "limit_reconciliation":
        raise EvidenceError("evidence kind must be limit_reconciliation")
    try:
        reconciliation = CustomerLimitReconciliation.model_validate_json(
            path.read_text()
        )
    except (OSError, ValueError, ValidationError) as error:
        raise EvidenceError(f"invalid limit reconciliation: {error}") from error
    if reconciliation.contract_sha256 != sha256_file(CONTRACT):
        raise EvidenceError("limit reconciliation targets a different contract")
    if reconciliation.standard != standard["recommendation"]:
        raise EvidenceError("limit reconciliation standard differs from selected bandwidth")
    if reconciliation.standard_edition != standard["edition"]:
        raise EvidenceError("limit reconciliation edition differs from selected standard")
    if standard.get("full_text_status") == "verified_local":
        _, text, _, text_sha = LOCAL_STANDARDS["P.1120"]
        if (
            not text.is_file()
            or sha256_file(text) != text_sha
            or reconciliation.standard_sha256 != text_sha
        ):
            raise EvidenceError(
                "limit reconciliation is not bound to the verified standard text"
            )
    expected = {
        item["id"]
        for transport in ("wired", "wireless", "bt")
        for item in _items_for(bandwidth, transport)
    }
    mapped = {row.item_id for row in reconciliation.mappings}
    if mapped != expected:
        missing = sorted(expected - mapped)
        extra = sorted(mapped - expected)
        raise EvidenceError(
            "limit reconciliation must cover selected bandwidth exactly "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    return reconciliation


def _load_catalog() -> dict:
    return json.loads(CATALOG.read_text())


def _standard_for(bandwidth: str) -> dict[str, object]:
    return _load_catalog()["standards"][bandwidth]


def _items_for(bandwidth: str, transport: str) -> list[dict]:
    contract = json.loads(CONTRACT.read_text())
    return [item for item in contract["items"]
            if item["bandwidth"] == bandwidth and item["transport"] == transport]


def validate_run_request(request: RunCreate) -> RunCreate:
    """Reject declarations that cannot plausibly document the prescribed scenario."""
    canonical = {
        row["id"]: row for row in _load_catalog()["noise_scenarios"]
    }
    seen = set()
    for scenario in request.fixture.noise_scenarios:
        if scenario.id in seen:
            raise EvidenceError(f"duplicate noise scenario {scenario.id}")
        seen.add(scenario.id)
        expected = canonical[scenario.id]
        speed = expected.get(
            "vehicle_speed_kmh", expected.get("vehicle_speed_min_kmh", 0)
        )
        checks = {
            "speed": abs(scenario.vehicle_speed_kmh - speed) <= 0.1,
            "windows": scenario.windows == expected["windows"],
            "wipers": scenario.wipers == expected["wipers"],
            "turn signal": scenario.turn_signal == expected["turn_signal"],
            "background talkers": scenario.background_talkers == expected["background_talkers"],
            "wind speed": scenario.wind_speed_mps <= expected["wind_speed_max_mps"],
            "precipitation": scenario.precipitation == expected["precipitation"],
            "temperature": (
                expected["temperature_c"][0]
                <= scenario.temperature_c
                <= expected["temperature_c"][1]
            ),
        }
        if "fan" in expected:
            checks["fan"] = scenario.fan_setting == expected["fan"]
        if "airflow" in expected:
            checks["airflow"] = scenario.airflow == expected["airflow"]
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise EvidenceError(
                f"{scenario.id}: declared conditions differ ({', '.join(failed)})"
            )
    points = [row.point for row in request.fixture.calibration_points]
    if len(points) != len(set(points)):
        raise EvidenceError("calibration points must be unique")
    return request


def _contract_item(item: dict) -> ContractItem:
    return ContractItem(
        **{key: item.get(key) for key in ContractItem.model_fields}
    )


def _safe_component(value: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value))


def _evidence_path(artifact: Artifact) -> Path:
    if any(part in {"", ".", ".."} for part in artifact.path.split("/")):
        raise EvidenceError("invalid evidence path")
    base = EVIDENCE.resolve()
    path = (base / artifact.path).resolve(strict=False)
    if not path.is_relative_to(base):
        raise EvidenceError("evidence path escapes inbox")
    current = base
    for part in artifact.path.split("/"):
        current /= part
        if current.is_symlink():
            raise EvidenceError("symlink evidence is forbidden")
    if not path.is_file():
        raise EvidenceError("evidence artifact does not exist")
    if sha256_file(path) != artifact.sha256:
        raise EvidenceError("evidence SHA-256 differs")
    return path


def verify_artifact(artifact: Artifact | None, *, kind: str | None = None) -> str | None:
    if artifact is None:
        return None
    _evidence_path(artifact)
    if kind is not None and artifact.kind != kind:
        raise EvidenceError(f"evidence kind must be {kind}")
    return artifact.sha256


def _all_artifacts(request: RunCreate) -> list[Artifact]:
    result = [*request.fixture.evidence, *request.dut.evidence]
    result += [point.evidence for point in request.fixture.calibration_points
               if point.evidence is not None]
    result += [scenario.evidence for scenario in request.fixture.noise_scenarios
               if scenario.evidence is not None]
    result += [request.hardware.evidence, request.method.package_evidence,
               request.method.polqa_evidence, request.method.signal_pack_evidence,
               request.standard_evidence, request.review_evidence,
               request.limit_reconciliation, request.measurement_bundle]
    return [row for row in result if row is not None]


def save_evidence(filename: str, content_type: str, data: bytes) -> EvidenceRecord:
    if not data or len(data) > MAX_EVIDENCE_BYTES:
        raise EvidenceError("evidence must be 1 byte to 64 MiB")
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "-", Path(filename).name).strip(".-") or "evidence.bin"
    identifier = new_id()
    relative = f"{identifier[:12]}/{identifier[12:]}-{clean[:120]}"
    destination = EVIDENCE / relative
    with _EVIDENCE_LOCK:
        destination.parent.mkdir(parents=True, exist_ok=False)
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".upload")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            destination.chmod(0o440)
        finally:
            Path(temporary).unlink(missing_ok=True)
        record = EvidenceRecord(
            artifact=Artifact(path=relative, sha256=sha256_file(destination),
                              kind="unclassified"),
            original_name=Path(filename).name[:240] or clean,
            size_bytes=len(data),
            content_type=content_type[:160] or "application/octet-stream",
            uploaded_at=utc_now(),
        )
        destination.with_suffix(destination.suffix + ".json").write_text(
            record.model_dump_json(indent=2) + "\n"
        )
    return record


def evidence_with_namespace(record: EvidenceRecord, namespace: str) -> EvidenceRecord:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", namespace):
        raise EvidenceError("invalid evidence namespace")
    original = _evidence_path(record.artifact)
    namespace_root = EVIDENCE / namespace
    if namespace_root.exists() and not namespace_root.is_dir():
        raise EvidenceError("evidence namespace collides with a file")
    namespace_root.mkdir(parents=True, exist_ok=True)
    if any(namespace_root.iterdir()):
        raise EvidenceError("evidence namespace already contains an artifact")
    target = namespace_root / original.name
    os.replace(original, target)
    metadata = original.with_suffix(original.suffix + ".json")
    metadata.unlink(missing_ok=True)
    artifact = record.artifact.model_copy(update={
        "path": str(target.relative_to(EVIDENCE))
    })
    moved = record.model_copy(update={"artifact": artifact})
    target.with_suffix(target.suffix + ".json").write_text(
        moved.model_dump_json(indent=2) + "\n"
    )
    return moved


def classify_evidence(artifact: Artifact, kind: str) -> EvidenceRecord:
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", kind):
        raise EvidenceError("invalid evidence kind")
    path = _evidence_path(artifact)
    metadata = path.with_suffix(path.suffix + ".json")
    record = EvidenceRecord.model_validate_json(metadata.read_text()).model_copy(
        update={"artifact": artifact.model_copy(update={"kind": kind})}
    )
    metadata.write_text(record.model_dump_json(indent=2) + "\n")
    return record


def verify_evidence_manifest() -> dict[str, object]:
    checked, failed = 0, []
    if EVIDENCE.is_dir():
        for metadata in sorted(EVIDENCE.glob("**/*.json")):
            try:
                record = EvidenceRecord.model_validate_json(metadata.read_text())
                _evidence_path(record.artifact)
                checked += 1
            except (OSError, ValueError, ValidationError):
                failed.append(str(metadata.relative_to(EVIDENCE)))
    return {"checked": checked, "failed": failed}


def evaluate_gates(request: RunCreate) -> tuple[list[GateResult], list[str], RunMode]:
    gates: list[GateResult] = []
    blockers: list[str] = []

    def gate(name: str, passed: bool, detail: str, *,
             blocking: tuple[RunMode, ...] = (RunMode.FORMAL,)) -> None:
        gates.append(GateResult(gate=name, passed=passed, detail=detail,
                                blocking_for=list(blocking)))
        if not passed:
            blockers.append(name)

    def evidence(artifact: Artifact | None, name: str, kind: str | None = None) -> bool:
        try:
            verify_artifact(artifact, kind=kind)
            return artifact is not None
        except EvidenceError:
            return False

    physical_modes = (RunMode.PRECOMPLIANCE, RunMode.FORMAL)
    points = {point.point: point for point in request.fixture.calibration_points}
    complete_points = REQUIRED_POINTS <= set(points)
    gate("fixture_calibration_points", complete_points,
         "MRP/DRP/HFRP/POI send/receive calibration points present"
         if complete_points else
         f"missing calibration points: {sorted(REQUIRED_POINTS - set(points))}",
         blocking=physical_modes)
    point_evidence = complete_points and all(
        point.evidence is not None and evidence(
            point.evidence, f"calibration:{point.point}", "calibration")
        for point in points.values()
    )
    gate("fixture_calibration_artifacts", point_evidence,
         "every calibration point has verified server-side evidence"
         if point_evidence else
         "one or more calibration artifacts are absent or fail SHA-256 verification",
         blocking=physical_modes)
    gate("fixture_quiet_room", request.fixture.background_noise_dba <= -64,
         f"background {request.fixture.background_noise_dba:.1f} dB(A); "
         "P.1120 requires <= -64 dBPa(A) for specified tests",
         blocking=physical_modes)
    gate("fixture_reverberation", request.fixture.reverberation_rt60_ms < 200,
         f"RT60 {request.fixture.reverberation_rt60_ms:.1f} ms; "
         "engineering precheck for controlled vehicle/cabin acoustics",
         blocking=physical_modes)
    age = utc_now() - request.fixture.calibrated_at
    gate("fixture_age", timedelta(0) <= age <= timedelta(days=366),
         "calibration must be current and not dated in the future",
         blocking=physical_modes)

    scenarios = request.fixture.noise_scenarios
    scenario_ids = {item.id for item in scenarios}
    scenario_evidence = bool(scenarios) and all(
        item.evidence is not None and evidence(
            item.evidence, f"noise:{item.id}", "noise_scenario")
        for item in scenarios
    )
    gate("p1120_noise_scenarios",
         request.mode is not RunMode.SIMULATION and
         REQUIRED_SCENARIOS <= scenario_ids and scenario_evidence,
         "C1-C4 present and evidence verified; declared C5 is also verified"
         if request.mode is not RunMode.SIMULATION and
         REQUIRED_SCENARIOS <= scenario_ids and scenario_evidence else
         "formal noise testing needs C1-C4 recordings plus verified scenario evidence",
         blocking=physical_modes)
    if "C5" in scenario_ids:
        gate("p1120_noise_scenario_c5", scenario_evidence,
             "declared C5 high-speed scenario must have verified evidence",
             blocking=physical_modes)

    gate("dut_network_profile",
         bool(request.dut.network_profiles) and
         all(profile.strip() for profile in request.dut.network_profiles),
         "codec/network profile explicitly identified", blocking=physical_modes)
    gate("dut_volume_setting", bool(request.dut.volume_setting.strip()),
         "fixed or enumerated volume condition supplied", blocking=physical_modes)
    gate("dut_audio_path", bool(request.dut.audio_path.strip()),
         "acoustic -> DUT -> network/POI path supplied", blocking=physical_modes)
    dut_evidence = bool(request.dut.evidence) and all(
        evidence(artifact, f"dut:{index}", None)
        for index, artifact in enumerate(request.dut.evidence)
    )
    gate("dut_evidence", dut_evidence,
         "DUT path/firmware evidence verified" if dut_evidence else
         "formal run needs firmware/path/configuration evidence",
         blocking=physical_modes)

    required_caps = {"playback", "capture", "timestamps", "level_calibration"}
    if request.bandwidth.value in {"SWB", "FB"}:
        required_caps.add("background_noise_control")
    if request.hardware.backend != "simulation":
        required_caps |= {"four_loudspeaker_noise", "subwoofer"}
    missing = required_caps - set(request.hardware.capabilities)
    gate("hardware_capabilities", not missing,
         f"missing capabilities: {sorted(missing)}", blocking=physical_modes)
    gate("hardware_calibration_id", bool(request.hardware.calibration_id.strip()),
         "backend calibration identity present", blocking=physical_modes)
    hardware_evidence = evidence(
        request.hardware.evidence, "hardware", "hardware_calibration")
    gate("hardware_evidence", hardware_evidence,
         "hardware calibration evidence verified" if hardware_evidence else
         "formal run needs a verified backend/hardware calibration record",
         blocking=physical_modes)

    method = request.method
    package_hash = bool(
        method.package_sha256 and method.package_evidence and
        method.package_sha256 == method.package_evidence.sha256 and
        evidence(method.package_evidence, "method_package", "method_package")
    )
    gate("method_package_hash", package_hash,
         "method package evidence is content-addressed and double-verified",
         blocking=physical_modes)
    polqa_hash = bool(
        method.polqa_build_sha256 and method.polqa_evidence and
        method.polqa_build_sha256 == method.polqa_evidence.sha256 and
        evidence(method.polqa_evidence, "polqa", "polqa_build")
    )
    gate("polqa_build_hash", polqa_hash,
         "approved P.863/POLQA build is content-addressed",
         blocking=physical_modes)
    signal_hash = bool(
        method.signal_pack_sha256 and method.signal_pack_evidence and
        method.signal_pack_sha256 == method.signal_pack_evidence.sha256 and
        evidence(method.signal_pack_evidence, "signals", "p501_signal_pack")
    )
    gate("signal_pack_hash", signal_hash,
         "P.501 signal pack is content-addressed", blocking=physical_modes)
    gate("p835_method_version", bool(method.p835_version.strip()),
         "noise quality method/version is declared", blocking=physical_modes)

    standard = _standard_for(request.bandwidth.value)
    standard_verified = standard["full_text_status"] == "verified_local"
    if standard_verified:
        pdf, text, pdf_sha, text_sha = LOCAL_STANDARDS["P.1120"]
        standard_verified = (
            pdf.is_file() and text.is_file() and
            sha256_file(pdf) == pdf_sha and sha256_file(text) == text_sha
        )
    standard_verified = standard_verified or evidence(
        request.standard_evidence, "standard", "standard_text")
    if standard["full_text_status"] != "verified_local" and request.standard_evidence:
        # An uploaded standard must be deliberately classified in a body-specific
        # namespace. A generic PDF cannot unlock formal adjudication.
        standard_verified = request.standard_evidence.path.startswith(
            f"{standard['recommendation']}-standard/"
        )
    gate("standard_text_verified", standard_verified,
         "a content-verified full standard text is available"
         if standard_verified else
         "the current P.1100/P.1110 body is unavailable or unverified")
    try:
        reconciliation = _verify_reconciliation(
            request.limit_reconciliation,
            bandwidth=request.bandwidth.value,
            standard=standard,
        )
        selected_contract_rows = sum(
            len(_items_for(request.bandwidth.value, transport))
            for transport in ("wired", "wireless", "bt")
        )
        reconciliation_error = None
    except EvidenceError as error:
        reconciliation = None
        selected_contract_rows = 0
        reconciliation_error = str(error)
    gate(
        "customer_limit_reconciliation",
        reconciliation is not None,
        f"all {selected_contract_rows} bandwidth rows are mapped to official "
        "clauses with a bound standard identity"
        if reconciliation is not None else
        reconciliation_error or "clause-by-clause customer limit reconciliation is absent",
    )

    bundle = evidence(request.measurement_bundle, "bundle", "measurement_bundle")
    gate("measurement_bundle", bundle,
         "server-verified measurement bundle present" if bundle else
         "execute needs an imported or issued measurement bundle")
    review = evidence(request.review_evidence, "review", "independent_review")
    gate("independent_review", review,
         "independent review artifact verified" if review else
         "formal completion requires independent review")

    ready = not blockers
    effective = request.mode
    if request.mode is RunMode.FORMAL and not ready:
        effective = RunMode.PRECOMPLIANCE
        blockers.append("formal_mode_downgraded")
    return gates, blockers, effective


def create_run(request: RunCreate) -> RunRecord:
    # Verify every quoted artifact before it becomes part of an audit snapshot.
    validate_run_request(request)
    for artifact in _all_artifacts(request):
        _evidence_path(artifact)
    items = _items_for(request.bandwidth.value, request.transport.value)
    gates, blockers, effective = evaluate_gates(request)
    record = RunRecord(
        run_id=new_id(), created_at=utc_now(),
        status=RunStatus.BLOCKED if blockers else RunStatus.READY,
        gate_status="blocked" if blockers else "ready",
        requested_mode=request.mode, effective_mode=effective,
        standard=_standard_for(request.bandwidth.value),
        contract={"path": str(CONTRACT.relative_to(ROOT)),
                  "sha256": sha256_file(CONTRACT),
                  "rows_for_selection": len(items)},
        request_sha256=canonical_sha256(request), request=request,
        gates=gates, blockers=blockers,
        selected_items=[_contract_item(item) for item in items],
        measured_items=0, not_measured_items=len(items),
    )
    destination = RUNS / record.run_id
    with _LOCK:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "request.json").write_text(request.model_dump_json(indent=2) + "\n")
        (destination / "run.json").write_text(record.model_dump_json(indent=2) + "\n")
        (destination / "gates.json").write_text(
            json.dumps([gate.model_dump(mode="json") for gate in gates], indent=2) + "\n")
    return record


def get_run(run_id: str) -> RunRecord | None:
    path = RUNS / run_id / "run.json" if _safe_component(run_id) else None
    if not path or not path.is_file():
        return None
    try:
        return RunRecord.model_validate_json(path.read_text())
    except (OSError, ValueError, ValidationError):
        # Pre-v2 plans were intentionally thin. They remain on disk as
        # historical artifacts but are not presented as current executable runs.
        return None


def list_runs(limit: int = 20) -> list[RunRecord]:
    if not RUNS.is_dir():
        return []
    paths = sorted(RUNS.glob("*/run.json"), key=lambda path: path.stat().st_mtime,
                   reverse=True)
    rows: list[RunRecord] = []
    for path in paths:
        try:
            rows.append(RunRecord.model_validate_json(path.read_text()))
        except (OSError, ValueError, ValidationError):
            continue
        if len(rows) >= max(1, min(limit, 100)):
            break
    return rows


def write_run_record(record: RunRecord) -> None:
    destination = RUNS / record.run_id
    if not destination.is_dir():
        raise EvidenceError("unknown run")
    temporary = destination / "run.json.next"
    temporary.write_text(record.model_dump_json(indent=2) + "\n")
    os.replace(temporary, destination / "run.json")


def load_measurement_bundle(artifact: Artifact) -> MeasurementBundle:
    path = _evidence_path(artifact)
    if artifact.kind != "measurement_bundle":
        raise EvidenceError("measurement artifact kind must be measurement_bundle")
    try:
        return MeasurementBundle.model_validate_json(path.read_text())
    except ValidationError as error:
        raise EvidenceError(f"invalid measurement bundle: {error}") from error


def compile_measurement_evidence(
    *,
    item_id: str,
    unit: str,
    kind: str,
    reference: Path,
    capture: Path,
    max_delay_ms: float,
    start_s: float,
    end_s: float | None,
    mask: dict | None,
    delay_samples: int,
) -> tuple[dict, dict]:
    """Run an evidence compiler and persist paired traces in the evidence inbox."""
    for path in (reference, capture):
        resolved = path.resolve()
        if not resolved.is_file():
            raise EvidenceError(f"measurement trace does not exist: {path}")
        if not resolved.is_relative_to(BUILD.resolve()):
            raise EvidenceError("measurement traces must be under the project build directory")
    reference_blob = reference.read_bytes()
    capture_blob = capture.read_bytes()
    reference_record = save_evidence(
        reference.name, "audio/wav", reference_blob
    )
    capture_record = save_evidence(
        capture.name, "audio/wav", capture_blob
    )
    safe_item = re.sub(r"[^A-Za-z0-9_.-]+", "-", item_id).strip(".-")
    if not safe_item:
        raise EvidenceError("measurement item id cannot form an evidence namespace")
    stamp = new_id()[:12]
    reference_record = evidence_with_namespace(
        reference_record, f"measurement-{safe_item}-{stamp}-reference"
    )
    capture_record = evidence_with_namespace(
        capture_record, f"measurement-{safe_item}-{stamp}-capture"
    )
    reference_record = classify_evidence(
        reference_record.artifact, "measurement_trace"
    )
    capture_record = classify_evidence(
        capture_record.artifact, "measurement_trace"
    )
    row = build_measurement_bundle(
        item_id=item_id, unit=unit, kind=kind,
        reference=reference, capture=capture,
        max_delay_ms=max_delay_ms, start_s=start_s, end_s=end_s,
        mask=mask, delay_samples=delay_samples,
        trace_path=capture_record.artifact.path,
    )
    row["trace"] = capture_record.artifact.model_dump(mode="json")
    row["source_identity"] = {
        "reference": reference_record.artifact.path,
        "reference_sha256": reference_record.artifact.sha256,
        "capture": capture_record.artifact.path,
        "capture_sha256": capture_record.artifact.sha256,
    }
    return row, {
        "reference": reference_record.model_dump(mode="json"),
        "capture": capture_record.model_dump(mode="json"),
    }


def write_measurement_bundle(measurements: list[dict], backend: str = "imported") -> tuple[MeasurementBundle, EvidenceRecord]:
    bundle = MeasurementBundle.model_validate({
        "schema_version": 1,
        "backend": {
            "name": backend,
            "version": "evidence-compiler-v1",
            "identity_sha256": canonical_sha256({
                "backend": backend,
                "measurements": measurements,
            }),
        },
        "measurements": measurements,
    })
    record = save_evidence(
        "measurement-bundle.json", "application/json",
        (bundle.model_dump_json(indent=2) + "\n").encode(),
    )
    record = evidence_with_namespace(
        record, f"measurement-bundle-{new_id()[:12]}"
    )
    record = classify_evidence(record.artifact, "measurement_bundle")
    return bundle, record


def adjudicate_numeric(item: ContractItem, value: float, *, backend: str) -> ItemResult:
    lower = item.lower_limit if isinstance(item.lower_limit, (int, float)) else None
    upper = item.upper_limit if isinstance(item.upper_limit, (int, float)) else None
    status: str = "review_required"
    margin: float | None = None
    if lower is not None and upper is not None:
        status = "pass" if lower <= value <= upper else "fail"
        margin = min(value - float(lower), float(upper) - value)
    elif lower is not None:
        status = "pass" if value >= float(lower) else "fail"
        margin = value - float(lower)
    elif upper is not None:
        status = "pass" if value <= float(upper) else "fail"
        margin = float(upper) - value
    reason = "customer workbook numeric interval"
    if item.interpretation == "requires_curve_or_limit_definition":
        status, margin = "review_required", None
        reason = "curve/mask and vertical placement require independent review"
    if backend == "simulation":
        status = "review_required"
        reason = "simulation value retained as diagnostic; no acoustic item auto-passes"
    elif backend != "imported":
        status = "review_required"
        reason = f"backend {backend!r} is not an accepted measurement source"
    return ItemResult(
        item_id=item.id, family=item.family, test_name=item.test_name,
        unit=item.unit, status=status, value=value, lower_limit=item.lower_limit,
        upper_limit=item.upper_limit, margin=margin, reason=reason,
        adjudication_basis="customer_workbook",
    )


def plan_from_contract(record: RunRecord, backend: str) -> list[ItemResult]:
    return [ItemResult(
        item_id=item.id, family=item.family, test_name=item.test_name,
        unit=item.unit, status="not_measured", lower_limit=item.lower_limit,
        upper_limit=item.upper_limit,
        reason=("simulation mode: official acoustic evidence is out of scope"
                if backend == "simulation" else
                "no server-verified measurement supplied"),
        adjudication_basis="customer_workbook",
    ) for item in record.selected_items]


def execute_imported(record: RunRecord) -> ExecutionRecord:
    artifact = record.request.measurement_bundle
    if artifact is None:
        return _finish_execution(record, backend="unavailable",
                                 message="No measurement bundle is available; no values were fabricated.")
    bundle = load_measurement_bundle(artifact)
    by_item = {measurement.item_id: measurement for measurement in bundle.measurements}
    requested = {item.id for item in record.selected_items}
    unknown = sorted(set(by_item) - requested)
    if unknown:
        raise EvidenceError(
            f"measurement bundle contains unknown or unselected item ids: {unknown[:5]}"
        )
    results: list[ItemResult] = []
    for item in record.selected_items:
        measurement = by_item.get(item.id)
        if measurement is None:
            results.append(ItemResult(
                item_id=item.id, family=item.family, test_name=item.test_name,
                unit=item.unit, status="not_measured", lower_limit=item.lower_limit,
                upper_limit=item.upper_limit,
                reason="measurement bundle has no value for this item",
                adjudication_basis="customer_workbook"))
            continue
        if measurement.unit != item.unit:
            results.append(ItemResult(
                item_id=item.id, family=item.family, test_name=item.test_name,
                unit=item.unit, status="invalid", value=measurement.value,
                lower_limit=item.lower_limit, upper_limit=item.upper_limit,
                method_id=measurement.method_id, evidence_sha256=artifact.sha256,
                recorded_at=measurement.recorded_at,
                reason=f"measurement unit {measurement.unit!r} differs from contract unit {item.unit!r}",
                adjudication_basis="customer_workbook"))
            continue
        if measurement.trace is not None:
            try:
                verify_artifact(measurement.trace)
            except EvidenceError as error:
                results.append(ItemResult(
                    item_id=item.id, family=item.family, test_name=item.test_name,
                    unit=item.unit, status="invalid", value=measurement.value,
                    lower_limit=item.lower_limit, upper_limit=item.upper_limit,
                    method_id=measurement.method_id,
                    evidence_sha256=measurement.trace.sha256,
                    recorded_at=measurement.recorded_at,
                    reason=f"measurement trace rejected: {error}",
                    adjudication_basis="customer_workbook"))
                continue
        results.append(adjudicate_numeric(item, measurement.value, backend="imported").model_copy(
            update={"method_id": measurement.method_id,
                    "evidence_sha256": measurement.trace.sha256
                    if measurement.trace else artifact.sha256,
                    "recorded_at": measurement.recorded_at}))
    return _finish_execution(record, backend=bundle.backend.name, results=results,
                             message="Imported measurements adjudicated against customer workbook limits.")


def _finish_execution(record: RunRecord, *, backend: str, message: str,
                      results: list[ItemResult] | None = None) -> ExecutionRecord:
    results = results or plan_from_contract(record, backend)
    execution_id = new_id()
    destination = EXECUTIONS / execution_id
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "results.json"
    output.write_text(json.dumps({
        "schema_version": 1, "run_id": record.run_id,
        "execution_id": execution_id, "standard_compliance": False,
        "items": [item.model_dump(mode="json") for item in results],
    }, indent=2, ensure_ascii=False) + "\n")
    summary = dict(Counter(item.status for item in results))
    execution = ExecutionRecord(
        execution_id=execution_id, run_id=record.run_id, created_at=utc_now(),
        status="completed", mode=record.effective_mode, backend=backend,
        message=message,
        result=Artifact(path=str(output.relative_to(EXECUTIONS.parent)),
                        sha256=sha256_file(output), kind="execution_results"),
        summary=summary, items=results)
    (destination / "execution.json").write_text(execution.model_dump_json(indent=2) + "\n")
    write_run_record(record.model_copy(update={
        "status": RunStatus.COMPLETED,
        "gate_status": "blocked" if record.blockers else "ready",
        "measured_items": len(results) - summary.get("not_measured", 0),
        "not_measured_items": summary.get("not_measured", 0),
    }))
    return execution


def execute_run(record: RunRecord) -> ExecutionRecord:
    with _EXECUTION_LOCK:
        if record.effective_mode is RunMode.SIMULATION:
            return _finish_execution(
                record, backend="simulation",
                results=plan_from_contract(record, "simulation"),
                message="Simulation enumerated selected rows; no acoustic item auto-passed.")
        return execute_imported(record)


def get_execution(execution_id: str) -> ExecutionRecord | None:
    path = EXECUTIONS / execution_id / "execution.json" if _safe_component(execution_id) else None
    return ExecutionRecord.model_validate_json(path.read_text()) if path and path.is_file() else None


def list_executions(run_id: str) -> list[ExecutionRecord]:
    if not EXECUTIONS.is_dir():
        return []
    rows = []
    for path in EXECUTIONS.glob("*/execution.json"):
        try:
            row = ExecutionRecord.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
        if row.run_id == run_id:
            rows.append(row)
    return sorted(rows, key=lambda row: row.created_at, reverse=True)
