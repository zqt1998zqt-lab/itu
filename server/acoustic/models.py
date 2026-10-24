"""Validated domain models for P.1100/P.1110/P.1120 acoustic qualification."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


SAFE_RELATIVE_PATH = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*(?:/[a-zA-Z0-9][a-zA-Z0-9_.-]*)*$"
SHA256 = r"^[a-f0-9]{64}$"


class Bandwidth(StrEnum):
    NB = "NB"
    WB = "WB"
    SWB = "SWB"
    FB = "FB"


class Transport(StrEnum):
    WIRED = "wired"
    WIRELESS = "wireless"
    BT = "bt"


class RunMode(StrEnum):
    SIMULATION = "simulation"
    PRECOMPLIANCE = "precompliance"
    FORMAL = "formal"


class RunStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Artifact(BaseModel):
    """A client may quote a server-issued reference; the server re-verifies it."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(pattern=SAFE_RELATIVE_PATH)
    sha256: str = Field(pattern=SHA256)
    kind: str = Field(min_length=1, max_length=80)


class CalibrationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point: Literal["HATS_MRP", "HATS_DRP", "HFRP", "POI_SEND", "POI_RECEIVE"]
    sensitivity_dbv_per_pa: float | None = None
    reference_db_spl: float | None = Field(default=None, ge=0, le=160)
    dc_v: float | None = None
    evidence: Artifact | None = None


class NoiseScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Literal["C1", "C2", "C3", "C4", "C5"]
    vehicle_speed_kmh: float = Field(ge=0, le=400)
    fan_setting: str = Field(min_length=1, max_length=80)
    airflow: str = Field(min_length=1, max_length=120)
    windows: Literal["up", "down", "mixed"]
    wipers: Literal["on", "off"]
    turn_signal: Literal["on", "off"]
    background_talkers: Literal["none", "present"] = "none"
    road_surface: str = Field(min_length=1, max_length=160)
    wind_speed_mps: float = Field(ge=0, le=80)
    precipitation: str = Field(min_length=1, max_length=80)
    temperature_c: float = Field(ge=-50, le=80)
    evidence: Artifact | None = None


class FixtureProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixture_id: str = Field(min_length=1, max_length=100)
    fixture_type: Literal["HATS", "HEAD_TORSO", "SIMULATED_CABIN"]
    room: Literal["anechoic", "vehicle", "simulated_vehicle"]
    zone_id: str = Field(default="driver", min_length=1, max_length=100)
    hats_position: str = Field(default="manufacturer defined", min_length=1, max_length=240)
    background_noise_dba: float = Field(ge=-160, le=120)
    reverberation_rt60_ms: float = Field(ge=0, le=5000)
    calibrated_at: datetime
    calibration_points: list[CalibrationPoint]
    noise_scenarios: list[NoiseScenario] = Field(default_factory=list)
    evidence: list[Artifact] = Field(default_factory=list)


class DutProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    firmware: str = Field(min_length=1, max_length=120)
    interface: Literal["acoustic", "electrical"]
    network_profiles: list[str] = Field(min_length=1)
    volume_setting: str = Field(min_length=1, max_length=100)
    audio_path: str = Field(min_length=1, max_length=240)
    evidence: list[Artifact] = Field(default_factory=list)


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = Field(min_length=1, max_length=120)
    capabilities: list[Literal[
        "playback", "capture", "loopback", "timestamps", "network_impairment",
        "level_calibration", "hats_rotation", "background_noise_control",
        "four_loudspeaker_noise", "subwoofer", "measurement_import",
        "dut_profile_evidence"
    ]]
    calibration_id: str = Field(min_length=1, max_length=120)
    execution_identity: str | None = Field(default=None, max_length=200)
    evidence: Artifact | None = None


class MethodProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package_id: str = Field(min_length=1, max_length=120)
    package_sha256: str | None = Field(default=None, pattern=SHA256)
    package_evidence: Artifact | None = None
    polqa_build_sha256: str | None = Field(default=None, pattern=SHA256)
    polqa_evidence: Artifact | None = None
    p835_version: str = Field(min_length=1, max_length=80)
    signal_pack_id: str = Field(min_length=1, max_length=120)
    signal_pack_sha256: str | None = Field(default=None, pattern=SHA256)
    signal_pack_evidence: Artifact | None = None


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: RunMode = RunMode.PRECOMPLIANCE
    bandwidth: Bandwidth
    transport: Transport
    fixture: FixtureProfile
    dut: DutProfile
    hardware: HardwareProfile
    method: MethodProfile
    standard_evidence: Artifact | None = None
    limit_reconciliation: Artifact | None = None
    review_evidence: Artifact | None = None
    measurement_bundle: Artifact | None = None
    confirm_formal: bool = False

    @model_validator(mode="after")
    def formal_attestation(self) -> "RunCreate":
        if self.mode is RunMode.FORMAL and not self.confirm_formal:
            raise ValueError("formal runs require explicit operator attestation")
        return self


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gate: str
    passed: bool
    detail: str
    blocking_for: list[RunMode]


class ContractItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    source_cell: str
    test_name: str
    family: str
    direction: str
    unit: str
    lower_limit: float | str | None = None
    upper_limit: float | str | None = None
    limit_source: str
    interpretation: str
    optional: bool = False


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    run_id: str
    created_at: datetime
    status: RunStatus
    gate_status: Literal["ready", "blocked"] = "blocked"
    requested_mode: RunMode
    effective_mode: RunMode
    standard: dict[str, object]
    contract: dict[str, str | int]
    request_sha256: str
    request: RunCreate
    gates: list[GateResult]
    blockers: list[str]
    selected_items: list[ContractItem]
    measured_items: int
    not_measured_items: int
    standard_compliance: Literal[False] = False
    limitation: str = (
        "A plan is evidence collection, not an accreditation certificate. Formal "
        "completion requires standard text, method packages, physical evidence and review."
    )


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm_execute: bool = False


class MeasurementCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1, max_length=120)
    unit: str = Field(min_length=1, max_length=120)
    kind: Literal["loopback", "transfer", "noise"]
    reference_path: str = Field(min_length=1, max_length=500)
    capture_path: str = Field(min_length=1, max_length=500)
    max_delay_ms: float = Field(default=1000, gt=0, le=5000)
    start_s: float = Field(default=0, ge=0)
    end_s: float | None = Field(default=None, gt=0)
    mask: dict | None = None
    delay_samples: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def transfer_mask_contract(self) -> "MeasurementCompileRequest":
        if self.kind == "transfer":
            if self.mask is None:
                raise ValueError("transfer measurement requires an official mask object")
            required = {
                "frequency_hz", "lower_db", "upper_db", "normalized_at_hz"
            }
            missing = sorted(required - self.mask.keys())
            if missing:
                raise ValueError(
                    "transfer mask missing required field(s): "
                    + ", ".join(missing)
                )
        return self


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1, max_length=120)
    value: float
    unit: str = Field(min_length=1, max_length=120)
    method_id: str = Field(min_length=1, max_length=180)
    recorded_at: datetime
    trace: Artifact | None = None
    source_identity: dict[str, str] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=1000)


class MeasurementBackendIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=100)
    identity_sha256: str = Field(pattern=SHA256)
    execution_log: Artifact | None = None


class MeasurementBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    backend: MeasurementBackendIdentity
    measurements: list[Measurement] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_items(self) -> "MeasurementBundle":
        ids = [row.item_id for row in self.measurements]
        if len(ids) != len(set(ids)):
            raise ValueError("measurement bundle contains duplicate item_id values")
        return self


class LimitMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1, max_length=120)
    clause: str = Field(min_length=1, max_length=120)
    table_or_figure: str = Field(min_length=1, max_length=120)
    official_limit: float | str
    customer_limit: float | str
    disposition: Literal[
        "same", "customer_tighter", "customer_looser", "not_standardized", "review"
    ]
    note: str = Field(default="", max_length=500)


class CustomerLimitReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    contract_sha256: str = Field(pattern=SHA256)
    standard: Literal["P.1100", "P.1110", "P.1120"]
    standard_edition: str = Field(min_length=1, max_length=40)
    standard_sha256: str = Field(pattern=SHA256)
    prepared_at: datetime
    reviewer: str = Field(min_length=1, max_length=160)
    mappings: list[LimitMapping] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_mappings(self) -> "CustomerLimitReconciliation":
        ids = [row.item_id for row in self.mappings]
        if len(ids) != len(set(ids)):
            raise ValueError("limit reconciliation contains duplicate item_id values")
        return self


class ItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    family: str
    test_name: str
    unit: str
    status: Literal["pass", "fail", "review_required", "invalid", "not_measured"]
    value: float | None = None
    lower_limit: float | str | None = None
    upper_limit: float | str | None = None
    margin: float | None = None
    method_id: str | None = None
    evidence_sha256: str | None = None
    recorded_at: datetime | None = None
    reason: str
    adjudication_basis: Literal["customer_workbook", "official_standard", "review"]


class ExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    execution_id: str
    run_id: str
    created_at: datetime
    status: Literal["running", "completed", "failed"]
    mode: RunMode
    backend: str
    message: str
    result: Artifact | None = None
    summary: dict[str, int] = Field(default_factory=dict)
    items: list[ItemResult] = Field(default_factory=list)
    standard_compliance: Literal[False] = False


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact: Artifact
    original_name: str
    size_bytes: int = Field(ge=0)
    content_type: str
    uploaded_at: datetime
    uploader: str = Field(default="local-operator", min_length=1, max_length=100)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def canonical_sha256(value: BaseModel | dict) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
