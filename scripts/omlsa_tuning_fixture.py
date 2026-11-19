#!/home/zhangqingtian/.venv/bin/python
"""Fail-closed access to the fixed OM-LSA tuning evaluation fixture."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ID = "omlsa-tuning-v1"
FIXTURE_ROOT = PROJECT_ROOT / "benchmarks" / "fixtures" / FIXTURE_ID
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
SCHEMA = "itu_omlsa_tuning_fixture.v1"
FIXTURE_MANIFEST_SHA256 = (
    "318c7098c8e069ac7ef6bc6bafa6c993bdb992f82c07f71cabfb2d69f595f847"
)
FIXTURE_MANIFEST_BYTES = 12_016

UPSTREAM_SCRIPT_RELATIVE = "scripts/evaluate_omlsa_tuning.py"
UPSTREAM_SCRIPT_SHA256 = (
    "7fbe7806442238fd954092e52cca0198d93554fe09eb23218c0d1457053de522"
)
UPSTREAM_SCRIPT_BYTES = 51_821

EXPECTED_MODEL_SHA256 = (
    "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
)
DEV_REFERENCE_SHA256 = (
    "dbe939a3c3fba3b9c8905455f34387448c5ca4a8cc0a64016d561485648660c5"
)
HOLDOUT_REFERENCES: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    "02": (
        "54d93a7656b85cd1c5e5699727418072a35bac00a075f7d25858e208906069d5",
        (2026111402, 2026111403),
        (6, 12),
    ),
    "03": (
        "07c70f7c15159f6814660fae6485adb3e02b601a637ba08ce356ae324482dc6b",
        (2026111402, 2026111403),
        (12, 18),
    ),
    "04": (
        "ae18dc343e073cbad381d55785a323ed2fa5201b621828583a4d80ac6671f431",
        (2026111402, 2026111403),
        (6, 18),
    ),
}
REFERENCE_SHA256 = {
    DEV_REFERENCE_SHA256,
    *(reference[0] for reference in HOLDOUT_REFERENCES.values()),
}
ASSESSMENT_SHA256 = {
    "02": "40ede870dd8c3a9b2d0653413f03f21e79a55b29e2312c6931da762aa99d7e90",
    "03": "264e1a44aba9399acbf8d5f966b57b86f03782665da99348411e84847724be66",
    "04": "d3d5376a4f3ff4909a056fa9b23fbf2fecc1c4f470317a797c93ab27fbab9b87",
}
ASSESSMENT_BYTES = {"02": 138_237, "03": 138_216, "04": 138_238}
REFERENCE_BYTES = 384_044

RATE = 16_000
DURATION_SECONDS = 12
SAMPLES = RATE * DURATION_SECONDS
HOP = 160
FRAME = 320

DATASET_ID = "cf29475e343052eaffa73dfebad1b0b0"
DATASET_AUDIO_DIR = (
    PROJECT_ROOT
    / "build"
    / "acoustic-qualification"
    / "acoustic-scene-condition-datasets"
    / DATASET_ID
    / "assets"
    / "audio"
)
HOLDOUT_BATCH_ID = "e8c06dab3bf544be809070682954a806"
HOLDOUT_BATCH_DIR = (
    PROJECT_ROOT
    / "build"
    / "acoustic-qualification"
    / "scene-condition-batches"
    / HOLDOUT_BATCH_ID
)
DEV_MANIFEST_SOURCE_PATH = Path(
    "/tmp/itu-ns-near-single-20261114/20261113T165610Z/stationary-white/evidence.json"
)


class FixtureError(RuntimeError):
    """The fixed evaluation fixture or its source lock is invalid."""


@dataclass(frozen=True)
class ArtifactLock:
    artifact_id: str
    kind: str
    role: str
    source_case_id: str
    source_path: str
    relative_path: str
    sha256: str
    bytes: int
    path: Path
    reference_binding: Mapping[str, Any] | None = None

    def evidence(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "role": self.role,
            "source_case_id": self.source_case_id,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "fixture_path": str(self.path),
        }
        if self.reference_binding is not None:
            result["reference_binding"] = dict(self.reference_binding)
        return result


@dataclass(frozen=True)
class CaseProtocol:
    case_id: str
    role: str
    source_case_id: str
    noise_seed: int
    target_snr_db: float
    mos_eligible: bool
    reference_artifact_id: str
    assessment_artifact_id: str | None
    synthetic: bool

    def evidence(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "source_case_id": self.source_case_id,
            "noise_seed": self.noise_seed,
            "target_snr_db": self.target_snr_db,
            "mos_eligible": self.mos_eligible,
            "reference_artifact_id": self.reference_artifact_id,
            "assessment_artifact_id": self.assessment_artifact_id,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class FixtureLock:
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: int
    fixture_root: Path
    upstream_script: ArtifactLock
    artifacts: tuple[ArtifactLock, ...]
    cases: tuple[CaseProtocol, ...]
    source_store: Mapping[str, Any]
    dev_manifest_origin: Mapping[str, Any]

    def artifact(self, artifact_id: str) -> ArtifactLock:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise FixtureError(f"unknown artifact id in validated fixture: {artifact_id}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(data: bytes, label: Path) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FixtureError(f"JSON is not UTF-8: {label}: {error}") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, FixtureError) as error:
        raise FixtureError(f"cannot parse JSON {label}: {error}") from error


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FixtureError(f"{label} must be a list")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixtureError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureError(f"{label} must be an integer")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixtureError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FixtureError(f"{label} must be finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    text = _require_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FixtureError(f"{label} must be a lowercase SHA-256")
    return text


def _safe_relative_path(value: Any, label: str) -> str:
    text = _require_str(value, label)
    if "\\" in text:
        raise FixtureError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise FixtureError(f"{label} escapes the fixture root: {text}")
    return path.as_posix()


def _assert_no_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise FixtureError(f"{label} contains a symlink component: {current}")


def _assert_regular_nonsymlink(path: Path, root: Path, label: str) -> Path:
    _assert_no_symlink_components(root, f"{label} root")
    _assert_no_symlink_components(path, label)
    if path.is_symlink():
        raise FixtureError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        raise FixtureError(f"{label} is missing: {path}")
    if not path.is_file():
        raise FixtureError(f"{label} must be a regular file: {path}")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise FixtureError(f"{label} escapes its allowed root: {path}") from error
    return resolved


def _assert_absolute_source_path(value: Any, label: str) -> str:
    text = _require_str(value, label)
    if not Path(text).is_absolute():
        raise FixtureError(f"{label} must preserve an absolute source path")
    return text


def _recursive_strings(value: Any) -> Sequence[str]:
    strings: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return strings


def assessment_reference_binding(
    assessment: Mapping[str, Any],
    expected_sha256: str,
    canonical_original_reference_path: Path,
) -> dict[str, Any]:
    """Bind the original assessment to its original hash/path, never a migrated path."""

    expected = _require_sha256(expected_sha256, "expected reference SHA-256")
    input_hashes = _require_mapping(
        assessment.get("input_sha256"), "assessment.input_sha256"
    )
    declared = _require_sha256(
        input_hashes.get("reference"), "assessment.input_sha256.reference"
    )
    if declared != expected:
        raise FixtureError(
            "assessment input_sha256.reference disagrees with the fixed reference"
        )
    actual_hashes = assessment.get("actual_input_sha256")
    if actual_hashes is not None:
        actual = _require_mapping(actual_hashes, "assessment.actual_input_sha256")
        actual_reference = _require_sha256(
            actual.get("reference"), "assessment.actual_input_sha256.reference"
        )
        if actual_reference != expected:
            raise FixtureError(
                "assessment actual_input_sha256.reference disagrees with the fixed reference"
            )

    canonical = os.path.abspath(os.path.normpath(str(canonical_original_reference_path)))
    basename = f"{expected}.wav"
    explicit_paths = sorted(
        {
            os.path.abspath(os.path.normpath(text))
            for text in _recursive_strings(assessment)
            if text.lower().endswith(".wav") and Path(text).name == basename
        }
    )
    if explicit_paths and explicit_paths != [canonical]:
        raise FixtureError(
            "assessment explicit reference path disagrees with the original canonical path: "
            f"expected {canonical}, got {explicit_paths}"
        )

    return {
        "json_field": "input_sha256.reference",
        "sha256": expected,
        "canonical_original_reference_path": canonical,
        "explicit_reference_paths": explicit_paths,
        "explicit_path_present": bool(explicit_paths),
        "fixture_path_comparison": "not_performed",
    }


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load a non-symlink manifest without resolving any original source path."""

    path = Path(manifest_path).expanduser().absolute()
    _assert_no_symlink_components(path, "fixture manifest")
    if path.is_symlink():
        raise FixtureError(f"fixture manifest must not be a symlink: {path}")
    if not path.exists():
        raise FixtureError(f"fixture manifest is missing: {path}")
    if not path.is_file():
        raise FixtureError(f"fixture manifest must be a regular file: {path}")
    data = path.read_bytes()
    document = _decode_json(data, path)
    manifest = _require_mapping(document, "fixture manifest")
    if manifest.get("schema") != SCHEMA:
        raise FixtureError(
            f"fixture manifest schema must be {SCHEMA}, got {manifest.get('schema')!r}"
        )
    if manifest.get("fixture_id") != FIXTURE_ID:
        raise FixtureError(
            f"fixture id must be {FIXTURE_ID}, got {manifest.get('fixture_id')!r}"
        )
    return dict(manifest)


def _known_case_protocols() -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = [
        {
            "case_id": "dev-near_single-01-white-seed2026111401-snr12",
            "role": "dev",
            "source_case_id": "sce1-wb-wired-near_single-01-conversation-v2",
            "noise_seed": 2026111401,
            "target_snr_db": 12.0,
            "mos_eligible": True,
            "reference_artifact_id": f"reference:{DEV_REFERENCE_SHA256}",
            "assessment_artifact_id": None,
            "synthetic": False,
        }
    ]
    for suffix, (reference_sha, seeds, snrs) in sorted(HOLDOUT_REFERENCES.items()):
        source_case_id = f"sce1-wb-wired-near_single-{suffix}-conversation-v2"
        for seed, snr in zip(seeds, snrs, strict=True):
            cases.append(
                {
                    "case_id": f"holdout-near_single-{suffix}-white-seed{seed}-snr{snr}",
                    "role": "holdout",
                    "source_case_id": source_case_id,
                    "noise_seed": seed,
                    "target_snr_db": float(snr),
                    "mos_eligible": True,
                    "reference_artifact_id": f"reference:{reference_sha}",
                    "assessment_artifact_id": f"assessment:{suffix}",
                    "synthetic": False,
                }
            )
    return tuple(cases)


KNOWN_CASE_PROTOCOLS = _known_case_protocols()


def _validate_protocol(protocol_value: Any) -> tuple[CaseProtocol, ...]:
    protocol = _require_mapping(protocol_value, "manifest.protocol")
    expected_scalars = {
        "rate": RATE,
        "duration_seconds": DURATION_SECONDS,
        "samples": SAMPLES,
        "hop": HOP,
        "frame": FRAME,
    }
    for key, expected in expected_scalars.items():
        actual = _require_int(protocol.get(key), f"manifest.protocol.{key}")
        if actual != expected:
            raise FixtureError(
                f"manifest.protocol.{key} must be {expected}, got {actual}"
            )
    if protocol.get("dev_reference_sha256") != DEV_REFERENCE_SHA256:
        raise FixtureError("manifest dev reference SHA-256 changed")
    holdouts = _require_mapping(
        protocol.get("holdout_references"), "manifest.protocol.holdout_references"
    )
    if set(holdouts) != set(HOLDOUT_REFERENCES):
        raise FixtureError("manifest holdout suffixes changed")
    for suffix, expected in HOLDOUT_REFERENCES.items():
        record = _require_list(
            holdouts[suffix], f"manifest.protocol.holdout_references.{suffix}"
        )
        if len(record) != 3:
            raise FixtureError(f"holdout reference record {suffix} must have three fields")
        reference_sha = _require_sha256(
            record[0], f"holdout reference {suffix} SHA-256"
        )
        seeds = _require_list(record[1], f"holdout reference {suffix} seeds")
        snrs = _require_list(record[2], f"holdout reference {suffix} SNRs")
        if reference_sha != expected[0]:
            raise FixtureError(f"holdout reference {suffix} SHA-256 changed")
        if tuple(_require_int(value, f"holdout {suffix} seed") for value in seeds) != expected[1]:
            raise FixtureError(f"holdout reference {suffix} seeds changed")
        if tuple(_require_int(value, f"holdout {suffix} SNR") for value in snrs) != expected[2]:
            raise FixtureError(f"holdout reference {suffix} SNRs changed")

    raw_cases = _require_list(protocol.get("cases"), "manifest.protocol.cases")
    if len(raw_cases) != len(KNOWN_CASE_PROTOCOLS):
        raise FixtureError(
            f"fixture must contain {len(KNOWN_CASE_PROTOCOLS)} cases, got {len(raw_cases)}"
        )
    parsed: list[CaseProtocol] = []
    for index, raw_value in enumerate(raw_cases):
        raw = _require_mapping(raw_value, f"manifest.protocol.cases[{index}]")
        case = CaseProtocol(
            case_id=_require_str(raw.get("case_id"), f"case[{index}].case_id"),
            role=_require_str(raw.get("role"), f"case[{index}].role"),
            source_case_id=_require_str(
                raw.get("source_case_id"), f"case[{index}].source_case_id"
            ),
            noise_seed=_require_int(
                raw.get("noise_seed"), f"case[{index}].noise_seed"
            ),
            target_snr_db=_require_number(
                raw.get("target_snr_db"), f"case[{index}].target_snr_db"
            ),
            mos_eligible=raw.get("mos_eligible") is True,
            reference_artifact_id=_require_str(
                raw.get("reference_artifact_id"),
                f"case[{index}].reference_artifact_id",
            ),
            assessment_artifact_id=(
                None
                if raw.get("assessment_artifact_id") is None
                else _require_str(
                    raw.get("assessment_artifact_id"),
                    f"case[{index}].assessment_artifact_id",
                )
            ),
            synthetic=raw.get("synthetic") is True,
        )
        parsed.append(case)

    if tuple(case.evidence() for case in parsed) != KNOWN_CASE_PROTOCOLS:
        raise FixtureError("fixture case ids/roles/source ids/seeds/SNRs changed")
    if len({case.case_id for case in parsed}) != len(parsed):
        raise FixtureError("fixture case ids must be unique")
    return tuple(parsed)


def _validate_artifacts(
    artifacts_value: Any,
    fixture_root: Path,
) -> tuple[ArtifactLock, ...]:
    raw_artifacts = _require_list(artifacts_value, "manifest.artifacts")
    if len(raw_artifacts) != 7:
        raise FixtureError(f"fixture must contain exactly 7 artifacts, got {len(raw_artifacts)}")

    artifacts: list[ArtifactLock] = []
    seen_ids: set[str] = set()
    for index, raw_value in enumerate(raw_artifacts):
        raw = _require_mapping(raw_value, f"manifest.artifacts[{index}]")
        artifact_id = _require_str(
            raw.get("artifact_id"), f"artifact[{index}].artifact_id"
        )
        if artifact_id in seen_ids:
            raise FixtureError(f"duplicate artifact id: {artifact_id}")
        seen_ids.add(artifact_id)
        kind = _require_str(raw.get("kind"), f"artifact[{index}].kind")
        if kind not in {"reference", "assessment"}:
            raise FixtureError(f"artifact[{index}].kind is unsupported: {kind}")
        role = _require_str(raw.get("role"), f"artifact[{index}].role")
        if role not in {"dev", "holdout"}:
            raise FixtureError(f"artifact[{index}].role is unsupported: {role}")
        relative_path = _safe_relative_path(
            raw.get("relative_path"), f"artifact[{index}].relative_path"
        )
        source_path = _assert_absolute_source_path(
            raw.get("source_path"), f"artifact[{index}].source_path"
        )
        sha256 = _require_sha256(raw.get("sha256"), f"artifact[{index}].sha256")
        size = _require_int(raw.get("bytes"), f"artifact[{index}].bytes")
        if size < 0:
            raise FixtureError(f"artifact[{index}].bytes must be non-negative")
        artifact_path = fixture_root / Path(relative_path)
        resolved = _assert_regular_nonsymlink(
            artifact_path, fixture_root, f"artifact {artifact_id}"
        )
        actual_size = resolved.stat().st_size
        actual_sha = sha256_file(resolved)
        if actual_size != size:
            raise FixtureError(
                f"artifact {artifact_id} byte count changed: expected {size}, got {actual_size}"
            )
        if actual_sha != sha256:
            raise FixtureError(
                f"artifact {artifact_id} SHA-256 changed: expected {sha256}, got {actual_sha}"
            )
        binding = raw.get("reference_binding")
        if kind == "reference":
            if binding is not None:
                raise FixtureError(f"reference artifact {artifact_id} cannot have a binding")
            if Path(source_path).name != f"{sha256}.wav":
                raise FixtureError(
                    f"reference artifact {artifact_id} source filename does not match its SHA"
                )
            if sha256 not in REFERENCE_SHA256:
                raise FixtureError(f"reference artifact {artifact_id} uses an unknown SHA")
        else:
            binding = dict(_require_mapping(binding, f"artifact {artifact_id}.reference_binding"))
        artifacts.append(
            ArtifactLock(
                artifact_id=artifact_id,
                kind=kind,
                role=role,
                source_case_id=_require_str(
                    raw.get("source_case_id"), f"artifact[{index}].source_case_id"
                ),
                source_path=source_path,
                relative_path=relative_path,
                sha256=sha256,
                bytes=size,
                path=resolved,
                reference_binding=(
                    MappingProxyType(binding) if binding is not None else None
                ),
            )
        )

    references = {artifact.sha256 for artifact in artifacts if artifact.kind == "reference"}
    if references != REFERENCE_SHA256:
        raise FixtureError(
            f"fixture reference set changed: expected {sorted(REFERENCE_SHA256)}, got {sorted(references)}"
        )
    assessments = [artifact for artifact in artifacts if artifact.kind == "assessment"]
    if len(assessments) != 3:
        raise FixtureError(f"fixture must contain exactly 3 assessments, got {len(assessments)}")
    for artifact in assessments:
        suffix = next(
            (
                key
                for key in ASSESSMENT_SHA256
                if artifact.artifact_id == f"assessment:{key}"
            ),
            None,
        )
        if suffix is None:
            raise FixtureError(f"unknown assessment artifact id: {artifact.artifact_id}")
        if artifact.sha256 != ASSESSMENT_SHA256[suffix]:
            raise FixtureError(f"assessment {suffix} SHA-256 changed")
        if artifact.bytes != ASSESSMENT_BYTES[suffix]:
            raise FixtureError(f"assessment {suffix} byte count changed")
        expected_case_id = f"sce1-wb-wired-near_single-{suffix}-conversation-v2"
        if artifact.source_case_id != expected_case_id:
            raise FixtureError(f"assessment {suffix} source case id changed")
        if artifact.source_path.endswith(
            f"/cases/{expected_case_id}/assessment.json"
        ) is False:
            raise FixtureError(f"assessment {suffix} original source path changed")
    return tuple(artifacts)


def _validate_assessment_bindings(
    artifacts: Sequence[ArtifactLock],
    source_store: Mapping[str, Any],
) -> None:
    dataset_audio_dir = Path(
        _assert_absolute_source_path(
            source_store.get("dataset_audio_dir"),
            "manifest.source_store.dataset_audio_dir",
        )
    )
    holdout_batch_id = _require_str(
        source_store.get("holdout_batch_id"),
        "manifest.source_store.holdout_batch_id",
    )
    _assert_absolute_source_path(
        source_store.get("holdout_batch_dir"),
        "manifest.source_store.holdout_batch_dir",
    )
    if holdout_batch_id != HOLDOUT_BATCH_ID:
        raise FixtureError("manifest holdout batch id changed")
    if source_store.get("dataset_id") != DATASET_ID:
        raise FixtureError("manifest source dataset id changed")

    for artifact in artifacts:
        if artifact.kind != "assessment":
            continue
        document = _decode_json(artifact.path.read_bytes(), artifact.path)
        assessment = _require_mapping(document, f"assessment {artifact.artifact_id}")
        if assessment.get("case_id") != artifact.source_case_id:
            raise FixtureError(
                f"assessment {artifact.artifact_id} case_id does not match its source case"
            )
        if assessment.get("batch_id") != holdout_batch_id:
            raise FixtureError(f"assessment {artifact.artifact_id} batch id changed")
        if assessment.get("dataset_id") != DATASET_ID:
            raise FixtureError(f"assessment {artifact.artifact_id} dataset id changed")
        suffix = artifact.artifact_id.split(":", 1)[1]
        expected_reference_sha = HOLDOUT_REFERENCES[suffix][0]
        canonical_original_path = dataset_audio_dir / f"{expected_reference_sha}.wav"
        actual_binding = assessment_reference_binding(
            assessment,
            expected_reference_sha,
            canonical_original_path,
        )
        if dict(artifact.reference_binding or {}) != actual_binding:
            raise FixtureError(
                f"assessment {artifact.artifact_id} reference binding changed"
            )


def _validate_upstream_script(
    upstream_value: Any,
    project_root: Path,
) -> ArtifactLock:
    upstream = _require_mapping(upstream_value, "manifest.upstream_script")
    relative_path = _safe_relative_path(
        upstream.get("path"), "manifest.upstream_script.path"
    )
    if relative_path != UPSTREAM_SCRIPT_RELATIVE:
        raise FixtureError("manifest upstream script path changed")
    source_path = _assert_absolute_source_path(
        upstream.get("source_path"), "manifest.upstream_script.source_path"
    )
    expected_path = project_root / Path(relative_path)
    if Path(source_path) != expected_path:
        raise FixtureError("manifest upstream script source_path changed")
    resolved = _assert_regular_nonsymlink(
        expected_path, project_root, "upstream evaluation script"
    )
    sha256 = _require_sha256(
        upstream.get("sha256"), "manifest.upstream_script.sha256"
    )
    size = _require_int(upstream.get("bytes"), "manifest.upstream_script.bytes")
    if sha256 != UPSTREAM_SCRIPT_SHA256:
        raise FixtureError("manifest upstream evaluation script SHA-256 changed")
    if size != UPSTREAM_SCRIPT_BYTES:
        raise FixtureError("manifest upstream evaluation script byte count changed")
    actual_size = resolved.stat().st_size
    actual_sha = sha256_file(resolved)
    if actual_size != size or actual_sha != sha256:
        raise FixtureError(
            "upstream evaluation script bytes changed: "
            f"expected {sha256}/{size}, got {actual_sha}/{actual_size}"
        )
    return ArtifactLock(
        artifact_id="upstream:evaluate_omlsa_tuning",
        kind="upstream_script",
        role="evaluator",
        source_case_id="omlsa-tuning-protocol",
        source_path=source_path,
        relative_path=relative_path,
        sha256=sha256,
        bytes=size,
        path=resolved,
    )


def _validate_case_artifact_links(
    cases: Sequence[CaseProtocol],
    artifacts: Sequence[ArtifactLock],
) -> None:
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    for case in cases:
        reference = by_id.get(case.reference_artifact_id)
        if reference is None or reference.kind != "reference":
            raise FixtureError(f"case {case.case_id} has no reference artifact")
        expected_sha = (
            DEV_REFERENCE_SHA256
            if case.role == "dev"
            else HOLDOUT_REFERENCES[
                next(
                    suffix
                    for suffix in HOLDOUT_REFERENCES
                    if case.source_case_id
                    == f"sce1-wb-wired-near_single-{suffix}-conversation-v2"
                )
            ][0]
        )
        if reference.sha256 != expected_sha:
            raise FixtureError(f"case {case.case_id} reference SHA-256 changed")
        if case.role == "dev":
            if case.assessment_artifact_id is not None:
                raise FixtureError("dev case cannot have an assessment artifact")
        else:
            assessment = by_id.get(case.assessment_artifact_id or "")
            if assessment is None or assessment.kind != "assessment":
                raise FixtureError(f"case {case.case_id} has no assessment artifact")
            if assessment.source_case_id != case.source_case_id:
                raise FixtureError(f"case {case.case_id} assessment source case changed")


def freeze_source_lock(
    manifest_path: Path = MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
) -> FixtureLock:
    """Validate and freeze every identity the current evaluator is allowed to read."""

    path = Path(manifest_path).expanduser().absolute()
    if path.is_symlink():
        raise FixtureError(f"fixture manifest must not be a symlink: {path}")
    manifest = load_manifest(path)
    manifest_data = path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_data)
    if manifest_sha256 != FIXTURE_MANIFEST_SHA256:
        raise FixtureError(
            "fixture manifest SHA-256 changed: "
            f"expected {FIXTURE_MANIFEST_SHA256}, got {manifest_sha256}"
        )
    if len(manifest_data) != FIXTURE_MANIFEST_BYTES:
        raise FixtureError(
            "fixture manifest byte count changed: "
            f"expected {FIXTURE_MANIFEST_BYTES}, got {len(manifest_data)}"
        )
    fixture_root = path.parent.resolve(strict=True)
    project_root = Path(project_root).expanduser().absolute().resolve(strict=True)

    upstream = _validate_upstream_script(manifest.get("upstream_script"), project_root)
    source_store = dict(
        _require_mapping(manifest.get("source_store"), "manifest.source_store")
    )
    dev_manifest_origin = dict(
        _require_mapping(
            manifest.get("dev_manifest_origin"), "manifest.dev_manifest_origin"
        )
    )
    _assert_absolute_source_path(
        dev_manifest_origin.get("source_path"),
        "manifest.dev_manifest_origin.source_path",
    )
    if not isinstance(dev_manifest_origin.get("present_at_preparation"), bool):
        raise FixtureError("dev manifest presence record must be boolean")
    if dev_manifest_origin["present_at_preparation"]:
        _require_sha256(
            dev_manifest_origin.get("sha256"),
            "manifest.dev_manifest_origin.sha256",
        )
        _require_int(
            dev_manifest_origin.get("bytes"),
            "manifest.dev_manifest_origin.bytes",
        )
    else:
        if (
            dev_manifest_origin.get("sha256") is not None
            or dev_manifest_origin.get("bytes") is not None
        ):
            raise FixtureError("missing dev manifest must not claim a hash or byte count")

    artifacts = _validate_artifacts(manifest.get("artifacts"), fixture_root)
    cases = _validate_protocol(manifest.get("protocol"))
    _validate_case_artifact_links(cases, artifacts)
    _validate_assessment_bindings(artifacts, source_store)

    return FixtureLock(
        manifest_path=path,
        manifest_sha256=manifest_sha256,
        manifest_bytes=len(manifest_data),
        fixture_root=fixture_root,
        upstream_script=upstream,
        artifacts=artifacts,
        cases=cases,
        source_store=MappingProxyType(source_store),
        dev_manifest_origin=MappingProxyType(dev_manifest_origin),
    )


def assert_upstream_protocol(
    dev_reference_sha256: str,
    holdout_references: Mapping[str, tuple[str, Sequence[int], Sequence[int]]],
    expected_model_sha256: str,
) -> None:
    if dev_reference_sha256 != DEV_REFERENCE_SHA256:
        raise FixtureError("upstream dev reference SHA-256 changed")
    normalized: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
    for suffix, record in holdout_references.items():
        if len(record) != 3:
            raise FixtureError(f"upstream holdout record {suffix} is malformed")
        normalized[str(suffix)] = (
            str(record[0]),
            tuple(int(value) for value in record[1]),
            tuple(int(value) for value in record[2]),
        )
    if normalized != HOLDOUT_REFERENCES:
        raise FixtureError("upstream holdout reference protocol changed")
    if expected_model_sha256 in REFERENCE_SHA256:
        raise FixtureError("model SHA-256 must not be used as a reference WAV SHA-256")
    if expected_model_sha256 != EXPECTED_MODEL_SHA256:
        raise FixtureError("upstream scorer model SHA-256 changed")


def fixture_evidence(lock: FixtureLock) -> dict[str, Any]:
    return {
        "schema": "itu_omlsa_tuning_fixture_evidence.v1",
        "fixture_id": FIXTURE_ID,
        "manifest": {
            "path": str(lock.manifest_path),
            "sha256": lock.manifest_sha256,
            "bytes": lock.manifest_bytes,
        },
        "upstream_script": lock.upstream_script.evidence(),
        "source_store": dict(lock.source_store),
        "dev_manifest_origin": dict(lock.dev_manifest_origin),
        "artifacts": [artifact.evidence() for artifact in lock.artifacts],
        "cases": [case.evidence() for case in lock.cases],
        "source_lock": {
            "artifact_count": len(lock.artifacts),
            "case_count": len(lock.cases),
            "reference_count": sum(
                artifact.kind == "reference" for artifact in lock.artifacts
            ),
            "assessment_count": sum(
                artifact.kind == "assessment" for artifact in lock.artifacts
            ),
            "verified": True,
        },
    }


def load_dev_spec(
    case_spec_type: Any,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[Any, dict[str, Any]]:
    lock = freeze_source_lock(manifest_path)
    case = next(case for case in lock.cases if case.role == "dev")
    reference = lock.artifact(case.reference_artifact_id)
    spec = case_spec_type(
        case_id=case.case_id,
        role=case.role,
        source_case_id=case.source_case_id,
        reference_path=reference.path,
        reference_sha256=reference.sha256,
        assessment_path=None,
        noise_seed=case.noise_seed,
        target_snr_db=case.target_snr_db,
        mos_eligible=case.mos_eligible,
        synthetic=case.synthetic,
    )
    evidence = fixture_evidence(lock)
    evidence.update(
        {
            "path": str(lock.manifest_path),
            "exists": True,
            "mode": "fixed_fixture",
            "fixed_dev_metadata": case.evidence(),
            "reference": reference.evidence(),
        }
    )
    return spec, evidence


def load_holdout_specs(
    case_spec_type: Any,
    manifest_path: Path = MANIFEST_PATH,
) -> list[Any]:
    lock = freeze_source_lock(manifest_path)
    specs: list[Any] = []
    for case in lock.cases:
        if case.role != "holdout":
            continue
        reference = lock.artifact(case.reference_artifact_id)
        assessment = lock.artifact(case.assessment_artifact_id or "")
        specs.append(
            case_spec_type(
                case_id=case.case_id,
                role=case.role,
                source_case_id=case.source_case_id,
                reference_path=reference.path,
                reference_sha256=reference.sha256,
                assessment_path=assessment.path,
                noise_seed=case.noise_seed,
                target_snr_db=case.target_snr_db,
                mos_eligible=case.mos_eligible,
                synthetic=case.synthetic,
            )
        )
    if len(specs) != 6:
        raise FixtureError(f"holdout invariant failed: expected 6 examples, got {len(specs)}")
    return specs
