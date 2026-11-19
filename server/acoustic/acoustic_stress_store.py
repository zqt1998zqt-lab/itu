"""Validated immutable store for echo and double-talk stress workloads."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf

from . import service
from . import acceptance_scene_conditions as legacy
from .acceptance_contract import required_conditions
from .acoustic_stress_profiles import (
    CATEGORIES,
    StressProfile,
    stress_profile_for_dataset,
)


StressDatasetError = legacy.SceneConditionDatasetError
ROLES = legacy.ROLES

SCHEMA_VERSION = "acoustic-stress-preview-v1"
DATASET_KIND = "acoustic_stress_preview"
TEST_SCOPE = "stress"
SCENE = "baseline"
CONTRACT_ITEM_COUNT = 0
DURATION_S = 12.0
EXPECTED_CASE_COUNT = 100
EXPECTED_OBSERVATION_COUNT = 100
RUNTIME_BINDING_SCHEMA = "acoustic-stress-runtime-binding-v1"
RUNTIME_REQUIRED_SOURCES = frozenset(
    {
        "server/acoustic/acoustic_stress_profiles.py",
        "scripts/create-acoustic-stress-datasets.py",
        "scripts/acoustic_stress_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "scripts/acceptance_signal_design.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
    }
)
ARTIFACT_NAMES = frozenset(
    {"manifest.json", "assets.json", "summary.json", "quality.json"}
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_name",
        "dataset_kind",
        "test_scope",
        "mode",
        "scene",
        "contract_item_count",
        "preview_condition_group_count",
        "standard_compliance",
        "formal_test_status",
        "cases",
    }
)
RUNTIME_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "status",
        "source_sha256",
        "artifact_sha256",
    }
)

_LOAD_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_LOAD_CACHE_LOCK = Lock()
_ACTIVITY_TOLERANCE_S = 1e-6
_ACTIVITY_TEMPLATES = {
    ("echo", "steady_echo"): (
        (0.0, 12.0, "far_only"),
    ),
    ("echo", "convergence"): (
        (0.0, 0.5, "quiet"),
        (0.5, 12.0, "far_only"),
    ),
    ("echo", "delay_change"): (
        (0.0, 12.0, "far_only"),
    ),
    ("echo", "path_change"): (
        (0.0, 12.0, "far_only"),
    ),
    ("echo", "noisy_echo"): (
        (0.0, 12.0, "far_only"),
    ),
    ("double_talk", "balanced"): (
        (0.0, 12.0, "double_talk"),
    ),
    ("double_talk", "near_dominant"): (
        (0.0, 12.0, "double_talk"),
    ),
    ("double_talk", "far_dominant"): (
        (0.0, 12.0, "double_talk"),
    ),
    ("double_talk", "transition"): (
        (0.0, 0.5, "quiet"),
        (0.5, 2.0, "far_only"),
        (2.0, 5.0, "near_only"),
        (5.0, 8.0, "far_only"),
        (8.0, 11.0, "double_talk"),
        (11.0, 12.0, "noise_only"),
    ),
    ("double_talk", "short_overlap"): (
        (0.0, 0.5, "quiet"),
        (0.5, 6.5, "near_only"),
        (6.5, 7.0, "double_talk"),
        (7.0, 12.0, "far_only"),
    ),
}
_ECHO_PATH_TEMPLATES = {
    "steady_echo": ((0.0, 12.0, "steady", (20.0, 25.0)),),
    "convergence": ((0.5, 12.0, "steady", (20.0, 25.0)),),
    "delay_change": (
        (0.0, 6.0, "steady", (20.0, 25.0)),
        (6.0, 12.0, "late", (80.0, 90.0)),
    ),
    "path_change": (
        (0.0, 6.0, "steady", (20.0, 25.0)),
        (6.0, 12.0, "alternate", (20.0, 25.0)),
    ),
    "noisy_echo": ((0.0, 12.0, "steady", (20.0, 25.0)),),
}
_ECHO_TAP_OFFSETS_MS = {
    "steady": {
        0: (0.0, 11.5, 27.0, 53.0),
        1: (0.0, 15.5, 34.0, 61.0),
    },
    "late": {
        0: (0.0, 17.0, 39.0, 71.0),
        1: (0.0, 21.0, 46.0, 83.0),
    },
    "alternate": {
        0: (0.0, 5.0, 39.0, 67.0),
        1: (0.0, 8.0, 43.0, 79.0),
    },
}
_ECHO_RAW_GAINS = {
    "steady": {
        0: (1.0, 0.45, 0.24, 0.11),
        1: (0.96, 0.36, 0.29, 0.15),
    },
    "late": {
        0: (0.82, 0.51, 0.31, 0.17),
        1: (0.79, 0.47, 0.35, 0.21),
    },
    "alternate": {
        0: (0.78, 0.22, 0.61, 0.33),
        1: (0.71, 0.55, 0.18, 0.46),
    },
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(float(value)))
    )


def dataset_root(dataset_id: str) -> Path:
    if not isinstance(dataset_id, str) or len(dataset_id) != 32:
        raise StressDatasetError("acoustic stress dataset id is invalid")
    if any(character not in "0123456789abcdef" for character in dataset_id):
        raise StressDatasetError("acoustic stress dataset id is invalid")
    return (
        Path(service.BUILD)
        / "acoustic-stress-datasets"
        / dataset_id
    )


def _stress_rate(mode: str) -> int:
    if mode.startswith("SWB-"):
        return 48_000
    if mode.startswith("NB-"):
        return 8_000
    if mode.startswith("WB-"):
        return 16_000
    raise StressDatasetError(f"unsupported acoustic stress mode: {mode}")


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    return legacy._read_json(path)


def _artifact_sha256(root: Path, name: str) -> tuple[str, bytes]:
    path = root / name
    try:
        metadata = path.lstat()
    except OSError as error:
        raise StressDatasetError(
            f"acoustic stress artifact is missing: {path}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise StressDatasetError(
            f"acoustic stress artifact is not a regular file: {path}"
        )
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), payload


def _validate_runtime_binding(
    root: Path,
    dataset_id: str,
) -> dict[str, str]:
    binding, _ = _read_json(root / "runtime-contract-binding.json")
    if (
        not isinstance(binding, dict)
        or set(binding) != set(RUNTIME_BINDING_FIELDS)
        or binding.get("schema_version") != RUNTIME_BINDING_SCHEMA
        or binding.get("dataset_id") != dataset_id
        or binding.get("status") != "reconciled"
        or not isinstance(binding.get("source_sha256"), dict)
        or not binding["source_sha256"]
        or not RUNTIME_REQUIRED_SOURCES.issubset(
            set(binding["source_sha256"])
        )
        or not isinstance(binding.get("artifact_sha256"), dict)
        or set(binding["artifact_sha256"]) != ARTIFACT_NAMES
    ):
        raise StressDatasetError("acoustic stress runtime binding is invalid")

    project_root = Path(__file__).resolve().parents[2]
    source_hashes: dict[str, str] = {}
    for relative, expected in binding["source_sha256"].items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise StressDatasetError(
                "acoustic stress runtime source path is unsafe"
            )
        source = (project_root / relative).resolve(strict=True)
        if (
            not source.is_relative_to(project_root)
            or not source.is_file()
            or not isinstance(expected, str)
            or legacy._SHA256.fullmatch(expected) is None
            or legacy._sha256_file(source) != expected
        ):
            raise StressDatasetError(
                f"acoustic stress runtime source hash differs: {relative}"
            )
        source_hashes[relative] = expected

    artifact_hashes: dict[str, str] = {}
    for name, expected in binding["artifact_sha256"].items():
        if not isinstance(expected, str) or legacy._SHA256.fullmatch(expected) is None:
            raise StressDatasetError(
                f"acoustic stress artifact hash is invalid: {name}"
            )
        actual, _ = _artifact_sha256(root, name)
        if actual != expected:
            raise StressDatasetError(
                f"acoustic stress artifact hash differs: {name}"
            )
        artifact_hashes[name] = actual
    return {
        "runtime_binding": hashlib.sha256(
            (root / "runtime-contract-binding.json").read_bytes()
        ).hexdigest(),
        **{f"artifact:{name}": digest for name, digest in artifact_hashes.items()},
        **{f"source:{name}": digest for name, digest in source_hashes.items()},
    }


def _store_signature(root: Path) -> tuple[Any, ...]:
    signature = list(legacy._store_signature(root))
    for relative in (
        "runtime-contract-binding.json",
        "summary.json",
        "quality.json",
    ):
        metadata = (root / relative).lstat()
        signature.extend(
            (
                relative,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        )
    return tuple(signature)


def _same_time(first: Any, second: float) -> bool:
    return (
        _finite_number(first)
        and abs(float(first) - float(second)) <= _ACTIVITY_TOLERANCE_S
    )


def activity_template(
    scope: str,
    category: str,
) -> tuple[tuple[float, float, str], ...]:
    try:
        return _ACTIVITY_TEMPLATES[(scope, category)]
    except (KeyError, TypeError) as error:
        raise StressDatasetError(
            "stress workload category has no activity template"
        ) from error


def _validate_echo_path_segments(
    provenance: Mapping[str, Any],
    *,
    scope: str,
    category: str,
    rate: int,
    require_crossbind: bool,
) -> None:
    if scope != "echo":
        return
    render_metadata = provenance.get("render_metadata")
    if not isinstance(render_metadata, Mapping):
        raise StressDatasetError(
            "echo stress provenance.render_metadata must be an object"
        )
    if require_crossbind and (
        render_metadata.get("workload_scope") != scope
        or render_metadata.get("workload_category") != category
        or render_metadata.get("sample_rate_hz") != rate
        or render_metadata.get("activity_segments")
        != provenance.get("activity_segments")
        or not _same_time(
            render_metadata.get("analysis_start_s"),
            provenance.get("analysis_start_s"),
        )
    ):
        raise StressDatasetError(
            "echo render_metadata is not cross-bound to provenance"
        )
    path_segments = render_metadata.get("echo_path_segments")
    expected = _ECHO_PATH_TEMPLATES[category]
    if not isinstance(path_segments, list) or len(path_segments) != len(expected):
        raise StressDatasetError(
            "echo path segments differ from the category template"
        )
    for index, (raw_segment, template) in enumerate(
        zip(path_segments, expected, strict=True)
    ):
        start, end, variant, delays = template
        if not isinstance(raw_segment, Mapping):
            raise StressDatasetError(
                f"echo path segment {index} must be an object"
            )
        paths = raw_segment.get("paths")
        if (
            set(raw_segment) != {"start_s", "end_s", "paths"}
            or not _same_time(raw_segment.get("start_s"), start)
            or not _same_time(raw_segment.get("end_s"), end)
            or not isinstance(paths, list)
            or len(paths) != 2
        ):
            raise StressDatasetError(
                f"echo path segment {index} differs from the category template"
            )
        for channel, path in enumerate(paths):
            if (
                not isinstance(path, Mapping)
                or set(path)
                != {
                    "channel",
                    "variant",
                    "delay_ms",
                    "tap_samples",
                    "tap_offsets_ms",
                    "tap_gains",
                    "echo_gain_linear",
                    "tap_l1",
                }
                or path.get("channel") != channel
                or path.get("variant") != variant
                or not _finite_number(path.get("delay_ms"))
                or abs(float(path["delay_ms"]) - delays[channel]) > 1e-6
            ):
                raise StressDatasetError(
                    f"echo path segment {index} channel {channel} is invalid"
                )
            delay_ms = float(delays[channel])
            offsets = _ECHO_TAP_OFFSETS_MS[variant][channel]
            raw_gains = _ECHO_RAW_GAINS[variant][channel]
            normalizer = sum(abs(gain) for gain in raw_gains)
            expected_samples = [
                int(round(delay_ms * rate / 1000.0))
                + int(round(offset_ms * rate / 1000.0))
                for offset_ms in offsets
            ]
            expected_offsets = [
                round(delay_ms + offset_ms, 9)
                for offset_ms in offsets
            ]
            expected_gains = [
                float(gain / normalizer)
                for gain in raw_gains
            ]
            samples = path.get("tap_samples")
            offsets_value = path.get("tap_offsets_ms")
            gains = path.get("tap_gains")
            tap_l1 = path.get("tap_l1")
            if (
                not isinstance(samples, list)
                or len(samples) != len(expected_samples)
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value != expected
                    for value, expected in zip(samples, expected_samples)
                )
                or not isinstance(offsets_value, list)
                or len(offsets_value) != len(expected_offsets)
                or any(
                    not _finite_number(value)
                    or abs(float(value) - expected) > 1e-9
                    for value, expected in zip(offsets_value, expected_offsets)
                )
                or not isinstance(gains, list)
                or len(gains) != len(expected_gains)
                or any(
                    not _finite_number(value)
                    or abs(float(value) - expected) > 1e-12
                    for value, expected in zip(gains, expected_gains)
                )
                or not _finite_number(tap_l1)
                or abs(float(tap_l1) - sum(abs(value) for value in expected_gains))
                > 1e-12
                or not _finite_number(path.get("echo_gain_linear"))
                or abs(float(path["echo_gain_linear"]) - 1.0) > 1e-12
            ):
                raise StressDatasetError(
                    f"echo path segment {index} channel {channel} taps differ"
                )


def validate_activity_provenance(
    provenance: Mapping[str, Any],
    *,
    scope: str,
    mode: str,
    require_source_flags: bool = True,
) -> list[dict[str, Any]]:
    """Validate the exact generator timeline and source provenance."""
    if not isinstance(provenance, Mapping):
        raise StressDatasetError("stress provenance must be an object")
    category = provenance.get("workload_category")
    if not isinstance(category, str) or category not in CATEGORIES.get(scope, {}):
        raise StressDatasetError("stress workload_category is invalid")
    rate = _stress_rate(mode)
    if require_source_flags:
        if rate not in {8_000, 16_000, 48_000}:
            raise StressDatasetError("stress activity rate is unsupported")
        expected_conversation_mode = (
            "far_single"
            if scope == "echo"
            else "transition"
            if category == "transition"
            else "double_talk"
        )
        expected_analysis_start = (
            0.5
            if category in {"convergence", "transition", "short_overlap"}
            else 0.0
        )
        if (
            provenance.get("scope") != scope
            or provenance.get("mode") != mode
            or provenance.get("scene") != SCENE
            or provenance.get("conversation_mode")
            != expected_conversation_mode
            or not _same_time(
                provenance.get("analysis_start_s"),
                expected_analysis_start,
            )
            or provenance.get("engineering_preview") is not True
            or provenance.get("engineering_combination") is not True
            or provenance.get("independent_source_recordings") is not False
            or provenance.get("actual_vehicle_recording") is not False
            or provenance.get("within_case_source_looping") is not False
        ):
            raise StressDatasetError(
                "stress provenance scope/mode/flags differ from generator"
            )
    if scope == "echo" and (
        require_source_flags or "render_metadata" in provenance
    ):
        _validate_echo_path_segments(
            provenance,
            scope=scope,
            category=category,
            rate=rate,
            require_crossbind=require_source_flags,
        )
    segments = provenance.get("activity_segments")
    expected_segments = activity_template(scope, category)
    if not isinstance(segments, list) or len(segments) != len(expected_segments):
        raise StressDatasetError(
            "stress activity_segments differ from the category template"
        )
    normalized: list[dict[str, Any]] = []
    for index, (raw_segment, expected) in enumerate(
        zip(segments, expected_segments, strict=True)
    ):
        if (
            not isinstance(raw_segment, Mapping)
            or set(raw_segment) != {"start_s", "end_s", "label"}
        ):
            raise StressDatasetError(
                f"stress activity segment {index} has invalid fields"
            )
        start, end, label = expected
        if (
            not _same_time(raw_segment.get("start_s"), start)
            or not _same_time(raw_segment.get("end_s"), end)
            or raw_segment.get("label") != label
        ):
            raise StressDatasetError(
                f"stress activity segment {index} differs from category template"
            )
        normalized.append(
            {
                "start_s": float(start),
                "end_s": float(end),
                "label": label,
            }
        )
    if abs(normalized[-1]["end_s"] - DURATION_S) > _ACTIVITY_TOLERANCE_S:
        raise StressDatasetError(
            "stress activity segments must cover the 12 second clip"
        )
    return normalized


def _validate_activity_segments(
    provenance: Mapping[str, Any],
    *,
    scope: str,
    rate: int,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    resolved_mode = mode if mode is not None else provenance.get("mode")
    if not isinstance(resolved_mode, str) or _stress_rate(resolved_mode) != rate:
        raise StressDatasetError("stress activity mode/rate is invalid")
    return validate_activity_provenance(
        provenance,
        scope=scope,
        mode=resolved_mode,
        require_source_flags=True,
    )


def _validate_case(
    case: dict[str, Any],
    *,
    profile: StressProfile,
    root: Path,
    assets: dict[str, Any],
    contract_by_id: dict[str, dict[str, Any]],
    seen_case_ids: set[str],
    category_indices: dict[str, set[int]],
    referenced_assets: set[str],
) -> None:
    if set(case) != set(legacy._CASE_FIELDS):
        raise StressDatasetError(
            "acoustic stress case fields differ from the legacy case contract"
        )
    case_id = case.get("case_id")
    if (
        not isinstance(case_id, str)
        or not case_id
        or case_id in seen_case_ids
    ):
        raise StressDatasetError(
            "acoustic stress case ids must be unique non-empty strings"
        )
    seen_case_ids.add(case_id)
    item_ids = case.get("contract_item_ids")
    if (
        not isinstance(item_ids, list)
        or len(item_ids) != 1
        or not isinstance(item_ids[0], str)
        or item_ids[0] not in contract_by_id
    ):
        raise StressDatasetError(
            f"stress case {case_id!r} must bind exactly one scope contract row"
        )
    item_id = item_ids[0]
    contract = contract_by_id[item_id]
    expected_conditions = required_conditions(contract)
    if case.get("conditions") != expected_conditions:
        raise StressDatasetError(
            f"stress case {case_id!r} conditions do not match required_conditions"
        )
    families = case.get("families")
    if (
        not isinstance(families, list)
        or families != [contract["family"]]
    ):
        raise StressDatasetError(
            f"stress case {case_id!r} families do not match its contract row"
        )

    provenance = case.get("provenance")
    if not isinstance(provenance, dict):
        raise StressDatasetError(
            f"stress case {case_id!r} provenance is invalid"
        )
    category = provenance.get("workload_category")
    index = provenance.get("category_index")
    if (
        not isinstance(category, str)
        or category not in CATEGORIES[profile.scope]
        or not isinstance(index, int)
        or isinstance(index, bool)
        or not 1 <= index <= 20
    ):
        raise StressDatasetError(
            f"stress case {case_id!r} workload category/index is invalid"
        )
    category_indices[category].add(index)
    rate = _stress_rate(profile.mode)
    activity_segments = _validate_activity_segments(
        provenance,
        scope=profile.scope,
        rate=rate,
        mode=profile.mode,
    )
    if (
        provenance.get("contract_item_id") != item_id
        or provenance.get("conditions") != expected_conditions
    ):
        raise StressDatasetError(
            f"stress case {case_id!r} provenance contract/conditions differ"
        )

    signals = case.get("signals")
    hashes = case.get("signals_sha256")
    if (
        not isinstance(signals, dict)
        or set(signals) != set(ROLES)
        or not isinstance(hashes, dict)
        or set(hashes) != set(ROLES)
    ):
        raise StressDatasetError(
            f"stress case {case_id!r} has incomplete signal roles"
        )
    for role in ROLES:
        relative = signals[role]
        expected_sha = hashes[role]
        if (
            not isinstance(expected_sha, str)
            or legacy._SHA256.fullmatch(expected_sha) is None
        ):
            raise StressDatasetError(
                f"stress case {case_id!r} has invalid {role} hash"
            )
        normalized = legacy._safe_relative_path(relative)
        referenced_assets.add(normalized)
        metadata = assets.get(normalized)
        if not isinstance(metadata, dict):
            raise StressDatasetError(
                f"stress asset is missing: {normalized}"
            )
        if metadata.get("sha256") != expected_sha:
            raise StressDatasetError(
                f"stress asset hash differs: {normalized}"
            )
        path = legacy.asset_path(root, normalized)
        try:
            info = sf.info(path)
        except (OSError, RuntimeError, ValueError) as error:
            raise StressDatasetError(
                f"stress WAV is unreadable: {normalized}: {error}"
            ) from error
        if (
            int(info.samplerate) != rate
            or int(info.frames) != int(rate * DURATION_S)
            or int(info.channels) != (2 if role == "microphone" else 1)
        ):
            raise StressDatasetError(
                f"stress WAV shape differs for {normalized}"
            )
        if int(metadata.get("size_bytes")) != path.stat().st_size:
            raise StressDatasetError(
                f"stress asset size differs: {normalized}"
            )
        if legacy._sha256_file(path) != expected_sha:
            raise StressDatasetError(
                f"stress asset content differs: {normalized}"
            )
    audio: dict[str, np.ndarray] = {}
    for role in ROLES:
        path = legacy.asset_path(root, signals[role])
        samples, _ = sf.read(path, dtype="float64", always_2d=True)
        audio[role] = np.asarray(samples[:, 0], dtype=np.float64)
    if profile.scope == "echo":
        if np.max(np.abs(audio["reference"])) > 1e-7:
            raise StressDatasetError(
                f"echo case {case_id!r} reference is not near-zero"
            )
        far_segments = [
            segment
            for segment in activity_segments
            if segment["label"] == "far_only"
        ]
        far_active = np.concatenate(
            [
                audio["far_end"][
                    int(round(segment["start_s"] * rate)):
                    int(round(segment["end_s"] * rate))
                ]
                for segment in far_segments
            ]
        )
        mic_active = np.concatenate(
            [
                audio["microphone"][
                    int(round(segment["start_s"] * rate)):
                    int(round(segment["end_s"] * rate))
                ]
                for segment in far_segments
            ]
        )
        if (
            not len(far_active)
            or float(np.sqrt(np.mean(np.square(far_active)))) <= 1e-7
            or float(np.sqrt(np.mean(np.square(mic_active)))) <= 1e-7
        ):
            raise StressDatasetError(
                f"echo case {case_id!r} lacks active far_only evidence"
            )
    elif profile.scope == "double_talk":
        for segment in activity_segments:
            if segment["label"] != "double_talk":
                continue
            start = int(round(segment["start_s"] * rate))
            end = int(round(segment["end_s"] * rate))
            if (
                float(np.sqrt(np.mean(np.square(audio["reference"][start:end]))))
                <= 1e-7
                or float(np.sqrt(np.mean(np.square(audio["far_end"][start:end]))))
                <= 1e-7
            ):
                raise StressDatasetError(
                    f"double-talk case {case_id!r} lacks both active ends"
                )


def _load_stress_dataset_uncached(dataset_id: str) -> dict[str, Any]:
    profile = stress_profile_for_dataset(dataset_id)
    root = dataset_root(dataset_id)
    if root.is_symlink():
        raise StressDatasetError(
            "acoustic stress dataset store cannot be a symbolic link"
        )
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise StressDatasetError(
            f"acoustic stress dataset store is missing: {root}"
        ) from error
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise StressDatasetError(
            "acoustic stress dataset store is not a regular directory"
        )
    hashes = _validate_runtime_binding(resolved_root, dataset_id)
    manifest, manifest_bytes = _read_json(resolved_root / "manifest.json")
    assets, assets_bytes = _read_json(resolved_root / "assets.json")
    if set(manifest) != set(MANIFEST_FIELDS):
        raise StressDatasetError(
            "acoustic stress manifest fields differ from the legacy set"
        )
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": DATASET_KIND,
        "test_scope": profile.scope,
        "mode": profile.mode,
        "scene": SCENE,
        "contract_item_count": profile.contract_item_count,
        "preview_condition_group_count": EXPECTED_CASE_COUNT,
        "standard_compliance": False,
        "formal_test_status": "not_measured",
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise StressDatasetError(
                f"acoustic stress manifest field {key!r} is invalid"
            )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise StressDatasetError(
            "acoustic stress manifest must contain exactly 100 cases"
        )
    if not isinstance(assets, dict):
        raise StressDatasetError("acoustic stress assets must be an object")

    contracts = profile.contract_rows
    contract_by_id = {row["id"]: row for row in contracts}
    if len(contract_by_id) != profile.contract_item_count:
        raise StressDatasetError(
            "acoustic stress profile contract rows are not unique"
        )
    category_indices = {
        category: set() for category in CATEGORIES[profile.scope]
    }
    seen_case_ids: set[str] = set()
    referenced_assets: set[str] = set()
    for case in cases:
        _validate_case(
            case,
            profile=profile,
            root=resolved_root,
            assets=assets,
            contract_by_id=contract_by_id,
            seen_case_ids=seen_case_ids,
            category_indices=category_indices,
            referenced_assets=referenced_assets,
        )
    for category, indices in category_indices.items():
        if indices != set(range(1, 21)):
            raise StressDatasetError(
                f"stress category {category!r} must contain indices 1..20"
            )
    if set(assets) != referenced_assets:
        raise StressDatasetError(
            "acoustic stress assets contain unreferenced or missing paths"
        )
    return {
        "dataset_id": dataset_id,
        "dataset_name": profile.dataset_name,
        "dataset_kind": DATASET_KIND,
        "schema_version": SCHEMA_VERSION,
        "test_scope": profile.scope,
        "mode": profile.mode,
        "scene": SCENE,
        "root": resolved_root,
        "manifest": manifest,
        "assets": assets,
        "cases": cases,
        "hashes": {
            "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
            "assets": hashlib.sha256(assets_bytes).hexdigest(),
            **hashes,
        },
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_assets_sha256": hashlib.sha256(assets_bytes).hexdigest(),
        "counts": {
            "cases": len(cases),
            "observations": len(cases),
            "categories": len(CATEGORIES[profile.scope]),
        },
        "category_counts": {
            category: len(indices)
            for category, indices in category_indices.items()
        },
        "contracts": contracts,
        "contract_by_id": contract_by_id,
        "contract_item_count": profile.contract_item_count,
        "contract_item_ids": profile.contract_item_ids,
        "preview_condition_group_count": len(cases),
        "observation_count": len(cases),
        "case_count": len(cases),
        "referenced_assets": referenced_assets,
    }


def load_stress_dataset(dataset_id: str) -> dict[str, Any]:
    profile = stress_profile_for_dataset(dataset_id)
    root = dataset_root(profile.dataset_id)
    if root.is_symlink():
        raise StressDatasetError(
            "acoustic stress dataset store cannot be a symbolic link"
        )
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise StressDatasetError(
            f"acoustic stress dataset store is missing: {root}"
        ) from error
    try:
        signature = _store_signature(resolved_root)
    except OSError as error:
        raise StressDatasetError(
            f"cannot inspect acoustic stress dataset: {error}"
        ) from error
    cache_key = str(resolved_root)
    with _LOAD_CACHE_LOCK:
        cached = _LOAD_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
    value = _load_stress_dataset_uncached(dataset_id)
    with _LOAD_CACHE_LOCK:
        _LOAD_CACHE[cache_key] = (signature, dict(value))
    return value


__all__ = [
    "ARTIFACT_NAMES",
    "DATASET_KIND",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_OBSERVATION_COUNT",
    "MANIFEST_FIELDS",
    "ROLES",
    "RUNTIME_BINDING_FIELDS",
    "RUNTIME_BINDING_SCHEMA",
    "RUNTIME_REQUIRED_SOURCES",
    "SCENE",
    "SCHEMA_VERSION",
    "StressDatasetError",
    "TEST_SCOPE",
    "dataset_root",
    "load_stress_dataset",
]
