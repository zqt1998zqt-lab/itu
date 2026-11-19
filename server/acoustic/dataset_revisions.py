"""Versioned dataset identities and guarded activation for the ITU previews."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any

from . import acoustic_stress_profiles as stress_profiles
from . import scene_mode_conversation_profiles as scene_profiles
from . import service
from .acceptance_scope import ACCEPTANCE_MODES


DATASET_REVISION = "measurement-20261117-r1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVATION_PATH = (
    PROJECT_ROOT / "build" / "dataset-revision-20261117" / "ACTIVATION.json"
)

SCENES = ("sce1", "sce2", "sce3", "sce4")
MODES = tuple(sorted(ACCEPTANCE_MODES))
STRESS_SCOPES = ("echo", "double_talk")

SCENE_ROOT = "acoustic-scene-condition-datasets"
STRESS_ROOT = "acoustic-stress-datasets"
ARTIFACT_NAMES = frozenset(
    {
        "manifest.json",
        "summary.json",
        "assets.json",
        "quality.json",
        "runtime-contract-binding.json",
    }
)

_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SCENE_BINDING_SCHEMA = "scene-condition-runtime-binding-v2"
_STRESS_BINDING_SCHEMA = "acoustic-stress-runtime-binding-v1"
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "status",
        "source_sha256",
        "artifact_sha256",
    }
)
_SCENE_REQUIRED_SOURCES = frozenset(
    {
        "server/acoustic/scene_conversation_profiles.py",
        "scripts/acceptance_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
        "server/acoustic/dataset_revisions.py",
        "scripts/create-revised-acoustic-datasets.py",
        "scripts/dataset_revision_sources.py",
        "scripts/dataset_revision_signal_design.py",
    }
)
_STRESS_REQUIRED_SOURCES = frozenset(
    {
        "server/acoustic/acoustic_stress_profiles.py",
        "scripts/create-acoustic-stress-datasets.py",
        "scripts/acoustic_stress_signal_design.py",
        "scripts/create-background-noise-conversation-dataset.py",
        "scripts/create-background-noise-scene-condition-dataset.py",
        "scripts/acceptance_signal_design.py",
        "server/acoustic/acceptance_scope.py",
        "cpp/evaluation/nv200-test-contract.json",
        "server/acoustic/dataset_revisions.py",
        "scripts/create-revised-acoustic-datasets.py",
        "scripts/dataset_revision_sources.py",
        "scripts/dataset_revision_signal_design.py",
        "scripts/create-acoustic-stress-mode-datasets.py",
    }
)


class DatasetRevisionError(ValueError):
    """A revision identity or activation receipt is invalid."""


class DatasetActivationError(DatasetRevisionError):
    """A published activation receipt failed closed."""


@dataclass(frozen=True)
class RevisionStressProfile(stress_profiles.StressProfile):
    """Legacy stress schema with a revision-scoped internal dataset identity."""

    @property
    def dataset_id(self) -> str:
        return _stress_dataset_id(self.scope, self.mode)


def _scene_dataset_id(scene: str, mode: str) -> str:
    return hashlib.sha256(
        f"{DATASET_REVISION}:scene:{scene}:{mode}".encode("utf-8")
    ).hexdigest()[:32]


def _stress_dataset_id(scope: str, mode: str) -> str:
    return hashlib.sha256(
        f"{DATASET_REVISION}:stress:{scope}:{mode}".encode("utf-8")
    ).hexdigest()[:32]


def _revision_scene_profile(
    scene: str,
    mode: str,
) -> scene_profiles.ModeConversationProfile:
    try:
        legacy = scene_profiles.MODE_CONVERSATION_PROFILES[(scene, mode)]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unknown scene-condition scene/mode: {scene}/{mode}"
        ) from error
    return scene_profiles.ModeConversationProfile(
        scene=legacy.scene,
        mode=mode,
        dataset_id=_scene_dataset_id(scene, mode),
        dataset_name=legacy.dataset_name,
        contract_item_ids=legacy.contract_item_ids,
        snr_db=legacy.snr_db,
        sample_rate_hz=scene_profiles.sample_rate_for_mode(mode),
        dataset_kind=legacy.dataset_kind,
        schema_version=legacy.schema_version,
        contract_item_count=legacy.contract_item_count,
        case_count=legacy.case_count,
        speech_case_count=legacy.speech_case_count,
        cng_case_count=legacy.cng_case_count,
        observation_count=legacy.observation_count,
        duration_s=legacy.duration_s,
    )


REVISION_SCENE_PROFILES = {
    (scene, mode): _revision_scene_profile(scene, mode)
    for scene in SCENES
    for mode in MODES
}
REVISION_STRESS_PROFILES = {
    (scope, mode): RevisionStressProfile(scope, mode)
    for scope in STRESS_SCOPES
    for mode in MODES
}

_REVISION_IDS = {
    profile.dataset_id
    for profile in (
        *REVISION_SCENE_PROFILES.values(),
        *REVISION_STRESS_PROFILES.values(),
    )
}
if len(_REVISION_IDS) != 48:
    raise RuntimeError("dataset revision identities are not unique")


def revision_scene_profile(
    scene: str,
    mode: str,
) -> scene_profiles.ModeConversationProfile:
    try:
        return REVISION_SCENE_PROFILES[(scene, mode)]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unknown scene-condition scene/mode: {scene}/{mode}"
        ) from error


def revision_stress_profile(scope: str, mode: str) -> RevisionStressProfile:
    try:
        return REVISION_STRESS_PROFILES[(scope, mode)]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unknown acoustic stress scope/mode: {scope}/{mode}"
        ) from error


def _register_revision_profiles() -> int:
    for profile in REVISION_SCENE_PROFILES.values():
        existing = scene_profiles.PROFILE_BY_DATASET_ID.get(profile.dataset_id)
        if existing is not None and existing != profile:
            raise DatasetRevisionError(
                f"scene dataset id collides with an existing identity: "
                f"{profile.dataset_id}"
            )
        scene_profiles.PROFILE_BY_DATASET_ID[profile.dataset_id] = profile
    for profile in REVISION_STRESS_PROFILES.values():
        existing = stress_profiles.BY_DATASET_ID.get(profile.dataset_id)
        if existing is not None and existing != profile:
            raise DatasetRevisionError(
                f"stress dataset id collides with an existing identity: "
                f"{profile.dataset_id}"
            )
        stress_profiles.BY_DATASET_ID[profile.dataset_id] = profile
    return len(REVISION_SCENE_PROFILES) + len(REVISION_STRESS_PROFILES)


def _activate_revision_defaults() -> None:
    scene_keys = set(REVISION_SCENE_PROFILES)
    stress_keys = set(REVISION_STRESS_PROFILES)
    if not scene_keys.issubset(scene_profiles.MODE_CONVERSATION_PROFILES):
        missing = sorted(
            scene_keys - set(scene_profiles.MODE_CONVERSATION_PROFILES)
        )
        raise DatasetActivationError(
            f"scene default inventory is incomplete: {missing}"
        )
    if not stress_keys.issubset(stress_profiles.PROFILES):
        missing = sorted(stress_keys - set(stress_profiles.PROFILES))
        raise DatasetActivationError(
            f"stress default inventory is incomplete: {missing}"
        )

    # Mutate the imported dictionaries in place so functions that captured them
    # during module import observe the one-time switch.
    scene_profiles.MODE_CONVERSATION_PROFILES.update(REVISION_SCENE_PROFILES)
    stress_profiles.PROFILES.update(REVISION_STRESS_PROFILES)


def install_revision_profiles(*, activate: bool = False) -> dict[str, object]:
    """Register all 48 revision identities and optionally switch defaults."""
    registered = _register_revision_profiles()
    if activate:
        _activate_revision_defaults()
    return {
        "revision": DATASET_REVISION,
        "registered": registered,
        "activated": bool(activate),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise DatasetActivationError(
            f"{label} cannot be read: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DatasetActivationError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise DatasetActivationError(
            f"{label} fields differ (missing={missing}, extra={extra})"
        )


def _dataset_root(dataset_id: str, kind: str) -> Path:
    root_name = SCENE_ROOT if kind == "scene" else STRESS_ROOT
    return Path(service.BUILD) / root_name / dataset_id


def _verify_artifact(
    root: Path,
    name: str,
    expected: str,
    *,
    dataset_id: str,
) -> str:
    path = root / name
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DatasetActivationError(
            f"dataset {dataset_id} artifact is missing: {name}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise DatasetActivationError(
            f"dataset {dataset_id} artifact is not a regular file: {name}"
        )
    try:
        actual = _sha256_file(path)
    except OSError as error:
        raise DatasetActivationError(
            f"dataset {dataset_id} artifact cannot be read: {name}: {error}"
        ) from error
    if actual != expected:
        raise DatasetActivationError(
            f"dataset {dataset_id} artifact hash differs: {name}"
        )
    return actual


def _verify_source_binding(
    source_sha256: object,
    *,
    kind: str,
    dataset_id: str,
) -> None:
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise DatasetActivationError(
            f"dataset {dataset_id} source binding is empty or invalid"
        )
    required = (
        _SCENE_REQUIRED_SOURCES
        if kind == "scene"
        else _STRESS_REQUIRED_SOURCES
    )
    missing = sorted(required - set(source_sha256))
    if missing:
        raise DatasetActivationError(
            f"dataset {dataset_id} source binding is incomplete: {missing}"
        )
    for relative, expected in source_sha256.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise DatasetActivationError(
                f"dataset {dataset_id} source path is unsafe: {relative!r}"
            )
        if (
            not isinstance(expected, str)
            or _SHA256_PATTERN.fullmatch(expected) is None
        ):
            raise DatasetActivationError(
                f"dataset {dataset_id} source hash is invalid: {relative}"
            )
        try:
            source = (PROJECT_ROOT / relative).resolve(strict=True)
        except OSError as error:
            raise DatasetActivationError(
                f"dataset {dataset_id} source is missing: {relative}"
            ) from error
        if not source.is_relative_to(PROJECT_ROOT) or not source.is_file():
            raise DatasetActivationError(
                f"dataset {dataset_id} source escapes the project: {relative}"
            )
        try:
            actual = _sha256_file(source)
        except OSError as error:
            raise DatasetActivationError(
                f"dataset {dataset_id} source cannot be read: {relative}"
            ) from error
        if actual != expected:
            raise DatasetActivationError(
                f"dataset {dataset_id} source hash differs: {relative}"
            )


def _verify_runtime_binding(
    root: Path,
    *,
    dataset_id: str,
    kind: str,
    artifact_sha256: dict[str, str],
) -> None:
    binding = _read_json_object(
        root / "runtime-contract-binding.json",
        f"dataset {dataset_id} runtime binding",
    )
    _exact_fields(
        binding,
        _BINDING_FIELDS,
        f"dataset {dataset_id} runtime binding",
    )
    expected_schema = (
        _SCENE_BINDING_SCHEMA
        if kind == "scene"
        else _STRESS_BINDING_SCHEMA
    )
    if (
        binding.get("schema_version") != expected_schema
        or binding.get("dataset_id") != dataset_id
        or binding.get("status") != "reconciled"
    ):
        raise DatasetActivationError(
            f"dataset {dataset_id} runtime binding identity differs"
        )

    expected_artifacts = {
        name: artifact_sha256[name]
        for name in ARTIFACT_NAMES
        if name != "runtime-contract-binding.json"
    }
    if binding.get("artifact_sha256") != expected_artifacts:
        raise DatasetActivationError(
            f"dataset {dataset_id} runtime artifact binding differs"
        )
    _verify_source_binding(
        binding.get("source_sha256"),
        kind=kind,
        dataset_id=dataset_id,
    )


def _expected_receipt_profiles() -> dict[str, tuple[str, str, str]]:
    expected: dict[str, tuple[str, str, str]] = {}
    for (scene, mode), profile in REVISION_SCENE_PROFILES.items():
        expected[profile.dataset_id] = ("scene", scene, mode)
    for (scope, mode), profile in REVISION_STRESS_PROFILES.items():
        expected[profile.dataset_id] = ("stress", scope, mode)
    return expected


def _validate_receipt(receipt: dict[str, Any]) -> None:
    _exact_fields(receipt, frozenset({"revision", "datasets"}), "activation")
    if receipt.get("revision") != DATASET_REVISION:
        raise DatasetActivationError(
            f"activation revision differs: {receipt.get('revision')!r}"
        )
    datasets = receipt.get("datasets")
    if not isinstance(datasets, list):
        raise DatasetActivationError("activation datasets must be a list")
    if len(datasets) != 48:
        raise DatasetActivationError(
            f"activation must contain 48 datasets, got {len(datasets)}"
        )

    expected = _expected_receipt_profiles()
    seen: set[str] = set()
    for index, row in enumerate(datasets):
        label = f"activation dataset[{index}]"
        if not isinstance(row, dict):
            raise DatasetActivationError(f"{label} must be an object")
        kind = row.get("kind")
        if kind not in {"scene", "stress"}:
            raise DatasetActivationError(f"{label} kind is invalid")
        discriminator = "scene" if kind == "scene" else "scope"
        _exact_fields(
            row,
            frozenset(
                {
                    "dataset_id",
                    "kind",
                    "mode",
                    discriminator,
                    "artifacts",
                }
            ),
            label,
        )
        dataset_id = row.get("dataset_id")
        if (
            not isinstance(dataset_id, str)
            or _ID_PATTERN.fullmatch(dataset_id) is None
        ):
            raise DatasetActivationError(f"{label} dataset_id is invalid")
        if dataset_id in seen:
            raise DatasetActivationError(
                f"activation dataset id is duplicated: {dataset_id}"
            )
        seen.add(dataset_id)
        if dataset_id not in expected:
            raise DatasetActivationError(
                f"activation dataset id is not part of this revision: "
                f"{dataset_id}"
            )
        expected_kind, expected_discriminator, expected_mode = expected[
            dataset_id
        ]
        if (
            kind != expected_kind
            or row.get(discriminator) != expected_discriminator
            or row.get("mode") != expected_mode
        ):
            raise DatasetActivationError(
                f"activation dataset identity differs: {dataset_id}"
            )

        artifacts = row.get("artifacts")
        if (
            not isinstance(artifacts, dict)
            or set(artifacts) != ARTIFACT_NAMES
        ):
            raise DatasetActivationError(
                f"dataset {dataset_id} artifact inventory is incomplete"
            )
        normalized_artifacts: dict[str, str] = {}
        for name, digest in artifacts.items():
            if (
                not isinstance(digest, str)
                or _SHA256_PATTERN.fullmatch(digest) is None
            ):
                raise DatasetActivationError(
                    f"dataset {dataset_id} artifact hash is invalid: {name}"
                )
            normalized_artifacts[name] = digest

        root = _dataset_root(dataset_id, kind)
        for name in sorted(ARTIFACT_NAMES):
            _verify_artifact(
                root,
                name,
                normalized_artifacts[name],
                dataset_id=dataset_id,
            )
        _verify_runtime_binding(
            root,
            dataset_id=dataset_id,
            kind=kind,
            artifact_sha256=normalized_artifacts,
        )

    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise DatasetActivationError(
            f"activation dataset inventory is incomplete: {missing}"
        )


def install_from_activation() -> dict[str, object]:
    """Register revision IDs, then switch defaults only for a valid receipt."""
    registered = _register_revision_profiles()
    try:
        metadata = ACTIVATION_PATH.lstat()
    except FileNotFoundError:
        return {
            "revision": DATASET_REVISION,
            "registered": registered,
            "activation": "absent",
            "activated": False,
        }
    except OSError as error:
        raise DatasetActivationError(
            f"activation path cannot be inspected: {ACTIVATION_PATH}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or ACTIVATION_PATH.is_symlink():
        raise DatasetActivationError(
            f"activation path is not a regular file: {ACTIVATION_PATH}"
        )

    receipt = _read_json_object(ACTIVATION_PATH, "activation receipt")
    _validate_receipt(receipt)
    _activate_revision_defaults()
    return {
        "revision": DATASET_REVISION,
        "registered": registered,
        "activation": "active",
        "activated": True,
    }


__all__ = [
    "ACTIVATION_PATH",
    "ARTIFACT_NAMES",
    "DATASET_REVISION",
    "DatasetActivationError",
    "DatasetRevisionError",
    "MODES",
    "REVISION_SCENE_PROFILES",
    "REVISION_STRESS_PROFILES",
    "RevisionStressProfile",
    "SCENES",
    "STRESS_SCOPES",
    "install_from_activation",
    "install_revision_profiles",
    "revision_scene_profile",
    "revision_stress_profile",
]
