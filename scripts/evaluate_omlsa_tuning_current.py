#!/home/zhangqingtian/.venv/bin/python
"""Current OM-LSA tuning evaluator backed by the fixed v1 fixture.

The historical ``scripts/evaluate_omlsa_tuning.py`` remains the scoring and CLI
implementation. This entrypoint loads that module with importlib, registers it
in ``sys.modules`` before execution, and replaces only its two input loaders
with fixture-backed equivalents.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
sys.dont_write_bytecode = True

import omlsa_tuning_fixture as fixture


LEGACY_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_omlsa_tuning.py"
LEGACY_MODULE_NAME = "itu_omlsa_tuning_legacy_entry"


class CurrentEvaluationError(RuntimeError):
    pass


def load_base_module(path: Path = LEGACY_SCRIPT) -> Any:
    """Load the historical evaluator with dataclass-safe sys.modules registration."""

    source = Path(path).expanduser().absolute()
    if source.is_symlink():
        raise CurrentEvaluationError(f"historical evaluator must not be a symlink: {source}")
    if not source.is_file():
        raise CurrentEvaluationError(f"historical evaluator is missing: {source}")
    existing = sys.modules.get(LEGACY_MODULE_NAME)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None or Path(existing_path).resolve() != source.resolve():
            raise CurrentEvaluationError(
                f"module name {LEGACY_MODULE_NAME} is already bound to another file"
            )
        return existing

    spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise CurrentEvaluationError(f"cannot construct import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module while exec_module is running.
    sys.modules[LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
_ORIGINAL_BASE_EVALUATE = None
_CONFIGURED_LOCK: fixture.FixtureLock | None = None


def _fixture_dev_spec() -> tuple[Any, dict[str, Any]]:
    return fixture.load_dev_spec(base.CaseSpec)


def _fixture_holdout_specs() -> list[Any]:
    return fixture.load_holdout_specs(base.CaseSpec)


def _inject_fixture_evidence(evidence: dict[str, Any], lock: fixture.FixtureLock) -> None:
    provenance = fixture.fixture_evidence(lock)
    evidence["fixture"] = provenance
    evidence["fixed_reference_sha256"] = sorted(
        artifact.sha256
        for artifact in lock.artifacts
        if artifact.kind == "reference"
    )
    evidence["fixture_source_lock"] = {
        "manifest_path": str(lock.manifest_path),
        "manifest_sha256": lock.manifest_sha256,
        "upstream_script_path": str(lock.upstream_script.path),
        "upstream_script_sha256": lock.upstream_script.sha256,
        "source_paths_accessed_at_evaluation": False,
        "fixture_paths_accessed_at_evaluation": True,
    }
    dev_manifest = evidence.get("dev_manifest")
    if isinstance(dev_manifest, dict):
        dev_manifest.update(
            {
                "fixture_manifest_path": str(lock.manifest_path),
                "fixture_manifest_sha256": lock.manifest_sha256,
                "fixture_id": lock.manifest_path.parent.name,
                "upstream_script_sha256": lock.upstream_script.sha256,
                "original_dev_manifest_path": lock.dev_manifest_origin.get(
                    "source_path"
                ),
                "original_dev_manifest_present_at_preparation": (
                    lock.dev_manifest_origin.get("present_at_preparation")
                ),
                "original_dev_manifest_accessed_at_evaluation": False,
            }
        )


def evaluate(args: Any) -> dict[str, Any]:
    if _ORIGINAL_BASE_EVALUATE is None:
        raise CurrentEvaluationError("historical evaluator is not configured")
    evidence = _ORIGINAL_BASE_EVALUATE(args)
    lock = fixture.freeze_source_lock()
    _inject_fixture_evidence(evidence, lock)
    return evidence


def configure() -> fixture.FixtureLock:
    """Bind the fixed fixture into the historical evaluator's two loader globals."""

    global _CONFIGURED_LOCK, _ORIGINAL_BASE_EVALUATE
    lock = fixture.freeze_source_lock()
    fixture.assert_upstream_protocol(
        base.DEV_REFERENCE_SHA256,
        base.HOLDOUT_REFERENCES,
        base.EXPECTED_MODEL_SHA256,
    )
    if _ORIGINAL_BASE_EVALUATE is None:
        _ORIGINAL_BASE_EVALUATE = base.evaluate

    base.DEV_MANIFEST = fixture.MANIFEST_PATH
    base.load_dev_spec = _fixture_dev_spec
    base.load_holdout_specs = _fixture_holdout_specs
    base.evaluate = evaluate
    _CONFIGURED_LOCK = lock
    return lock


def fixture_metadata() -> dict[str, Any]:
    lock = _CONFIGURED_LOCK or fixture.freeze_source_lock()
    return fixture.fixture_evidence(lock)


def main(argv: Sequence[str] | None = None) -> int:
    """Preserve the historical evaluator CLI and execute its main implementation."""

    return int(base.main(argv))


configure()


if __name__ == "__main__":
    raise SystemExit(main())
