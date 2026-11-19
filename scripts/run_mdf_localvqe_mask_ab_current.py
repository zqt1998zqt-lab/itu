#!/usr/bin/env python3
"""Current-data entrypoint for the matched post-RX MDF/LocalVQE mask A/B."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from server.acoustic import dataset_revisions_v2 as revisions


LEGACY = PROJECT / "scripts/run_mdf_localvqe_mask_ab_v2.py"
LEGACY_MODULE_NAME = "_itu_current_mdf_localvqe_mask_ab_v2_implementation"
SCHEMA = "itu-mdf-localvqe-mask-ab-current-v1"
EVALUATION_PROTOCOL_VERSION = "itu-mdf-localvqe-mask-ab-evaluation-v2"
PUBLISHED_SCHEMA = "itu-mdf-localvqe-mask-ab-v2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _load_current_legacy_module():
    activation = revisions.install_from_activation()
    if (
        activation.get("revision") != revisions.DATASET_REVISION
        or activation.get("activation") != "active"
        or activation.get("activated") is not True
    ):
        raise RuntimeError(
            "ACTIVATION-r2 must validate and activate before MDF A/B"
        )

    module = sys.modules.get(LEGACY_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, LEGACY)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load legacy MDF A/B runner: {LEGACY}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[LEGACY_MODULE_NAME] = module
        spec.loader.exec_module(module)

    echo = revisions.REVISION_STRESS_PROFILES[("echo", "WB-wired")]
    double_talk = revisions.REVISION_STRESS_PROFILES[
        ("double_talk", "WB-wired")
    ]
    module.ECHO_DATASET = echo.dataset_id
    module.DT_DATASET = double_talk.dataset_id
    module.SCHEMA = SCHEMA

    required = (
        "source_cases",
        "prepare_case",
        "run_case",
        "run_controls",
        "measure",
        "Engine",
        "main",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"legacy MDF A/B implementation is incomplete: {missing}")
    return module, activation


def _dataset_identity(module, profile) -> dict[str, str]:
    root = module.STRESS / profile.dataset_id
    manifest = root / "manifest.json"
    assets = root / "assets.json"
    return {
        "dataset_id": profile.dataset_id,
        "dataset_name": profile.dataset_name,
        "manifest_path": str(manifest),
        "manifest_sha256": digest(manifest),
        "assets_path": str(assets),
        "assets_sha256": digest(assets),
    }


def current_identity(module=None, activation=None) -> dict[str, object]:
    if module is None:
        module, activation = _load_current_legacy_module()
    echo = revisions.REVISION_STRESS_PROFILES[("echo", "WB-wired")]
    double_talk = revisions.REVISION_STRESS_PROFILES[
        ("double_talk", "WB-wired")
    ]
    return {
        "schema": "itu-mdf-localvqe-mask-ab-current-identity-v1",
        "data_revision": revisions.DATASET_REVISION,
        "activation": activation,
        "activation_path": str(revisions.ACTIVATION_PATH),
        "activation_sha256": digest(revisions.ACTIVATION_PATH),
        "datasets": {
            "echo": _dataset_identity(module, echo),
            "double_talk": _dataset_identity(module, double_talk),
        },
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "legacy_runner_path": str(LEGACY),
            "legacy_runner_sha256": digest(LEGACY),
            "post_rx_mixing": True,
            "same_inputs_across_routes": True,
            "same_controls": True,
            "fitted_delay_for_scoring": False,
        },
        "historical_comparability": {
            "published_schema": PUBLISHED_SCHEMA,
            "comparable": False,
            "reason": (
                "The published v2 result used the old 16-store identities. "
                "Current R2 WB-wired data has different manifests and assets; "
                "reuse of the scoring protocol does not make the datasets "
                "interchangeable."
            ),
        },
    }


def _install_write_hook(module) -> None:
    if getattr(module, "_current_identity_write_hook", False):
        return
    original = module.write_json

    def write_json(path: Path, value: dict) -> None:
        if path.name == "identity.json":
            payload = {
                **value,
                "current_data_identity": current_identity(
                    module, revisions.install_from_activation()
                ),
            }
        elif path.name == "result.json":
            identity = current_identity(
                module, revisions.install_from_activation()
            )
            payload = {
                **value,
                "data_identity": identity,
                "historical_comparability": identity[
                    "historical_comparability"
                ],
            }
        else:
            payload = value
        original(path, payload)

    module.write_json = write_json
    module._current_identity_write_hook = True


def source_cases_smoke(module, count: int) -> dict[str, object]:
    groups = {}
    expected_datasets = {
        "near_single": module.DT_DATASET,
        "far_single": module.ECHO_DATASET,
        "double_talk": module.DT_DATASET,
        "path_change": module.ECHO_DATASET,
    }
    for group in module.GROUPS:
        rows = module.source_cases(group, count)
        if len(rows) != count:
            raise RuntimeError(f"{group} source-case count mismatch")
        for row in rows:
            if row["dataset"] != expected_datasets[group]:
                raise RuntimeError(f"{group} did not resolve the current R2 dataset")
            if not all(path.is_file() for path in row["paths"].values()):
                raise RuntimeError(f"{group} source metadata points to a missing asset")
        groups[group] = [
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "dataset": row["dataset"],
                "manifest_sha256": row["manifest_sha256"],
                "category": row["category"],
                "paths": {
                    role: str(path) for role, path in row["paths"].items()
                },
            }
            for row in rows
        ]
    return {
        "schema": SCHEMA,
        "mode": "source_cases_only",
        "network_execution": False,
        "identity": current_identity(module, revisions.install_from_activation()),
        "groups": groups,
    }


def main() -> int:
    if "--source-cases-smoke" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source-cases-smoke", action="store_true")
        parser.add_argument("--cases", type=int, default=1)
        args = parser.parse_args()
        if not 1 <= args.cases <= 20:
            parser.error("--cases must be 1..20")
        module, _activation = _load_current_legacy_module()
        print(
            json.dumps(
                source_cases_smoke(module, args.cases),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    module, _activation = _load_current_legacy_module()
    _install_write_hook(module)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
