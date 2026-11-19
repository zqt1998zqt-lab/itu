#!/usr/bin/env python3
"""Run the legacy 16-by-5 VCP representative verifier against activated R2 data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from server.acoustic import dataset_revisions_v2 as revisions


LEGACY = PROJECT / "scripts/verify-acoustic-stress-vcp.py"
LEGACY_MODULE_NAME = "_itu_current_acoustic_stress_vcp_implementation"


def _load_current_legacy_module():
    activation = revisions.install_from_activation()
    if (
        activation.get("revision") != revisions.DATASET_REVISION
        or activation.get("activation") != "active"
        or activation.get("activated") is not True
    ):
        raise RuntimeError(
            "ACTIVATION-r2 must validate and activate before stress verification"
        )

    module = sys.modules.get(LEGACY_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, LEGACY)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load legacy stress verifier: {LEGACY}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[LEGACY_MODULE_NAME] = module
        spec.loader.exec_module(module)

    expected = revisions.REVISION_STRESS_PROFILES
    if set(expected) != set(module.PROFILES):
        raise RuntimeError("legacy verifier profile inventory is incomplete")
    if any(module.PROFILES[key] is not profile for key, profile in expected.items()):
        raise RuntimeError("legacy verifier did not retain the activated R2 profiles")

    categories = module.CATEGORIES
    representative_count = sum(
        len(categories[scope]) for scope, _mode in expected
    )
    if len(expected) != 16 or representative_count != 80:
        raise RuntimeError(
            "current verifier must preserve 16 profiles x 5 representatives"
        )
    if "server.app" in sys.modules:
        raise RuntimeError("current VCP verifier must not import server.app")
    return module, activation


def current_identity() -> dict[str, object]:
    module, activation = _load_current_legacy_module()
    return {
        "schema": "itu-current-acoustic-stress-vcp-identity-v1",
        "activation": activation,
        "activation_path": str(revisions.ACTIVATION_PATH),
        "profiles": [
            {
                "scope": scope,
                "mode": mode,
                "dataset_id": profile.dataset_id,
                "contract_item_count": profile.contract_item_count,
            }
            for (scope, mode), profile in sorted(module.PROFILES.items())
        ],
        "representative_count": sum(
            len(module.CATEGORIES[scope]) for scope, _mode in module.PROFILES
        ),
        "legacy_runner": str(LEGACY),
        "server_app_imported": "server.app" in sys.modules,
    }


def main() -> int:
    module, _activation = _load_current_legacy_module()
    if sys.argv[1:] == ["--check-current-profiles"]:
        print(json.dumps(current_identity(), sort_keys=True))
        return 0
    if sys.argv[1:]:
        raise SystemExit(
            "usage: verify-current-acoustic-stress-vcp.py "
            "[--check-current-profiles]"
        )
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
