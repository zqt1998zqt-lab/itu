from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from server.acoustic import (
    acoustic_stress_profiles,
    dataset_revisions_v2 as revisions,
)


ROOT = Path(__file__).resolve().parents[1]
VCP = ROOT / "scripts/verify-current-acoustic-stress-vcp.py"
MDF = ROOT / "scripts/run_mdf_localvqe_mask_ab_current.py"
TS = ROOT / "scripts/verify-current-acoustic-stress-model-snapshots.ts"
FIXTURE = (
    ROOT
    / "benchmarks/fixtures/acoustic-stress-model-schema-v1/metadata.json"
)
FIXTURE_SHA256 = (
    "d94a7b3721e1832fb61ab5bb91bbb16c2a42b9cb076e39a16727ccc2d212af61"
)
OLD_ECHO_ID = "0563b621e16a1a7702dd7994683ddda2"
OLD_DT_ID = "01d5e9176ea5abbbead40185fc9b4546"


@pytest.fixture(autouse=True)
def preserve_stress_registries():
    registries = (
        acoustic_stress_profiles.PROFILES,
        acoustic_stress_profiles.BY_DATASET_ID,
    )
    snapshots = [dict(registry) for registry in registries]
    yield
    for registry, prior in zip(registries, snapshots, strict=True):
        registry.clear()
        registry.update(prior)


def _load_mdf():
    spec = importlib.util.spec_from_file_location(
        "_test_current_mdf_localvqe_mask_ab", MDF
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def test_vcp_current_entry_activates_r2_without_importing_server_app():
    result = _run(
        [sys.executable, str(VCP), "--check-current-profiles"]
    )
    assert result.returncode == 0, result.stderr
    identity = json.loads(result.stdout)
    assert identity["activation"] == {
        "revision": "measurement-20261117-r2",
        "registered": 48,
        "activation": "active",
        "activated": True,
    }
    assert identity["representative_count"] == 80
    assert identity["server_app_imported"] is False
    profiles = {
        (row["scope"], row["mode"]): row["dataset_id"]
        for row in identity["profiles"]
    }
    assert profiles[("echo", "WB-wired")] == (
        revisions.REVISION_STRESS_PROFILES[("echo", "WB-wired")].dataset_id
    )
    assert profiles[("double_talk", "WB-wired")] == (
        revisions.REVISION_STRESS_PROFILES[
            ("double_talk", "WB-wired")
        ].dataset_id
    )
    assert OLD_ECHO_ID not in profiles.values()
    assert OLD_DT_ID not in profiles.values()


def test_mdf_current_entry_rebinds_only_data_identity_and_schema():
    module = _load_mdf()
    legacy, activation = module._load_current_legacy_module()
    echo = revisions.REVISION_STRESS_PROFILES[("echo", "WB-wired")]
    double_talk = revisions.REVISION_STRESS_PROFILES[
        ("double_talk", "WB-wired")
    ]
    assert activation["activated"] is True
    assert legacy.ECHO_DATASET == echo.dataset_id
    assert legacy.DT_DATASET == double_talk.dataset_id
    assert legacy.ECHO_DATASET != OLD_ECHO_ID
    assert legacy.DT_DATASET != OLD_DT_ID
    assert legacy.SCHEMA == "itu-mdf-localvqe-mask-ab-current-v1"

    identity = module.current_identity(legacy, activation)
    assert identity["data_revision"] == "measurement-20261117-r2"
    assert identity["datasets"]["echo"]["dataset_id"] == echo.dataset_id
    assert identity["datasets"]["double_talk"]["dataset_id"] == (
        double_talk.dataset_id
    )
    assert identity["datasets"]["echo"]["manifest_sha256"] == (
        "2f08577f1ae3cc4ff73e48bfa1297a1776e1207c52d9916b81470a6b02552157"
    )
    assert identity["evaluation_protocol"]["version"] == (
        "itu-mdf-localvqe-mask-ab-evaluation-v2"
    )
    assert identity["evaluation_protocol"]["post_rx_mixing"] is True
    assert identity["evaluation_protocol"]["same_inputs_across_routes"] is True
    assert identity["evaluation_protocol"]["same_controls"] is True
    assert identity["historical_comparability"]["comparable"] is False


def test_mdf_identity_hook_preserves_legacy_flat_identity(tmp_path):
    module = _load_mdf()
    legacy, _activation = module._load_current_legacy_module()
    module._install_write_hook(legacy)
    original = {
        "library": {
            "path": str(MDF),
            "sha256": hashlib.sha256(MDF.read_bytes()).hexdigest(),
        }
    }
    legacy.write_json(tmp_path / "identity.json", original)
    assert set(original) == {"library"}
    written = json.loads((tmp_path / "identity.json").read_text(encoding="utf-8"))
    assert written["library"] == original["library"]
    assert written["current_data_identity"]["data_revision"] == (
        "measurement-20261117-r2"
    )


def test_mdf_default_cli_delegates_legacy_cases_argument(monkeypatch):
    module = _load_mdf()
    observed = []

    class LegacyMain:
        @staticmethod
        def main():
            observed.append(list(sys.argv))
            return 0

    monkeypatch.setattr(
        module, "_load_current_legacy_module", lambda: (LegacyMain(), {})
    )
    monkeypatch.setattr(module, "_install_write_hook", lambda _module: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MDF),
            "--output",
            "/tmp/current-ab",
            "--library",
            "/tmp/library.so",
            "--cases",
            "5",
        ],
    )
    assert module.main() == 0
    assert observed[0][-2:] == ["--cases", "5"]


def test_source_cases_do_not_depend_on_the_old_16_store(tmp_path, monkeypatch):
    module = _load_mdf()
    legacy, _activation = module._load_current_legacy_module()
    stress_root = tmp_path / "stress"
    monkeypatch.setattr(legacy, "STRESS", stress_root)

    roles = ("reference", "far_end", "noise", "microphone")
    signal_hashes = {}
    for role in roles:
        path = tmp_path / f"{role}.wav"
        path.write_bytes(role.encode())
        signal_hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()

    def add_dataset(profile, cases):
        root = stress_root / profile.dataset_id
        root.mkdir(parents=True)
        signals = {}
        for role in roles:
            relative = f"assets/{role}.wav"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(role.encode())
            signals[role] = relative
        manifest = {
            "dataset_id": profile.dataset_id,
            "cases": [
                {
                    "case_id": case_id,
                    "signals": signals,
                    "signals_sha256": signal_hashes,
                    "provenance": {
                        "workload_category": category,
                        "render_metadata": {
                            "echo_gain_linear": 0.0,
                            "echo_path_segments": [],
                        },
                    },
                }
                for case_id, category in cases
            ],
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    echo = revisions.REVISION_STRESS_PROFILES[("echo", "WB-wired")]
    double_talk = revisions.REVISION_STRESS_PROFILES[
        ("double_talk", "WB-wired")
    ]
    add_dataset(
        echo,
        (
            ("echo-steady", "steady_echo"),
            ("echo-path", "path_change"),
        ),
    )
    add_dataset(double_talk, (("dt-balanced", "balanced"),))
    assert not (stress_root / OLD_ECHO_ID).exists()
    assert not (stress_root / OLD_DT_ID).exists()

    expected = {
        "near_single": double_talk.dataset_id,
        "far_single": echo.dataset_id,
        "double_talk": double_talk.dataset_id,
        "path_change": echo.dataset_id,
    }
    for group, dataset_id in expected.items():
        rows = legacy.source_cases(group, 1)
        assert len(rows) == 1
        assert rows[0]["dataset"] == dataset_id
        assert OLD_ECHO_ID not in str(rows[0]["paths"])
        assert OLD_DT_ID not in str(rows[0]["paths"])


def test_metadata_fixture_and_node_entry_use_no_wav_store():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert fixture["metadata_only"] is True
    assert fixture["wav_files_required"] is False
    assert fixture["historical_datasets"]["echo"][
        "expected_first_case_microphone_sha256"
    ] == "174cebb8239ba940c83943e66edd1bb1278535669760ab8740b6f44cbdc66837"
    assert len(fixture["historical_representative_snapshots"]) == 16
    assert {
        Path(row["snapshot_path"]).name
        for row in fixture["historical_representative_snapshots"]
    } == {"00000011.json"}
    assert {
        row["semantics"]
        for row in fixture["historical_cached_snapshots"]
    } == {
        "historical_v1_cached_snapshot_schema_only_not_current_r2_evidence"
    }

    result = _run(
        ["node", "--experimental-strip-types", str(TS)]
    )
    assert result.returncode == 0, result.stderr
    assert "PASS historical-fixture echo/WB-wired" in result.stdout
    assert "PASS historical-fixture double_talk/WB-wired" in result.stdout
    assert "PASS current-r2 echo/WB-wired" in result.stdout
    assert "PASS current-r2 double_talk/WB-wired" in result.stdout
    assert "PASS historical-schema echo/WB-wired" in result.stdout
    assert "PASS historical-schema double_talk/WB-wired" in result.stdout
    assert (
        "PASS historical-representatives profiles=16 "
        "completed_cases=80 snapshot=00000011"
    ) in result.stdout
    assert "PASS metadata-only metadata_files_read=" in result.stdout
    assert "wav_files_read=0" in result.stdout
    assert OLD_ECHO_ID not in next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("PASS current-r2 echo/")
    )
