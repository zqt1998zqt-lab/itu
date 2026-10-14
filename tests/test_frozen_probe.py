"""Replay metadata may differ, but frozen scene conditions must not."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
import probe_frozen_audio as probe


def test_replay_inventory_checks_original_freeze_and_scene_conditions(tmp_path):
    frozen = dict(scenes=[dict(id="a", files={"before.wav": {"sha256": "input"}},
                              windows={"near_only": [1, 3]})])
    frozen_path = tmp_path / "original.json"
    frozen_path.write_text(json.dumps(frozen))
    source = tmp_path / "replay"
    source.mkdir()
    replay = dict(frozen, environment={"cli_sha256": "replay-binary"})
    inventory = source / "inputs-manifest.json"
    inventory.write_text(json.dumps(replay))
    acceptance = dict(frozen_manifest_path=str(frozen_path),
                      frozen_manifest_sha256=probe.ev.digest(frozen_path))
    assert probe.accepted_inventory(source, acceptance) == replay
    replay["scenes"][0]["windows"]["near_only"] = [1, 4]
    inventory.write_text(json.dumps(replay))
    with pytest.raises(ValueError, match="conditions changed"):
        probe.accepted_inventory(source, acceptance)
    inventory.write_text(json.dumps(dict(scenes=[])))
    with pytest.raises(ValueError, match="inventory differs"):
        probe.accepted_inventory(source, acceptance)
    frozen_path.write_text(json.dumps(dict(scenes=[])))
    with pytest.raises(ValueError, match="baseline manifest hash mismatch"):
        probe.accepted_inventory(source, acceptance)


def test_variant_profile_precedes_module_overrides(tmp_path):
    scene = dict(family="ns", stress="none")
    argv, profile = probe.variant_invocation(scene, "cli", tmp_path,
        ["--ns-level", "1", "--profile", "vehicle_stream"])
    assert profile == "vehicle_stream"
    assert argv[:3] == ["cli", "--profile", "vehicle_stream"]
    assert argv.count("--profile") == 1 and "--no-aec" in argv
    assert argv[-2:] == ["--ns-level", "1"]
    with pytest.raises(ValueError, match="only one"):
        probe.variant_invocation(scene, "cli", tmp_path,
            ["--profile", "general", "--profile", "vehicle_stream"])
    with pytest.raises(ValueError, match="value missing"):
        probe.variant_invocation(scene, "cli", tmp_path, ["--profile"])
    with pytest.raises(ValueError, match="unknown processing profile"):
        probe.variant_invocation(scene, "cli", tmp_path, ["--profile", "unknown"])
