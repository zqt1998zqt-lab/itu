"""Coverage must preserve audited failures and reject incomplete inventories."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("eval_coverage", Path(__file__).resolve().parents[1] / "cpp/tools/summarize_eval_coverage.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def sample():
    scenes = [dict(id="a", family="ns", noise="tcar", snr_db=-10, near_speaker="1"),
              dict(id="b", family="ns", noise="tcar", snr_db=5, near_speaker="2"),
              dict(id="c", family="combined", noise="car_hvac", snr_db=5, near_speaker="1"),
              dict(id="d", family="aec", noise="none", snr_db=None, near_speaker="2")]
    details = [dict(id=s["id"], passed=i != 0,
                    gates=[dict(status="fail" if i == 0 else "pass", segment="near_only", metric="stoi_change")])
               for i, s in enumerate(scenes)]
    return dict(scenes=scenes), dict(scenes=4, passed_scenes=3, counts={"pass": 3, "fail": 1}, details=details)


def test_vehicle_and_snr_groups_preserve_real_failures():
    manifest, acceptance = sample()
    report = coverage.summarize(manifest, acceptance)
    assert report["total"]["failed_scenes"] == 1
    assert report["vehicle"]["scenes"] == 3 and report["vehicle"]["passed_scenes"] == 2
    assert report["vehicle_noise_types"] == ["car_hvac", "tcar"]
    assert report["vehicle_families"]["ns"]["counts"]["fail"] == 1
    low = next(row for row in report["ns_noise_snr"] if row["snr_db"] == -10)
    assert low["passed_scenes"] == 0 and low["failed_metrics"] == {"near_only.stoi_change": 1}
    assert sum(row["scenes"] for row in report["by_snr_db"]) == 4
    assert next(row for row in report["by_snr_db"] if row["value"] is None)["scenes"] == 1


@pytest.mark.parametrize("defect", ["missing", "duplicate", "empty", "false_totals"])
def test_incomplete_or_inconsistent_inventory_is_rejected(defect):
    manifest, acceptance = deepcopy(sample())
    if defect == "missing":
        acceptance["details"].pop()
    elif defect == "duplicate":
        acceptance["details"].append(acceptance["details"][0])
    elif defect == "empty":
        manifest["scenes"] = []
        acceptance["details"] = []
    else:
        acceptance["passed_scenes"] = 4
    with pytest.raises(ValueError):
        coverage.summarize(manifest, acceptance)


def test_playable_subset_covers_every_speaker_and_vehicle_condition():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
    from eval_profiles import expanded_matrix, VEHICLE_NOISES
    from publish_audio_eval import representative_ids
    people = {str(i): {"sex": "M" if i < 20 else "F"} for i in range(40)}
    scenes = expanded_matrix(people)
    details = [dict(id=s["id"], passed=True) for s in scenes]
    selected = representative_ids(scenes, details, sharded=True)
    audio = [s for s in scenes if s["id"] in selected]
    assert {s["near_speaker"] for s in audio} == set(people)
    assert {"tcar", *VEHICLE_NOISES} <= {s["noise"] for s in audio}
    assert {s["family"] for s in audio} == {"ns", "aec", "combined", "stress", "agc", "bf"}
    assert {s["snr_db"] for s in audio if s["family"] == "ns"} == set(range(-10, 31, 5))
    assert {s["family"] for s in audio if s.get("playback_source") == "radio"} == {"aec", "combined"}


def test_radio_is_grouped_as_playback_and_not_noise():
    manifest, acceptance = sample()
    manifest["scenes"][3]["playback_source"] = "radio"
    report = coverage.summarize(manifest, acceptance)
    assert report["vehicle"]["scenes"] == 4
    assert "none" not in report["vehicle_noise_types"]
    assert next(row for row in report["by_playback"] if row["value"] == "radio")["scenes"] == 1
    assert all(row["value"] != "car_radio" for row in report["by_noise"])


def test_generator_lock_rejects_source_changes_and_unlocked_partial_batch(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
    from run_expanded_eval import pin_generator
    root = tmp_path / "new"
    root.mkdir()
    original = pin_generator(root, "first-source")
    assert pin_generator(root, "first-source") == original
    with pytest.raises(ValueError, match="generation sources changed"):
        pin_generator(root, "changed-source")
    partial = tmp_path / "partial"
    (partial / "cases/example").mkdir(parents=True)
    (partial / "cases/example/scene.json").write_text("{}")
    with pytest.raises(ValueError, match="no pre-generation version lock"):
        pin_generator(partial, "first-source")
