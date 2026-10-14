"""The continuous sequence starts with speech and retains adverse transitions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
from eval_profiles import expanded_matrix
from run_vehicle_stream import select_timeline, sequence_beam_delays, segment_timing, stream_verdict, validate_stream_frames
from check_audio_acceptance import check_scene, RULES
from run_vehicle_regression import summarize
from build_vehicle_array_dataset import spatial_case
import run_audio_eval as ev
import numpy as np
import json


def test_vehicle_timeline_starts_with_speech_and_retains_reference_faults():
    metadata = {str(i): {"sex": "M" if i < 20 else "F"} for i in range(40)}
    scenes = expanded_matrix(metadata)
    timeline = select_timeline(scenes, 0)
    label, start, begin, end = timeline[0]
    assert label == "cold_start_speech" and begin == 2 and end == 9
    assert start["family"] == "ns" and start["playback_source"] == "none"
    assert sum(end - begin for _, _, begin, end in timeline) == 235
    assert {s["stress"] for _, s, _, _ in timeline} >= {"wrong_reference", "clock_300ppm", "path_change"}
    assert {s["noise"] for _, s, _, _ in timeline} >= {"car_window_wind", "car_engine_ramp"}
    assert all(s["noise"] != "car_passenger" and label != "unreferenced_passenger"
               for label, s, _, _ in timeline)
    assert {s["playback_source"] for _, s, _, _ in timeline} >= {"none", "radio"}
    assert all(s["mics"] == 1 and s["pair"] == 0 for _, s, _, _ in timeline)
    with pytest.raises(ValueError, match="missing"):
        select_timeline([s for s in scenes if s["stress"] != "path_change"], 0)


def test_curated_timeline_prefers_corrected_sweep_without_losing_stress():
    metadata = {str(i): {"sex": "M" if i < 20 else "F"} for i in range(40)}
    scenes = expanded_matrix(metadata)
    corrected = [dict(s, id="v4-" + s["id"], profile="curated-v4") for s in scenes if s["family"] == "ns"]
    timeline = select_timeline(scenes + corrected, 0)
    assert all(scene["id"].startswith("v4-") for _, scene, _, _ in timeline if scene["family"] == "ns")
    assert len(timeline) == 9 and sum(end-begin for _, _, begin, end in timeline) == 235


@pytest.mark.parametrize("microphones", [1, 2, 4])
def test_vehicle_timeline_never_selects_catalog_excluded_records(microphones):
    metadata = {str(i): {"sex": "M" if i < 20 else "F"} for i in range(40)}
    scenes = [dict(scene, mics=microphones) for scene in expanded_matrix(metadata)]
    excluded = [dict(scene, id="excluded-" + scene["id"], catalog_excluded=True,
                     profile="curated-v4" if scene["family"] == "ns" else scene["profile"])
                for scene in scenes]
    timeline = select_timeline(excluded + scenes, 0, microphones)
    assert len(timeline) == 9
    assert all(not scene["id"].startswith("excluded-") for _, scene, _, _ in timeline)
    with pytest.raises(ValueError, match="missing"):
        select_timeline(excluded, 0, microphones)


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_stream_timing_rejects_dropped_duplicate_and_mistimed_frames(rate):
    rows = [dict(frame=str(i), time_s=str(i / 100), process_ms="0.2",
                 release_lateness_ms="0.1", completion_late="0") for i in range(103)]
    delay = 3 * (rate // 100)
    validate_stream_frames(rows, 100, delay, rate)
    # 480 samples require three drain frames.
    with pytest.raises(ValueError, match="extra"):
        validate_stream_frames(rows + [dict(rows[-1], frame="103", time_s="1.03")], 100, delay, rate)
    with pytest.raises(ValueError, match="missing"):
        validate_stream_frames(rows[:20] + rows[21:], 100, delay, rate)
    duplicate = [dict(row) for row in rows]
    duplicate[20]["frame"] = "19"
    with pytest.raises(ValueError, match="unordered"):
        validate_stream_frames(duplicate, 100, delay, rate)
    duplicate[20]["frame"] = "20"
    duplicate[20]["time_s"] = "0.21"
    with pytest.raises(ValueError, match="mistimed"):
        validate_stream_frames(duplicate, 100, delay, rate)


def test_vehicle_array_selection_requires_real_channels_and_fixed_geometry():
    metadata = {str(i): {"sex": "M" if i < 20 else "F"} for i in range(40)}
    mono = expanded_matrix(metadata)
    timeline = select_timeline(mono, 0)
    assert sequence_beam_delays(timeline, 1) is None
    array = [dict(scene, mics=4, beam_delays_samples=[3.5, 2.3, 1.1, 0])
             for _, scene, _, _ in timeline]
    selected = select_timeline(mono + array, 0, 4)
    assert len(selected) == 9 and all(scene["mics"] == 4 for _, scene, _, _ in selected)
    assert sequence_beam_delays(selected, 4) == [3.5, 2.3, 1.1, 0]
    with pytest.raises(ValueError, match="missing"):
        select_timeline(mono + array, 0, 2)
    with pytest.raises(ValueError, match="microphones"):
        select_timeline(mono, 0, 3)
    array[-1]["beam_delays_samples"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="fixed calibrated"):
        sequence_beam_delays(select_timeline(array, 0, 4), 4)
    array[-1]["beam_delays_samples"] = [3.5, 2.3, 1.1]
    with pytest.raises(ValueError, match="fixed calibrated"):
        sequence_beam_delays(select_timeline(array, 0, 4), 4)


@pytest.mark.parametrize("microphones", [2, 4])
def test_array_derivation_replaces_inherited_mono_geometry(microphones, tmp_path):
    scene = dict(id="mono", family="ns", mics=1, files={}, pair=0,
                 near_speaker="0", far_speaker="1", near_dbfs=-30,
                 microphone_positions_m=[[2., 2., 1.2]], near_position_m=[3.3, 3.2, 1.5],
                 near_arrival_samples=[0.], beam_delays_samples=[0.],
                 noise="white", snr_db=10, ser_db=0, rt60_s=.15,
                 echo_delay_ms=80, stress="none", seed=42, playback_source="none")
    case = spatial_case(scene, microphones)
    rng = np.random.default_rng(16)
    result = ev.make_case(case, {"0": rng.normal(size=32000), "1": rng.normal(size=32000)},
                          {}, tmp_path / "array", {})
    pcm = ev.read(tmp_path / "array/before.wav")
    assert pcm.shape == (24 * ev.RATE, microphones)
    assert len(result["beam_delays_samples"]) == microphones
    assert len(result["microphone_positions_m"]) == microphones
    assert not np.array_equal(pcm[:, 0], pcm[:, 1])
    assert scene["microphone_positions_m"] == [[2., 2., 1.2]]
    with pytest.raises(ValueError, match="mono source"):
        spatial_case(dict(scene, mics=2), microphones)


def test_paced_segment_uses_only_its_own_deadlines_and_release_lateness():
    rows = [dict(frame=str(i), process_ms="0.2", release_lateness_ms="0.1",
                 completion_late="0") for i in range(20)]
    rows[3].update(release_lateness_ms="12", completion_late="1")
    _, early, _ = segment_timing(rows, 0, .1, True)
    _, late, _ = segment_timing(rows, .1, .2, True)
    assert early["completion_deadline_misses"] == 1
    assert early["release_lateness_max_ms"] == 12
    assert late["completion_deadline_misses"] == 0
    assert late["release_lateness_max_ms"] == .1
    with pytest.raises(ValueError, match="missing"):
        segment_timing(rows[:4] + rows[5:], 0, .1, True)


def test_missing_required_measurement_and_scheduled_overrun_block_acceptance():
    unavailable = dict(status="unavailable", optional=False)
    verdict = stream_verdict([dict(passed=False, gates=[unavailable])])
    assert not verdict["all_available_passed"] and not verdict["required_measurements_complete"]
    rules = json.loads(RULES.read_text())
    scene = dict(id="deadline", family="ns", stress="none", windows={})
    row = dict(mode="full", segments={}, process_frames_over_10ms=0,
               output_levels=dict(peak_dbfs=-1, clipping_percent=0),
               engine=dict(paced_10ms=True, process_p95_ms=.2, process_max_ms=.5,
                           process_rtf=.02, completion_deadline_misses=1))
    failed = check_scene(scene, row, rules)
    assert not failed["passed"]
    assert [g["metric"] for g in failed["gates"] if g["status"] == "fail"] == ["completion_deadline_misses"]
    row["engine"]["completion_deadline_misses"] = 0
    assert check_scene(scene, row, rules)["passed"]
    del row["engine"]["completion_deadline_misses"]
    assert not check_scene(scene, row, rules)["passed"]


def test_regression_keeps_runtime_and_quality_failures_separate():
    gates = [dict(segment="near_only", metric="stoi", status="pass", optional=False),
             dict(segment="runtime", metric="process_p95_ms", status="fail", optional=False)]
    failure = dict(label="cold_start_speech", **gates[1])
    report = dict(details=[dict(label="cold_start_speech", passed=False, gates=gates)],
                  failures=[failure], all_available_passed=False, required_measurements_complete=True)
    result = summarize([(0, "84", report)])
    assert result["phases"] == 1 and result["passed_phases"] == 0
    assert result["quality_passed_phases"] == 1 and result["failed_gates"] == 1
    gates[0]["status"] = "unavailable"
    assert summarize([(0, "84", report)])["quality_passed_phases"] == 0
    with pytest.raises(ValueError, match="nonempty"):
        summarize([])
