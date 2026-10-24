"""Replay summaries must expose and block candidate regressions."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "cpp/tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "frozen_replay", TOOLS / "run_frozen_repair_replay.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def row(original, current, *, same=False, engine=None):
    return {
        "original_status": original,
        "acceptance": {"status": current},
        "same_output_as_original": same,
        "metrics": {"engine": engine or {}},
    }


def test_regressions_block_admission_and_are_reported_separately():
    summary = replay.admission_summary(
        [
            row("fail", "pass"),
            row("pass", "fail"),
            row("pass", "pass"),
        ]
    )
    assert summary["fixes"] == 1
    assert summary["regressions"] == 1
    assert not summary["candidate_eligible"]
    assert summary["admission_status"] == "blocked_regressions"
    assert summary["transitions"]["pass->fail"] == 1


def test_zero_regressions_are_required_for_eligibility():
    summary = replay.admission_summary(
        [
            row("fail", "pass"),
            row("pass", "pass"),
            row("fail", "fail"),
        ]
    )
    assert summary["fixes"] == 1
    assert summary["regressions"] == 0
    assert summary["candidate_eligible"]
    assert summary["admission_status"] == "eligible"


def test_all_unchanged_outputs_are_blocked_as_no_effect():
    summary = replay.admission_summary(
        [row("fail", "fail", same=True), row("pass", "pass", same=True)]
    )
    assert not summary["candidate_eligible"]
    assert summary["admission_status"] == "blocked_no_effect"


def test_dependency_change_blocks_admission_before_result_counts():
    summary = replay.admission_summary(
        [row("fail", "pass")],
        identity_verified=False,
        changed_dependencies=["cpp/src/audio_engine.cpp"],
    )
    assert not summary["candidate_eligible"]
    assert summary["admission_status"] == "blocked_identity"
    assert summary["changed_dependencies"] == ["cpp/src/audio_engine.cpp"]


def test_expected_gain_mode_is_verified_from_engine_telemetry():
    speech = {
        "agc2_enabled": True,
        "agc2_speech_gain_policy": True,
        "agc2_adaptive_enabled": False,
    }
    summary = replay.admission_summary(
        [row("fail", "pass", engine=speech)], expected_gain_mode="speech"
    )
    assert summary["admission_status"] == "eligible"

    wrong = dict(speech, agc2_enabled=False)
    summary = replay.admission_summary(
        [row("fail", "pass", engine=wrong)], expected_gain_mode="speech"
    )
    assert not summary["candidate_eligible"]
    assert summary["admission_status"] == "blocked_config"
    assert summary["config_errors"] == ["AGC speech mode differs: (False, True, False)"]


def test_stage_taps_are_required_when_gain_analysis_is_requested():
    summary = replay.admission_summary(
        [row("fail", "pass")],
        diagnostic_stages=True,
        expected_gain_mode="speech",
    )
    assert summary["admission_status"] == "blocked_config"
    assert summary["config_errors"] == ["AGC stage taps missing"]
