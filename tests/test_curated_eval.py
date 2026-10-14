import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp/tools"))
import pytest
from run_curated_eval import summarize, select_cases


def test_historical_failure_remains_visible_with_nonapplicable_speech_gate():
    failed = dict(segment="noise_only", metric="total_attenuation_db", status="fail")
    rows = [dict(variant="test", sample_rate=16000, track="stress",
                 historical_threshold_verdict=dict(passed=False, gates=[failed]),
                 applicable_gates_passed=True, not_applicable_gates=[failed])]
    result = summarize(rows)[0]
    assert result["historical_passed"] == 0
    assert result["applicable_gates_passed"] == 1
    assert result["not_applicable_gates"] == 1
    assert result["failed_gates"]["noise_only/total_attenuation_db"] == 1


def test_full_cartesian_selection_and_exclusion_negative_controls():
    scenes = [dict(id=f"{p}/{snr}", pair=p, snr_db=snr, noise="tcar", profile="curated-v4",
                   family="ns", mics=1, split="development") for p in (0, 1) for snr in (0, 10)]
    assert len(select_cases(scenes, {0, 1}, {0, 10})) == 4
    for broken in (scenes[:-1], scenes+[scenes[0]], [dict(s, catalog_excluded=True) for s in scenes]):
        with pytest.raises(ValueError, match="complete"):
            select_cases(broken, {0, 1}, {0, 10})
    with pytest.raises(ValueError, match="complete"):
        select_cases(scenes, {0, 1}, {0, 10}, splits=["holdout"])
