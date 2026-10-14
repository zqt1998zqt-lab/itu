from pathlib import Path
import sys
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cpp/tools"))
import ecnr_v2_evidence as e
def test_paired_median_is_not_difference_of_medians():
    pairs=[dict(baseline=dict(available=True,poly={k:b for k in ("SIG","BAK","OVRL")}),candidate=dict(available=True,poly={k:c for k in ("SIG","BAK","OVRL")})) for b,c in ((1,2),(2,2),(5,4))]
    p=e.score_summary(pairs,3)
    assert p["status"]=="completed"
    assert p["scores"][0]["paired_delta_median"]==0
    p=e.score_summary(pairs,4)
    assert p["status"]=="unavailable"
def test_stage_envelope_keeps_original_amplitude_and_declared_time(tmp_path):
    path=tmp_path/"raw.wav";sf.write(path,np.array([.05,-.02,1.2,-1.1]),16000,subtype="FLOAT")
    row=e.stage_record("test",path,2,tmp_path/"web","s","candidate","test")
    assert row["waveform"]["max"][2]>1
    assert row["timeline_offset_ms"]==-.125
    assert row["trimmed_leading_samples"]==0
