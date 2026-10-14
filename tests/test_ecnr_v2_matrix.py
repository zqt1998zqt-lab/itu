from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cpp/tools"))
import run_ecnr_v2_matrix as m
def report(status):
    g=dict(segment="near_only",metric="raw_sisdr_db",status=status,operator=">=",threshold=19,optional=False,value=20)
    return dict(identity={k:None for k in m.PAIR_KEYS},details=[dict(label="speech",gates=[g])])
def test_new_failure_and_repair_identified():
    assert len(m.compare(report("pass"),report("fail"))[0]["new_failures"])==1
    assert len(m.compare(report("fail"),report("pass"))[0]["repaired_gates"])==1
    assert len(m.compare(report("pass"),report("unavailable"))[0]["new_failures"])==1
def test_input_and_contract_drift_rejected():
    b=report("pass");c=deepcopy(b);c["identity"]["sample_rate"]=48000
    with pytest.raises(ValueError,match="identity"):m.compare(b,c)
    c=deepcopy(b);c["details"][0]["gates"][0]["threshold"]=18
    with pytest.raises(ValueError,match="contract"):m.compare(b,c)
def test_optional_unavailable_preserved():
    b=report("unavailable");b["details"][0]["gates"][0]["optional"]=True
    assert m.quality(b["details"][0]["gates"])
    assert not m.compare(b,b)[0]["new_failures"]
