from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cpp/tools"))
from ecnr_gate_contract import paired_contract
def gate(metric="total_attenuation_db",status="pass",value=20,optional=False,threshold=15):
    return dict(segment="far_only",metric=metric,status=status,value=value,optional=optional,threshold=threshold,operator=">=")
def test_existing_pcm_bound_is_comparable_and_kept():
    before=[gate()]
    after=[gate(status="unavailable",value=None,optional=True),gate("attenuation_lower_bound_db",value=70)]
    b,c,changes=paired_contract(before,after)
    assert c[("far_only","total_attenuation_db")]["actual_metric"]=="attenuation_lower_bound_db"
    assert changes["candidate"][0]["actual_gates"]==after
def test_failed_bound_cannot_be_a_pass():
    b,c,_=paired_contract([gate()],[gate(status="unavailable",value=None,optional=True),gate("attenuation_lower_bound_db",status="fail",value=10)])
    assert c[("far_only","total_attenuation_db")]["status"]=="fail"
def test_arbitrary_gate_or_threshold_change_refused():
    with pytest.raises(ValueError):paired_contract([gate()],[gate(status="unavailable",value=None,optional=True),gate("attenuation_lower_bound_db",threshold=14)])
    with pytest.raises(ValueError):paired_contract([gate()],[gate("invented",value=40)])
