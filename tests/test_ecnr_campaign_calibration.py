import importlib.util, sys
from pathlib import Path
import numpy as np
import pytest
TOOLS=Path(__file__).resolve().parents[1]/"cpp/tools"
sys.path.insert(0,str(TOOLS))
import ecnr_campaign_calibration as c

def fixture():
    rate=24000;t=np.arange(rate)/rate;delays=np.array([0,1.18/16000,2.35/16000,3.55/16000])
    a=np.stack([.15*np.sin(2*np.pi*1000*(t-d))+.12*np.sin(2*np.pi*2000*(t-d)) for d in delays],axis=1)
    return rate,delays,a
def test_actual_geometry_tones_and_negative_controls():
    rate,delays,a=fixture()
    assert c.audit_geometry(a,rate,delays)["accepted"]
    reverse=a.copy();reverse[:,-1]*=-1
    assert not c.audit_geometry(reverse,rate,delays)["accepted"]
    gain=a.copy();gain[:,2]*=2
    assert not c.audit_geometry(gain,rate,delays)["accepted"]
    assert not c.audit_geometry(a,rate,delays*3)["accepted"]
    assert not c.audit_geometry(a,rate,delays,context="double_talk")["accepted"]
def test_geometry_checks_declared_delay_against_positions():
    positions=np.array([[2.0175,2,1.2],[1.9825,2,1.2]]);near=np.array([3.3,3.2,1.5])
    delay=(np.linalg.norm(positions-near,axis=1)-np.min(np.linalg.norm(positions-near,axis=1)))/343*16000
    scene=dict(mics=2,microphone_positions_m=positions.tolist(),near_position_m=near.tolist(),
               near_arrival_samples=delay.tolist(),beam_delays_samples=(max(delay)-delay).tolist())
    assert np.allclose(c.geometry(scene)*16000,delay)
    scene["beam_delays_samples"]=[0,0]
    with pytest.raises(ValueError,match="disagree"): c.geometry(scene)
