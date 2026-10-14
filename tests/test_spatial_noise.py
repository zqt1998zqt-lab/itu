import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cpp/tools"))
from spatial_noise import coherence,field,audit


@pytest.mark.parametrize("model",["spherical","cylindrical","corcos"])
def test_covariance_and_generated_field(model):
    positions=np.array([[0,0,0],[.035,0,0],[.070,0,0],[.105,0,0]])
    matrices=coherence(np.linspace(0,8000,513),positions,model)
    assert np.linalg.eigvalsh(matrices).min()>-1e-9
    rng=np.random.default_rng(55)
    data=field(rng.normal(size=16000*20),positions,16000,rng,model)
    check=audit(data,positions,16000,model)
    assert check["bins"]>20 and check["coherence_complex_mae"]<.16
    # Repeated mono channels cannot pass the field audit.
    bad=audit(np.repeat(data[:,:1],4,axis=1),positions,16000,model)
    assert bad["coherence_complex_mae"]>check["coherence_complex_mae"]+.1
