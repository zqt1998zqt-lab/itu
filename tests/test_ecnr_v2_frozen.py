import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cpp/tools"))
import run_ecnr_v2_frozen as f
def test_cleanup_cannot_remove_inputs(tmp_path):
    outside=tmp_path/"source.wav";outside.write_bytes(b"old")
    owned=tmp_path/"owned";owned.mkdir()
    with pytest.raises(ValueError):f.owned_unlink(outside,owned)
    link=owned/"input.wav";link.symlink_to(outside)
    with pytest.raises(ValueError):f.owned_unlink(link,owned)
    fresh=owned/"result.wav";fresh.write_bytes(b"new");f.owned_unlink(fresh,owned)
    assert outside.read_bytes()==b"old" and not fresh.exists()
def proof(tmp_path,delta=0):
    p=tmp_path/"proof.json";p.write_text(json.dumps(dict(complete=True,identity_verified=True,
      quality=dict(total_phases=270,baseline_pass=184,candidate_pass=185,new_failed_gates=0),
      perceptual=dict(status="completed",scores=[dict(metric=k,paired_delta_median=delta) for k in ("SIG","OVRL")]))))
    return p
@pytest.mark.parametrize("delta",[0,-.001])
def test_minimal_claims_without_identity_cannot_start_full(tmp_path,delta):
    # Aggregate PASS claims are insufficient, even with a non-declining median.
    with pytest.raises(ValueError,match="proof"):f.require_proof(proof(tmp_path,delta))
def test_incomplete_cannot_start_full(tmp_path):
    p=proof(tmp_path);d=json.loads(p.read_text());d["complete"]=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="proof"):f.require_proof(p)
