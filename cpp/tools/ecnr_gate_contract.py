"""Compare the frozen attenuation rule including its existing PCM16 bound branch."""
from __future__ import annotations
def canonical_gates(gates):
    selected={(g["segment"],g["metric"]):dict(g) for g in gates if g["segment"]!="runtime"}
    if len(selected)!=sum(g["segment"]!="runtime" for g in gates):raise ValueError("duplicate quality gate")
    changes=[]
    for key,g in list(selected.items()):
        if key[1]!="attenuation_lower_bound_db":continue
        parent_key=(key[0],"total_attenuation_db");parent=selected.get(parent_key)
        if (parent is None or parent["status"]!="unavailable" or parent.get("value") is not None or not parent["optional"]
            or g["optional"] or g["operator"]!=parent["operator"] or g["threshold"]!=parent["threshold"]):
            raise ValueError("not the existing valid censored attenuation branch")
        effective=dict(g);effective.update(metric="total_attenuation_db",optional=False,
            actual_metric="attenuation_lower_bound_db",comparison_basis="existing PCM16 conservative bound")
        selected[parent_key]=effective;del selected[key]
        changes.append(dict(segment=key[0],branch="existing PCM16 lower bound",actual_gates=[parent,g]))
    return selected,changes
def paired_contract(before,after):
    b,bc=canonical_gates(before);c,cc=canonical_gates(after)
    if set(b)!=set(c):raise ValueError("original quality gate inventory differs")
    for k in b:
        if any(b[k][field]!=c[k][field] for field in ("operator","threshold","optional")):
            raise ValueError("original quality gate contract differs")
    return b,c,dict(baseline=bc,candidate=cc)
