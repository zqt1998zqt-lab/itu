#!/usr/bin/env python3
"""Independent digital fixtures gate the ECNR campaign; never use speech for calibration."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy import signal
import ecnr_preflight as pre
from run_vehicle_stream import select_timeline, sequence_beam_delays

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)

def geometry(scene):
    positions=np.asarray(scene["microphone_positions_m"],dtype=float)
    near=np.asarray(scene["near_position_m"],dtype=float)
    if positions.shape != (scene["mics"],3) or near.shape!=(3,) or not np.all(np.isfinite(positions)):
        raise ValueError("finite actual microphone geometry required")
    arrival=(np.linalg.norm(positions-near,axis=1)-np.min(np.linalg.norm(positions-near,axis=1)))/343.0
    declared=np.asarray(scene["near_arrival_samples"],dtype=float)/16000
    beam=np.asarray(scene["beam_delays_samples"],dtype=float)/16000
    if not np.allclose(arrival,declared,atol=1e-10,rtol=0) or not np.allclose(beam,np.max(arrival)-arrival,atol=1e-10,rtol=0):
        raise ValueError("geometry and declared beam delays disagree")
    return arrival

def audit_geometry(mics, rate, delays_s, *, context="digital_calibration"):
    values=np.asarray(mics,dtype=float); delays=np.asarray(delays_s,dtype=float)
    if values.ndim!=2 or values.shape[1]!=len(delays): raise ValueError("calibration channel inventory differs")
    comparisons=[]
    for i in range(1,len(delays)):
        a,b=0,i
        if delays[b]<delays[a]: a,b=b,a
        spacing=(delays[b]-delays[a])*343*1000
        result=pre.audit_array_phase(values[:,a],values[:,b],rate,geometry="endfire" if spacing else "broadside",
            spacing_mm=float(spacing),context=context)
        comparisons.append(dict(channels=[a,b],declared_relative_delay_s=float(delays[b]-delays[a]),result=result))
    if not comparisons:
        # Mono still uses a separate broadband reference gain/delay calibration below.
        return dict(accepted=True,comparisons=[],scope="mono; no inter-microphone phase")
    return dict(accepted=all(x["result"]["accepted"] for x in comparisons),comparisons=comparisons)

def run(dataset, output):
    dataset=Path(dataset).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((dataset/"inputs-manifest.json").read_text())
    rng=np.random.default_rng(20261011)
    base_frequencies=rng.uniform(.04,.40,180); phases=rng.uniform(-math.pi,math.pi,180)
    def broadband(t): return .012*np.sum(np.sin(2*math.pi*frequencies[:,None]*t+phases[:,None]),axis=0)
    def tones(t): return .15*np.sin(2*math.pi*1000*t+.3)+.12*np.sin(2*math.pi*2000*t-.4)
    results=[]
    for rate in (16000,24000,48000):
        frequencies=base_frequencies*rate
        t=np.arange(round(1.25*rate))/rate; ref=broadband(t)
        delays=13.372+np.array([-.01,0,.005,-.005,.01,0,.005])
        normal=np.stack([broadband(t-d/1000) for d in delays])
        for kind,expected in (("normal",True),("delay_jump",False),("polarity",False),("gain_mismatch",False)):
            data=normal.copy()
            if kind=="delay_jump": data[3,len(t)//2:]=broadband(t[len(t)//2:]-(delays[3]+4)/1000)
            if kind=="polarity": data*=-1
            if kind=="gain_mismatch": data*=2
            result=pre.audit_bulk_delay(ref,data,rate,context="digital_calibration")
            gains=[10*math.log10(float(np.mean(record**2))/float(np.mean(ref**2))) for record in data]
            result["gain_measurement"]="stationary independent broadband RMS ratio per complete record; interpolation-biased projection gain retained separately"
            result["input_gain_db_range"]=[min(gains),max(gains)]
            result["gain_accepted"]=max(abs(x) for x in gains)<=1
            result["accepted"]=result["accepted"] and result["gain_accepted"]
            name=f"r{rate}-reference-{kind}"; fixture=output/(name+".npz")
            np.savez_compressed(fixture,reference=ref,recordings=data,sample_rate=rate)
            row=dict(id=name,expected_accepted=expected,actual_accepted=result["accepted"],
                     expectation_met=result["accepted"]==expected,fixture=str(fixture),fixture_sha256=sha(fixture),result=result)
            save(output/(name+".json"),row); results.append(row)
        for mics in (1,2,4):
            for pair in (0,2):
                timeline=select_timeline(manifest["scenes"],pair,mics); scene=timeline[0][1]
                sequence_beam_delays(timeline,mics); arrival=geometry(scene)
                if mics==1:
                    results.append(dict(id=f"r{rate}-m1-p{pair:02}-geometry",expected_accepted=True,actual_accepted=True,
                                        expectation_met=True,microphone_positions_m=scene["microphone_positions_m"],beam_delays_samples=[0]))
                    continue
                tone_t=np.arange(rate)/rate
                normal_array=np.stack([tones(tone_t-d) for d in arrival],axis=1)
                for kind,expected in (("normal",True),("polarity",False),("gain_mismatch",False)):
                    data=normal_array.copy()
                    if kind=="polarity": data[:,-1]*=-1
                    if kind=="gain_mismatch": data[:,-1]*=2
                    result=audit_geometry(data,rate,arrival)
                    name=f"r{rate}-m{mics}-p{pair:02}-{kind}";fixture=output/(name+".npz")
                    np.savez_compressed(fixture,microphones=data,sample_rate=rate,relative_delays_s=arrival,
                                        microphone_positions_m=scene["microphone_positions_m"],near_position_m=scene["near_position_m"])
                    row=dict(id=name,expected_accepted=expected,actual_accepted=result["accepted"],
                        expectation_met=result["accepted"]==expected,fixture=str(fixture),fixture_sha256=sha(fixture),
                        geometry_scene_id=scene["id"],microphone_positions_m=scene["microphone_positions_m"],
                        near_position_m=scene["near_position_m"],beam_delays_samples=scene["beam_delays_samples"],
                        result=result)
                    save(output/(name+".json"),row);results.append(row)
    identity=dict(dataset_manifest_sha256=sha(dataset/"inputs-manifest.json"),tool_sha256=sha(__file__),
                  preflight_sha256=sha(pre.__file__),numpy_version=np.__version__,
                  rates=[16000,24000,48000],microphones=[1,2,4],pairs=[0,2],seed=20261011,
                  source="independent deterministic broadband and 1/2kHz calibration, no speech",
                  scope="digital reconstruction using actual declared simulation geometry; no real cabin or double-talk proof")
    report=dict(schema_version=1,status="passed" if all(x["expectation_met"] for x in results) else "failed",
                complete=True,identity=identity,cases=results,
                summary=f'{sum(x["expectation_met"] for x in results)}/{len(results)} independent normal/negative expectations')
    save(output/"report.json",report)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
    report=run(args.dataset,args.root);print(json.dumps({k:report[k] for k in ("status","summary")}))
    return 0 if report["status"]=="passed" else 1
if __name__=="__main__": raise SystemExit(main())
