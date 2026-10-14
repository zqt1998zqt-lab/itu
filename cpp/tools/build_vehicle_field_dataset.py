#!/usr/bin/env python3
"""Independent sound-field extension; frozen sources and rules remain immutable."""
import argparse
import json
from pathlib import Path
import run_audio_eval as ev


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--speakers",type=int,default=2)
    p.add_argument("--snrs",type=int,nargs="+",default=[0,15,30],help="Diagnostic default; e.g. -10 -5 0 5 10 15 20 25 30")
    a=p.parse_args(); dataset=a.dataset.resolve(); root=a.root.resolve()
    if root==dataset or root.is_relative_to(dataset): p.error("extension must be outside frozen corpus")
    speakers,noises=ev.sources(dataset); names=sorted(speakers)
    if a.speakers<1 or len(names)<max(3,a.speakers): p.error("at least three source speakers required")
    root.mkdir(parents=True,exist_ok=False); (root/"sources").symlink_to(dataset/"sources",target_is_directory=True)
    cases=[]; cache={}
    for pair in range(a.speakers):
        for m in (1,2,4):
            for label,model,family in (("cold",None,"ns"),("noise_step","spherical","ns"),
                    ("diffuse","spherical","ns"),("cylindrical","cylindrical","ns"),
                    ("wind","corcos","ns"),("multirender","spherical","combined")):
                for snr in sorted(set(a.snrs)):
                    case=dict(id=f"field-p{pair}-m{m}-{label}-snr{snr}",pair=pair,
                        near_speaker=names[pair],far_speaker=names[(pair+1)%len(names)],
                        second_far_speaker=names[(pair+2)%len(names)],near_dbfs=-26,mics=m,
                        family=family,noise="tcar",snr_db=snr,ser_db=-5,rt60_s=.15,
                        echo_delay_ms=30,stress="none",seed=202610100+len(cases),profile="field-v1",pcm_subtype="FLOAT",
                        field_label=label,room_dimensions_m=[3.6,1.8,1.5],
                        microphone_positions_m=[[1.4+(ch-(m-1)/2)*.035,.9,1.3] for ch in range(m)],
                        near_position_m=[.8,.65,1.1],render_positions_m=[[.2,.3,.8],[.2,1.5,.8]])
                    if model and m>1: case["spatial_noise_model"]=model
                    if label=="cold": case["near_windows"]=[[0,10],[12,22]]
                    if label=="noise_step": case["noise_steps"]=[dict(time_s=11,gain_db=9)]
                    if label in ("diffuse","cylindrical","multirender"): case["near_rt60_s"]=.15
                    if label=="multirender": case["reference_channels"]=2
                    result=ev.make_case(case,speakers,noises,root/"cases"/case["id"],cache)
                    if label=="cold": result["windows"]["cold_start"]=[0,.5]
                    if label=="noise_step": result["windows"]["transition_noise"]=[10,12]
                    (root/"cases"/case["id"]/"scene.json").write_text(json.dumps(result,indent=2)+"\n")
                    cases.append(result); print(case["id"],flush=True)
    result=dict(identity=dict(source_manifest_sha256=ev.digest(dataset/"sources/manifest.json"),
        generator_sha256=ev.digest(Path(__file__)),acoustic_generator_sha256=ev.digest(Path(ev.__file__)),
        field_generator_sha256=ev.digest(Path(__file__).with_name("spatial_noise.py")),
        source_rate=16000,native_highband=False,scope="simulated cabin paths and PSD-colored fields; not measured cabin RIR"),scenes=cases)
    (root/"inputs-manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(scenes=len(cases),manifest_sha256=ev.digest(root/"inputs-manifest.json"))))


if __name__=="__main__": main()
