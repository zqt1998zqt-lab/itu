#!/usr/bin/env python3
"""Recompute field-dataset identity, waveform and spatial-coherence evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import run_audio_eval as ev
from spatial_noise import audit


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    manifest=a.dataset/"inputs-manifest.json"; scenes=json.loads(manifest.read_text())["scenes"]
    hashes=wav=clipped=nonfinite=0; fields=[]; reconstruction=[]
    for scene in scenes:
        folder=a.dataset/"cases"/scene["id"]
        for name,record in scene["files"].items():
            path=folder/name; assert path.stat().st_size==record["bytes"] and ev.digest(path)==record["sha256"]
            hashes+=1
            if path.suffix==".wav":
                x,rate=sf.read(path); assert rate==16000
                nonfinite+=int((~np.isfinite(x)).sum()); clipped+=int((np.abs(x)>=32767/32768).sum()); wav+=1
        before=ev.read(folder/"before.wav")
        components=sum(ev.read(folder/name) for name in ("clean-mics.wav","noise-component.wav","echo-component.wav"))
        difference=(before-components)*32768
        reconstruction.append(dict(id=scene["id"],max_lsb=float(np.abs(difference).max()),rms_lsb=float(np.sqrt(np.mean(difference**2)))))
        if scene.get("spatial_noise_model"):
            check=audit(ev.read(folder/"noise-component.wav"),scene["microphone_positions_m"],16000,scene["spatial_noise_model"])
            fields.append(dict(id=scene["id"],**check))
    report=dict(manifest_sha256=ev.digest(manifest),scenes=len(scenes),files_verified=hashes,wav_files=wav,
        clipped_samples=clipped,nonfinite_samples=nonfinite,
        counts={key:dict(Counter(str(s.get(key,1 if key=="reference_channels" else "missing")) for s in scenes))
                for key in ("near_speaker","mics","snr_db","field_label","reference_channels")},
        fields=fields,reconstruction=reconstruction,
        passed=clipped==0 and nonfinite==0 and all(r["max_lsb"]<=2 for r in reconstruction)
               and all(f["bins"]>20 and f["coherence_complex_mae"]<.16 for f in fields))
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("fields","reconstruction")},indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
