#!/usr/bin/env python3
"""Separate simulated echo delay from reverberation with fixed source pairs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import json
from pathlib import Path
import shutil

import soundfile as sf

import run_controlled_native_scenes as ctl


def main():
    project=Path(__file__).resolve().parents[2]
    source=project/"build/controlled-native-full-20261017"
    root=project/"build/failure-cause-analysis-20261017/aec-factorial"
    identity=json.loads((source/"identity.json").read_text());manifest=Path(identity["source_manifest"])
    for p,h in identity["dependencies"].items():
        if ctl.ev.digest(Path(p))!=h:raise ValueError(f"identity differs: {p}")
    groups=[]
    plan=json.loads((source/"plan.json").read_text())["groups"]
    for rate in (16000,24000,48000):
        base=next(g for g in plan if g["id"]==f"vehicle-{rate}-validation-m1-00")
        for rt60 in (.15,.45,.6):
            for delay in (20,80,180):
                g=copy.deepcopy(base);g.update(id=f"factorial-{rate}-rt{rt60:.2f}-delay{delay:03d}",
                    rt60_s=rt60,echo_delay_ms=delay,grid=[None],grid_kind="diagnostic_echo_factorial")
                groups.append(g)
    root.mkdir(parents=True,exist_ok=False);(root/"artifacts").mkdir()
    cli=root/"artifacts/itu_audio_cli";shutil.copy2(source/"artifacts/itu_audio_cli",cli);sha=ctl.ev.digest(cli)
    if sha!=identity["cli_sha256"]:raise ValueError("CLI differs")
    bindings={**identity["dependencies"],str(Path(__file__)):ctl.ev.digest(Path(__file__))}
    ctl.legacy.save(root/"identity.json",dict(dependencies=bindings,cli_sha256=sha,source_identity_sha256=ctl.ev.digest(source/"identity.json")))
    ctl.legacy.save(root/"plan.json",dict(groups=groups,scope="validation-source diagnostic factorial; not a new blind test"))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool,(root/"results.jsonl").open("x") as stream:
        for future in as_completed([pool.submit(ctl.run_group,(g,root,manifest,cli,sha)) for g in groups]):
            result=future.result();rows.extend(result)
            for r in result:stream.write(json.dumps(r)+"\n")
            stream.flush();print(json.dumps(dict(completed=len(rows))),flush=True)
    hashes={}
    for rate in (16000,24000,48000):
        values={name:set() for name in ("clean.wav","ref.wav","playback.wav")}
        for r in rows:
            if "scene" not in r or r["scene"]["sample_rate"]!=rate:continue
            for name in values:
                x,sr=sf.read(root/"cases"/r["scene"]["id"]/name,dtype="float64",always_2d=True)
                values[name].add((sr,x.shape,hashlib.sha256(x.tobytes()).hexdigest()))
        hashes[str(rate)]={name:len(v) for name,v in values.items()}
    changed=[p for p,h in bindings.items() if ctl.ev.digest(Path(p))!=h]
    errors=[r for r in rows if "error" in r]
    verified=not changed and not errors and len(rows)==27 and all(n==1 for v in hashes.values() for n in v.values())
    ctl.legacy.save(root/"report.json",dict(state="completed",cases=len(rows),execution_errors=errors,
        identity_verified=not changed,changed_dependencies=changed,paired_source_pcm_hash_counts=hashes,
        same_target_and_reference_verified=verified,standard_compliance=False))
    print(json.dumps(dict(completed=len(rows),verified=verified)),flush=True)


if __name__=="__main__":main()
