#!/usr/bin/env python3
"""Diagnostic: jointly disable two echo protection routes on all mono AEC cases."""
import json
from pathlib import Path
import shutil

import run_controlled_native_scenes as ctl


def main():
    project=Path(__file__).resolve().parents[2]
    source=project/"build/controlled-native-full-20261017"
    root=project/"build/failure-cause-analysis-20261017/aec-controls"
    rows=[json.loads(line) for line in (source/"results.jsonl").read_text().splitlines()]
    rows=[r for r in rows if r["scene"]["family"]=="aec" and r["scene"]["mics"]==1]
    assert len(rows)==27
    root.mkdir(parents=True,exist_ok=False);(root/"artifacts").mkdir()
    cli=root/"artifacts/itu_audio_cli";shutil.copy2(source/"artifacts/itu_audio_cli",cli);sha=ctl.ev.digest(cli)
    identity=json.loads((source/"identity.json").read_text())
    if sha!=identity["cli_sha256"]:raise ValueError("CLI differs")
    bindings={**identity["dependencies"],str(Path(__file__)):ctl.ev.digest(Path(__file__)),
              str(source/"results.jsonl"):ctl.ev.digest(source/"results.jsonl")}
    for p,h in bindings.items():
        if ctl.ev.digest(Path(p))!=h:raise ValueError(f"identity differs: {p}")
    result=[]
    for original in rows:
        scene=original["scene"];folder=root/"cases"/scene["id"];folder.mkdir(parents=True)
        for p in (source/"cases"/scene["id"]).iterdir():
            if p.is_file():(folder/p.name).symlink_to(p.resolve())
            elif p.name=="oracle":(folder/p.name).symlink_to(p.resolve(),target_is_directory=True)
        for name,meta in scene["files"].items():
            if ctl.ev.digest(folder/name)!=meta["sha256"]:raise ValueError("input changed")
        mode="diagnostic_joint_protection_off";prefix=folder/"runs"/mode/"result"
        command=ctl.command_for(scene,folder,prefix,str(cli))+["--no-reference-guard","--no-linear-speech-protection"]
        ctl.legacy.invoke(command,prefix,sha)
        measured=ctl.legacy.measure(scene,root,mode)
        result.append(dict(scene=scene,baseline=original["metrics"],diagnostic=measured["metrics"],
            baseline_acceptance=original["acceptance"],diagnostic_acceptance=measured["acceptance"]))
        print(json.dumps(dict(completed=len(result),id=scene["id"])),flush=True)
    changed=[p for p,h in bindings.items() if ctl.ev.digest(Path(p))!=h]
    ctl.legacy.save(root/"report.json",dict(cases=len(result),identity_verified=not changed,
        dependencies=bindings,cli_sha256=sha,changed_dependencies=changed,
        scope="diagnostic only, protective routes disabled; not a proposed default or final acceptance",standard_compliance=False))
    with (root/"results.jsonl").open("x") as stream:
        for row in result:stream.write(json.dumps(row)+"\n")


if __name__=="__main__":main()
