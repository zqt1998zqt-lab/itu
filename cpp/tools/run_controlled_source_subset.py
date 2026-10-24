#!/usr/bin/env python3
"""Run only newly acquired noise sources without repeating the parent matrix."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import run_controlled_native_scenes as controlled
import numpy as np


def run_subset_group(arguments):
    original=controlled.source_context
    def context(case,manifest):
        speech,noise,used,evidence=original(case,manifest)
        noise=dict(noise);evidence=dict(evidence)
        key=case["noise"]
        if key in noise:
            required=(24 if case["family"] in ("ns","agc") else 30)*case["sample_rate"]
            samples=len(noise[key])
            # Published CAVEMOVE recordings can be up to 74 ms short of 30 s.
            # Preserve raw bytes and identity; explicitly record an EOF wrap.
            if samples<required:
                if not samples or required-samples>case["sample_rate"]//10:
                    raise ValueError("source too short: refusing >100 ms periodic extension")
                noise[key]=np.pad(noise[key],(0,required-samples),mode="wrap")
                evidence[key]=dict(evidence[key],eof_wrap_samples=required-samples,
                    original_samples=samples,analysis_channel=0,
                    extension_scope="EOF periodic wrap to finish one 30 s scene; no SRC; same original recording")
        return speech,noise,used,evidence
    controlled.source_context=context
    try:
        return controlled.run_group(arguments)
    finally:
        controlled.source_context=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--cli",type=Path,required=True)
    parser.add_argument("--noise-prefix",required=True)
    parser.add_argument("--parent-sources",type=Path,required=True)
    parser.add_argument("--jobs",type=int,choices=range(1,9),default=8)
    parser.add_argument("--paced",action="store_true")
    args=parser.parse_args()
    manifest,root=args.sources.resolve(),args.root.resolve()
    legacy,ev=controlled.legacy,controlled.ev
    if ev.digest(legacy.acceptance.RULES)!=legacy.RULES_SHA:raise ValueError("rules changed")
    groups=[g for g in controlled.matrix(manifest) if g["noise"].startswith(args.noise_prefix)]
    source_rows=json.loads(manifest.read_text())["sources"]
    parent_rows=json.loads(args.parent_sources.read_text())["sources"]
    added=[e for e in source_rows if e["kind"]=="noise" and e["scene"].lower().startswith(args.noise_prefix)]
    if not added or any(e["sha256"] in {p["sha256"] for p in parent_rows if p["kind"]=="noise"} for e in added):
        raise ValueError("noise prefix must select only new recording hashes absent from the parent manifest")
    rates=sorted({g["sample_rate"] for g in groups})
    for rate in rates:
        for family in ("ns","combined"):
            if not any(g["sample_rate"]==rate and g["mics"]==1 and g["family"]==family for g in groups):
                raise ValueError(f"missing planned {family} negative-control coverage at {rate} Hz")
        if args.paced and not all(any(g["sample_rate"]==rate and g["mics"]==m and g["family"]=="combined" for g in groups) for m in (1,2,4)):
            raise ValueError(f"missing planned paced microphone coverage at {rate} Hz")
    plan=controlled.validate_plan(groups,manifest)
    root.mkdir(parents=True,exist_ok=False);(root/"artifacts").mkdir()
    cli=root/"artifacts/itu_audio_cli";shutil.copy2(args.cli.resolve(),cli);cli_sha=ev.digest(cli)
    dependencies=[Path(__file__),Path(controlled.__file__),Path(ev.__file__),Path(legacy.__file__),
                  Path(controlled.audit.__file__),Path(legacy.acceptance.__file__),
                  Path(controlled.__file__).with_name("spatial_noise.py"),legacy.acceptance.RULES,manifest,
                  args.parent_sources.resolve()]
    for i,path in enumerate(dependencies):shutil.copy2(path,root/"artifacts"/f"{i:02d}-{path.name}")
    bindings={str(p.resolve()):ev.digest(p) for p in dependencies}
    identity=dict(cli_sha256=cli_sha,dependencies=bindings,rules_sha256=legacy.RULES_SHA,
                  source_manifest=str(manifest),parent_source_manifest=str(args.parent_sources.resolve()),
                  new_noise_recordings=[e["sha256"] for e in added],noise_prefix=args.noise_prefix,sample_rate_conversion=False)
    legacy.save(root/"identity.json",identity);legacy.save(root/"plan.json",dict(groups=groups,summary=plan))
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool,(root/"results.jsonl").open("x") as stream:
        futures=[pool.submit(run_subset_group,(g,root,manifest,cli,cli_sha)) for g in groups]
        for future in as_completed(futures):
            for row in future.result():rows.append(row);stream.write(json.dumps(row)+"\n")
            stream.flush()
            report=dict(state="running",completed=sum("scene" in r for r in rows),total=plan["scenes"],
                        elapsed_s=time.monotonic()-started,summary=legacy.summary(rows),standard_compliance=False)
            legacy.save(root/"report.json",report);print(json.dumps({k:v for k,v in report.items() if k!="summary"}),flush=True)
    controls=[];paced=[]
    rates=sorted({g["sample_rate"] for g in groups})
    for rate in rates:
        for family,mode,segment in (("ns","negative_no_ns","noise_only"),("combined","negative_no_aec","far_only")):
            row=next((r for r in rows if "scene" in r and r["scene"]["sample_rate"]==rate and
                      r["scene"]["mics"]==1 and r["scene"]["family"]==family),None)
            if row is None:continue
            scene=row["scene"];folder=root/"cases"/scene["id"];prefix=folder/"runs"/mode/"result"
            legacy.invoke(controlled.command_for(scene,folder,prefix,str(cli),mode),prefix,cli_sha)
            result=legacy.measure(scene,root,mode)
            gate=next(g for g in result["acceptance"]["gates"] if g["segment"]==segment and g["metric"]=="total_attenuation_db")
            controls.append(dict(id=scene["id"],sample_rate=rate,mode=mode,verified=gate["status"]=="fail",value=gate["value"],threshold=gate["threshold"]))
        if args.paced:
            for mics in (1,2,4):
                row=next((r for r in rows if "scene" in r and r["scene"]["sample_rate"]==rate and
                          r["scene"]["mics"]==mics and r["scene"]["family"]=="combined"),None)
                if row is None:continue
                scene=row["scene"];folder=root/"cases"/scene["id"];prefix=folder/"runs/paced/result"
                legacy.invoke(controlled.command_for(scene,folder,prefix,str(cli),paced=True),prefix,cli_sha)
                result=legacy.measure(scene,root,"paced")
                paced.append(dict(id=scene["id"],sample_rate=rate,mics=mics,
                    output_matches_offline=result["metrics"]["output_sha256"]==row["metrics"]["output_sha256"],
                    runtime_gates=[g for g in result["acceptance"]["gates"] if g["segment"]=="runtime"]))
    changed=[p for p,h in bindings.items() if ev.digest(p)!=h]
    if ev.digest(cli)!=cli_sha:changed.append(str(cli))
    errors=[r for r in rows if "error" in r]
    expected_controls=len(rates)*2
    controls_verified=len(controls)==expected_controls and expected_controls>0 and all(c["verified"] for c in controls)
    paced_verified=not args.paced or (len(paced)==len(rates)*3 and all(p["output_matches_offline"] and
                    all(g["status"]=="pass" for g in p["runtime_gates"]) for p in paced))
    report.update(state="completed_with_errors" if errors or changed else "completed" if controls_verified and paced_verified else "completed_with_verification_failures",
                  completed=sum("scene" in r for r in rows),elapsed_s=time.monotonic()-started,
                  negative_controls=controls,paced_checks=paced,controls_verified=controls_verified,paced_verified=paced_verified,
                  identity_verified=not changed,changed_dependencies=changed,execution_errors=len(errors),plan=plan,
                  expected_negative_controls=expected_controls,
                  scope="supplementary new-source subset; parent matrix remains separate and immutable")
    legacy.save(root/"report.json",report);print(json.dumps(report))


if __name__=="__main__":main()
