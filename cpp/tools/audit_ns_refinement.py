#!/usr/bin/env python3
"""Audit complete experiment bindings, actual configs, PCM identity and replay."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,struct
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import refine_ns_streams as opt


def main():
    root=opt.ROOT;reports={};rows={};hash_cache={};errors=[]
    def digest(path):
        path=Path(path)
        if path not in hash_cache:hash_cache[path]=hashlib.sha256(path.read_bytes()).hexdigest()
        return hash_cache[path]
    phases=['round1','development-cold','development-clean','round2','round2-cold','round2-clean','final-vehicle','final-diagnostic','final-clean','final-cold']
    output_count=0;canonical={};cold_count=0
    for phase in phases:
        folder=root/phase;report=json.loads((folder/'report.json').read_text());identity=json.loads((folder/'identity.json').read_text())
        assert report['state']=='completed' and report['identity_verified'],phase
        for p,h in identity['dependencies'].items():
            if digest(p)!=h:errors.append(dict(phase=phase,dependency=p,error='identity drift'))
        data=[json.loads(l) for l in (folder/'results.jsonl').open()];rows[phase]=data
        assert len(data)==report['completed']==report['total']
        assert len({(r['variant'],r['original_run'],r['scene']['id']) for r in data})==len(data)
        for r in data:
            prefix=folder/r['variant']/r['original_run']/r['scene']['id'];c=r['config'];em=r['engine']
            for field in ('backend','strength','dry_max','snr_threshold_db','noise_gain','startup_ms','mild_strength','slew_ms'):
                a,b=em[field],c[field]
                if a!=b and not (isinstance(a,(int,float)) and abs(a-b)<1e-6):errors.append(dict(phase=phase,id=r['scene']['id'],field=field,error='actual config mismatch'))
            assert em['control_policy']==c['policy']
            assert em['aec']==False and em['agc']==False and em['hpf']==False
            assert digest(prefix/'result.after.wav')==r['output_sha256'];output_count+=1
            if r['scope'].startswith('cold'):
                inp=prefix/'cold-input.wav';assert digest(inp)==r['input_sha256'];x,sr=sf.read(inp,dtype='float32');info=sf.info(inp)
                canon=hashlib.sha256(struct.pack('<IIQ',sr,info.channels,len(x))+x.astype('<f4').tobytes()).hexdigest()
                key=(r['original_run'],r['scene']['id']);assert canonical.setdefault(key,canon)==canon
                cold_count+=1
            else:
                filename='clean.wav' if r['scope'].startswith('clean') else 'before.wav'
                inp=opt.PROJECT/'build'/r['original_run']/'cases'/r['scene']['id']/filename
                assert digest(inp)==r['input_sha256']
            target=opt.PROJECT/'build'/r['original_run']/'cases'/r['scene']['id']/'clean.wav'
            assert digest(target)==r['clean_sha256']
        reports[phase]=report
    # Frozen winners must reproduce each corresponding tuning output, not
    # merely a similar aggregate. Startup/cold fixtures compare PCM outputs.
    finals=json.loads((root/'final-variants.json').read_text());pairs=[]
    for c in finals:
        src=c['source_variant'];source_phase='round2' if '_r2_' in src else 'round1'
        tune={(r['original_run'],r['scene']['id']):r for r in rows[source_phase] if r['variant']==src}
        full={(r['original_run'],r['scene']['id']):r for r in rows['final-vehicle'] if r['variant']==c['name']}
        assert len(tune)==262
        for key,r in tune.items():
            f=full[key];same=r['output_sha256']==f['output_sha256'] and r['segments']==f['segments'] and r['onsets']==f['onsets'] and r['pesq_first']==f['pesq_first']
            pairs.append(dict(variant=c['name'],run=key[0],id=key[1],equal=same))
    assert len(pairs)==1048 and all(p['equal'] for p in pairs)
    for name in ('probe-validation','stream-probe-validation'):
        assert json.loads((root/name/'report.json').read_text())['passed']
    assert json.loads((root/'baseline-reproduction-v2.json').read_text())['passed']
    for name in ('perceptual','streams'):
        report=json.loads((root/name/'report.json').read_text());assert report['state']=='completed' and report['identity_verified'];reports[name]=report
        for p,h in json.loads((root/name/'identity.json').read_text())['dependencies'].items():assert digest(p)==h
    stream=reports['streams'];assert all(r['paced_pcm_bit_exact'] for r in stream['rows'])
    native=json.loads((root/'native24/report.json').read_text());assert native['state']=='completed'
    assert all(r['engine']['core_rate']==24000 and r['engine']['src_delay_samples']==0 for r in native['rows'])
    output=dict(passed=not errors,errors=errors,measurements=sum(len(x) for x in rows.values()),verified_output_files=output_count,verified_unique_file_hashes=len(hash_cache),canonical_cold_input_files=cold_count,unique_cold_PCM_groups=len(canonical),cold_PCM_equal_across_variants_and_phases=True,full_vs_tune_pairs=1048,full_vs_tune_all_exact=True,baseline_reproduction_pairs=1572,native24_rows=len(native['rows']),reports=reports,standard_compliance=False)
    opt.save(root/'final-audit.json',output);opt.save(root/'tune-full-equivalence.json',pairs)
    print(json.dumps({k:v for k,v in output.items() if k!='reports'},indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
