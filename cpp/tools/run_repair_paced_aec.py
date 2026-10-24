#!/usr/bin/env python3
"""Paced three-rate duplex AEC replay of existing, hash-verified array PCM."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

import run_frozen_repair_replay as replay
import measure_frozen_repair_v5 as metering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args, flags = parser.parse_known_args()
    requested = {f'vehicle-{rate}-validation-m2-02' for rate in (16000,24000,48000)}
    selected = [(run,row) for run,row in replay.select_cases('core')
                if run == 'controlled-native-full-20261017' and row['scene']['id'] in requested]
    if len(selected) != 3 or {r['scene']['sample_rate'] for _,r in selected} != {16000,24000,48000}:
        raise ValueError('missing three-rate fixed AEC coverage')
    for _,row in selected:
        if row['scene']['family'] != 'aec' or row['scene']['mics'] != 2:
            raise ValueError('expected actual two-microphone AEC inputs')
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/'artifacts').mkdir()
    cli = root/'artifacts/itu_audio_cli'
    shutil.copy2(args.cli,cli)
    ctl = replay.ctl
    dependencies = [Path(__file__),Path(replay.__file__),Path(metering.__file__),
                    Path(ctl.__file__),Path(ctl.legacy.__file__),Path(ctl.ev.__file__),
                    Path(ctl.legacy.acceptance.__file__),ctl.legacy.acceptance.RULES,cli]
    dependencies += [replay.PROJECT/'build'/run/'results.jsonl' for run,_ in selected]
    bindings = {str(p.resolve()):ctl.ev.digest(p) for p in dependencies}
    ctl.legacy.save(root/'identity.json',dict(dependencies=bindings,flags=flags,paced=True,
                    scope='three actual two-microphone duplex AEC streams, 30 s per rate'))
    results = []
    started = time.monotonic()
    with (root/'results.jsonl').open('x') as output:
        for run, original in selected:
            result = replay.replay((root,run,original,cli,flags,False,True))
            result = metering.remeasure((root,result))
            scene,engine = result['scene'],result['metrics']['engine']
            folder = root/run/'cases'/scene['id']/'runs/full'
            playback,rate = sf.read(folder/'result.rx-spk.wav',always_2d=True)
            reference,rate2 = sf.read(folder/'result.rx-ref.wav',always_2d=True)
            if rate != rate2 or not np.array_equal(playback,reference) or not np.any(playback):
                raise ValueError('AEC reference is not nonzero actual speaker playback')
            if not engine['aec_enabled'] or not engine['paced_10ms'] or engine['stream_resets'] != 0:
                raise ValueError('AEC/paced/continuous-state contract differs')
            expected = round(scene['seconds']*100)
            if engine['rx_processing_frames'] != expected or engine['reference_source'] != 'post_rx_processing':
                raise ValueError('RX processed more than once or reference tap differs')
            result['duplex_audit'] = dict(actual_speaker_reference_equal=True,
                rx_processing_frames=expected,stream_resets=0,paced=True)
            results.append(result)
            output.write(json.dumps(result)+'\n');output.flush()
            print(json.dumps(dict(completed=len(results),total=3,id=scene['id'],
                                  status=result['acceptance']['status'])),flush=True)
    changed = [path for path,sha in bindings.items() if ctl.ev.digest(Path(path)) != sha]
    report = dict(state='completed',cases=len(results),counts=Counter(r['acceptance']['status'] for r in results),
                  flags=flags,paced=True,seconds_per_case=30,identity_verified=not changed,
                  changed_dependencies=changed,elapsed_seconds=time.monotonic()-started,
                  duplex_audits_passed=len(results),standard_compliance=False,
                  runtime_failures=[dict(id=r['scene']['id'],gate=g) for r in results
                                    for g in r['acceptance']['gates'] if g['segment']=='runtime' and g['status']!='pass'])
    ctl.legacy.save(root/'report.json',report)
    print(json.dumps(report),flush=True)
    if changed:raise ValueError('dependencies changed')


if __name__ == '__main__':
    main()
