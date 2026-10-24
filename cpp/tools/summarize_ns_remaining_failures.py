#!/usr/bin/env python3
"""Join frozen frame evidence and real C++ ablations without changing acceptance."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_ns_remaining_frames as audit

P = audit.P
ROOT = P / 'build/ns-remaining-frame-audit-20261018'


def read(path):
    return [json.loads(line) for line in path.open()]


def key(r):
    return r['scope'], r['variant'], r['original_run'], r['scene']['id']


def folder(base, r):
    return base / r['scope'] / r['variant'] / r['original_run'] / r['scene']['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'summary')
    args = parser.parse_args()
    dependencies = [Path(__file__), ROOT / 'frames/results.jsonl', ROOT / 'ablations/results.jsonl',
                    ROOT / 'backend-core/results.jsonl', ROOT / 'dataset-audit/audit.json']
    hashes = {str(path): audit.sha(path) for path in dependencies}
    rows = read(ROOT / 'frames/results.jsonl')
    ablated = read(ROOT / 'ablations/results.jsonl') + read(ROOT / 'backend-core/results.jsonl')
    lookup = {(key(r), r['ablation']): r for r in ablated}
    dataset = json.loads((ROOT / 'dataset-audit/audit.json').read_text())
    sources = {(r['key']['original_run'], r['key']['scene_id']): r for r in dataset['failed_parent_scenes']}
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    ledger, windows = [], []
    groups = defaultdict(Counter)
    bins = defaultdict(Counter)
    for r in rows:
        group = groups[r['scope'] + ':' + r['variant']]
        group['failure_records'] += 1
        source = sources[(r['original_run'], r['scene']['id'])]
        integrity = source['objective_input_counts']
        bad = any(integrity[k] for k in ('files_missing_from_case_dir', 'file_scene_json_hash_mismatches', 'file_source_results_hash_mismatches'))
        bad |= any(integrity['nonfinite_samples'].values()) or any(integrity['clipped_samples_at_or_above_0_999'].values())
        bad |= source['component_reconstruction']['reconstruction']['samples_error_gt_1_lsb'] > 0
        record = dict(scope=r['scope'], variant=r['variant'], original_run=r['original_run'], scene=r['scene'],
                      domain=r['domain'], input_integrity_defect=bool(bad),
                      input_sha256=r['input_sha256'], clean_sha256=r['clean_sha256'], output_sha256=r['output_sha256'],
                      input_difficulty=dict(label_snr_db=r['scene']['snr_db'],
                                            clean_below60hz_ac_percent=source['clean_spectrum']['below_60hz_percent_of_ac'],
                                            clean_below100hz_ac_percent=source['clean_spectrum']['below_100hz_percent_of_ac']),
                      evidence=[], ablations={})
        for kind in ('no_rumble', 'steady_raw_off', 'startup_off', 'filter_only', 'bypass', 'bare_backend'):
            ab = lookup.get((key(r), kind))
            if ab:
                record['ablations'][kind] = dict(removed=ab['removed_failures'], added=ab['added_failures'],
                                                 main_pass=ab['ns_tradeoff_pass'], raw_pass=ab['raw_quality_pass'],
                                                 declared_delay_samples=ab['engine']['delay_samples'], invocation=ab['invocation'])
        with np.load(folder(ROOT / 'frames', r) / 'frames.npz') as f:
            for label, w in r['windows'].items():
                labels = sorted(set(w['triggered']))
                item = dict(scope=r['scope'], variant=r['variant'], original_run=r['original_run'], scene_id=r['scene']['id'],
                            sample_rate=r['scene']['sample_rate'], split=r['scene']['split'], noise=r['scene']['noise'],
                            snr_db=r['scene']['snr_db'], window=label, start_s=w['limits_s'][0], stop_s=w['limits_s'][1],
                            failures=';'.join(labels), input_integrity_defect=bool(bad))
                if label.startswith('near'):
                    group['failed_near_windows'] += 1
                    group['active_frame_appearances'] += w['active_frames']
                    group['newly_damaged_frame_appearances'] += w['newly_damaged_frames']
                    item.update(input_projection_db=w['before']['projection_gain_db'], output_projection_db=w['after']['projection_gain_db'],
                                input_sisdr_db=w['before']['sisdr_db'], output_sisdr_db=w['after']['sisdr_db'],
                                input_stoi=w['before']['stoi'], output_stoi=w['after']['stoi'],
                                newly_damaged_frames=w['newly_damaged_frames'],
                                newly_damaged_target_energy_percent=w['newly_damaged_target_energy_percent'],
                                frame_snr_median_db=w['input_snr_db']['median'])
                    damage = w['worst_damage_run']
                    item['longest_new_damage_ms'] = damage['frames'] * 10
                    item['longest_new_damage_start_s'] = None if damage['start_frame'] is None else w['limits_s'][0] + damage['start_frame'] / 100
                    if label + ':projection_gain' in labels:
                        group['projection_failed_windows'] += 1
                        outside = w['before_projection_already_outside_gate']
                        group['projection_input_already_failed'] += outside
                        group['projection_input_passed_output_failed'] += not outside
                        if not outside:
                            record['evidence'].append(label + ': input projection inside gate, output outside')
                    if label + ':raw_sisdr' in labels:
                        band = w['common_above100_diagnostic']
                        recovered = band['after']['sisdr_db'] >= min(band['before']['sisdr_db'], 20) - 1
                        group['raw_failed_windows'] += 1
                        group['raw_common100_recovers'] += recovered
                        item['common100_raw_diagnostic_recovers'] = recovered
                    first, last = w['frame_start'], w['frame_stop']
                    active = f['target_active'][first:last]
                    bproj, aproj = f['input_projection_db'][first:last], f['output_projection_db'][first:last]
                    mask = active & (bproj >= -3) & (aproj < -6)
                    snr, level, cp = f['input_frame_snr_db'][first:last], f['clean_dbfs'][first:last], f['clean_power'][first:last]
                    for name, selected in (('snr_lt0', snr < 0), ('snr_0to15', (snr >= 0) & (snr < 15)), ('snr_ge15', snr >= 15),
                                           ('clean_ge_minus40', level >= -40), ('clean_lt_minus40', level < -40)):
                        b = bins[r['scope'] + ':' + r['variant'] + ':' + name]
                        b['active_frame_appearances'] += int(np.sum(active & selected))
                        b['newly_damaged_frame_appearances'] += int(np.sum(mask & selected))
                        b['clean_energy'] += float(cp[active & selected].sum())
                        b['newly_damaged_clean_energy'] += float(cp[mask & selected].sum())
                else:
                    group['failed_noise_windows'] += 1
                    group['noise_target_nonzero'] += bool(w['clean_nonzero_samples'])
                    group['noise_input_quantization_floor'] += w['input_power_below_1e_minus12']
                    item.update(attenuation_db=w['attenuation']['total_attenuation_db'],
                                top10_input_energy_percent=w['top10_input_energy_percent'],
                                top10_output_energy_percent=w['top10_output_energy_percent'],
                                top10_attenuation_db=w['top10_attenuation_db'], other90_attenuation_db=w['other90_attenuation_db'])
                    group['noise_top10_fails_other90_pass'] += w['top10_attenuation_db'] < 6 <= w['other90_attenuation_db']
                    for margin in ('20', '50', '100', '250', '500'):
                        value = w['trim_boundary_diagnostic'][margin]['total_attenuation_db']
                        passes = value is not None and value >= 6
                        group['noise_trim' + margin + '_passes'] += passes
                        item['trim' + margin + '_attenuation_db'] = value
                    record['evidence'].append(label + ': target exactly zero, actual residual suppression below gate')
                for kind in ('no_rumble', 'steady_raw_off', 'startup_off', 'filter_only', 'bare_backend'):
                    ab = lookup.get((key(r), kind))
                    if ab:
                        item[kind + '_removed'] = ';'.join(z for z in ab['removed_failures'] if z in labels)
                windows.append(item)
        # Evidence flags may overlap. They are not mutually exclusive diagnoses.
        record['conclusion_scope'] = 'processing effects and input difficulty can coexist; no claim of inherent unsolvability'
        ledger.append(record)
    for value in bins.values():
        value['newly_damaged_clean_energy_percent'] = 100 * value['newly_damaged_clean_energy'] / max(value['clean_energy'], 1e-30)
    summary = dict(failure_records=len(rows), unique_parent_scenes=len(sources),
                   analyzed_frame_appearances=sum(r['frame_count'] for r in rows), failed_windows=len(windows),
                   groups=dict(groups), frame_bins=dict(bins),
                   all_failure_labels=dict(Counter(f.split(':')[-1] for r in rows for f in r['failures'] + r['raw_failures'])),
                   objective_integrity=dataset['objective_integrity_counts'],
                   max_mix_reconstruction_error_lsb=max(s['component_reconstruction']['reconstruction']['max_error_lsb'] for s in sources.values()),
                   no_input_integrity_defect_records=sum(not r['input_integrity_defect'] for r in ledger),
                   records_with_processing_effect_evidence=sum(bool(r['evidence']) for r in ledger),
                   ablation_calls=len(ablated), algorithm_or_dataset_is_not_binary=True,
                   thresholds_unchanged=True, standard_compliance=False)
    audit.save(out / 'report.json', summary)
    with (out / 'record-ledger.jsonl').open('x') as stream:
        for row in ledger:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    fields = sorted({field for row in windows for field in row})
    with (out / 'window-ledger.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(windows)
    # Concrete examples selected to expose distinct mechanisms, not a ranking.
    examples = []
    rain = next(r for r in rows if r['scope']=='vehicle' and r['variant']=='webrtc_v4_balanced' and r['scene']['id']=='vehicle-noise-24000-holdout-m1-sim_rain_impulse-snr-10')
    impulse = next(r for r in rows if r['scope']=='vehicle' and r['variant']=='speex_v4_balanced' and r['scene']['id']=='vehicle-noise-16000-development-m1-sim_indicator-snr-10')
    clean = next(r for r in rows if r['scope']=='clean' and r['variant']=='speex_v4_noise' and lookup[(key(r),'filter_only')]['raw_failures'])
    cold = next(r for r in rows if r['scope']=='cold' and r['variant']=='webrtc_v4_balanced' and 'near_startup:projection_gain' in lookup[(key(r),'startup_off')]['added_failures'])
    for number, r, label, title in ((1,rain,'near_only','Low-SNR speech: processing adds target suppression'),
                                     (2,impulse,'noise_only','Impulsive noise: energy-weighted residual is dominated by bursts'),
                                     (3,clean,'near_only','Clean input: fixed high-pass versus unfiltered diagnostic'),
                                     (4,cold,'near_startup','Cold start: protection versus disabled startup')):
        original_row = next(x for x in read(audit.R / ('final-'+r['scope']) / 'results.jsonl') if key(dict(x,scope=r['scope'])) == key(r))
        clean_pcm, before, original, source, crop = audit.source_arrays(r['scope'], original_row)
        rate = r['scene']['sample_rate']
        f = audit.frames(clean_pcm, before, original, rate)
        lo, hi = r['scene']['windows'][label]
        lo = max(0, lo - (.5 if number==2 else 0))
        hi = min(len(clean_pcm)/rate, hi)
        sl = (f['time_s'] >= lo) & (f['time_s'] < hi)
        t = f['time_s'][sl]
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        ax[0].plot(t, f['clean_dbfs'][sl], label='target', lw=1)
        ax[0].plot(t, f['input_dbfs'][sl], label='input', lw=1, alpha=.8)
        ax[0].plot(t, f['output_dbfs'][sl], label='repaired output', lw=1, alpha=.8)
        ax[0].set(ylabel='10ms RMS (dBFS)', ylim=(-95, 0))
        ax[0].legend(loc='lower right', ncols=3)
        active = f['target_active'][sl]
        if number==2:
            ax[1].plot(t, f['noise_attenuation_db'][sl], lw=1)
            ax[1].axhline(6, color='r', ls='--', label='6dB noise gate')
            ax[1].set(ylabel='10ms noise attenuation (dB)', ylim=(-10, 25))
        else:
            ax[1].plot(t, np.where(active, f['input_projection_db'][sl], np.nan), label='input', lw=1)
            ax[1].plot(t, np.where(active, f['output_projection_db'][sl], np.nan), label='repaired output', lw=1)
            ax[1].axhline(-3, color='r', ls='--')
            ax[1].axhline(-6, color='r', ls=':', alpha=.6)
            ax[1].set(ylabel='10ms target projection (dB)', ylim=(-30, 20))
        if number in (3,4):
            kind = 'filter_only' if number==3 else 'startup_off'
            ab = lookup[(key(r),kind)]
            path = folder(ROOT/'ablations'/kind,r)/'result.after.wav'
            audio, actual_rate = audit.sf.read(path)
            audio = audio[ab['engine']['delay_samples']:]
            ff = audit.frames(clean_pcm,before,audio,rate)
            ax[1].plot(t, np.where(active,ff['output_projection_db'][sl],np.nan),label=kind,lw=1,alpha=.8)
            ax[2].plot(t, f['output_nmse_db'][sl], label='repaired output', lw=1)
            ax[2].plot(t, ff['output_nmse_db'][sl], label=kind, lw=1, alpha=.8)
            ax[2].set(ylabel='10ms target error / target (dB)',ylim=(-45,30))
        else:
            control = audit.R/('final-'+r['scope'])/r['variant']/r['original_run']/r['scene']['id']/'result.csv'
            c = np.genfromtxt(control,delimiter=',',names=True)
            mask = (c['time_s']>=lo)&(c['time_s']<hi)
            ax[2].plot(c['time_s'][mask], c['confidence'][mask], label='input confidence', lw=1)
            ax[2].plot(c['time_s'][mask], c['quiet_mix_target'][mask], label='raw mix target', lw=1)
            ax[2].plot(c['time_s'][mask], c['rumble_mix_target'][mask], label='rumble target', lw=1)
            ax[2].set(ylabel='Input-time control targets',ylim=(-.05,1.05))
        for axis in ax:
            axis.grid(alpha=.2)
            axis.legend(loc='upper right',fontsize=8)
        ax[-1].set_xlabel('Input time after cold crop (s)' if r['scope']=='cold' else 'Input time (s)')
        fig.suptitle(title+'\n'+r['variant']+' / '+r['scene']['id'],fontsize=10)
        figure = out/f'example-{number}.png'
        fig.savefig(figure,dpi=160)
        plt.close(fig)
        example = dict(number=number,key=key(r),window=label,figure=str(figure.relative_to(P)),
                       whole_window=r['windows'].get(label, original_row['segments'][label]),
                       comparison_ablation='filter_only' if number==3 else 'startup_off' if number==4 else None)
        if example['comparison_ablation']:
            ab = lookup[(key(r),example['comparison_ablation'])]
            example['comparison_window']=ab['segments'][label]
        examples.append(example)
    audit.save(out/'examples.json',examples)
    changed = [path for path, digest in hashes.items() if audit.sha(Path(path)) != digest]
    audit.save(out/'identity.json',dict(dependencies=hashes,identity_verified=not changed,changed_dependencies=changed))
    if changed:
        raise ValueError('summary dependency drift')
    print(json.dumps({k:summary[k] for k in ('failure_records','unique_parent_scenes','failed_windows','ablation_calls','all_failure_labels','no_input_integrity_defect_records')}))


if __name__=='__main__':
    main()
