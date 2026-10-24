#!/usr/bin/env python3
"""Verify retained quality evidence against the final C++ integration build."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import soundfile as sf

from verify_ns_vehicle_v4 import B, P, R, bindings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def records(phase):
    rows = [json.loads(line) for line in (R / phase / 'results.jsonl').open()]
    keyed = {(r['variant'], r['original_run'], r['scene']['id']): r for r in rows}
    require(rows and len(keyed) == len(rows), 'empty or duplicate result identity: ' + phase)
    report = read(R / phase / 'report.json')
    require(report['state'] == 'completed' and report['identity_verified'], 'incomplete phase: ' + phase)
    require(report['completed'] == report['total'] == len(rows), 'phase count differs: ' + phase)
    return keyed


def gates(row):
    failures, raw = [], []
    for label, item in row['segments'].items():
        if label.startswith('near'):
            before, after = item['before'], item['after']
            if after['stoi'] is not None and before['stoi'] is not None and after['stoi'] < before['stoi'] - .02:
                failures.append(label + ':stoi')
            if not -3 <= after['projection_gain_db'] <= 18:
                failures.append(label + ':projection_gain')
            if after['sisdr_db'] < min(before['sisdr_db'], 20) - 1:
                raw.append(label + ':raw_sisdr')
        elif label.startswith('noise_only'):
            attenuation = item.get('total_attenuation_db')
            if attenuation is None:
                attenuation = item.get('attenuation_lower_bound_db')
            if attenuation is not None and attenuation < 6:
                failures.append(label + ':attenuation')
    if row['engine']['clipped_samples_before_write'] > 0:
        failures.append('clipping')
    require(failures == row['failures'] and raw == row['raw_failures'], 'stored gates differ')
    require(row['ns_tradeoff_pass'] == (not failures), 'main gate flag differs')
    require(row['raw_quality_pass'] == (not failures and not raw), 'combined raw gate flag differs')


def verify_cold_header(old_path, expected_digest, phase, origin, rate):
    # libsndfile FLOAT WAV embeds wall time in PEAK. Prove that this is the
    # ONLY differing byte range by recovering the exact recorded file hash.
    payload = bytearray(old_path.read_bytes())
    offset, timestamp_offset = 12, None
    while offset + 8 <= len(payload):
        tag, size = struct.unpack_from('<4sI', payload, offset)
        if tag == b'PEAK':
            require(size >= 8, 'invalid WAV PEAK chunk')
            timestamp_offset = offset + 12
            break
        offset += 8 + size + (size % 2)
    require(timestamp_offset is not None, 'cold WAV has no timestamp metadata')
    before, sr = sf.read(origin / 'before.wav')
    clean, cr = sf.read(origin / 'clean.wav')
    retained, rr = sf.read(old_path)
    hop = rate // 100
    power = np.mean(clean[:len(clean) // hop * hop].reshape(-1, hop) ** 2, axis=1)
    positions = np.flatnonzero(power >= 10 ** (-65 / 10))
    require(len(positions) > 0, 'cold fixture has no signal')
    cropped = before[int(positions[0]) * hop:].astype(np.float32).astype(np.float64)
    require(sr == cr == rr == rate and np.array_equal(cropped, retained), 'cold PCM differs from frozen parent crop')
    start = int((R / phase / 'identity.json').stat().st_mtime) - 1
    stop = int((R / phase / 'report.json').stat().st_mtime) + 2
    for timestamp in range(start, stop):
        struct.pack_into('<I', payload, timestamp_offset, timestamp)
        if hashlib.sha256(payload).hexdigest() == expected_digest:
            return dict(peak_timestamp=timestamp, pcm_samples=len(retained), exact_file_hash_reconstructed=True)
    raise RuntimeError('cold input hash cannot be explained by WAV PEAK timestamp alone')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bindings()
    overlay = read(R / 'engine-integration-freeze-v2.json')
    binary_sha256 = sha(B)
    digest_cache = {}

    def check(path, expected):
        if path not in digest_cache:
            digest_cache[path] = sha(path)
        require(digest_cache[path] == expected, 'hash mismatch: ' + str(path))

    phases, total, cold_header_checks = [], 0, []
    for scope, expected in (('vehicle', 2541), ('clean', 264), ('cold', 264), ('native24', 72)):
        old_phase, new_phase = 'final-' + scope, 'final-r5-' + scope
        old, new = records(old_phase), records(new_phase)
        require(old.keys() == new.keys() and len(old) == expected, 'replay keys differ: ' + scope)
        for phase in (old_phase, new_phase):
            for path, expected_digest in read(R / phase / 'identity.json')['dependencies'].items():
                relative = str(Path(path).relative_to(P)) if Path(path).is_relative_to(P) else path
                change = overlay['changes'].get(relative)
                if change and change['before_sha256'] == expected_digest:
                    expected_digest = change['after_sha256']
                check(Path(path), expected_digest)
        for key, a in old.items():
            b = new[key]
            for field in ('config', 'scene', 'clean_sha256', 'output_sha256',
                          'segments', 'onsets', 'failures', 'raw_failures', 'ns_tradeoff_pass',
                          'raw_quality_pass', 'pesq_first', 'mean_stoi_improvement',
                          'mean_sisdr_improvement_db', 'mean_noise_attenuation_db'):
                require(a[field] == b[field], 'r4/r5 field differs: ' + repr((key, field)))
            gates(a)
            gates(b)
            require(b['binary_sha256'] == binary_sha256, 'r5 result uses another binary')
            folder = R / old_phase / a['variant'] / a['original_run'] / a['scene']['id']
            require(a['wav_retained'] and not b['wav_retained'], 'unexpected retained-WAV policy')
            check(folder / 'result.after.wav', a['output_sha256'])
            origin = P / 'build' / a['original_run'] / 'cases' / a['scene']['id']
            source = origin / ('clean.wav' if scope == 'clean' else 'before.wav')
            if scope == 'cold':
                source = folder / 'cold-input.wav'
            check(source, a['input_sha256'])
            check(origin / 'clean.wav', a['clean_sha256'])
            if a['input_sha256'] != b['input_sha256']:
                require(scope == 'cold', 'non-cold input identity differs')
                proof = verify_cold_header(source, b['input_sha256'], new_phase, origin, a['scene']['sample_rate'])
                cold_header_checks.append(dict(key=key, old_sha256=a['input_sha256'], new_sha256=b['input_sha256'], **proof))
        phases.append(dict(scope=scope, rows=len(old), output_sha256_bitexact=len(old),
                           quality_fields_identical=len(old), gates_recomputed=2 * len(old),
                           counts={name: dict(cases=sum(r['variant'] == name for r in old.values()),
                                              passed=sum(r['ns_tradeoff_pass'] for r in old.values() if r['variant'] == name),
                                              raw_passed=sum(r['raw_quality_pass'] for r in old.values() if r['variant'] == name))
                                   for name in sorted({r['variant'] for r in old.values()})}))
        total += len(old)

    engine = read(R / 'engine-cli-equivalence-final' / 'report.json')
    require(engine['state'] == 'completed' and len(engine['rows']) == 9, 'engine matrix incomplete')
    require(all(r['bitexact'] and r['max_abs_difference'] == 0 for r in engine['rows']), 'engine PCM differs')
    require(len({(r['variant'], r['rate']) for r in engine['rows']}) == 9, 'duplicate engine pair')
    check(R / 'engine-build/cpp/itu_audio_cli', engine['engine_sha256'])
    streams = read(R / 'streams-r5' / 'report.json')
    require(streams['state'] == 'completed' and streams['identity_verified'] and len(streams['rows']) == 10, 'stream matrix incomplete')
    require(all(r['paced_pcm_bitexact'] and r['reset_negative_different_samples'] > 0 and r['paced']['paced_deadline_misses'] == 0
                for r in streams['rows']), 'stream verification failed')
    baseline = read(R / 'baseline-reproduction-r5' / 'report.json')
    require(baseline['all_bitexact'] and len(baseline['rows']) == 6, 'legacy baseline changed')

    files = ['selection-freeze.json', 'engine-integration-freeze-v2.json',
             'artifacts/final-r4/identity.json', 'artifacts/final-r5/identity.json',
             'engine-cli-equivalence-final/report.json', 'streams-r5/report.json',
             'baseline-reproduction-r5/report.json', 'ctest-integration-final.log', 'pytest.log',
             'independent-final-numeric/results.json', 'agent-model-verification.json']
    for scope in ('vehicle', 'clean', 'cold', 'native24'):
        for prefix in ('final-', 'final-r5-'):
            files.extend(prefix + scope + '/' + name for name in ('identity.json', 'report.json', 'results.jsonl'))
    hashes = {str(R / name): sha(R / name) for name in files}
    for name in ('cpp/tools/verify_ns_vehicle_engine.py', 'cpp/tools/verify_ns_vehicle_v4.py',
                 'cpp/tools/audit_ns_vehicle_v4.py', 'cpp/tests/test_vehicle_ns.cpp',
                 'cpp/CMakeLists.txt', 'cpp/include/itu/audio_engine.hpp',
                 'cpp/tools/itu_audio_cli.cpp', 'cpp/src/audio_engine.cpp',
                 'cpp/src/native_pipeline.cpp', 'cpp/include/itu/vehicle_ns.hpp',
                 'cpp/src/vehicle_ns.cpp', 'cpp/src/vehicle_ns_controller.hpp',
                 'cpp/src/vehicle_rumble_filter.hpp'):
        hashes[str(P / name)] = sha(P / name)
    bindings()
    report = dict(state='completed', evidence_audit_passed=True, r4_r5_equivalent_rows=total,
                  hash_verified_files=len(digest_cache), phases=phases,
                  actual_engine_bitexact_pairs=9, legacy_baseline_bitexact_pairs=6,
                  paced_streams=10, paced_deadline_misses=0,
                  paced_max_process_ms=max(r['paced']['max_process_ms'] for r in streams['rows']),
                  cold_wav_timestamp_only_differences=cold_header_checks,
                  quality_all_passed=False, standard_compliance=False,
                  processor_binary_sha256=binary_sha256, evidence_sha256=hashes,
                  scope='frozen NS isolation quality and local aarch64 streaming; not full-chain ITU certification')
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('state', 'r4_r5_equivalent_rows', 'hash_verified_files', 'actual_engine_bitexact_pairs', 'paced_max_process_ms')}))


if __name__ == '__main__':
    main()
