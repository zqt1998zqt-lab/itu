#!/usr/bin/env python3
"""Train/export a native-rate, causal GRU corrector. GTCRN is offline only.

Preparation reads only the currently accepted mono NS/no-playback cases.
Development trains; validation selects checkpoints; holdout is never loaded.
The C++ extractor is the feature specification, not a Python approximation.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
LIMIT = 1.3815510558
ORDER = ('gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0',
         'gru.bias_hh_l0', 'head.weight', 'head.bias')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

class Student(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        if hidden_size not in (32, 64):
            raise ValueError('supported GRU widths are 32 and 64')
        self.gru = nn.GRU(128, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 128)

    def forward(self, x, hidden=None):
        y, h = self.gru(x, hidden)
        y = self.head(y)
        return y[..., :64].sigmoid(), y[..., 64:].tanh()*LIMIT, h

def export(model, rate, path):
    state = model.state_dict()
    weights = np.concatenate([state[k].detach().cpu().numpy().ravel() for k in ORDER]).astype('<f4')
    hidden = model.gru.hidden_size
    parameters = 3*hidden*(128+hidden)+6*hidden+hidden*128+128
    if hidden not in (32, 64) or len(weights) != parameters or not np.isfinite(weights).all():
        raise ValueError('invalid deployment weights')
    payload = b'ITUNSGR1'+struct.pack('<6I', 1, rate, 64, hidden, 1, len(weights))+weights.tobytes()
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte)*1099511628211) & ((1 << 64)-1)
    Path(path).write_bytes(payload+struct.pack('<Q', checksum))

def prepare_one(task):
    probe, root, run, scene = task
    origin = PROJECT/'build'/run/'cases'/scene['id']
    rate = scene['sample_rate']
    for name in ('before.wav', 'clean.wav'):
        if sha(origin/name) != scene['files'][name]['sha256']:
            raise ValueError('source identity changed: '+str(origin/name))
    info, ci = sf.info(origin/'before.wav'), sf.info(origin/'clean.wav')
    if info.channels != 1 or ci.channels != 1 or info.samplerate != rate or info.frames != ci.frames or ci.samplerate != rate:
        raise ValueError('requires native mono input and aligned target')
    token = hashlib.sha256((run+'/'+scene['id']).encode()).hexdigest()[:20]
    prefix = root/'features'/token
    raw = prefix.with_suffix('.f32')
    command = [str(probe), '--mic', str(origin/'before.wav'), '--clean', str(origin/'clean.wav'),
               '--features', str(raw), '--out', str(prefix), '--floor-db', '-18']
    result = subprocess.run(command, capture_output=True, text=True)
    result.check_returncode()
    rows = np.fromfile(raw, dtype='<f4').reshape(-1, 448)
    if not len(rows) or not np.isfinite(rows).all():
        raise ValueError('nonfinite/empty extracted features')
    packed = prefix.with_suffix('.npz')
    np.savez_compressed(packed, x=rows[:, :128], p_stat=rows[:, 128:192],
                        xi=rows[:, 192:256], gamma=rows[:, 256:320],
                        cp=rows[:, 320:384], np=rows[:, 384:448])
    for ext in ('.f32', '.after.wav', '.csv', '.json'):
        prefix.with_suffix(ext).unlink()
    return dict(run=run, id=scene['id'], split=scene['split'], rate=rate,
                speaker=scene['near_speaker'], noise=scene['noise'], snr_db=scene['snr_db'],
                origin=str(origin), features=str(packed), features_sha256=sha(packed),
                input_sha256=sha(origin/'before.wav'), clean_sha256=sha(origin/'clean.wav'),
                frames=len(rows))

def prepare(args):
    import repair_ns_vehicle_v4 as ev
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'features').mkdir()
    selected = [(run, row['scene']) for run, row in ev.sources('vehicle')
                if row['scene']['split'] in ('development', 'validation')]
    if not selected:
        raise ValueError('no accepted training/validation scenes')
    if any(s['family'] != 'ns' or s.get('far_schedule') or s['mics'] != 1 for _, s in selected):
        raise ValueError('training selection contains playback or multichannel data')
    tasks = [(args.probe.resolve(), root, run, s) for run, s in selected]
    rows = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for future in as_completed([pool.submit(prepare_one, task) for task in tasks]):
            rows.append(future.result())
            if len(rows) % 25 == 0:
                print('prepare', len(rows), '/', len(tasks), flush=True)
    rows.sort(key=lambda r: (r['rate'], r['split'], r['run'], r['id']))
    save(root/'manifest.json', dict(schema=1, rows=rows, probe_sha256=sha(args.probe),
         extractor_sha256=sha(PROJECT/'cpp/tools/hybrid_ns_probe.cpp'),
         training_script_sha256=sha(__file__), source_selection='accepted vehicle NS, development/validation only',
         feature_columns=448, external_vad=False, resampled=False,
         counts=dict(Counter(str(r['rate'])+':'+r['split'] for r in rows)),
         holdout_loaded=False))
    print('prepared', len(rows), flush=True)

def band_matrix(rate):
    hop = rate//100
    nfft = 1 << (2*hop-1).bit_length()
    centers = np.r_[np.linspace(0, 1000, 24), 1000+3000*np.arange(1, 17)/16,
                    4000+(rate/2-4000)*np.arange(1, 25)/24]
    hz = np.arange(nfft//2+1)*rate/nfft
    left = np.clip(np.searchsorted(centers, hz, side='left')-1, 0, 62)
    w = np.clip((hz-centers[left])/(centers[left+1]-centers[left]), 0, 1)
    m = np.zeros((len(hz), 64), dtype=np.float32)
    m[np.arange(len(hz)), left] = 1-w
    m[np.arange(len(hz)), left+1] += w
    m /= np.maximum(m.sum(axis=0), 1e-12)
    return m, nfft

def teacher_targets(rows, root):
    """Aligned offline GTCRN waveform -> 20/10 ms gain target, NEVER SPP."""
    spec = importlib.util.spec_from_file_location('itu_offline_gtcrn', root/'gtcrn.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    teacher = mod.GTCRN().eval()
    checkpoint = torch.load(root/'checkpoints/model_trained_on_dns3.tar', map_location='cpu', weights_only=True)
    teacher.load_state_dict(checkpoint['model'])
    window = torch.hann_window(512).sqrt()
    m, nfft = band_matrix(16000)
    sine = np.sin(np.pi*(np.arange(320)+.5)/320).astype(np.float32)
    record = dict(revision='502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73',
                  source_sha256=sha(root/'gtcrn.py'), weights_sha256=sha(root/'checkpoints/model_trained_on_dns3.tar'),
                  native_rate=16000, offline_centered_stft=True, waveform_alignment_samples=0,
                  runtime_teacher=False, target='spectral amplitude gain, not speech probability',
                  unavailable_rates=[24000, 48000])
    # The upstream network is time-causal on spectral frames. Centered offline
    # STFT/ISTFT uses boundary context; exclude 512 samples in this prefix check.
    torch.manual_seed(123)
    first = torch.randn(4096)*.01
    def enhance(x):
        with torch.inference_mode():
            z = torch.stft(x, 512, 256, window=window, return_complex=True)
            y = teacher(torch.view_as_real(z)[None])[0]
            return torch.istft(torch.view_as_complex(y.contiguous()), 512, 256,
                               window=window, length=len(x))
    a = enhance(torch.cat((first, torch.randn(2048)*.01)))
    b = enhance(torch.cat((first, torch.randn(2048)*.01)))
    error = float((a[:3584]-b[:3584]).abs().max())
    record['prefix_maxabs_excluding_boundary'] = error
    if error > 2e-6:
        raise ValueError('offline teacher spectral prefix check failed')
    for i, row in enumerate(rows):
        if row['rate'] != 16000:
            continue
        path = Path(row['features'])
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        cache_id = hashlib.sha256((record['source_sha256']+record['weights_sha256']+'gain-v1').encode()).digest()
        if 'teacher_gain' in data:
            if 'teacher_id' not in data or data['teacher_id'].tobytes()!=cache_id:
                raise ValueError('cached teacher labels lack matching identity; use a fresh feature copy')
            continue
        x, rate = sf.read(row.get('input_path', Path(row['origin'])/'before.wav'), dtype='float32')
        x = (np.clip(np.rint(x*32768), -32768, 32767)/32768).astype(np.float32)
        y = enhance(torch.from_numpy(x)).numpy()
        def power(v):
            v = np.pad(v, (160, 160))
            frames = np.lib.stride_tricks.sliding_window_view(v, 320)[::160][:row['frames']]
            return (abs(np.fft.rfft(frames*sine, n=nfft))**2) @ m
        data['teacher_gain'] = np.clip(np.sqrt(power(y)/np.maximum(power(x), 1e-16)), .01, 1).astype(np.float32)
        data['teacher_id'] = np.frombuffer(cache_id,dtype=np.uint8).copy()
        np.savez_compressed(path, **data)
        row['features_sha256'] = sha(path)
        if i % 25 == 0:
            print('offline teacher', i, '/', len(rows), flush=True)
    return record

def load_rows(manifest, rate, split, label_schema):
    loaded = []
    for row in manifest['rows']:
        if row['rate'] != rate or row['split'] != split:
            continue
        path = Path(row['features'])
        if sha(path) != row['features_sha256']:
            raise ValueError('training data identity drift')
        with np.load(path) as z:
            data = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        cp, noise = data['cp'], data['np']
        # Train a local speech-dominance EVENT with proper Bernoulli BCE.
        # The sigmoid prediction is its probability, not an ideal ratio mask.
        ratio = cp/(noise+1e-12)
        data['p_target'] = (ratio>10**(-5/10)).float() if label_schema=='event_v2' else ratio/(ratio+.1)
        data['p_target'] *= (cp > 1e-9).float()
        data['d_target'] = (torch.log(ratio.clamp_min(1e-4))-torch.log(data['xi'].clamp_min(1e-3))).clamp(-LIMIT, LIMIT)
        loaded.append(data)
    if not loaded:
        raise ValueError('missing rate/split training data')
    return loaded

def loss(model, x, p, d, teacher, teacher_weight):
    pp, dd, _ = model(x)
    value = F.binary_cross_entropy(pp, p)+.18*F.smooth_l1_loss(dd, d)
    if teacher is not None and teacher_weight:
        # Simple amplitude-gain surrogate used ONLY as auxiliary distillation.
        # The deployed controller always uses the full OM-LSA equation.
        h1 = (x[..., :64]*6).exp().sub(1).clamp_min(1e-3)
        h1 = ((h1*dd.exp())/(1+h1*dd.exp())).sqrt().clamp(.12589254, 1)
        gain = (pp*h1.log()+(1-pp)*np.log(.12589254)).exp()
        value = value+teacher_weight*F.smooth_l1_loss(gain, teacher)
    return value

def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = args.root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if any(r['split'] not in ('development','validation') for r in manifest['rows']):
        raise ValueError('training/teacher manifest must exclude holdout')
    teacher_record = None
    if args.teacher_root:
        teacher_record = teacher_targets(manifest['rows'], args.teacher_root.resolve())
        save(root/'manifest.json', manifest)
        save(root/'teacher-identity.json', teacher_record)
    # Teacher construction/checks consume RNG; keep the no-KD initialization
    # and minibatch order reproducible when only the teacher loss changes.
    torch.manual_seed(args.seed)
    output = root/args.name
    output.mkdir(parents=True, exist_ok=False)
    save(output/'training-manifest.json', manifest)
    summary = {}
    for rate in args.rates:
        train_data, val_data = (load_rows(manifest, rate, s, args.label_schema) for s in ('development', 'validation'))
        model = Student(args.hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        chunks = [(i, k) for i, data in enumerate(train_data)
                  for k in range(0, len(data['x'])-args.sequence+1, args.sequence)]
        best = float('inf')
        history = []
        started = time.monotonic()
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(len(chunks))
            total = 0
            for at in range(0, len(order), args.batch):
                batch = [chunks[i] for i in order[at:at+args.batch]]
                def stack(key):
                    return torch.stack([train_data[i][key][k:k+args.sequence] for i, k in batch])
                teacher = stack('teacher_gain') if rate == 16000 and all('teacher_gain' in train_data[i] for i, _ in batch) else None
                value = loss(model, stack('x'), stack('p_target'), stack('d_target'), teacher,
                             args.teacher_weight if rate == 16000 else 0)
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                total += float(value.detach())*len(batch)
            model.eval()
            with torch.inference_mode():
                validation = np.mean([float(loss(model, data['x'][None], data['p_target'][None], data['d_target'][None],
                    data.get('teacher_gain')[None] if 'teacher_gain' in data else None,
                    args.teacher_weight if rate == 16000 else 0)) for data in val_data])
            history.append(dict(epoch=epoch+1, training_loss=total/len(chunks), validation_loss=float(validation)))
            if validation < best:
                best = validation
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(args.name, rate, history[-1], flush=True)
        model.load_state_dict(best_state)
        model.eval()
        checkpoint = output/f'student-{rate}.pt'
        torch.save(model.state_dict(), checkpoint)
        deployment = output/f'student-{rate}.bin'
        export(model, rate, deployment)
        save(output/f'training-{rate}.json', dict(history=history, best_validation_loss=best,
             development_cases=len(train_data), validation_cases=len(val_data),
             development_frames=sum(len(d['x']) for d in train_data), epochs=args.epochs,
             seed=args.seed, hidden_size=args.hidden_size, parameters=sum(p.numel() for p in model.parameters()),
             learning_rate=args.lr, batch=args.batch, sequence=args.sequence,
             spp_label_schema=args.label_schema,local_event_snr_threshold_db=-5,local_clean_power_floor=1e-9,
             truncated_bptt_state_reset_frames=args.sequence, holdout_loaded=False,
             teacher_weight=args.teacher_weight if rate == 16000 and any('teacher_gain' in d for d in train_data) else 0,
             elapsed_seconds=time.monotonic()-started, model_sha256=sha(deployment)))
        summary[str(rate)] = dict(model=str(deployment), sha256=sha(deployment), bytes=deployment.stat().st_size,
                                 validation_loss=best)
    save(output/'models.json', dict(models=summary, teacher_in_runtime=False,
         manifest_sha256=sha(output/'training-manifest.json'), training_script_sha256=sha(__file__),
         teacher_identity=teacher_record, runtime='portable C++ FP32', default_enabled=False))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--probe', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--name', default='supervised-v1')
    p.add_argument('--rates', nargs='+', type=int, default=[16000, 24000, 48000])
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--sequence', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--hidden-size', type=int, choices=(32,64), default=32)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--teacher-root', type=Path)
    p.add_argument('--teacher-weight', type=float, default=0)
    p.add_argument('--label-schema', choices=('event_v2','ratio_v1'), default='event_v2')
    args = p.parse_args()
    if args.prepare:
        if not args.probe or not 1 <= args.jobs <= 8:
            p.error('preparation requires --probe and jobs 1..8')
        prepare(args)
    else:
        if not set(args.rates) <= {16000, 24000, 48000} or min(args.epochs, args.sequence, args.batch, args.threads) < 1:
            p.error('invalid training dimensions/rates')
        train(args)

if __name__ == '__main__':
    main()
