#!/usr/bin/env python3
"""Causal feature/ConvGRU/gain-loss ablations; old trainer and format stay intact."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path
import struct
import time
import numpy as np
from scipy.special import exp1
import torch
from torch import nn
from torch.nn import functional as F
from train_hybrid_ns import LIMIT, ORDER, Student, export as legacy_export, load_rows, save, sha


class ExtendedStudent(nn.Module):
    def __init__(self, hidden_size=32, convolution=False):
        super().__init__()
        self.contract = 3 if convolution else 2
        if convolution:
            self.conv = nn.Conv1d(4, 4, 3, padding=1)
        self.gru = nn.GRU(256, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 128)

    def forward(self, x, hidden=None):
        if self.contract == 3:
            batch, frames, _ = x.shape
            bands = x.reshape(batch*frames, 4, 64)
            x = (bands+self.conv(bands).tanh()).reshape(batch, frames, 256)
        y, h = self.gru(x, hidden)
        y = self.head(y)
        return y[..., :64].sigmoid(), y[..., 64:].tanh()*LIMIT, h


def load_student(state):
    width = state['gru.weight_hh_l0'].shape[1]
    inputs = state['gru.weight_ih_l0'].shape[1]
    model = Student(width) if inputs == 128 else ExtendedStudent(width, 'conv.weight' in state)
    model.load_state_dict(state)
    return model


def export(model, rate, path):
    if isinstance(model, Student):
        return legacy_export(model, rate, path)
    order = (('conv.weight', 'conv.bias') if model.contract == 3 else ())+ORDER
    weights = np.concatenate([model.state_dict()[key].detach().cpu().numpy().ravel() for key in order]).astype('<f4')
    width = model.gru.hidden_size
    count = 3*width*(256+width)+6*width+width*128+128+(52 if model.contract == 3 else 0)
    if width not in (32, 64) or len(weights) != count or not np.isfinite(weights).all():
        raise ValueError('invalid extended deployment weights')
    payload = b'ITUNSGR2'+struct.pack('<6I', 2, rate, 64, width, model.contract, count)+weights.tobytes()
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte)*1099511628211) & ((1 << 64)-1)
    Path(path).write_bytes(payload+struct.pack('<Q', checksum))


class Exp1(torch.autograd.Function):
    """Exact scalar E1 with analytic derivative, training CPU only."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.from_numpy(exp1(x.detach().numpy()).astype(np.float32))

    @staticmethod
    def backward(ctx, gradient):
        x, = ctx.saved_tensors
        return gradient*(-torch.exp(-x)/x)


def band_gain(pp, dd, data, rate):
    # Same static equations and .5/.5 mixes as C++; band projection and no
    # temporal synthesis make this a surrogate, not an exact waveform loss.
    floor = 10**(-18/20)
    confidence = (2*pp-1).square()
    mix = .5*confidence
    p = (1-mix)*data['p_stat']+mix*pp
    xi = (data['xi']*(.5*confidence*dd).exp()).clamp(1e-3, 1e8)
    w = xi/(1+xi)
    v = (w*data['gamma']).clamp_min(1e-12)
    h1 = (w*(.5*Exp1.apply(v)).exp()).clamp(floor, 1)
    gain = (p*h1.log()+(1-p)*np.log(floor)).exp()
    speech = ((pp-.85)/.15).clamp(0, 1)*mix*(xi/.1).clamp(0, 1)
    speech_floor = ((1-speech)*np.log(floor)+speech*np.log(10**(-3/20))).exp()
    gain = torch.maximum(gain, speech_floor)
    base_w = data['xi']/(1+data['xi'])
    base_v = (base_w*data['gamma']).clamp_min(1e-12)
    base_h1 = (base_w*(.5*Exp1.apply(base_v)).exp()).clamp(floor, 1)
    base_gain = (data['p_stat']*base_h1.log()+(1-data['p_stat'])*np.log(floor)).exp()
    quiet = 16*(rate//100)/(12*32768**2)
    trusted = ((data['xi']-3.16227766)/6.83772234).clamp(0, 1)*((data['p_stat']-.9)/.1).clamp(0, 1)*(1-data['noise']/quiet).clamp(0, 1)
    return torch.maximum(gain, (1-trusted)*gain+trusted*base_gain).clamp(floor, 1)


def loss(model, data, rate, gain_weight):
    pp, dd, _ = model(data['x'])
    supervised = F.binary_cross_entropy(pp, data['p_target'])+.18*F.smooth_l1_loss(dd, data['d_target'])
    if not gain_weight:
        return supervised
    gain = band_gain(pp, dd, data, rate)
    cp, noise = data['cp'], data['np']
    total = cp+noise
    valid = total > 1e-9
    # Penalize target attenuation four times as strongly as residual noise.
    cost = (4*(1-gain).square()*cp+gain.square()*noise)/total.clamp_min(1e-9)
    output = (cost*valid).sum()/valid.sum().clamp_min(1)
    return supervised+gain_weight*output


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = args.root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if any(r['split'] not in ('development', 'validation') for r in manifest['rows']):
        raise ValueError('holdout prohibited during training')
    output = root/args.name
    output.mkdir(parents=True, exist_ok=False)
    save(output/'training-manifest.json', manifest)
    summary = {}
    for rate in args.rates:
        train_data, val_data = (load_rows(manifest, rate, s, 'event_v2') for s in ('development', 'validation'))
        model = ExtendedStudent(args.hidden_size, args.architecture == 'conv')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        chunks = [(i, k) for i, d in enumerate(train_data) for k in range(0, len(d['x'])-args.sequence+1, args.sequence)]
        best = float('inf');history = [];started = time.monotonic();best_state = None
        for epoch in range(args.epochs):
            model.train();order = rng.permutation(len(chunks));total = 0
            for at in range(0, len(order), args.batch):
                batch = [chunks[i] for i in order[at:at+args.batch]]
                keys = ('x', 'p_target', 'd_target', 'p_stat', 'xi', 'gamma', 'cp', 'np', 'noise')
                data = {key:torch.stack([train_data[i][key][k:k+args.sequence] for i,k in batch]) for key in keys}
                value = loss(model, data, rate, args.gain_loss_weight)
                if not torch.isfinite(value):
                    raise ValueError('nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);value.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3);optimizer.step()
                total += float(value.detach())*len(batch)
            model.eval()
            with torch.inference_mode():
                validation = np.mean([float(loss(model, {k:v[None] for k,v in d.items()}, rate, args.gain_loss_weight)) for d in val_data])
            history.append(dict(epoch=epoch+1, training_loss=total/len(chunks), validation_loss=float(validation)))
            if validation < best:
                best = validation;best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            print(args.name, rate, history[-1], flush=True)
        model.load_state_dict(best_state);model.eval()
        torch.save(model.state_dict(), output/f'student-{rate}.pt')
        deployment = output/f'student-{rate}.bin';export(model, rate, deployment)
        save(output/f'training-{rate}.json', dict(history=history, best_validation_loss=best,development_cases=len(train_data),validation_cases=len(val_data),
             seed=args.seed,epochs=args.epochs,hidden_size=args.hidden_size,contract=model.contract,parameters=sum(p.numel() for p in model.parameters()),
             learning_rate=args.lr,batch=args.batch,sequence=args.sequence,gain_loss_weight=args.gain_loss_weight,
             gain_loss='band static controller surrogate: 4*speech_distortion+residual_noise',teacher_weight=0,holdout_loaded=False,
             elapsed_seconds=time.monotonic()-started,model_sha256=sha(deployment)))
        summary[str(rate)] = dict(model=str(deployment),sha256=sha(deployment),bytes=deployment.stat().st_size,validation_loss=best)
    save(output/'models.json',dict(models=summary,architecture=args.architecture,teacher_in_runtime=False,default_enabled=False,
         manifest_sha256=sha(output/'training-manifest.json'),training_script_sha256=sha(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--architecture',choices=('enriched','conv'),required=True)
    p.add_argument('--gain-loss-weight',type=float,default=0)
    p.add_argument('--hidden-size',type=int,choices=(32,64),default=32)
    p.add_argument('--rates',nargs='+',type=int,choices=(16000,24000,48000),default=[16000,24000,48000])
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--sequence',type=int,default=100)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=.002)
    args = p.parse_args()
    if min(args.epochs,args.threads,args.sequence,args.batch)<1 or not 0<=args.gain_loss_weight<=1:
        p.error('invalid training limits')
    train(args)
