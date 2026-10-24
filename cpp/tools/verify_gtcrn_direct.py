#!/usr/bin/env python3
"""Verify the real GTCRN weights, causal stream and native C++ frontend."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pcm16(x):
    """Match both native C++ probes: lround, ties away from zero."""
    x = np.clip(np.asarray(x, dtype=np.float32)*32768, -32768, 32767)
    return (np.copysign(np.floor(np.abs(x.astype(np.float64))+.5), x)/32768).astype(np.float32)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def spectral_stream(runtime, spectra, reset_each=False):
    cache_shapes = [(2, 1, 16, 16, 33), (2, 3, 1, 1, 16), (2, 1, 33, 16)]
    states = [np.zeros(s, dtype=np.float32) for s in cache_shapes]
    outputs = []
    for t in range(spectra.shape[2]):
        if reset_each:
            states = [np.zeros(s, dtype=np.float32) for s in cache_shapes]
        values = dict(zip(['mix', 'conv_cache', 'tra_cache', 'inter_cache'],
                          [spectra[:, :, t:t+1].copy(), *states]))
        out, *states = runtime.run(None, values)
        outputs.append(out)
    return np.concatenate(outputs, axis=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.root.resolve(); out = args.output.resolve() if args.output else root/'verification'; out.mkdir(exist_ok=False)
    model_path = args.model.resolve() if args.model else root/'upstream/stream/onnx_models/gtcrn_simple.onnx'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('itu_gtcrn_official_reference', root/'upstream/gtcrn.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.GTCRN().eval()
    model.load_state_dict(torch.load(root/'upstream/checkpoints/model_trained_on_dns3.tar', map_location='cpu', weights_only=False)['model'])
    runtime = session(model_path)
    x, rate = sf.read(args.source, dtype='float32')
    assert rate == 16000 and x.ndim == 1
    x = pcm16(x[:8*rate]); sf.write(out/'input.wav', x, rate, subtype='FLOAT')
    window = torch.hann_window(512).sqrt()
    z = torch.stft(torch.from_numpy(x), 512, 256, window=window,
                   center=True, pad_mode='constant', return_complex=True)
    z = torch.view_as_real(z)[None].numpy()
    with torch.no_grad():
        expected = model(torch.from_numpy(z)).numpy()
    actual = spectral_stream(runtime, z)
    spec_error = float(np.max(np.abs(expected-actual)))
    enhanced = torch.istft(torch.view_as_complex(torch.from_numpy(actual[0]).contiguous()),
                           512, 256, window=window, length=len(x)).numpy()
    expected_wave = torch.istft(torch.view_as_complex(torch.from_numpy(expected[0]).contiguous()),
                                512, 256, window=window, length=len(x)).numpy()
    command = [str(args.probe.resolve()), '--mic', str(out/'input.wav'), '--model', str(model_path),
               '--out', str(out/'cpp'), '--dump-f32', str(out/'cpp.f32')]
    proc = subprocess.run(command, capture_output=True, text=True)
    save(out/'invocation.json', dict(argv=command, returncode=proc.returncode, stderr=proc.stderr))
    proc.check_returncode()
    raw = np.fromfile(out/'cpp.f32', dtype='<f4')
    cpp_wave = raw[480:480+len(x)]
    waveform_error = float(np.max(np.abs(cpp_wave-enhanced)))
    official_error = float(np.max(np.abs(cpp_wave-expected_wave)))
    # Spectral network causality and cache negative control use the same spectra.
    prefix = spectral_stream(runtime, z[:, :, :101])
    prefix_equal = np.array_equal(prefix, actual[:, :, :101])
    reset = spectral_stream(runtime, z, reset_each=True)
    reset_difference = float(np.max(np.abs(reset-actual)))
    # End-to-end causality: change only later input and compare all callbacks
    # returned before that later input was supplied. No fitted sample alignment.
    changed = x.copy(); boundary = 3*rate
    changed[boundary:] = np.random.default_rng(20261018).normal(0, .05, len(x)-boundary)
    sf.write(out/'changed.wav', changed, rate, subtype='FLOAT')
    cmd2 = [str(args.probe.resolve()), '--mic', str(out/'changed.wav'), '--model', str(model_path),
            '--out', str(out/'changed'), '--dump-f32', str(out/'changed.f32')]
    subprocess.run(cmd2, capture_output=True, check=True)
    changed_raw = np.fromfile(out/'changed.f32', dtype='<f4')
    audio_prefix_equal = np.array_equal(raw[:boundary], changed_raw[:boundary])
    selftest = subprocess.run([str(args.probe.resolve()), '--self-test'], capture_output=True, text=True, check=True)
    engine = json.loads((out/'cpp.json').read_text())
    checks = dict(spectral_checkpoint_parity=spec_error <= 1e-3,
                  cpp_onnx_waveform_parity=waveform_error <= 1/32768,
                  cpp_official_waveform_parity=official_error <= 1/32768,
                  spectral_prefix_causal=prefix_equal, waveform_prefix_causal=audio_prefix_equal,
                  reset_cache_negative_control=reset_difference > 1e-4,
                  exact_length=len(raw) == len(x)+480,
                  runtime_model_executed=engine['inferences'] > 0 and not engine['identity_model'],
                  identity_frontend=json.loads(selftest.stdout)['passed'])
    save(out/'report.json', dict(passed=all(checks.values()), checks=checks,
         spectral_max_abs_error=spec_error, cpp_onnx_max_abs_error=waveform_error,
         cpp_official_max_abs_error=official_error, reset_cache_max_abs_difference=reset_difference,
         spectral_tolerance=1e-3, waveform_tolerance=1/32768,
         boundary='zero padding for causal startup and finite tail; no reflection or RMS normalization',
         delay_samples=480, callback_samples=160, hop_samples=256,
         checkpoint_sha256=hashlib.sha256((root/'upstream/checkpoints/model_trained_on_dns3.tar').read_bytes()).hexdigest(),
         onnx_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
         parameters=sum(p.numel() for p in model.parameters()),
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         providers=runtime.get_providers(), identity_frontend=json.loads(selftest.stdout), engine=engine))
    print(json.dumps(checks), flush=True)
    if not all(checks.values()):
        raise RuntimeError('GTCRN verification failed; preserve evidence before evaluation')


if __name__ == '__main__':
    main()
