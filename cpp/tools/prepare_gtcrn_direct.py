#!/usr/bin/env python3
"""Export pinned DNS3 weights with upstream streaming conversion, not a guessed ONNX identity."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
import onnx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root.resolve()
    out = root/'model'; out.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(20261018)
    sys.path.insert(0, str(root/'upstream/stream'))
    from gtcrn_stream import StreamGTCRN
    from modules.convert import convert_to_stream
    spec = importlib.util.spec_from_file_location('itu_gtcrn_dns3', root/'upstream/gtcrn.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.GTCRN().eval()
    checkpoint = root/'upstream/checkpoints/model_trained_on_dns3.tar'
    expected = 'a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
    original.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    stream = StreamGTCRN().eval(); convert_to_stream(stream, original)
    spectrum = torch.randn(1, 257, 12, 2)*.2
    states = [torch.zeros(2, 1, 16, 16, 33), torch.zeros(2, 3, 1, 1, 16), torch.zeros(2, 1, 33, 16)]
    with torch.no_grad():
        expected_output = original(spectrum)
        outputs = []
        for i in range(spectrum.shape[2]):
            value, *states = stream(spectrum[:, :, i:i+1].clone(), *states)
            outputs.append(value)
        error = float((expected_output-torch.cat(outputs, dim=2)).abs().max())
    assert error < 1e-4, error
    path = out/'gtcrn-dns3-stream.onnx'
    dummy = (torch.randn(1, 257, 1, 2), torch.zeros(2, 1, 16, 16, 33),
             torch.zeros(2, 3, 1, 1, 16), torch.zeros(2, 1, 33, 16))
    torch.onnx.export(stream, dummy, str(path),
        input_names=['mix','conv_cache','tra_cache','inter_cache'],
        output_names=['enh','conv_cache_out','tra_cache_out','inter_cache_out'],
        opset_version=11, dynamo=False)
    onnx.checker.check_model(str(path))
    (out/'identity.json').write_text(json.dumps(dict(checkpoint_sha256=expected,
        onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        stream_conversion_max_abs_error=error, opset=11,
        normalization=False, sample_rate=16000, seed_for_export_probe=20261018,
        source_revision='502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73'),indent=2)+'\n')
    print(path, 'stream-conversion-error', error, flush=True)


if __name__ == '__main__':
    main()
