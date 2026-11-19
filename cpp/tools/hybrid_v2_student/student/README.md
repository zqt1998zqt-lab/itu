# ITUNSR02 hybrid residual student v2

Single-head causal C64 residual model for `dpdf-frame-v1-db` and
`dpdf-frame-v2-basegain-db`. This directory does not train and does not evaluate
MOS or runtime quality.

## Model interface

```python
from student.model import ResidualStudent

model = ResidualStudent(input_dim=320, hidden=64)  # or input_dim=256
delta_db, hidden_out = model(x, hidden)
state0 = model.initial_state(batch_size, device=None)
delta_1, state1 = model.step(frame, state0)
```

- `x`: `[B,T,input_dim]`, raw feature values with no implicit scaling.
- `hidden`: `[B,64]`.
- `delta_db`: `[B,T,64] = 12*tanh(head(...))`.
- `hidden_out`: `[B,64]`; silence never resets it.
- `step`: `[B,1,input_dim] -> [B,1,64] + [B,64]`.

| input | feature contract | version | params | MAC/s @100fps |
|---:|---|---:|---:|---:|
| 256 | `dpdf-frame-v1-db` | 1 | 45,568 | 4.5056M |
| 320 | `dpdf-frame-v2-basegain-db` | 2 | 49,664 | 4.9152M |

The 256-D contract is the loss-only control. The 320-D contract appends one
64-band group to the v1 groups:

```text
20*log10(max(project(basegain_257 -> 64), 1e-8))
```

`basegain` is projected as a linear amplitude gain first and logged afterwards.
It is not logged per FFT bin before projection. Power/noise use
`10*log10(max(projected_power,1e-8))`. The normalized triangular FFT512 band map
is the same as v1.

Projected `basegain` must be finite in `[0,1]`. Power/noise must share dtype and
device. The explicit previous-log-power state must share dtype/device with power.
`eps` must be finite and strictly positive. These are preparation-time validation
checks; the GRU callback performs no value-dependent device synchronization.

## Checkpoint and export

```python
from student.export import export_checkpoint
manifest = export_checkpoint(checkpoint_path, output_dir, trained=True)
```

Required trained checkpoint shape:

```python
{
  "state_dict": model.state_dict(),
  "metadata": {
    "input_dim": 256 or 320,
    "steps": >0,
    "feature_contract": "dpdf-frame-v1-db" or "dpdf-frame-v2-basegain-db",
    "training_manifest_sha256": "64 hex",
    "run_id": "non-empty",
    "trained": True,
    "fixture": False,
    "gain_mode": "residual_db",
    "floor_db": number,
    "base_floor_db": number
  }
}
```

`model_sha256` is optional; if supplied it must equal the deployed `model.py`
hash. Export rejects key/shape mismatch, non-finite weights, and `|weight| > 100`.
`steps` cannot be negative; zero is allowed only for an untrained fixture.
`floor_db` and `base_floor_db` must be finite in `[-40,0]`. Unknown input widths
are rejected. `model.json` records runtime head units, residual limit 12 dB and
`explicit_no_silence_reset` state policy.

Formal trained checkpoints require `fixture=false`. A native test clone may set
the binary header `trained=1` while retaining sidecar `fixture=true`; such a file
is test-only and always has `quality_use=forbidden`.

## ITUNSR02 binary

`weights.bin` is:

```text
ASCII8 ITUNSR02
LE u32[14]
LE f32 weights
LE u64 FNV1A64 over all preceding bytes
```

The 14 header fields are:

```text
version=2, sample_rate=16000, bands=64, input_dim=256/320,
projection=64, hidden=64, head=64, parameter_count=45568/49664,
trained=0/1, feature_version=1/2, fft=512, hop=160, window=320, reserved=0
```

Weight order:

```text
projection.weight, projection.bias,
gru.weight_ih_l0, gru.weight_hh_l0,
gru.bias_ih_l0, gru.bias_hh_l0,
head.weight, head.bias
```

GRU rows remain PyTorch reset-after `[r;z;n]`. The old `ITUNSR01` loader must not
accept this file.

`model.json` includes feature contract/version, input dimension, gain mode,
floor/base floor, model hash, dataset SHA, run id, trained/fixture flags, steps,
parameter/MAC counters, and SHA-256 hashes for weights/ONNX/checkpoint.

`step.onnx` has explicit `hidden_in[B,64]`, dynamic batch, one spectral frame, and
outputs `delta_db[B,1,64]` plus `hidden_out[B,64]`.

## Structure verification

```bash
/home/zhangqingtian/.venv/bin/python student/verify.py \
  --output-dir verification/student/structure
```

The command verifies both input widths and writes:

```text
verification/student/structure/model-256/
verification/student/structure/model-320/
verification/student/structure/verification.json
```

Each variant checks feature chunk/sequence parity, v2 base-gain log semantics,
negative/non-finite rejection, sequence-vs-step, future-prefix causality,
silent-state finiteness, ITUNSR02 header/order/checksum, and ONNX Runtime parity.

ORT and Torch each feed their **own** `hidden_out` into their next step for the
continuous 32-frame runtime metric, with tolerance `1e-3`. A separate same-state
operator metric uses `1e-5`. If a same-state sample exceeds `1e-5`, that strict
flag remains false; it is not hidden by changing the fixed seeds `8802/8810` or
by sharing hidden state.

Random structure fixtures are exported with `trained=0, fixture=true`; a
deployment loader must reject them.

## Future native parity

```bash
/home/zhangqingtian/.venv/bin/python student/verify.py \
  --native-probe native/hybrid_v2_probe \
  --checkpoint train/<run>/best.pt \
  --output-dir verification/<run>
```

Native GRU mode reads rows of `input_dim` and writes `delta64 | hidden64`.

For 320-D band mode, each row is:

```text
[power257 | noise257 | basegain257] = 771 values
```

and each output row is:

```text
[features320 | power64 | noise64 | basegain64] = 512 values
```

Feature strict max-abs `1e-4` is reported separately from the declared runtime
feature bound `1e-3`; an extreme-dynamic-range strict miss remains visible and
does not get hidden. The 320 band output is checked by metric: projected
power/noise use relative error `1e-4`; projected gain uses max-abs `2e-6`.
Recurrent native runtime tolerance is `1e-3`. Native and PyTorch maintain
independent states. No
runtime-quality or teacher input is accepted by verification.
