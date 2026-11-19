# Distilled C64 residual-noise student

Single-head causal student for the DPDF-guided residual-noise experiment. It
does not train, run a teacher, read quality labels, or evaluate MOS.

## Model interface

```python
from student.model import ResidualStudent

model = ResidualStudent(hidden=64)
residual_db, hidden_out = model(features, hidden)
state0 = model.initial_state(batch_size, device=device)
residual_db_1, state1 = model.step(frame, state0)
```

- `features`: `[B,T,256]`, raw finite dB feature values. No external scaling is
  applied; the trainable `projection` layer consumes them directly.
- `hidden`: `[B,64]`.
- `residual_db`: `[B,T,64]`, `12*tanh(head(...))`, bounded to `[-12,12]` dB.
- `hidden_out`: `[B,64]` after the final frame.
- `step` input: `[B,1,256]`; outputs remain time-singleton `[B,1,64]` plus
  `[B,64]` state.
- `initial_state` returns zeros with shape `[B,64]`. Silence never resets either
  feature state or GRU state; reset is caller-explicit only.

Architecture:

```text
256 -> Linear64 -> ReLU -> unidirectional GRU64 -> Linear64 -> 12*tanh
```

Exact accounting: **45,568 parameters**, **182,272 FP32 weight bytes**, **256
FP32 state bytes/item**, **45,056 MAC/frame**, and **4.5056 MMAC/s at 100 fps**.
MACs include the input dense layer, all three GRU gates, and the head. Bias adds,
ReLU, sigmoid/tanh and other pointwise operations are excluded.

## Feature contract

`dpdf-frame-v1-db` uses 16 kHz, FFT512 sine-STFT, hop 160, window 320. The model
sees 64 **projected bands**, not 257 raw FFT bins. Four channel-major groups are
ordered as:

```text
[logpower_db(64), imcra_noise_db(64),
 logpostSNR_db(64), delta_logpower_db(64)]
```

`projected_band_matrix()` and `project_fft_power_to_bands()` reproduce the native
Hz-domain band map. `compose_causal_frame_features_with_state()` accepts projected
power/noise and an explicit `CausalFeatureState(previous_log_power)`. One-frame
and three-frame chunks must equal full-sequence features exactly within `1e-4`.

This contract is not compatible with existing TinyNsCorrector weights or feature
scaling. The aligned teacher waveform is only used to generate training targets:
apply the declared 640-sample teacher alignment/tail policy, re-analyse it through
the student FFT512 sine-STFT, and aggregate to the same 64 projected bands.

## Checkpoint and export

Training code calls:

```python
from student.export import export_checkpoint
manifest = export_checkpoint(checkpoint_path, output_dir, trained=True)
```

Checkpoint schema:

```python
{
  "state_dict": model.state_dict(),
  "metadata": {
    "steps": int,
    "training_manifest_sha256": "64 hex characters",
    "run_id": "non-empty string",
    "trained": True,
    "fixture": False,
    "feature_contract": "dpdf-frame-v1-db"
  }
}
```

Trained export requires `steps > 0`, a non-zero training-manifest SHA and a run
id. Non-finite weights or `abs(weight) > 100` are rejected. A `trained=False`
export is an explicit structural fixture; it may use zero steps and a zero
manifest placeholder.

The three deployment files are:

- `weights.bin`: 64-byte header, FP32 little-endian weights, 8-byte checksum.
- `step.onnx`: dynamic batch, one frame, inputs `features[B,1,256]` and
  `hidden_in[B,64]`; outputs `residual_db[B,1,64]` and `hidden_out[B,64]`.
- `model.json`: hashes and version/training identity.

### `weights.bin`

Header is exactly:

```text
ASCII8  ITUNSR01
LE u32  14 values
```

| field | value |
|---|---:|
| format version | 1 |
| sample rate | 16000 |
| projected bands | 64 |
| feature dim | 256 |
| projection units | 64 |
| hidden size | 64 |
| head units | 64 |
| parameter count | 45568 |
| trained | 1 or 0 |
| feature version | 1 |
| FFT size | 512 |
| hop samples | 160 |
| window samples | 320 |
| reserved | 0 |

Weight order is fixed:

```text
projection.weight, projection.bias,
gru.weight_ih_l0, gru.weight_hh_l0,
gru.bias_ih_l0, gru.bias_hh_l0,
head.weight, head.bias
```

The final eight bytes are little-endian FNV-1A64 over every preceding byte. A
deployment loader must reject `trained=0`; the structural fixture is for parity
only and cannot be used for quality claims.

`model.json` includes `sha256`/`weights_sha256`, `onnx_sha256`,
`checkpoint_sha256`, model and feature contract, `trained`, `fixture`, `steps`,
`headunits`, training-manifest SHA, `runid`, and header fields.

## Verification

Structure-only verification writes a random fixture to the requested directory:

```bash
/home/zhangqingtian/.venv/bin/python student/verify.py \
  --output-dir verification/model-structure
```

It checks parameter/MAC accounting, chunked features, sequence-vs-step,
future-prefix causality, 32-frame nonzero-state ONNX parity, 32-frame silent
state finiteness/no reset, and binary header/order/checksum/hash readback. The
random `weights.bin` has `trained=0`.

Optional native parity:

```bash
/home/zhangqingtian/.venv/bin/python student/verify.py \
  --native-probe native/residual_ns_probe \
  --checkpoint train/<run>/checkpoint.pt \
  --output-dir verification/native-parity \
  --fixture
```

`--fixture` creates a new checkpoint in the output directory with header
`trained=1`, `metadata.fixture=true`, `metadata.steps=1`, and
`quality_use=forbidden`. It never rewrites the source checkpoint or its existing
`weights.bin`.

The current native probe is invoked twice:

```text
residual_ns_probe --band-input band_input.f32 --out band-output.f32
residual_ns_probe --gru-input features.f32 --model weights.bin --out gru-output.f32
```

Input files:

```text
band_input.f32   # 128 rows, [FFT512 power 257 | FFT512 noise 257]
features.f32     # 128*256 LE f32 fixed-seed raw dB GRU input
```

Output files:

```text
band-output.f32  # 128 rows: features 256 | power 64 | noise 64
gru-output.f32   # 128 rows: residual_db 64 | hidden 64
```

The native GRU starts from zero hidden state and consumes the supplied raw dB
`features.f32`; its residual/hidden outputs are compared with PyTorch per step.
The FFT power/noise path separately tests the native band map and four feature
groups against the Python replica. Feature extraction must use
`10*log10(max(P,1e-8))`, and residual output must be `12*tanh(head_logit)`.
The same trained checkpoint is also checked against ONNX Runtime for 32 nonzero
state frames and 32 physical-silence frames, including finite/bounded outputs.
At each step the ONNX `hidden_out` is fed back to both runtimes for the next
step, so the `1e-5` threshold tests operator parity without double-counting two
independent floating-point state trajectories. ORT and native parity share the
fixed raw-dB fixture seed `8802`.

`native-parity.json` therefore contains two distinct concepts:

- Native-vs-PyTorch 128-frame recurrent parity uses each runtime's own state and
  has a `1e-3` runtime-difference tolerance.
- The ORT state block is a **same-state one-step operator metric** with a
  `1e-5` tolerance; it is not continuous independent-state parity.

For the stricter continuous-state diagnostic, run:

```bash
/home/zhangqingtian/.venv/bin/python student/verify.py \
  --checkpoint train/kd-then-clean-floor-21/best.pt \
  --output-dir verification/trained-kd-floor21 \
  --independent-state-parity --independent-state-frames 32
```

This writes `independent-state-parity.json` without modifying the existing
native receipt. PyTorch and ORT each feed their own `hidden_out` into their own
next step. The report contains both the same-state `1e-5` operator result and
the independent-state `1e-3` runtime bound for fixed seeds `8802` and `8810`.
If a same-state sample exceeds `1e-5` but remains below the declared `1e-3`
runtime bound, the report keeps `pass=false` for that strict metric while the
overall runtime-parity status may still pass. This records floating-point
runtime ordering differences without mislabeling them as a model failure.
The parity run also asserts that one negative raw FFT power bin is rejected and,
when `--fixture` is used, that the native loader rejects the original
`trained=0` `model-structure/weights.bin`.
Hard limits: projected-band relative error and feature max-abs error `<=1e-4`;
residual dB max-abs error `<=1e-3`; hidden max-abs error `<=1e-3`. A detailed
`native-parity.json` is written before any assertion failure.
The receipt records the native executable SHA-256 immediately before and after
the run; parity fails if those hashes differ, preventing a mixed-version result.

No teacher, clean label, runtime-quality signal, MOS, or perceptual score is an
input to either verification path.
