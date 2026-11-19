# C64 single-head student training contract

This directory trains the deployable `ResidualStudent(hidden=64)` only. It does
not run the teacher, inspect eval/holdout audio, compute MOS/DNSMOS, choose a
checkpoint by perceptual score, or export a model. Export remains
`student/export.py::export_checkpoint`.

## Data boundary

`features/manifest.json` has a top-level `rows` list. Each row must contain
`id`, `split` (`train` or `dev`), `language`, `npzpath`, and `npz_sha256`
(`hash`/`sha256` are accepted for older manifests). Paths are relative to the
manifest directory. The trainer filters `split` before opening any file, so a
`test`/`eval` row is never loaded. The self-test includes a poisoned `test` row
with a nonexistent path to enforce this boundary.

Each NPZ is 16 kHz, mono, 100 frames/s, with `T` frames:

| key | shape | use |
|---|---|---|
| `x` | `[T,256]` | already-computed raw dB feature groups: log power, IMCRA noise, post-SNR, delta log power |
| `base_gain` | `[T,257]` | native OMLSA base amplitude gain |
| `mix_spec` | `[T,257] complex64` | student/native FFT512 sine-STFT mixture |
| `clean_spec` | `[T,257] complex64` | clean reference, labels only |
| `teacher_spec` | `[T,257] complex64` | aligned teacher STFT, KD labels only |
| `power`/`noise` | `[T,64]` optional | projected-band diagnostics |

`teacher_spec` must already be aligned to the student timeline (declared 640
sample teacher alignment/tail policy). The trainer does not silently realign or
index-match teacher FFT320 bins. It computes the teacher target through the
student 257-bin complex mixture. The self-test does not claim a real teacher
quality result.

## Frequency and gain contract

The 64 centers are exactly the native `NsFeatureBands` centers at 16 kHz:

```text
b=0..23:  1000*b/23
b=24..39: 1000+3000*(b-23)/16
b=40..63: 4000+4000*(b-39)/24
```

For FFT512, each of 257 Hz bins uses its two neighboring centers with clamped
weight. Two matrices are kept distinct:

- projection `[257,64]`: column-normalized, for aggregating target power to bands;
- interpolation `[257,64]`: every row sums to one, for expanding neural delta dB.

The model returns 64 residual dB values. They are interpolated to 257 bins and
the final amplitude mask is applied exactly once:

```text
g_final = clamp(base_gain * 10^(delta_db/20), absolute_floor, 1)
```

`--absolute-floor-db` defaults to `-9` and is never hard-coded into the model.
The initial/base frontend floor is recorded as `base_frontend_floor_db=-9`.
The current bounded head is `12*tanh`, so from a `-9` base the reachable floor
is at most `-21` dB. `-30` is accepted as a separately named experiment for
diagnosis, but it is not claimed reachable by this head; metadata records
`reachable_floor_db` and the residual limit.

## Objective

All masks are amplitude gains. `ls_projection(C, X)` is
`clamp(real(C * conj(X)) / (|X|^2 + eps), floor, 1)` and is used only to create
clean/teacher labels. It is not a runtime feature and must not be used for a
quality report.

The clean supervised objective is a masked sum of:

1. compressed complex error `|compress(Y)-compress(C)|^2`, where
   `compress(z)=|z|^0.3*z/|z|`;
2. compressed magnitude error `(|Y|^0.3-|C|^0.3)^2`;
3. `0.5 * SmoothL1(raw interpolated residual, clipped residual target, beta=1 dB)`.

The third term uses `clip(20log10(clean_gain/base_gain), -12, 12)` so it still
has a gradient when the final mask is floor-clipped. Clean power supplies a
fixed power weight, and bins at least 30 dB below the frame peak get a fixed
`0.5` weak-clean boost. This is one gain head; there is no SPP, xi, or extra
quality head.

KD is the log-amplitude-gain SmoothL1 error between the predicted final 257-bin
gain and the teacher LS mask. Frames/bins where teacher power is below half the
clean power, with clean energy present, use KD weight `0.25`; all others use
`1.0`. KD epochs use `KD + 0.25 * clean_loss`. The second phase uses clean loss
alone. Noise-only clean labels are zero and therefore target the fixed floor;
all losses use `eps=1e-8` and reject non-finite values.

## Schedule, state, and selection

- Both modes use the same seed and initialization. Initialization is torch's
  seeded default for projection/GRU plus a zero-initialized head, so both modes
  start at residual 0 dB.
- `kd_then_clean`: 8 KD epochs followed by 12 clean epochs.
- `supervised`: 20 clean epochs. `kd_epochs + clean_epochs` must be 20 for both
  modes, so the optimization budget is identical.
- Each epoch shuffles fixed streams with a seeded generator. A batch keeps its
  hidden state across all 200-frame chunks; hidden is detached only at the
  TBPTT boundary, never reset every 2 seconds. Padding is masked per frame.
- Torch uses two threads, deterministic settings, CUDA FP32, no autocast, and no
  quantization. AdamW uses the CLI learning rate, weight decay `1e-4`, and
  gradient clipping at norm `5`.
- Dev is evaluated with the clean supervised loss only. `best-kd.pt` is the
  best dev-clean checkpoint in the KD phase (KD mode only); `best.pt` is the
  best dev-clean checkpoint in the clean phase; `last.pt` is always updated.
  No eval/MOS result participates in selection.

## Checkpoints and logs

Every checkpoint is:

```python
{"state_dict": model.state_dict(), "metadata": {...}}
```

Metadata includes `trained=True`, `fixture=False`, `steps`/`optimizer_steps`,
`training_manifest_sha256`, `run_id`, `feature_contract="dpdf-frame-v1-db"`,
`model_sha256`, `teacher_sha256`, `native_probe_sha256`, `architecture`, `seed`,
`mode`, `stage`, `selection_metric="dev_clean_loss"`, `absolute_floor_db`,
`base_frontend_floor_db`, `reachable_floor_db`, `residual_limit_db`, and the
schedule/dimensions. The native probe is locked at SHA256
`4d1b3f139214bfad11a6c80e635635026dd2b0c3ce98fc5c4992c8ea4ce5209a`; training
fails if the binary differs. The input manifest may also carry
`native_probe.sha256`; if present it must match.

`training.jsonl` and `training.csv` contain one row per epoch: total/KD/clean
loss, learning rate, mean pre-clip grad norm, valid frames, optimizer steps,
wall time, GPU peak allocated MiB, and dev clean loss. `run.json` records the
run identity, hashes, floor constraints, and actual wall time.

## CLI and commands

```bash
cd /home/zhangqingtian/project/itu/build/ns-distilled-c64-20261116
/home/zhangqingtian/.venv/bin/python train_student.py --self-test --output-dir verification/training-smoke

# KD then clean, initial -9 dB version
/home/zhangqingtian/.venv/bin/python train_student.py \
  --manifest features/manifest.json --output-dir train/kd-then-clean-floor-9 \
  --device cuda --seed 20261116 --mode kd_then_clean \
  --kd-epochs 8 --clean-epochs 12 --batch-size 8 --chunk-frames 200 \
  --learning-rate 0.001 --absolute-floor-db -9

# Supervised control, same initialization/budget
/home/zhangqingtian/.venv/bin/python train_student.py \
  --manifest features/manifest.json --output-dir train/supervised-floor-9 \
  --device cuda --seed 20261116 --mode supervised \
  --kd-epochs 8 --clean-epochs 12 --batch-size 8 --chunk-frames 200 \
  --learning-rate 0.001 --absolute-floor-db -9
```

The requested relative form `.venv/bin/python ...` works when that environment
is linked inside the build directory; the current remote environment is at
`/home/zhangqingtian/.venv`.

## Resource expectation

For 128 40-second train streams and 24 dev streams, the CPU cache is about
4.0 GiB for train plus 0.75 GiB for dev in KD mode (teacher spectra included);
supervised mode omits teacher spectra. A 200-frame batch-8 chunk is small
(under roughly 0.1 GiB of spectral tensors before autograd; model state is
256 bytes per stream), with expected GPU peak below 1 GiB on the configured
GB10. The 20-epoch loop is 6,400 optimizer steps per mode (128/8 batches times
20 chunks times 20 epochs). It should be measured from the first epoch log;
the design target is minutes, not hours. The self-test runs one short KD TBPTT
batch, one clean TBPTT batch, dev selection, checkpoint round-trip, and export
metadata check on GPU in a few seconds.
