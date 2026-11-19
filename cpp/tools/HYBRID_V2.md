# VCP Hybrid V2 usage

`vcp_hybrid_v2` is the native 16 kHz mono VCP C64 profile plus an optional
ITUNSR02 synthesis-gain controller and causal pitch post-filter. It derives
`barrot_vcp_c64_config()`: AEC off, OM-LSA strength 0, aligned policy, no raw
mix, default OM-LSA tuning, and no legacy repair/rumble/guard stages.

## Command line

Pitch off:

```bash
itu_audio_cli \
  --profile vcp_hybrid_v2 \
  --mic input-16k-mono.wav \
  --hybrid-v2-model weights.bin \
  --hybrid-v2-mix 1 \
  --hybrid-v2-floor-db -21 \
  --hybrid-v2-pitch-strength 0 \
  --diagnostic-stages \
  --out prefix
```

Pitch DSP arm:

```bash
itu_audio_cli \
  --profile vcp_hybrid_v2 \
  --mic input-16k-mono.wav \
  --hybrid-v2-model weights.bin \
  --hybrid-v2-mix 1 \
  --hybrid-v2-floor-db -21 \
  --hybrid-v2-pitch-strength 0.5 \
  --diagnostic-stages \
  --out prefix
```

Omit `--hybrid-v2-model` for the statistical-only control. A pitch-only DSP
comparison may also omit the model while setting an explicit nonzero
`--hybrid-v2-pitch-strength`.

The profile is fixed to native 16 kHz, 160-sample frames, mono, AEC off, and
enabled OM-LSA. The legacy NS/model flags are rejected for this profile. Model
paths are never hard-coded.

## Model contract

- Format: `ITUNSR02`, FFT 512, hop 160, window 320, 64 projection/hidden/output
  units.
- Input dimensions: 256 parameters with feature version 1, or 320 parameters
  with feature version 2.
- A non-empty model path is loaded during construction. Missing, corrupt,
  untrained, checksum-failing, or wrong-dimension models throw; there is no
  runtime switch to an untrained model.
- Final real 320 checkpoint:
  `train/newloss320/export/weights.bin`,
  SHA256 `662377b8b2c011046226d2b457beb3af45486071fde6ae8273ee37be1fe0ac63`.
  Its sidecar reports `fixture:false`; selected export step 7,600 of 8,000 total
  training steps.

## Runtime outputs and limits

- The V2 gain observer changes synthesis gain only; OM-LSA PSD, DD, IMCRA and
  noise state remain independent.
- Pitch observation uses the same PCM16-quantized OM-LSA input before NS. The
  post-gain complex hook changes wet257 only and adds zero latency.
- Declared NS delay is 160 samples. Pitch strength zero does not construct the
  pitch filter and is byte-identical to the disabled arm.
- `prefix.json` and `prefix.frames.csv` expose independent `hybrid_v2_*` model,
  gain, pitch, raw-mix and fallback counters. Timing is the full OM-LSA plus V2
  chain, not pure neural-network timing.
- Runtime fallback keeps the OM-LSA base gain/wet spectrum. A construction-time
  model load failure does not fall back.

## Verification boundary

The fixed corrected binaries are CLI `61c46886...` and native probe
`cca8c9ab...`, with native header `bb762558...`. On the frozen first Chinese,
English speech, and English clean cases, pitch 0 and 0.5 produced 384,160-sample
outputs with `max_abs=0.0` and bit-exact probe/CLI samples. The full receipt is
under `verification/vcp/` and the three-case summary is
`verification/integrated-real320/final-fixed-3cases/summary.json`.

These checks establish frontend/native parity and telemetry semantics only. No
MOS, acoustic-quality, generalization, or pure-neural runtime claim is made.
