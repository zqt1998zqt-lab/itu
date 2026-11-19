# VCP adaptive candidate

`vcp_adaptive` is an opt-in, frame-causal NS experiment based on `vcp_mild`. Select it explicitly; the existing profiles remain available. The fixed evaluated CLI is `build/vcp-adaptive-20261114/build-delivered/cpp/itu_audio_cli` (SHA256 `c4eb33250432f63b6b228be8c896a507cf15d0fc111755f15cc319fe4557499e`).

```bash
build/vcp-adaptive-20261114/build-delivered/cpp/itu_audio_cli \
  --profile vcp_adaptive --no-aec --mic capture.wav --out build/adaptive-example
```

For per-stage G/S/N, use the existing stage scorer in the configured evaluation environment:

```bash
python cpp/tools/vcp_stage_mos.py \
  --cli build/vcp-adaptive-20261114/build-delivered/cpp/itu_audio_cli \
  --profile vcp_adaptive --mic capture.wav \
  --output-dir build/adaptive-stage-example -- --no-aec
```

The output directory must be new. The scorer retains actual PCM taps, original-amplitude and −30 LUFS G/S/N, the pinned DNSMOS model, declared-delay crops and hashes. Input audio is not normalized for enhancement. G/S/N mean OVRL/SIG/BAK; their arithmetic average is not a separate G-MOS metric.

## Controls and processing

- **Pause noise tracking**: `adaptive_noise_tracking` uses the engine's current/past capture spectral-excess evidence, with release and confirmation periods. It reduces held speech protection during confirmed pauses, reopens only bounded upward PSD updates and continuously lowers the synthesis floor by at most 9 dB relative to the configured floor. It does not read clean audio, clip schedules, MOS or future samples.
- **High-SNR transparency**: `high_snr_transparency` restores individual synthesis gains only with current local excess and midband evidence. It does not introduce a fullband dry bypass.
- **Low-band protection**: `low_band_protection` applies bounded restoration in 80–400 Hz with midband speech evidence and local SNR support. It does not restore all low frequencies during pauses.
- A narrow cold-start tone safeguard avoids treating an isolated 250–1200 Hz tone as a pause. Protection is limited to nearby bins and expires as soon as continuity is broken. Preserving an isolated tonal noise is an explicit tradeoff.
- Estimator and decision-directed H1 state do not consume the corrected synthesis gain. Processing remains allocation-free per 10 ms frame; declared NS delay stays one hop (160/240/480 samples at 16/24/48 kHz).

Each control defaults to false in `OmlsaTuningConfig`. `vcp_adaptive` enables all three; `vcp_mild` leaves them false. Single-factor switches are:

```text
--omlsa-adaptive-noise-tracking / --no-omlsa-adaptive-noise-tracking
--omlsa-high-snr-transparency / --no-omlsa-high-snr-transparency
--omlsa-low-band-protection / --no-omlsa-low-band-protection
```

With AEC enabled, construction retains the established WebRTC route and clears OM-LSA tuning, including these controls. No live backend switching is introduced. Existing C ABI layout/profile IDs are unchanged; this candidate is exposed through C++ and CLI. The C++ configuration layout grows, so rebuild C++ clients against the new headers; C++ binary ABI compatibility is not claimed. Direct `--no-ns` combined with inherited numeric OM-LSA tuning retains the existing validation error; clear numeric tuning explicitly before disabling NS.

## Measured results and limitations

These are internal synthetic-mixture regressions, with 24 s processing, fixed 0–12 s scoring, three overlapping DNSMOS hops and −30 LUFS scoring copies. Scores were produced with the frozen r2 CLI and bound to the delivered binary only after all nine stage PCM outputs matched exactly on all 62 adaptive cases. The scored model SHA is `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`.

| Group | Mild G/S/N | Adaptive G/S/N |
|---|---|---|
| Mandarin 40 | 2.760 / 3.303 / 3.402 | 2.825 / 3.350 / 3.535 |
| English 10 | 2.920 / 3.506 / 3.500 | 2.916 / 3.446 / 3.548 |
| New speaker/noise holdout 4 | 2.950 / 3.492 / 3.409 | 3.053 / 3.565 / 3.592 |

Mandarin G improves in 27/40 pairs, but the median gain is only about 0.012. English S declines in 9/10 and STOI in 10/10; do not promote this as a universal improvement. The four new holdout cases cover only one speaker and remain below the established `vcp` baseline's G/S/N. The 20 Mandarin pairs excluded from this turn's parameter development were already inspected in prior diagnosis; they are not a globally unseen benchmark.

At Mandarin SNR10, G changes 2.967→2.972 and S 3.429→3.418. The transparency control itself has a small positive Mandarin S tendency but slightly negative all-language G/N; high-SNR transparency is not established as solved. Low-band restoration has small mixed benefits. Development ablations identify pause tracking as the main contributor (mean G +0.065 over its disabled counterpart).

The 22–24 s pure-noise attenuation changes 7.702→7.766 dB for 32 car-cabin pairs and 3.847→5.545 dB for eight outdoor-street pairs. Do not confuse this with the scorer's separate low-energy-frame metric inside 0–12 s. The original English/Mandarin waveform, noise and clean references were retained; no oracle gate or 2800 Hz notch is part of this candidate.

## Verification and evidence

- `ctest-delivered.log`: nine focused C++/CLI/engine tests, including allocation, causality, cold-tone continuity, missing-evidence sentinels and backward compatibility.
- `structural/delivered/`: 46 command checks; old `vcp_mild` compatibility, AEC route equality, disabled-control equality and native-rate tap contracts.
- `eval/delivered-equivalence/`: all 62 adaptive cases, all stage PCM equal to the scored version; source/binary SHA stable.
- `eval/delivery/`: case-level and stage-level CSV, original report paths, consolidated provenance and exact source hashes.
- `eval/ablation-r2/`: 36 calls for three single-factor comparisons.
- `eval/transparency-check/`: 25 high-SNR transparency comparisons.
- `integration-receipt.json`: source-before checks, preserved backups and the exact ten-file integration scope.

The candidate has not been deployed to a service or made a default. Low-SNR speech/noise discrimination, actual devices, continuous natural conversation and larger unseen-speaker evaluation remain open.
