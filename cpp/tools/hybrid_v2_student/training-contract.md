# Hybrid NS v2 training contract

2026-11-15. All formal runs remain separate from frozen v1/C64 weights.

- Data: original 128 train/24 dev feature rows plus 32 train/8 dev Common Voice Chinese streams. Source identities and licenses stay in the expansion manifests. Common Voice originals are references; absence of background noise is not guaranteed. A separate speaker/recording-isolated newtest16 is excluded from training and checkpoint selection.
- Three supervised ablations share the exact data order, seed 20261117, 20 epochs, batch 8, AdamW lr 0.001, weight decay 1e-4, gradient clip 5, TBPTT 200 frames. Hidden state persists across the entire 40-second stream, including pauses. No teacher is consumed this round.
- `oldloss256`: the exact prior supervised compressed-complex + compressed-magnitude + 0.5 dB-residual Huber loss. The copied source hash is recorded.
- `newloss256`: same model and inputs, new speech-preservation objective.
- `newloss320`: append `20 log10(max(project(linear base_gain257),1e-8))` in 64 bands. Project first, then take the logarithm. No additional clipping or feature scaling.
- Model: I→Linear64/ReLU→GRU64→Linear64→12tanh; residual dB, base OM-LSA floor -9dB, final floor -21dB. Input256 is 45,568 parameters/4.5056MMAC/s; input320 is 49,664/4.9152MMAC/s. All runtime DSP costs are additional.

New objective, applied to one final gain g:

`g = clip(base_gain * 10^(interpolate(delta64)/20), 10^(-21/20), 1)`.

On each TBPTT chunk and each stream, `D = sum_valid(|C|²+|N|²)+eps` with `N=X-C` labels only. Speech distortion is `sum(w_t*(1-g)²*|C|²)/D`. Noise residue is `sum_valid(g²*|N|²)/D`, including pauses and pure-noise streams. The speech coefficient is 2, or 4 for streams whose clean-active reference SNR exceeds 20dB. Speech weights are precomputed on the complete stream: active energy exceeds max(p95*1e-5,eps); weak active frames are below p95*0.03; onset/tail frames fall within 20 frames of a clean activity transition; weights are 3 for those protected active frames and 1 for other active frames. Label lookahead used only to weight training targets does not enter runtime inputs.

The tiny auxiliary is `0.01*SmoothL1(raw_delta/12, clipped(20log10(IRM_amplitude/base))/12, beta=.1)`, ignoring only truly undefined clean+noise bins. IRM is an amplitude target, not SPP. The old unnormalized 0.5 dB-Huber is absent from the new objective. Main/aux ratios and gradients are logged rather than assumed balanced.

The validation objective is this same new speech-preservation criterion for all three runs. It averages valid per-stream/per-chunk ratios, rather than claiming equivalence to one globally energy-pooled ratio. No DNSMOS/PESQ/test result selects checkpoints. Best/last checkpoints carry the model source hash, manifest SHA, input contract, floor and gain-mode metadata.

Training allocation uses complete stream batches. With 160 train streams, 20 epochs comprise 8,000 optimizer steps (400 per epoch). All three runs receive the same budget; no batch is silently truncated to the old 6,400-step budget.

Harmonic filtering is a separate inference ablation using the exact newloss320 weights at fixed strength 0.5. It has no additional neural head and cannot be attributed to loss/input improvements. Default remains c4eb unless paired quality and regression evidence supports a later change.
