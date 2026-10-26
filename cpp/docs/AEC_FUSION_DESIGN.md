# Fused Speex AEC

`vcp_fused_aec` keeps Speex MDF as the primary long-tail canceller and adds a
causal short-tail NLMS shadow estimator. The two PCM outputs are never added.
The shadow branch estimates a short acoustic path from the conditioned render
and capture, and the controller blends residuals:

```
fused_error = (1 - w) * mdf_error + w * nlms_error
```

`w` is bounded to `[0.02, 0.85]` and updated with a frame-rate EMA. It can
rise only after the primary and shadow residual powers are measured while the
reference is trusted. It decays toward the safe floor on an unverified
reference or double talk. NLMS adaptation is frozen while the existing
near-end hold is active. This lets the primary MDF keep its 500 ms long-tail
coverage while the shadow can react to short path changes without replacing
the established post-AEC context, coherence residual suppression, NS, EQ,
AGC, DRC, or limiter stages.

The fused path is deliberately restricted to one microphone and one
synchronous render channel and does not support Speex RES. The normal `vcp`
and WebRTC profiles remain unchanged. Both `vcp_fused_aec` and `vcp_unified`
use the fused AEC; `vcp_unified` additionally selects the UnifiedNs four-expert
noise suppressor. The C++ presets are
`barrot_vcp_fused_aec_config()` and `barrot_vcp_unified_config()`. The C API
profiles are `ITU_AUDIO_PROFILE_VCP_FUSED_AEC` and
`ITU_AUDIO_PROFILE_VCP_UNIFIED`; the matching CLI profiles are
`vcp_fused_aec` and `vcp_unified`.
The CLI also accepts `--aec-fused-learning-rate` and
`--aec-fused-shadow-taps-ms` (1..64 ms, clamped internally to 16..1024 taps).

Per-frame diagnostics are exported in `FrameInfo` and `frames.csv`:
`aec_fused_shadow_weight`, primary/shadow error powers,
`aec_fused_double_talk`, and `aec_fused_updates`. The JSON report records the
profile, algorithm name, and shadow parameters. The new regression test
compares baseline Speex MDF and the fused path on white-noise playback with a
path change and a near-end tone; it also checks bounds, invalid multichannel
configuration, finite output, stable declared latency, and near-end gain.

This is an engineering candidate, not a production acceptance claim. Board
latency/RTF and an independent holdout of real vehicle recordings remain
required before selecting it as the default.
