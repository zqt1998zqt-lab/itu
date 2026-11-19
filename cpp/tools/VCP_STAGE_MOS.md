# VCP stage MOS runner

`vcp_stage_mos.py` executes the VCP CLI once with `--diagnostic-stages`, then scores only the
`pipeline_stages` taps named by that run's `<prefix>.stages.json`. It never reruns the engine
with modules disabled to manufacture a stage trajectory. Each channel is scored separately;
the runner does not downmix.

## Requirements

Run the examples from the project root. Use the existing `ecnr_perceptual.py` environment and install the pinned loudness dependency:

```bash
python3 -m pip install -r cpp/tools/requirements-stage-mos.txt
```

The DNSMOS model, source commit, and model SHA are validated by the existing scorer and its
`DEFAULT_ASSETS`. `--assets-dir` is available for an equivalent pinned test asset directory.

## Usage

The selected window is on the shared input-capture timeline and must contain near speech. In a
duplex run, choose a window that also contains far-end speech when RX stage scores are needed;
otherwise the RX stages correctly remain unavailable/null. TX and RX intentionally share this
window. For example:

```bash
python3 cpp/tools/vcp_stage_mos.py \
  --cli build/vcp-stage-mos-20261114/cpp/itu_audio_cli \
  --profile vcp \
  --mic capture.wav \
  --rx-in rx.wav \
  --kind near_only \
  --window 1.0 8.0 \
  --output-dir runs/stage-mos-20261114 \
  -- --aec-backend speex_fused_cpu --beam-delays 0,11.25,22.5
```

All engine-specific options belong after the literal `--`. The runner adds `--profile`,
`--mic`, `--rx-in`, `--echo`, `--out`, and `--diagnostic-stages` itself, so passthrough copies
of those options are rejected. Multiple microphones require the CLI's calibrated
`--beam-delays` argument after `--`.

When there is no far-end reference, make that state explicit in the one engine run:

```bash
python3 cpp/tools/vcp_stage_mos.py \
  --cli build/vcp-stage-mos-20261114/cpp/itu_audio_cli --profile vcp --mic capture.wav \
  --output-dir runs/stage-mos-no-aec -- --no-aec
```

For actual playback-reference replay, use `--echo playback.wav` instead of `--rx-in`.

The output directory must not exist. On success, the final stdout line is the absolute path
to `stage-mos.json`. The same directory also contains `stage-mos.csv`, `stage-mos.md`,
engine logs, the raw run artifacts, and FLOAT normalized WAVs under `normalized/`.

## Measurement protocol

- G is DNSMOS `OVRL`; S is `SIG`; N is `BAK`. Full scorer metadata, raw values, polynomial
  MOS, and hop scores are retained.
- Stage availability uses a conjunction across all physical channels:
  `original_available = all(channel original)`, `normalized_available = all(channel -30)`, and
  `stage.available = original_available and normalized_available`. Partial counts and each
  unavailable score reason remain explicit in `unavailable_scores`.
- `original_scores` use the actual stage level. `normalized_scores` use the same cropped,
  windowed segment resampled to 16 kHz, then normalized to `--loudness-target` (default
  `-30` LUFS). Loudness preparation always scales by `0.7 / peak` before K-weighted
  measurement; original and prepared LUFS, the constant gain, and reread error are retained.
  Both scoring paths share the same sample duration before the scorer's documented short-input
  repeated-doubling policy. A normalized waveform reaching full scale is reported as
  `normalization_clipping` instead of being accepted.
- The fixed stage delay is removed before the window is applied. Native-rate input length is
  checked against metadata to within one sample; a larger mismatch is an error.
- `measurement_domain` is copied verbatim into the report. In particular, a LocalVQE AEC tap
  that includes the model's own noise suppression is not relabeled as pure AEC.
- `tx_input` and `rx_input` may be the runner's original `--mic`/`--rx-in` path or a source
  copy inside the new run directory. A copy must equal the input after float32 conversion,
  sample for sample; both source and stage SHAs are reported. All other taps must belong to
  this run's output directory.
- Every declared pipeline-stage file must exist. `capture_available: false` only marks an
  existing unavailable capture; it does not permit a missing tap. A missing file fails the run
  instead of producing a normal report.
- Audio above full scale is retained. Its peak and clipped-sample count are reported; it is
  never silently clamped.
- Non-finite PCM is JSON-safe evidence: peak and clipped count are null, while `finite: false`
  and `nonfinite_samples` are recorded. That channel's original and normalized scores are
  unavailable with `nonfinite_audio`; it is not resampled, normalized, or scored.
- Unavailable scores keep null G/S/N and an explicit reason. A bypassed module
  (`enabled: false`) still scores its real captured bypass PCM and is marked
  `status: bypassed`; only `capture_available: false` suppresses scoring.
- Adjacent deltas use the same direction, channel index, and `order` from the same metadata.
  A multi-microphone `tx_input` is not treated as one-to-one with a mono `tx_bf`; that delta is
  a ch0 reference with explicit `selected_reference_channel_0` mapping. `capture_channels`
  distinguishes physical channels from copied file channels, so copied AEC channels are not
  counted as independent physical channels. `diagnostic: true` entries (including
  `tx_ns_selected`) are scored but excluded from the main-chain delta.

The runner does not infer whether a clip is near speech. Pure noise is not a valid near-only
role guess: choose a capture window that contains the intended near talker, or use a role
explicitly supported by the scorer.

DNSMOS P.835 here is a non-personalized model estimate. It is not human MOS, POLQA, or an ITU
listening-test certification.
