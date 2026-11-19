# VCP mild candidate

This opt-in candidate is available as `--profile vcp_mild`. The existing `vcp` profile remains the rollback route. The binary used for final verification is `build/vcp-opt-20261114/cpp/itu_audio_cli`.

From the ITU project root, run the existing stage scorer in its evaluation Python environment:

```bash
python3 cpp/tools/vcp_stage_mos.py \
  --cli build/vcp-opt-20261114/cpp/itu_audio_cli \
  --profile vcp_mild --mic capture.wav \
  --output-dir build/mild-stage-test -- --no-aec
```

The new output directory contains every stage WAV, actual-level and post-enhancement -30 LUFS G/S/N, adjacent-stage changes, and JSON/CSV/Markdown reports. G/S/N are DNSMOS OVRL/SIG/BAK predictions, not listening-test MOS. Inputs and enhancement are never loudness-normalized for inference.

## Processing routes

- With AEC disabled, use native OM-LSA/IMCRA, noise-level 0 (a -9 dB spectral gain floor), speech-qualified upward noise-update scale 0.5, speech recovery 0.7, and release 120 ms. The floor does not imply every sample or noise segment loses exactly 9 dB.
- With AEC enabled, retain the established WebRTC NS route. This selection occurs during construction; there is no runtime backend switching or estimator reset. The experiment that applied the new NS to AEC residuals failed the far-only attenuation guard and is not selected by the final profile.
- The bounded utterance gain uses a +3 dB cap, rise 3 dB/s, fall 6 dB/s, and 120 ms hold. It resets at physical far activity; fixed/adaptive/disabled AGC report the bounded envelope as effectively disabled.
- The optional `--omlsa-speech-floor-db -6` or `-3` is available for controlled experiments. Neither is enabled by the final profile: they reduced noise-quality gains in the development comparison.
- `--no-bounded-speech-gain` disables only the new envelope. `--no-mild-near-only-ns` disables only the conditional route selection, retaining the explicitly configured backend and OM parameters. It is not a full rollback. To change from OM-LSA to a different backend, explicitly restore OM tuning to defaults, or start from `--profile vcp`.
- `mild_ns_route` reports the selected construction route; always read it with `ns_enabled`. A disabled NS does not process audio even if the selected route has a name.

## Measured result

The seven previously inspected clips are development diagnostics. On six mixtures excluding the original example, normalized output G/S/N changed from 2.644100/3.520543/2.951798 to 2.746350/3.556674/3.123991. One high-SNR development clip loses G; do not describe this as a win on every clip.

Before viewing new results, the base mild parameters were selected over the stronger -6/-3 dB speech-floor experiments. Twelve new clean waveforms, each mixed with a fixed new seed and 3–18 dB stationary white noise, then produced:

| Metric | Established VCP | Mild |
|---|---:|---:|
| G / OVRL at -30 LUFS | 2.355035 | 2.560081 |
| S / SIG | 3.403021 | 3.501897 |
| N / BAK | 2.540348 | 2.858080 |
| STOI | 0.873839 | 0.892281 |
| SI-SDR, dB | 11.277971 | 15.202744 |
| Later quiet-region attenuation, dB relative to input | -6.336057 | -8.533503 |

G improves on 12/12 new mixtures; SIG improves on 11/12. The new waveform SHAs differ from the development SHAs, but source recordings may recur, so this is not a speaker-disjoint or recording-independent evaluation.

## Known tradeoffs

Weak clean-dominant cells have a larger mean loss on the twelve new mixtures: -2.3725 to -3.8317 dB. Strong clean-dominant 4–8 kHz cells change from -1.5884 to -4.0562 dB. The candidate improves the measured MOS/intelligibility metrics but does not meet the earlier proposed weak/HF preservation goals.

Four clean guards have average G +0.0602, but average SI-SDR -4.0963 dB and STOI -0.000795; one clean clip has a large SI-SDR drop from 47.32 to 28.41 dB. This is a material fidelity limitation, even though most non-reference MOS scores rise. Do not call the candidate transparent on clean audio.

The final AEC route must be assessed with its own final far-only and double-talk records; the discarded OM residual-chain scores do not describe the delivered route. Pure echo must not be scored with a near-speech DNSMOS role.

The candidate has not been deployed to a running service or promoted as a product default. Broader nonstationary-noise, new-speaker, device and listening validation remain open.

## Evidence

- `build/vcp-opt-20261114/validation-comparison/results.json`: 12 paired noisy mixtures, stage MOS, STOI, SI-SDR, spectrum, clips, source/CLI SHAs.
- `build/vcp-opt-20261114/clean-guards/results.json`: four clean fidelity guards.
- `build/vcp-opt-20261114/duplex-guards-final/results.json`: delivered AEC-route double-talk results.
- `build/vcp-opt-20261114/v6-final-verification.json`: byte-exact far-only and selected near-only equivalence.
- `build/vcp-opt-20261114/final-nearonly-equivalence/verification.json`: all twelve final near-only output bytes equal the scored outputs.
- `build/vcp-opt-20261114/cli-candidate-validation-v6.log`: profile, explicit flags, defaults, parameter validation, rates and stage exports.
- `build/vcp-opt-20261114/ctest-v6.log`: focused C++ and CLI regression checks.

The final binary identity is recorded in the verification receipt. Earlier development binary identities are kept with their original records; waveform equivalence is required before reusing their scores.
