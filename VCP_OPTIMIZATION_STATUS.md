# VCP optimization checkpoint — 2026-10-24

The enhancement page (`#enhancement`) combines the BF/AEC/NS/EQ/AGC/DRC chain,
paired audio graphics, and evidence for the tested development candidates.
The graph engine uses shared input identities, alignment and physical power
calibration; it must not substitute internal plots for audio quality gates.

## Latest diagnostic result

The minima/history bootstrap candidate remains rejected. On the four-microphone
0 dB cold-start window, its SI-SDR delta was -1.12084288094 dB and STOI delta
was -0.08039659381 against both frozen references. The original limits remain
SI-SDR delta >= -0.1 dB and STOI delta >= -0.02, together with the existing
noise, echo, gain and clipping requirements.

Two frozen SG component inputs were inspected with the option off/on. All four
full 2,402-hop wet outputs matched their reference bytes. The first 60 hops
show minima changes at hop 1, estimator decision and raw noise PSD changes in
the same hop, then gamma, DD prior, spectral gain and wet changes at hop 2.
All five recurrence checks passed for every case/role. The four-microphone
case accumulated much larger spectral changes; gamma and gain are not
monotonic at every frequency bin.

This is component evidence. The rejected whole-chain runs do not have a
same-run independent SG input capture, so the traces do not quantify the
causal share of the whole-chain SI-SDR/STOI loss. Minima reseeding and ring
clock reset are still a bundled intervention.

## Next experiment

Keep PSD, DD, smoothing and ring timing continuous; freeze one gradual
minima/history migration design before testing. First bridge same-run SG
input/state/output with the failed whole-chain cold window, then apply the
original gates. Expand through the known cases, controls, 24 and 72 scenes
only after the preceding gates pass. Independent holdout and device validation
remain separate. No new quality improvement or production promotion is claimed.

## Evidence location and identity

Large evaluation assets, raw traces, experimental build trees and internal
records retain the repository's existing separate-storage policy. Runtime
reports live under `public/evaluation/`; they are not supplied by this source
commit. Keep the matching asset release when deploying the web shell.

- Latest report: `public/evaluation/vcp-joint-structure/report.json`
- Report SHA-256: `97620b264cc11442450459d40e63ae46d4df0952624759e5b332409c0f8aa093`
- Manifest SHA-256: `f4db6720c5e00143cec3b466986153474dc45f582f01e0e03c77a39cfedd053c`
- Component analysis SHA-256: `4b3303985e2fdec64fb345ad9d73a7d13f9825cb1081a47f2b67052adea10b40`
- Delivery ZIP SHA-256: `0e149e3c72fb73c8fc3df72d196dc6b3c8ea04156397e4b1dbbda1e389bf14dc`

Publication validation completed six desktop/mobile checks with 388 image
loads and no page errors, preserving ten candidates, 23 eight-image audio
groups and ten diagnostic figures. Production CLI and graph engine identities
were unchanged by the diagnostic publication. These are evidence snapshot
checks, not a fresh-clone deployment or independent acoustic validation.
