# Frozen minima/history propagation diagnostic

These are the frozen probe, observer-only patch, analyzer, protocol and analysis
plan used for the 2026-10-24 component diagnosis. They are retained as source
artifacts; copying them here does not replay audio or enable a runtime option.
Only probe.cpp line endings are normalized to LF; the original probe source
SHA-256 is 6a4b761cc584d0e8fb1599127f384be6216a3d9df1af72deb5017fd43740aa5b.

The protocol intentionally names the original frozen ITU evidence paths and
hashes. The analyzer requires that evidence tree, including both inputs, four
reference outputs, role traces and pre-run identity locks. It is not a
standalone fresh-clone test. Do not regenerate or reinterpret missing evidence
as a passing audio result. Apply observer.patch only to the exact rejected
candidate source identified in the evidence lock, not directly to production.

See VCP_OPTIMIZATION_STATUS.md from the repository root for the result
and limitations. The latest diagnosis does not admit the rejected candidate.
