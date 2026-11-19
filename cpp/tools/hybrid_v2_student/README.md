# Hybrid NS v2 source snapshot

Versioned sources for the isolated `build/ns-hybrid-v2-20261115` experiment.
Copy this directory into a fresh `project/build/<run>/` directory before using scripts that derive project paths. Data, checkpoints, native binaries and frozen scorer assets remain referenced by manifests and delivery receipts. This source snapshot does not copy audio, models, caches, or credentials.

Model magic is ITUNSR02. Both input256 and input320 are intentional. v1 c64_student sources and ITUNSR01 remain separate. The corrected runtime probe is `native/hybrid_v2_probe-fixed` (SHA cca8c9ab5955b2bdde7912203c1c61a1ffdcbd45b124c26c78d952ab7540c152); never reuse the superseded projection-offset binary.

Current data references include Common Voice originals that may retain real background noise. No candidate is promoted to defaults based merely on successful training or passing inference checks.
