# ETSI TS 103 281 Model A configuration files

Source: ETSI TS 103 281 V1.3.2 (2026-07), Annex A.2.

Archive:

`https://www.etsi.org/deliver/etsi_ts/103200_103299/103281/01.03.02_60/ts_103281v010302p0.zip`

The three `*.model` files are the random-forest regression configurations
published with the standard. They are not a complete standalone implementation:
Model A also requires the preprocessing, hearing-model, reference-optimisation
and feature-extraction stages described in clauses 6.3.2 through 6.3.8.

Official Annex C test vectors are not copied into this repository. The verifier
script downloads them from the ETSI docbox when requested.

`annex-c-validation.json` records the complete result of running the verifier
against all 48 official conditions. It is bound to the SHA-256 of
`server/quality/ts103281.py`; a source change invalidates the evidence until
the verifier is rerun. The current evidence reports 0/48 conditions within the
normative +/-0.1 MOS tolerance, so the implementation is experimental and must
not be represented as TS 103 281 compliant.
