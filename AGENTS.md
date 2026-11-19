# ITU project agent instructions

## NS quality reference

- For NS implementation changes, comparisons, and tuning, use the frozen `dpdfnet2-16k-v1` reference described in `benchmarks/ns/README.md`. Run candidates and the reference on identical inputs/windows with the frozen scoring protocol; report paired ΔG/ΔS/ΔN, clean-reference scores, separate Chinese/English and speech/clean/continuous/noise groups, reference fidelity, latency and resource costs. Keep c4eb as the current processing/default regression baseline. Do not compare the historical aggregate G score across datasets, treat DPDFNet as a quality ceiling, or overwrite frozen v1 artifacts; version changes to weights or protocol.
