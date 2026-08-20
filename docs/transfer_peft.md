# V3.2 Quasi-to-Mobility spatial transfer

Quasi provides spatial pretraining only; PriST-RIS does not claim that Mobility non-pilot reconstruction was pretrained on Quasi.

The explicit spatial-only loader accepts a V3.2 Quasi checkpoint and loads structurally compatible backbone, coordinate encoders, shared prior encoder, observed-to-dense attention, first anchor refiner, and first anchor head. The Mobility q3 refiner/head and temporal components remain newly initialized. V3.1 sources are rejected because their prior and refinement paths are not compatible. Metadata records loaded, skipped, and newly initialized keys.

Fractions are nested 1%, 5%, 10%, 20%, and 100% subsets. Canonical protocols are:

- `target_only_scratch`
- `full_finetune`
- `frozen_spatial`
- `selective`

The fake `adapter_only` protocol was removed, reducing the matrix from 25 to 20 runs. Frozen-spatial and selective expose distinct trainable parameter sets.
