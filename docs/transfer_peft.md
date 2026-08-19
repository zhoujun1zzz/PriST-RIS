# V3.1 Quasi-to-Mobility spatial transfer

Quasi provides spatial pretraining only; PriST-RIS does not claim that future forecasting was pretrained on Quasi.

The explicit spatial-only loader accepts a V3.1 Quasi C/Full checkpoint and loads compatible backbone, coordinate encoders, prior feature encoder, shared anchor feature layer, and first anchor head. The Mobility second anchor head, temporal spatial encoder/bases, coefficient/trend module, and future correction remain newly initialized. Metadata records loaded, skipped, and newly initialized keys.

Fractions are nested 1%, 5%, 10%, 20%, and 100% subsets. Canonical protocols are:

- `target_only_scratch`
- `full_finetune`
- `frozen_spatial`
- `selective`

The fake `adapter_only` protocol was removed, reducing the matrix from 25 to 20 runs. Frozen-spatial and selective expose distinct trainable parameter sets.
