# V3.1 Quasi-to-Mobility spatial transfer

Quasi provides spatial pretraining only; PriST-RIS does not claim that Mobility non-pilot reconstruction was pretrained on Quasi.

The explicit spatial-only loader accepts a current-protocol V3.1 Quasi C/Full checkpoint and loads compatible backbone, coordinate encoders, prior feature encoder, observed-to-dense attention, shared anchor feature layer, and first anchor head. The Mobility q3 anchor head, temporal spatial encoder/bases, coefficient/trend module, and non-pilot correction remain newly initialized. A Quasi A/B checkpoint may still provide its compatible base weights, with the new attention initialized on the target. Pre-attention Quasi C/Full checkpoints are rejected rather than mislabeled as the new C ladder. Metadata records loaded, skipped, and newly initialized keys.

Fractions are nested 1%, 5%, 10%, 20%, and 100% subsets. Canonical protocols are:

- `target_only_scratch`
- `full_finetune`
- `frozen_spatial`
- `selective`

The fake `adapter_only` protocol was removed, reducing the matrix from 25 to 20 runs. Frozen-spatial and selective expose distinct trainable parameter sets.
