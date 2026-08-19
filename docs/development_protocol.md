# V3.1 development protocol

Development is validation-only and writes new artifacts under `runs/v3_1_dev/` or names prefixed `v31_`. Existing `runs/v3_dev/`, Ridge files, checkpoints, and V3.0 diagnostics are not deleted or overwritten.

Sequence:

1. Run audit and confirm all four train/validation sources, keys, shapes, and provenance.
2. Run the full test suite, including CUDA AMP tests when CUDA is available.
3. Fit Quasi Ridge on target block 0 and Mobility Ridge on blocks 0/1 using train only; choose regularization on validation.
4. Run C smoke first and inspect q0, q1, and observed-anchor aggregate.
5. Only after the spatial gate, run Full smoke and inspect q0–q5, observed, future, and overall diagnostics.
6. Use `mode=dev` only for the first Mobility seed-123 validation run. Do not start formal three-seed or broad HPO automatically.

Formal mode rejects AMP. Smoke/dev AMP is supported through an FP32 complex island. No development command authorizes test.
