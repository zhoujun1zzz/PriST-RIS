# V3.1 development protocol

Development is validation-only and writes new Mobility artifacts under `runs/v3_1_q0q3_dev/` or names prefixed `v31_q0q3_`. Existing pre-fix V3.1 runs, Ridge files, checkpoints, and diagnostics are not deleted or overwritten.

Sequence:

1. Run audit and confirm all four train/validation sources, keys, shapes, and provenance.
2. Run the full test suite, including CUDA AMP tests when CUDA is available.
3. Fit Quasi Ridge on target block 0 and a new Mobility Ridge on blocks 0/3 using train only; choose regularization on validation. Do not reuse `v31_ridge_mobility_dev4096.npz`.
4. Run C smoke first and inspect q0, q3, and pilot-anchor aggregate.
5. Only after the spatial gate, run Full smoke and inspect q0–q5, pilot, non-pilot, and overall diagnostics.
6. Use `mode=dev` only for the first Mobility seed-123 validation run. Do not start formal three-seed or broad HPO automatically.

Formal mode rejects AMP. Smoke/dev AMP is supported through an FP32 complex island. No development command authorizes test.

Pre-fix Mobility priors, C checkpoints, q0-only/true-residual ablations, and
Full smoke results remain preserved as provenance but are not valid inputs to
the q0/q3 canonical line. The true-residual experimental block change is not
part of this repair.
