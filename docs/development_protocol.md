# V3.1 development protocol

Development is validation-only and writes new attention-stage Mobility runs under `runs/v3_1_obsdense_dev/` or names prefixed `v31_obsdense_`. Existing pre-attention V3.1 runs, Ridge files, checkpoints, and diagnostics are not deleted or overwritten.

Sequence:

1. Run audit and confirm all four train/validation sources, keys, shapes, and provenance.
2. Run the full test suite, including CUDA AMP tests when CUDA is available.
3. Reuse the post-PR-#3 Mobility q0/q3 Ridge artifact, or refit it on train with validation-only regularization selection. Ridge does not depend on the C spatial architecture.
4. Run C smoke, then `scripts/micro_overfit_spatial.py` on exactly two TRAIN samples. Inspect correction/ideal RMS, cosine, and gradient groups.
5. Run a seed-123 short C gate for 5–10 epochs. If validation remains pinned to Ridge, stop immediately; do not spend 30 epochs.
6. Only if C improves clearly should it continue toward the spatial gate (`<=-19.0 dB` enters Full development; `<=-19.5 dB` is strong). Do not run Full before that decision.

Formal mode rejects AMP. Smoke/dev AMP is supported through an FP32 complex island. No development command authorizes test.

Pre-fix q0/q1 Mobility priors remain invalid. Post-PR-#3 q0/q3 Ridge artifacts remain valid, while pre-attention q0/q3 C/Full checkpoints are rejected by the new spatial protocol guard. Canonical residual blocks remain post-activation; true residual is an explicit ablation rather than part of this repair.
