# V3.2 development protocol

Development is validation-only and writes new runs under `runs/v3_2_dev/` or names prefixed `v32_`. Existing runs, Ridge files, checkpoints, and diagnostics are not deleted or overwritten.

Sequence:

1. Run audit and confirm all four train/validation sources, keys, shapes, and provenance.
2. Run the full CPU suite and the CUDA suite on GPU 3 when available.
3. Reuse the post-PR-#3 Mobility q0/q3 Ridge artifact, or refit it on train with validation-only regularization selection. Ridge does not depend on the C spatial architecture.
4. Verify B starts exactly at Ridge and run the synthetic/gradient/anchor-isolation tests.
5. Run only Mobility B, seed 123, mode dev, FP32, and stop after epoch 5. Inspect feature scales, learned correction, and validation NMSE.
6. If B fails to learn a useful residual over Ridge, stop. Do not run C or Full. Later stages require a separate evidence-based decision.

Formal mode rejects AMP. Smoke/dev AMP is supported through an FP32 complex island. No development command authorizes test.

Pre-fix q0/q1 Mobility priors remain invalid. Post-fix q0/q3 Ridge artifacts remain valid because data semantics are unchanged. All V3.1 model checkpoints are rejected by architecture and spatial protocol guards. Canonical residual blocks are now scaled true residuals.
