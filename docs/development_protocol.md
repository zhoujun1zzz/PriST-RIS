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

## S1 capacity/depth Pareto screen

After the Mobility-B functional gate and any already-running B versus B+Coord late-convergence control, the next independent research question is whether width or full-resolution depth can be reduced without leaving the accuracy–complexity Pareto frontier. The fixed candidates are B64, B48, D1 `(3,3,2)+final1`, and D2 `(2,2,2)+final2`; the existing B80 `(3,3,4)+final4` run is reused as reference rather than retrained.

Use `prist-ris screen-spatial` to inspect the plan and `--execute --confirm-gpu-free` only after `nvidia-smi -i 3` confirms GPU 3 is available. The executable queue requires `CUDA_VISIBLE_DEVICES=3`, is serial, uses seed 123 and TRAIN 4096 / VALIDATION 1800, and forces every screening run to stop at epoch 30. It does not enable AMP, a scheduler, coordinates, attention, temporal prediction, or TEST.

Continue a candidate toward 100 epochs only if epoch 30 is still clearly improving, or if it is within roughly 0.1–0.3 dB of B80 while materially reducing parameters, GMAC, or latency. Stop candidates that are clearly worse and flat, or more expensive without validation gain. S1 only identifies candidates; it does not change the canonical architecture. Multi-scale supervision and other mechanisms remain separate future PRs after the Pareto evidence is available.
