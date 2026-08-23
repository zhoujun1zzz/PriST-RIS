# PriST-RIS Temporal-Lite V1 protocol

## Fixed scientific scope

Temporal-Lite does not change the frozen Full PriST-RIS S3/T2 conclusion. It
tests whether the learned correction around the existing deterministic linear
trend can be compressed after the spatial model has already been fixed to
Lite-A.

V1 contains exactly one candidate, TL24:

- Lite-A spatial backbone: hidden 32, blocks 2/2/1, final refine 1;
- the full-data prior-guided Lite-A spatial model with direct-add RIS
  coordinates and SE;
- independent temporal hidden 24;
- deterministic linear-trend base plus a learned bounded trend-coefficient
  correction and rank-2 complex residual;
- no `FutureResidualCorrection`;
- no delta or curvature auxiliary loss;
- no architecture, width, rank, loss, or scheduler search.

The new `temporal_hidden` setting is backward compatible. When absent or null,
the effective temporal width remains equal to spatial `hidden`, preserving old
Full checkpoint state-dict shapes and numerical semantics.

Lite-A is not the hidden-80 Prior-S3 model. It retains the RIS-coordinate and
SE mechanisms selected during the full-model spatial study, but uses the frozen
Lite-A hidden-32, blocks-2/2/1 architecture.

For a non-pilot time `t`, TL24 implements

```text
H_hat_t = A0
          + (alpha_t + Delta_alpha_phi(t)) (A3 - A0)
          + sum_{r=1}^2 c_phi,r(t) B_phi,r
```

where `alpha_t` is the deterministic normalized time coordinate and
`Delta_alpha_phi(t) = 0.25 * tanh(alpha_head(...))`. In the compact paper form
`H_hat_t = H_trend_t + R_phi(t)`, the learned correction is
`R_phi(t) = Delta_alpha_phi(t)(A3-A0) + sum c_phi,r(t)B_phi,r`; it is not only
the rank-2 basis term.

## Provenance and complexity gates

Planning validates all of the following before producing commands:

- canonical nested seed-123 full TRAIN manifest with exactly 20,000 indices;
- fraction-1.00 Ridge fitted from the identical ordered subset;
- exact full-data Lite-A checkpoint and its explicit sample-index provenance;
- matching artifact SHA256 values and canonical Mobility semantics;
- `test_split_used=false` throughout.

Planning runs only on CPU. It loads the Lite-A checkpoint into a real
`prist_ris_full` graph and profiles batch-1 FP32 q0-q5 inference. Historical
`parameters`, `macs`, `gmacs`, `flops`, and `gflops` retain their neural-only
meaning. Temporal-Lite additionally derives Ridge storage and compute from the
validated artifact's complex coefficient shape and the canonical Mobility
input shape.

One complex Ridge coefficient counts as two real scalar parameter equivalents;
one complex multiply-accumulate counts as four real MACs. The budget gate uses
`total_parameter_real_equivalents = neural_parameters +
prior_parameter_real_equivalents` and `total_gmacs = neural_gmacs +
prior_gmacs`. Training is blocked unless these totals are below 1,112,904 and
6.337769472 respectively. A failed gate stops for human review; it never
changes width 24 or starts a search automatically.

## Formal execution

The fixed training protocol is temporal-only, seed 123, full TRAIN 20,000,
VALIDATION 1,800, batch 16/32, AdamW at `5e-4` with weight decay `1e-5`, fixed
learning rate, 30 epochs, `min_epochs=31`, patience 15, and FP32.

`temporal-lite --action run` performs one-GPU serial execution:

1. strict GPU preflight;
2. build or exactly reuse Lite-A q0/q3 TRAIN/VALIDATION anchor caches;
3. evaluate deterministic T1-Lite on VALIDATION;
4. train TL24 only if the complexity gate passed;
5. evaluate the exact best checkpoint from raw observations plus Ridge prior.

The cache stores anchors and sample indices, never targets. Its manifest records
the spatial checkpoint, Ridge, semantics, complete spatial configuration,
sample provenance, split counts, and TEST isolation. Exact completed artifacts
may be reused. Partial artifacts are rejected, and TL24 resumes only with an
explicit flag, an exact recorded spec, and its last checkpoint.

```bash
export CUDA_VISIBLE_DEVICES=0
prist-ris temporal-lite --action run \
  --sample-index-manifest "$FULL_SAMPLE_MANIFEST" \
  --prior "$FULL_RIDGE" --spatial-checkpoint "$LITE_A_FULL_CHECKPOINT" \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cuda:0 --workers 8 \
  --physical-gpu-index 0 --confirm-gpu-free \
  --output-root runs/temporal_lite_v1
```

No command reads TEST. No formal training starts during planning.

## Summary

Summaries report T1-Lite and TL24 q0-q5 VALIDATION diagnostics, including
anchor, non-pilot, interpolation, extrapolation, delta, and curvature errors.
TL24 also records best/last epochs, wall time, neural-only and Ridge-inclusive
complexity, LPAN-L budget status, and performance gaps versus the declared
LPAN-L, LPAN, and
`Direct_S3_T2_cache_composed_validation_seed123` references. The latter is a
VALIDATION-only cache composition used solely as a performance-gap reference;
it is not a validated standard end-to-end deployable Direct Full checkpoint.
`winner` remains null and selection is human-only.

Expected layout:

```text
runs/temporal_lite_v1/
  temporal_lite_plan.json
  manifests/specs/{anchor_cache,t1_lite,tl24}.json
  profiles/{t1_lite,tl24}.json
  anchor_cache/{train,validation}.h5
  anchor_cache/cache_manifest.json
  evaluations/{t1_lite_validation,tl24_best_validation}.json
  runs/tl24/
    paper_experiment_spec.json
    checkpoints/{best_checkpoint,last_checkpoint}.pth
    results/{training_history.csv,final_result.json}
  summaries/temporal_lite_summary.json
```
