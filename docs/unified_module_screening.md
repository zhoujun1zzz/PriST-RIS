# Unified S2/S3 and T0-T4 screening protocol

This framework adds optional, factor-isolated experiments without changing the canonical PriST-RIS V3.2 architecture or invalidating the completed D1 + RISCoord q0/q3 spatial checkpoint.

## Frozen reference and module definitions

S0 reuses the completed D1 + RISCoord result: validation NMSE `-20.25387859912702 dB`, best epoch 74, early-stop epoch 89. It is not retrained as a 30-epoch baseline.

- S2 enables `--spatial-multiscale-supervision`. The shared prior encoder, refiners, and heads supervise physical widths 4, 8, and 16. It adds no inference parameters or deployment graph nodes.
- S3 sets `--spatial-channel-attention se`. Zero-gated 3D SE preserves the baseline at initialization and changes the inference graph after training.
- S23 enables S2 and S3 together.
- T0 is a TRAIN/VALIDATION-only audit of adjacent differences, correlations, first differences, and second differences.
- T1 is an untrained deterministic linear trend through the cached q0/q3 spatial anchors.
- T2 adds a zero-output-initialized learned residual to T1.
- T3 adds delta loss with screening weight 0.1 and the same inference graph as T2.
- T4 adds curvature loss with screening weights delta=0.1 and curvature=0.05. It is scheduled only with `--include-curvature`.

The independent temporal switches are `--temporal-base-mode static|linear_trend`, `--temporal-learned-residual-enabled/--no-temporal-learned-residual-enabled`, `--temporal-delta-loss-weight`, and `--temporal-curvature-loss-weight`. The legacy `--temporal-mode` remains only for checkpoint and command compatibility.

## Exact commands

```bash
export PRIST_RIS_DATA_ROOT=/root/autodl-tmp/lpan
export PRIOR=/path/to/v31_q0q3_ridge_mobility_dev4096.npz
export SPATIAL_CHECKPOINT=/path/to/D1_RISCoord/checkpoints/best_checkpoint.pth
export ANCHOR_CACHE_ROOT=/path/to/cache/d1_riscoord_seed123
export CUDA_VISIBLE_DEVICES=3

# Optional CPU-friendly T0 audit.
prist-ris audit-temporal \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cpu --workers 8 \
  --max-train 4096 --max-validation 1800 \
  --output-json reports/generated/t0_temporal_audit.json \
  --output-csv reports/generated/t0_temporal_audit.csv

# Reuse the completed spatial checkpoint; do not fit Ridge again.
prist-ris cache-spatial-anchors \
  --checkpoint "$SPATIAL_CHECKPOINT" --prior "$PRIOR" \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cuda --workers 8 \
  --max-train 4096 --max-validation 1800 \
  --output-root "$ANCHOR_CACHE_ROOT"

# S2, S3, S23: serial 30-epoch runs with deterministic 30-to-40 decisions.
prist-ris screen-spatial-modules \
  --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8 --physical-gpu-index 3 \
  --execute --confirm-gpu-free

# T1 plus T2/T3 and optional T4 from the fixed cache.
prist-ris screen-temporal \
  --prior "$PRIOR" --spatial-checkpoint "$SPATIAL_CHECKPOINT" \
  --anchor-cache-root "$ANCHOR_CACHE_ROOT" \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cuda --workers 8 \
  --physical-gpu-index 3 --include-curvature \
  --execute --confirm-gpu-free
```

Run the complete queue in a persistent server session with:

```bash
nohup bash scripts/run_unified_screening_gpu3.sh \
  > unified_s2_s3_t0_t4_gpu3.log 2>&1 &
```

The launcher requires the four environment variables shown above. It displays GPU3, refuses to start when that GPU already has a compute PID, creates the cache only when no completed cache manifest exists, then runs spatial and temporal screens serially.

## Fixed continuation rules

The first invocation for each trained candidate uses `--epochs 40 --stop-after-epoch 30`, guaranteeing an exact epoch-30 checkpoint while keeping the resume configuration fixed. It resumes to epoch 40 if either condition is true:

1. best epoch at 30 is at least 26; or
2. validation NMSE improves by at least 0.05 dB from epoch 21 to epoch 30 and the best value is no worse than reference + 0.30 dB.

The summary recommends a 100-epoch follow-up only if the epoch-40 best occurs at or after epoch 36 or is within 0.10 dB of the reference. Recommendations do not execute by default. `--run-long-followups` explicitly permits serial resume of only the recommended candidates.

## Isolation and artifacts

The cache contains only q0/q3 predictions and sample indices. Its manifest records checkpoint SHA256, Ridge SHA256, data-semantics hash, spatial and position protocol versions, spatial supervision mode, channel-attention mode, and `target_cached=false`. Targets remain in the original TRAIN/VALIDATION datasets and are read only for temporal loss and metrics.

Every run records module switches, temporal weights, protocol versions, model-selection split, and `test_split_used=false` in configs, checkpoints, metadata, profiles, and summaries. Spatial ranking remains q0/q3-only; temporal ranking remains q0..q5-only. The runners reuse completed runs, reject any existing incomplete run directory, preserve optimizer/RNG/DataLoader generator state on resume, and never remove historical artifacts.
