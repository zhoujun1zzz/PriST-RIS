# PriST-RIS V3.2

PriST-RIS is a standalone PyTorch project for prior-guided spatio-temporal RIS channel reconstruction. Architecture version **3.2** repairs the spatial learning path while retaining the validated physical grid and Mobility q0/q3 data semantics.

V3.0 evidence remains reproducible at commit `f10c90ecd1f3bb4d3764e9aa709db9843be0f995`; legacy runs and checkpoints are not overwritten. Current checkpoints carry `architecture_version="3.2"`, `spatial_protocol_version="physical_stable_residual_position_v3"`, and `position_semantics_version="physical_ris_decoupled_v1"`. Mobility retains `mobility_contract_version="mobility_q0_q3_v1"` and its existing semantics hash, so post-fix q0/q3 Ridge artifacts remain reusable. Older model checkpoints without the decoupled position contract are rejected.

## Canonical model ladder

| Key | Physical grid | Ridge prior | Legacy default position bundle | Observed→dense attention | Trend temporal |
|---|---:|---:|---:|---:|---:|
| `prist_ris_a` | yes | no | no | no | no |
| `prist_ris_b` | yes | yes | no | no | no |
| `prist_ris_c` | yes | yes | yes | yes | no |
| `prist_ris_full` | yes | yes | yes | yes | yes |

C/Full retain their historical coupled default for reproducibility, but new experiments use six explicit switches that independently control backbone RIS coordinates, backbone antenna-index encoding, attention enablement, attention RIS coordinates, and attention antenna-index encoding. The old `coordinate_enabled` option is only a recorded compatibility alias. Antennas are never flattened into one global attention sequence. Mobility A/B/C return compact q0/q3 anchors; Full returns strict q0..q5 order.

## Frozen data and metric

- Quasi: `[B,1,32,64,2] -> [B,1,256,64,2]`.
- Mobility: `[B,2,32,64,2] -> [B,6,256,64,2]` within each sample.
- Mobility pilot times: q0 and q3; all six queries belong to the same frame.
- Observed RIS indices: `0,8,...,248`, validated as `(row=0..15, col={0,8})`.
- Physical progression: `16×2 -> 16×4 -> 16×8 -> 16×16`; antenna-index axis remains 64.
- Raw complex layout: grouped real channels followed by grouped imaginary channels.
- Main metric: per-sample linear NMSE, sample mean, then one dB conversion.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Validation-only development

```bash
export PRIST_RIS_DATA_ROOT=/root/autodl-tmp/lpan

prist-ris audit \
  --data-root "$PRIST_RIS_DATA_ROOT" \
  --output reports/generated/v31_q0q3_data_audit.json

prist-ris fit-prior \
  --domain mobility \
  --data-root "$PRIST_RIS_DATA_ROOT" \
  --max-train 4096 --max-validation 1800 \
  --workers 8 --batch-size 64 --eval-batch-size 64 \
  --output artifacts/v31_q0q3_ridge_mobility_dev4096.npz

prist-ris train \
  --domain mobility --model prist_ris_b --mode dev --seed 123 \
  --prior artifacts/v31_q0q3_ridge_mobility_dev4096.npz \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cuda \
  --workers 8 --batch-size 16 --eval-batch-size 32 \
  --epochs 30 --min-epochs 1 --patience 15 --stop-after-epoch 5 \
  --run-name v32_stable_mobility_B_gate5 --output-root runs/v3_2_dev
```

`audit` reads train and validation only. The first real-data gate is Mobility B, seed 123, FP32, stopped after epoch 5. Inspect Ridge equality at initialization, correction/ideal scale, validation NMSE, and gradient flow. Do not launch C or Full if B does not learn a useful residual. `evaluate --split test` remains protected by the exact freeze-manifest path/hash gate. No real-data improvement is claimed by the synthetic tests in this repository.

## S1 spatial capacity/depth screening

`screen-spatial` generates the fixed validation-only B64, B48, D1, and D2 plan without changing the canonical model. Every candidate is forced to stop at epoch 30 even when the general dev runner would otherwise consider extending a late-improving run.

```bash
export PRIOR=artifacts/v31_q0q3_ridge_mobility_dev4096.npz

# Inspect the exact commands without starting training.
prist-ris screen-spatial \
  --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8

# On the shared server, inspect GPU 3 before starting the serial queue.
nvidia-smi -i 3
CUDA_VISIBLE_DEVICES=3 nohup prist-ris screen-spatial \
  --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8 --execute --confirm-gpu-free \
  > s1_spatial_screen.log 2>&1 &
```

The queue runs one candidate at a time, invokes `nvidia-smi -i 3` before each candidate, refuses to overwrite incomplete runs, saves per-candidate profiles, and produces `summary.json` with best/last validation NMSE, best epoch, q0/q3, late-window improvement, parameters, GMAC/GFLOP, latency, peak memory, wall time, and the accuracy–GMAC Pareto frontier. Supply `--reference-run` and `--reference-profile` to compute reductions relative to the existing B80 result. TEST is never read.

## P1-P3 position semantics screen

`screen-position` generates three serial, fixed 30-epoch validation runs: P1 adds only backbone RIS coordinates by direct addition; P2 adds only zero-initialized gated backbone RIS coordinates; P3 enables observed-to-dense attention with RIS coordinates while leaving both backbone position paths and all antenna-index paths off. It does not schedule P4 or change the canonical model.

```bash
prist-ris screen-position \
  --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8

nvidia-smi -i 3
CUDA_VISIBLE_DEVICES=3 prist-ris screen-position \
  --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT" \
  --device cuda --workers 8 --execute --confirm-gpu-free
```

See `docs/` for the data, model, development, formal, transfer, isolation, and provenance contracts.

## Unified S2/S3 and T0-T4 screening

The optional module framework keeps the V3.2 spatial checkpoint contract intact. S2 adds shared-head supervision at physical widths 4/8/16 and is training-only; S3 adds gated 3D SE and changes the inference graph. The temporal path independently selects a static or deterministic linear-trend base, an optional learned residual, and optional delta/curvature losses.

```bash
# Plan only; no data or GPU work is started.
prist-ris screen-spatial-modules --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT"
prist-ris screen-temporal --prior "$PRIOR" \
  --spatial-checkpoint "$SPATIAL_CHECKPOINT" \
  --anchor-cache-root "$ANCHOR_CACHE_ROOT" \
  --data-root "$PRIST_RIS_DATA_ROOT" --include-curvature

# TRAIN/VALIDATION-only temporal statistics and fixed spatial-anchor cache.
prist-ris audit-temporal --data-root "$PRIST_RIS_DATA_ROOT"
prist-ris cache-spatial-anchors --checkpoint "$SPATIAL_CHECKPOINT" \
  --prior "$PRIOR" --data-root "$PRIST_RIS_DATA_ROOT" \
  --max-train 4096 --max-validation 1800 --output-root "$ANCHOR_CACHE_ROOT"
```

Both runners execute candidates serially for exactly 30 epochs. A candidate extends to 40 when its best epoch is at least 26, or when epoch 21 to 30 improves by at least 0.05 dB and its best score is within 0.30 dB of the fixed reference. A 100-epoch continuation is only recommended when the 40-epoch best is at or after epoch 36 or within 0.10 dB of the reference; it runs only with `--run-long-followups`. TEST is never opened by these workflows, the supplied Ridge artifact is never refit, completed runs are reused, and incomplete runs are never overwritten.

See [the unified screening protocol](docs/unified_module_screening.md) and [the GPU3 serial launcher](scripts/run_unified_screening_gpu3.sh) for exact execution commands.

## Paper experiment matrix

The validation-only paper framework generates deterministic data-efficiency and low-shot transfer plans without starting training:

```bash
prist-ris paper-matrix --action plan --phase data-efficiency --seeds 123
prist-ris paper-matrix --action plan --phase transfer --seeds 123
```

It binds Direct/Prior models to identical nested TRAIN subsets, fits each Ridge baseline from the same allowed fraction, records resumable cosine timing, and produces paper-ready JSON/CSV summaries. Formal execution remains an explicit later action and TEST stays locked. See [the paper experiment matrix protocol](docs/paper_experiment_matrix.md).
