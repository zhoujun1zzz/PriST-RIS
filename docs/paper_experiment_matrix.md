# Paper experiment matrix

The paper matrix validates the frozen D1 + RISCoord + SE spatial model. It does not search architecture, optimizer, scheduler, temporal rank, or loss weights. Every planner, runner, prior artifact, run spec, and summary is TRAIN/VALIDATION-only and records `test_split_used=false`.

## Frozen protocol

- Spatial: hidden 80, blocks `(3,3,2)`, one final refinement block, direct-add RIS coordinates, no antenna encoding, no observed-dense attention, no multi-scale supervision, gated SE enabled.
- Training: AdamW, learning rate `5e-4`, weight decay `1e-5`, cosine schedule to `5e-6`, exactly 100 FP32 epochs, batch 32, validation batch 64.
- Temporal: preferred formulation is T2 trend + learned residual. The final T2 checkpoint and S3 anchor cache remain intentionally unbound until the final validation run is available.
- Transfer: a final Quasi D1 + RISCoord + SE cosine checkpoint must be supplied explicitly at execution time. Planning does not substitute an older checkpoint.

The first plan creates `runs/paper_matrix/manifests/frozen_protocol.json`. An existing manifest or subset manifest is reused only when its content is identical.

## Plan

Planning writes deterministic nested subset manifests, fraction-specific Ridge jobs, exact experiment specs, commands, dependencies, sample counts, and a phase-specific plan. It does not inspect data, initialize CUDA, fit a prior, or start training.

```bash
prist-ris paper-matrix --action plan \
  --phase data-efficiency --seeds 123 \
  --output-root runs/paper_matrix

prist-ris paper-matrix --action plan \
  --phase transfer --seeds 123 \
  --output-root runs/paper_matrix
```

Seed 123 schedules:

- Data efficiency: 4 fractions × Direct-S3/Prior-S3 = 8 GPU runs.
- Transfer: 3 fractions × scratch/full_finetune/selective = 9 GPU runs.

The CPU Ridge jobs are dependencies and are not counted as GPU runs.

## Data-budget fairness

For every seed and fraction, Direct-S3 and Prior-S3 use the same deterministic subset manifest. Prior-S3 receives a separate Ridge artifact fitted on that exact TRAIN subset. The artifact records seed, fraction, sample count, manifest SHA256, ordered-index hash, Mobility semantics hash, TRAIN fit split, VALIDATION selection split, and TEST=false. An existing artifact with any mismatch is rejected rather than overwritten.

The formal Mobility fractions are 0.10/0.25/0.50/1.00 of 20,000 samples. Transfer uses 0.05/0.10/0.25. Fractions are nested independently for seeds 123, 456, and 789.

## Run

Formal execution is intentionally separate from planning. Run only on the server after checking GPU3 and supplying the real data path. Transfer additionally requires the final frozen Quasi checkpoint.

```bash
export CUDA_VISIBLE_DEVICES=3
nvidia-smi -i 3

prist-ris paper-matrix --action run \
  --phase data-efficiency --seeds 123 \
  --data-root /home/zhoujunyi/datasets/lpan \
  --device cuda:0 --workers 8 \
  --physical-gpu-index 3 --confirm-gpu-free \
  --output-root runs/paper_matrix

prist-ris paper-matrix --action run \
  --phase transfer --seeds 123 \
  --quasi-checkpoint /path/to/final_quasi_D1_RISCoord_SE_cosine.pth \
  --data-root /home/zhoujunyi/datasets/lpan \
  --device cuda:0 --workers 8 \
  --physical-gpu-index 3 --confirm-gpu-free \
  --output-root runs/paper_matrix
```

The runner is serial and rechecks GPU3 before each GPU experiment. Foreign compute PIDs cause a safe stop. A completed run is reused. An existing incomplete directory is rejected unless `--resume-incomplete` is explicit and the spec, training config, subset, prior, checkpoint, semantics, optimizer, scheduler, RNG, and DataLoader state all match.

## Summarize

```bash
prist-ris paper-matrix --action summarize \
  --phase all --final-spatial-run /path/to/final_S3_cosine_run \
  --output-root runs/paper_matrix
```

Outputs under `runs/paper_matrix/summaries/`:

- `summary.json`
- `data_efficiency.csv`
- `convergence_efficiency.csv`
- `transfer.csv`
- `complexity.csv`

Convergence uses P1 training histories and extracts the first epoch and cumulative wall-clock time reaching -18, -19, and -20 dB. NMSE reaches a threshold when it is less than or equal to that threshold. Seed aggregates report mean, sample standard deviation, and n. Spatial q0/q3 evidence is never ranked together with temporal q0..q5 evidence.

## Current unresolved dependency

The final temporal T2 checkpoint and its final S3 spatial-anchor cache are pending. The frozen manifest records this explicitly; no old T2, T3, or T4 artifact is silently promoted. The paper matrix planner remains usable, while any future action that requires the final temporal binding must reject the null binding.
