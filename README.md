# PriST-RIS V3.1

PriST-RIS is a standalone PyTorch project for prior-guided spatio-temporal RIS channel reconstruction. Architecture version **3.1** replaces the V3.0 flattened RIS axis with an explicitly validated physical `16×2` observation grid and a strong three-dimensional progressive backbone.

V3.0 evidence remains reproducible at commit `f10c90ecd1f3bb4d3764e9aa709db9843be0f995`; V3.1 does not overwrite its runs, checkpoints, Ridge artifacts, or logs. V3.1 checkpoints carry `architecture_version="3.1"` and ordinary resume/evaluate paths reject V3.0 checkpoints.

## Canonical model ladder

| Key | Physical-grid strong spatial | Dual Ridge prior | Coordinate encoding | Trend temporal |
|---|---:|---:|---:|---:|
| `prist_ris_a` | yes | no | no | no |
| `prist_ris_b` | yes | yes | no | no |
| `prist_ris_c` | yes | yes | yes | no |
| `prist_ris_full` | yes | yes | yes | yes |

Canonical V3.1 contains no cross-attention and accepts no old `v3_c_prior_crossattn` alias. Mobility A/B/C output the two spatial anchors A0/A1. Full returns `[B,6,256,64,2]` with exact `q0=A0`, `q1=A1`; only q2–q5 use `Delta=A1-A0`, trend-conditioned low-rank reconstruction, and optional future-only correction. Quasi returns one spatial anchor and does not pretend to pretrain forecasting.

## Frozen data and metric

- Quasi: `[B,1,32,64,2] -> [B,1,256,64,2]`.
- Mobility: `[B,2,32,64,2] -> [B,6,256,64,2]` within each sample.
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
  --output reports/generated/v31_data_audit.json

prist-ris fit-prior \
  --domain mobility \
  --data-root "$PRIST_RIS_DATA_ROOT" \
  --max-train 4096 --max-validation 1800 \
  --workers 8 --batch-size 64 --eval-batch-size 64 \
  --output artifacts/v31_ridge_mobility_dev4096.npz

prist-ris train \
  --domain mobility --model prist_ris_c --mode smoke --seed 123 \
  --prior artifacts/v31_ridge_mobility_dev4096.npz \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cuda \
  --workers 8 --batch-size 16 --eval-batch-size 32 \
  --run-name v31_mobility_C_smoke --output-root runs/v3_1_dev

prist-ris train \
  --domain mobility --model prist_ris_full --mode smoke --seed 123 \
  --prior artifacts/v31_ridge_mobility_dev4096.npz \
  --data-root "$PRIST_RIS_DATA_ROOT" --device cuda \
  --workers 8 --batch-size 16 --eval-batch-size 32 \
  --run-name v31_mobility_full_smoke --output-root runs/v3_1_dev
```

`audit` reads train and validation only. `evaluate --split test` remains protected by the exact freeze-manifest path/hash gate. Formal training remains FP32; AMP is development-only and its complex reconstruction runs inside an explicit FP32/complex64 island.

See `docs/` for the data, model, development, formal, transfer, isolation, and provenance contracts.
