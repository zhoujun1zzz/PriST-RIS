# PriST-RIS V3.2

PriST-RIS is a standalone PyTorch project for prior-guided spatio-temporal RIS channel reconstruction. Architecture version **3.2** repairs the spatial learning path while retaining the validated physical grid and Mobility q0/q3 data semantics.

V3.0 evidence remains reproducible at commit `f10c90ecd1f3bb4d3764e9aa709db9843be0f995`; legacy runs and checkpoints are not overwritten. V3.2 checkpoints carry `architecture_version="3.2"` and `spatial_protocol_version="physical_stable_residual_v2"`. Mobility retains `mobility_contract_version="mobility_q0_q3_v1"` and its existing semantics hash, so post-fix q0/q3 Ridge artifacts remain reusable. V3.1 model checkpoints are rejected because their learned upsampling, prior mixing, residual, and correction-head contracts differ.

## Canonical model ladder

| Key | Physical grid | Ridge prior | Coordinates | Observed→dense attention | Trend temporal |
|---|---:|---:|---:|---:|---:|
| `prist_ris_a` | yes | no | no | no | no |
| `prist_ris_b` | yes | yes | no | no | no |
| `prist_ris_c` | yes | yes | yes | yes | no |
| `prist_ris_full` | yes | yes | yes | yes | yes |

C/Full use physical-coordinate-aware residual cross-attention on the shared observation feature before per-anchor prior fusion; antennas are never flattened into one global attention sequence. Mobility A/B/C return a compact two-anchor tensor whose semantic times are q0/q3. B/C/Full start exactly at the Ridge prediction because their correction heads are zero initialized. Full returns `[B,6,256,64,2]` in q0..q5 order with exact `q0=A0` and `q3=A3`; the temporal path is unchanged in this repair.

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

See `docs/` for the data, model, development, formal, transfer, isolation, and provenance contracts.
