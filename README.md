# PriST-RIS

PriST-RIS (Prior-guided Structured Progressive Spatio-Temporal RIS reconstruction) is a standalone PyTorch research project for RIS channel reconstruction. It does not import or modify the earlier V1/LPAN repositories.

The canonical code identifiers are `prist_ris_a`, `prist_ris_b`, `prist_ris_c`, and `prist_ris_full`. Legacy `v3_*` strings are accepted only as input aliases and are normalized immediately; checkpoints, reports, and metadata always record **PriST-RIS**.

## Frozen tasks

| Domain | Observation | Prediction |
|---|---:|---:|
| Quasi-static | `[B,1,32,64,2]` | `[B,1,256,64,2]` |
| Mobility | `[B,2,32,64,2]` | `[B,6,256,64,2]` |

Complex values use grouped real/imaginary storage. Observed RIS indices are exactly `0,8,...,248`; the 256 RIS elements use `index = 16*row + column`. Mobility is an in-sample two-observed-block to six-query-block task. NMSE is computed per sample in linear scale, averaged, and converted to dB once.

## Architecture

- Structured progressive spatial reconstruction expands only the RIS axis: `32 -> 64 -> 128 -> 256`.
- Every local block separates a depthwise `1x3` RIS branch and a depthwise `3x1` antenna branch.
- A train-only Ridge artifact supplies an explicit dense spatial anchor.
- One optional four-head observed-to-dense residual cross-attention layer uses 32 observed K/V tokens and 256 dense Q tokens.
- Rank-2 or rank-3 complex temporal factorization aligns queries 0/1 to observed blocks and predicts future blocks without target access.
- A compact depthwise residual corrector refines the temporal output.

## Install and verify

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

CUDA is recommended for training. Formal runs are FP32; `--amp` is accepted only for smoke/development runs.

## Safe local workflow

Set the dataset root to a directory containing the existing `risce` and `risce-0` trees (or use the layouts documented in `docs/data_contract.md`).

```powershell
prist-ris audit --data-root D:\data
prist-ris fit-prior --domain quasi --data-root D:\data --output artifacts\ridge_quasi.npz
prist-ris fit-prior --domain mobility --data-root D:\data --output artifacts\ridge_mobility.npz
prist-ris profile --domain mobility --model prist_ris_full --device cuda
prist-ris train --domain mobility --model prist_ris_full --mode smoke --data-root D:\data --prior artifacts\ridge_mobility.npz
```

`audit` reads train and validation only. `evaluate --split test` is rejected unless an exact checkpoint hash is present in a valid freeze manifest with test explicitly unlocked.

## Experiment commands

```text
prist-ris audit
prist-ris profile
prist-ris fit-prior
prist-ris train
prist-ris tune
prist-ris ablate
prist-ris transfer
prist-ris import-baselines
prist-ris freeze
prist-ris evaluate
prist-ris report
```

Planning commands are non-destructive unless `--execute` is supplied. Formal three-seed commands are generated/run by `scripts/run_formal_protocol.py`. No long formal training is launched by repository setup.

## Reproducibility and provenance

Each run stores the invoked command, normalized config, semantics hash, sample indices, prior path/hash, parameter accounting, training history, best/last checkpoints, RNG state, DataLoader generator state, and validation result. Resume rejects config, prior, or data-semantics mismatches. A graceful `--stop-after-epoch` supports deterministic preemption without changing the frozen training config.

See `docs/` for the data, model, development, formal, baseline-reuse, PEFT, test-isolation, and result-provenance contracts.
