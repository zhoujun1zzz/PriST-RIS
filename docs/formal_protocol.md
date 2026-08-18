# Formal protocol

Formal training starts only after development checks pass. All selection remains validation-only and FP32.

Targeted tuning at seed 123:

1. Capacity: hidden 64/80/96, LR `5e-4`, rank 2, 30 epochs; rank by median validation linear NMSE over epochs 25-30, then best value.
2. Learning rate: `2e-4`, `5e-4`, `1e-3` at the selected width, 40 epochs; rank over epochs 31-40.
3. Mobility temporal rank: 2/3 at the selected width/LR, 40 epochs; rank over epochs 31-40.

The selected configuration is trained with seeds 123/456/789, maximum 100 epochs, minimum 40 epochs, patience 15, AdamW weight decay `1e-5`, gradient clipping 1.0, and FP32. Mobility batch size is fixed at 32; Quasi batch size may be independently benchmarked and is recorded. `scripts/run_formal_protocol.py --dry-run` emits the complete command set without training.

Mechanism ablation uses Mobility seed 123 and the same frozen optimization settings. The full reference checkpoint is reused rather than retrained. The five trained comparisons cover RIS-only control, structured progressive, prior guidance, cross-attention, and low-rank temporal without the compact residual.

Only after checkpoints, priors, baseline manifest, architecture, hyperparameters, ablation definitions, and PEFT protocol are frozen may test be unlocked. Test evaluation is a single reporting stage and must not feed back into selection.
