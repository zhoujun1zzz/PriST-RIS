# Development protocol

Development never uses test data.

1. Run `audit`; confirm the four train/validation files and semantics hashes.
2. Run the complete unit suite.
3. Fit Ridge candidates on train and select regularization on validation.
4. Run `profile` for shape, parameters, GMACs/GFLOPs, batch-1 latency, and CUDA peak memory.
5. Use `train --mode smoke` for a one-epoch 64/16-sample integration check.
6. Use `train --mode dev` for at most 4,096 train samples and 1,800 validation samples. It runs 30 epochs and extends to 45 only when the best epoch lies in 26-30.

AMP results are development diagnostics and are never treated as formal results. Formal runs reject AMP. The loss is sample-level linear NMSE plus a small Charbonnier term; ranking always uses validation linear NMSE.

Tests cover the frozen shapes, grouped-complex mapping, progressive widths, factorized branches, Ridge round-trip, observed-only attention, temporal alignment/ranks, target non-leakage, parameter ceiling, metric contract, test lock, checkpoint metadata, tiny overfit, and bitwise deterministic resume.
