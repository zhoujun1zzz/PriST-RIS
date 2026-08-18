# Quasi to Mobility transfer and PEFT

The transfer study uses one structurally compatible Quasi `prist_ris_full` checkpoint. Quasi and Mobility share the same four-channel input adapter, hidden width, stage depths, attention heads, and temporal rank. Target-only scratch therefore differs in initialization/protocol, not architecture.

Mobility train fractions are 1%, 5%, 10%, 20%, and 100%. A seed-123 permutation creates nested subsets of 200, 1,000, 2,000, 4,000, and 20,000 samples. The manifest records every exact index.

Five protocols are run at every fraction:

- `target_only_scratch`
- `full_finetune`
- `frozen_spatial`
- `selective`
- `adapter_only`

Pretrained protocols must load the same Quasi source checkpoint; structural mismatches are fatal. Every run reports total/trainable parameter counts, trainable names, and top-level trainable modules. Selection and comparison use Mobility validation only. `prist-ris transfer` writes the 25-run plan; add `--execute` to run it.
