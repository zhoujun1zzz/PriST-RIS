# Independent-test isolation

This gate applies unchanged to PriST-RIS V3.2. Freeze rejects checkpoints whose `architecture_version` is not exactly `3.2` or whose spatial protocol is not `physical_stable_residual_v2`; Mobility checkpoints must also carry the post-fix `mobility_q0_q3_v1` contract and matching semantics hash.

Test is locked by default at three layers:

1. Dataset construction rejects `split=test` unless explicitly authorized.
2. CLI audit excludes test by default.
3. CLI evaluation requires a valid freeze manifest and the exact frozen checkpoint path and SHA-256.

Before freezing, all tuning, early stopping, ablation, PEFT, baseline import, and reporting are validation-only. To freeze, commit the final code and run `prist-ris freeze` with every final checkpoint/prior and optional imported baseline manifest. Test unlock additionally requires `--unlock-test-after-freeze --confirm FREEZE_PRIOR_AND_MODELS`.

Changing one byte of the checkpoint invalidates authorization. A test result is for final reporting only and cannot trigger a new model, hyperparameter, epoch, or protocol choice.
