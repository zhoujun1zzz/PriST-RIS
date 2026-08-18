# Baseline reuse

PriST-RIS does not silently retrain trusted V1 baselines. `import-baselines` accepts only an audited `v1_repair_compact` JSON manifest and statuses `reused` or `rerun`. Supported method keys are explicitly allow-listed.

Every imported row records source project/commit/run/checkpoint, validation metric, input/output shapes, RIS indices, complex layout, metric contract, and semantics hash. When a checkpoint is present, its embedded semantics are verified; `--require-checkpoints` makes missing files fatal. No test metric or test-selection artifact is imported.

Example:

```powershell
prist-ris import-baselines --source external\v1_manifest.json --output external_results\baseline_manifest.json --require-checkpoints
```

The imported manifest is itself hashed into the final freeze manifest. Baseline rows are provenance records, not authorization to modify the V1 repository or runs.
