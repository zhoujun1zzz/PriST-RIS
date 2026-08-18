# Result provenance

Each run directory is immutable unless explicitly resumed. It contains:

```text
command.txt
config.json
metadata.json
checkpoints/{best_checkpoint,last_checkpoint}.pth
results/{training_history.csv,final_result.json}
manifests/{data_semantics,sample_indices,prior}.json
```

Checkpoints include model and optimizer state, completed epoch, best validation linear NMSE, normalized model/training configs, exact data semantics/hash, Ridge metadata/path/hash, Python/NumPy/PyTorch/CUDA RNG state, DataLoader generator state, staleness counter, and history. Resume requires exact equality of config, semantics, and prior metadata.

The profile convention is batch-1 FP32 single forward, with one MAC equal to two FLOPs. Parameters and trainable parameters are exact. Convolution and linear operations are hook-counted consistently; latency is a warm-start median and CUDA peak memory is reported when applicable.

Formal result summaries should state method, domain, seed, source commit, checkpoint hash, prior hash, validation/test status, linear NMSE, dB conversion, parameter counts, device, and wall-clock time. Smoke results are labeled `smoke_test` and must never be presented as formal performance.
