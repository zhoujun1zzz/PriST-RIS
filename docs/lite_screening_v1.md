# PriST-RIS-Lite V1 screening protocol

## Scope

This workflow asks only how far a prior-guided spatial residual backbone that
retains the S3-selected RIS-coordinate and SE mechanisms can be compressed. It
does not replace the frozen PriST-RIS model, change S3 or the
paper matrix, search additional candidates, or implement Temporal-Lite.

The only candidates are:

| Candidate | Hidden | Blocks per stage | Final refine blocks |
|---|---:|---|---:|
| Lite-A | 32 | 2,2,1 | 1 |
| Lite-B | 40 | 2,2,1 | 1 |

Both use Mobility q0/q3, a fraction-matched Ridge prior, direct-add RIS
coordinates, no antenna index,
no attention, no multiscale supervision, SE channel attention, and scaled true
residual blocks. Optimization is fixed to AdamW, LR `5e-4`, weight decay
`1e-5`, cosine to `5e-6`, 100 epochs, `min_epochs=101`, patience 15, batch
32/64, seed 123, and FP32.

## Provenance gate

Planning requires a canonical nested seed-123 sample manifest whose 25% entry
contains exactly 5,000 unique TRAIN indices. The Ridge artifact must record the
same manifest SHA256, ordered-index hash, fraction, sample count, Mobility
semantics hash, TRAIN fit split, VALIDATION selection split, and
`test_split_used=false`. A full-data or other-fraction prior is rejected.

`lite-screen --action plan` runs on CPU, creates deterministic experiment specs
and exact commands, and records Git HEAD plus both artifact hashes. It also
profiles batch-1 FP32 parameters, trainable parameters, MACs/GMACs, and
FLOPs/GFLOPs for both candidates.

The profiles cover only the q0/q3 spatial path. Therefore: spatial-only GMAC
cannot yet be claimed as an apples-to-apples end-to-end comparison with LPAN-L.
An end-to-end paper comparison requires a later unified spatial plus temporal
profile.

## Execution and recovery

`--action run` executes Lite-A then Lite-B serially. CUDA execution requires
`CUDA_VISIBLE_DEVICES` to match `--physical-gpu-index`, explicit
`--confirm-gpu-free`, and an empty foreign-compute-process query. Exact completed
runs are reused. An incomplete directory is rejected unless
`--resume-incomplete` is supplied and an exact-spec last checkpoint exists.
Nothing is deleted or overwritten automatically.

Every run inherits the existing training engine's exact experiment spec,
history, best/last checkpoints, configuration, semantics, prior, sample-index,
Git and timing provenance. TEST remains false.

## Summary and selection

`--action summarize` reports validation performance, best and last epochs,
wall time, complexity, and first crossings at -18, -19, and -20 dB. It reports
pairwise performance/complexity dominance but leaves `winner=null`.

The predeclared priorities are:

1. minimize spatial parameters and GMACs;
2. maximize 25% canonical VALIDATION performance;
3. prefer convergence efficiency.

If both candidates are poor, the workflow stops. Adding Lite-C requires a new
human decision and is not automatic.
