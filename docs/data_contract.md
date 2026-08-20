# PriST-RIS V3.2 data contract

V3.2 does not change the q0/q3 data semantics or semantics hash; it changes only the model-side spatial learning contract.

`resolve_dataset_source()` is the single source resolver used by Dataset, audit, and DataLoader. Every resolved source records domain, split, path, input/target keys, and provenance.

Quasi train uses `input_da/output_da`. Quasi validation first prefers a separate validation file with `Yd/Hd`, then falls back to `input_da_test/output_da_test` in the train HDF5. Quasi test uses a separate `Yd/Hd` file. Mobility uses `Yd/Hd` for all splits.

Raw shapes are strict:

- Quasi input `[2,32,64,N]`, target `[2,256,64,N]`.
- Mobility input `[4,32,64,N]`, target `[12,256,64,N]`.
- Input/target sample counts must match.
- Mobility counts are train 20,000, validation 1,800, test 9,000.
- Loaded samples must be finite.

Mobility uses grouped complex packing and the canonical pilot-time contract is
`obs_time_index=(0,3)` with `query_time=(0,1,2,3,4,5)`. The two observations
are sparse pilots at q0 and q3 inside one frame. q1/q2 lie between the pilots;
q4/q5 lie after the second pilot. This is frame-internal reconstruction, not
next-frame prediction or extrapolation from consecutive q0/q1 observations.

The 32 observed positions must be exactly `range(0,256,8)`. Row-major `index=16*row+column` maps them to rows 0–15 and columns `{0,8}`. `observations_to_physical_grid()` validates this before producing `[B,4,64,16,2]`. Quasi pads its absent second time block with zeros. `observation_mask` is not a model input.

Audit excludes test by default and reports `raw_input_shape`, `raw_target_shape`, keys, path, provenance, sample count, and semantics hash. Changing q0/q1 to q0/q3 changes the Mobility semantics hash, so pre-fix Ridge artifacts cannot be loaded by post-fix training.
