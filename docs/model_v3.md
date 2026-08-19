# PriST-RIS V3.1 model contract

Architecture version is `3.1`. Default spatial configuration is hidden 80, progressive block depths `[3,3,4]`, and four dense final-refinement blocks.

The backbone treats dimensions as antenna-index × RIS-row × RIS-column. A learned column upsampler performs `16×2 -> 16×4 -> 16×8 -> 16×16` without changing the 64 antenna-index or 16 RIS-row axes. Strong residual blocks use two full `3×3×3` Conv3d layers and no fixed 0.1 residual scale or BatchNorm.

Coordinate-enabled C/Full adds two independent encoders at the observed grid and every progressive stage:

- RIS coordinates use normalized physical row and the actual stage columns `{0,8}`, `{0,4,8,12}`, even columns, or all columns.
- Antenna encoding is explicitly named **antenna index encoding**; no physical BS geometry is claimed.

Mobility Ridge predicts A0/A3 from the two observed pilot blocks. A shared anchor feature layer and separate anchor heads produce learned residuals; B/C/Full add them to the dual prior. Their compact `[B,2,256,64,2]` output positions mean q0 and q3, not tensor-time q0 and q1. Quasi uses only A0.

Full preserves q0=A0 and q3=A3 exactly. For non-pilot q1/q2/q4/q5 it encodes A0, A3, and Delta=A3−A0, retains their separate pooled contexts, and uses normalized time `alpha=(t−0)/(3−0)`. The baseline is `A0 + alpha*Delta`; rank-2/3 complex residuals and learned trend adjustments are added on top. The optional physical-grid correction applies only to q1/q2/q4/q5. Results are scattered into strict q0..q5 order instead of concatenating compact anchors before temporal predictions.

Protocol metadata records `spatial_anchor_time_index=[0,3]` and either compact
`output_time_index=[0,3]` for A/B/C or `[0,1,2,3,4,5]` for Full. Training and
evaluation align compact prediction positions to target tensors by these
semantic time indices.

Canonical V3.1 contains no cross-attention. A/B/C are spatial models; Full is the temporal model.
