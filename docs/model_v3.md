# PriST-RIS V3.1 model contract

Architecture version is `3.1`. Default spatial configuration is hidden 80, progressive block depths `[3,3,4]`, and four dense final-refinement blocks.

The backbone treats dimensions as antenna-index × RIS-row × RIS-column. A learned column upsampler performs `16×2 -> 16×4 -> 16×8 -> 16×16` without changing the 64 antenna-index or 16 RIS-row axes. Strong residual blocks use two full `3×3×3` Conv3d layers and no fixed 0.1 residual scale or BatchNorm.

Coordinate-enabled C/Full adds two independent encoders at the observed grid and every progressive stage:

- RIS coordinates use normalized physical row and the actual stage columns `{0,8}`, `{0,4,8,12}`, even columns, or all columns.
- Antenna encoding is explicitly named **antenna index encoding**; no physical BS geometry is claimed.

Mobility Ridge predicts A0/A1 from both observed pilot blocks. A shared anchor feature layer and separate anchor heads produce learned residuals; B/C/Full add them to the dual prior. Quasi uses only A0.

Full preserves q0=A0 and q1=A1 exactly. For q2–q5 it encodes the spatial tensors A0, A1, and Delta=A1−A0, retains their separate pooled contexts, combines them with query time, and predicts rank-2/3 complex residuals plus learned trend coefficients. The optional physical-grid correction is applied only to q2–q5.

Canonical V3.1 contains no cross-attention. A/B/C are spatial models; Full is the temporal model.
