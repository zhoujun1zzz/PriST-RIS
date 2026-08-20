# PriST-RIS V3.1 model contract

Architecture version is `3.1`. Default spatial configuration is hidden 80, progressive block depths `[3,3,4]`, and four dense final-refinement blocks.

The backbone treats dimensions as antenna-index × RIS-row × RIS-column. A learned column upsampler performs `16×2 -> 16×4 -> 16×8 -> 16×16` without changing the 64 antenna-index or 16 RIS-row axes. Strong residual blocks use two full `3×3×3` Conv3d layers and no BatchNorm. Canonical `spatial_residual_style="post_activation"` retains `GELU(value+body(value))`; `scaled_true_residual` is exposed only as a controlled ablation and computes `value+0.1*GELU(body(value))`.

Coordinate-enabled C/Full adds two independent encoders at the observed grid and every progressive stage:

- RIS coordinates use normalized physical row and the actual stage columns `{0,8}`, `{0,4,8,12}`, even columns, or all columns.
- Antenna encoding is explicitly named **antenna index encoding**; no physical BS geometry is claimed.

Mobility Ridge predicts A0/A3 from the two observed pilot blocks. A shared anchor feature layer and separate anchor heads produce learned residuals; B/C/Full add them to the dual prior. Their compact `[B,2,256,64,2]` output positions mean q0 and q3, not tensor-time q0 and q1. Quasi uses only A0.

C and Full insert one `PhysicalObservedDenseResidualAttention` after physical-backbone and Ridge feature fusion, before the shared anchor feature/head path. Dense features plus dense 16×16 row/column coordinates form `Q=[B*64,256,H]`. The q0/q3 complex observations, physical observed row/column coordinates, semantic pilot-time descriptors, and antenna-index encoding form `K/V=[B*64,32,H]`. Four-head attention is computed independently per antenna and added as `features + 0.1*attention_delta`; it never replaces the backbone and never consumes `target_h`. The same module supports Quasi's single q0 pilot.

Full preserves q0=A0 and q3=A3 exactly. For non-pilot q1/q2/q4/q5 it encodes A0, A3, and Delta=A3−A0, retains their separate pooled contexts, and uses normalized time `alpha=(t−0)/(3−0)`. The baseline is `A0 + alpha*Delta`; rank-2/3 complex residuals and learned trend adjustments are added on top. The optional physical-grid correction applies only to q1/q2/q4/q5. Results are scattered into strict q0..q5 order instead of concatenating compact anchors before temporal predictions.

Protocol metadata records `spatial_anchor_time_index=[0,3]` and either compact
`output_time_index=[0,3]` for A/B/C or `[0,1,2,3,4,5]` for Full. Training and
evaluation align compact prediction positions to target tensors by these
semantic time indices.

Metadata records `spatial_protocol_version="physical_obsdense_attn_v1"`, the per-antenna `32_to_256` scope, four heads, residual scale 0.1, target usage false, and the selected spatial residual style. A/B do not instantiate observed-dense attention; C and Full do, and Full obtains q0/q3 from that same spatial path before temporal reconstruction.
