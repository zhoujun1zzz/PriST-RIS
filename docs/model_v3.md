# PriST-RIS V3.2 model contract

Architecture version is `3.2`. Default spatial configuration is hidden 80, progressive block depths `[3,3,4]`, and four dense final-refinement blocks.

The backbone treats dimensions as antenna-index × RIS-row × RIS-column. Parameter-free nearest-neighbor column expansion performs `16×2 -> 16×4 -> 16×8 -> 16×16` without changing the 64 antenna-index or 16 RIS-row axes. Strong residual blocks use two full `3×3×3` Conv3d layers and no BatchNorm. Canonical `spatial_residual_style="scaled_true_residual"` computes `value+0.1*GELU(body(value))`; `post_activation` is retained only as a legacy ablation.

Position semantics are split into six explicit flags: `backbone_ris_coordinate_enabled`, `backbone_antenna_index_enabled`, `attention_enabled`, `attention_ris_coordinate_enabled`, `attention_antenna_index_enabled`, plus `backbone_ris_coordinate_mode=off|direct_add|zero_init_gated`. The legacy `coordinate_enabled` flag remains only as a recorded compatibility alias and must not be mixed with explicit position flags.

- RIS coordinates use normalized physical row and the actual stage columns `{0,8}`, `{0,4,8,12}`, even columns, or all columns.
- Antenna encoding is explicitly named **antenna index encoding**; no physical BS geometry is claimed.

Backbone RIS coordinates may be directly added or multiplied by one learned scalar per injection stage. In `zero_init_gated` mode all four gates start at zero, so the initial backbone is exactly the position-blind B backbone. Gate gradients open first; RIS projection gradients appear after a gate becomes nonzero. Attention RIS coordinates are independent of backbone coordinates. P3 therefore keeps the existing Mobility q0/q3 aggregation into 32 observed RIS tokens per antenna while enabling only RIS-coordinate-aware observed/dense attention.

Mobility Ridge predicts A0/A3 from the two observed pilot blocks. The shared `Conv3d(2,H,1)` prior encoder is applied separately to each anchor, so q0 features never read q3 prior channels and vice versa. A true residual refiner and a separate correction head are used per anchor. B/C/Full correction heads are zero initialized, making their initial output exactly Ridge while allowing head gradients on step one and upstream gradients after the first update. Their compact `[B,2,256,64,2]` output positions mean q0 and q3. Quasi uses only A0.

C and Full insert one `PhysicalObservedDenseResidualAttention` on the shared observation feature before any per-anchor prior fusion. Dense features plus dense 16×16 row/column coordinates form `Q=[B*64,256,H]`. The q0/q3 complex observations, physical observed coordinates, semantic pilot-time descriptors, and antenna-index encoding form `K/V=[B*64,32,H]`. Attention is computed independently per antenna and never consumes `target_h`.

Full preserves q0=A0 and q3=A3 exactly. For non-pilot q1/q2/q4/q5 it encodes A0, A3, and Delta=A3−A0, retains their separate pooled contexts, and uses normalized time `alpha=(t−0)/(3−0)`. The baseline is `A0 + alpha*Delta`; rank-2/3 complex residuals and learned trend adjustments are added on top. The optional physical-grid correction applies only to q1/q2/q4/q5. Results are scattered into strict q0..q5 order instead of concatenating compact anchors before temporal predictions.

Protocol metadata records `spatial_anchor_time_index=[0,3]` and either compact
`output_time_index=[0,3]` for A/B/C or `[0,1,2,3,4,5]` for Full. Training and
evaluation align compact prediction positions to target tensors by these
semantic time indices.

Metadata records `spatial_protocol_version="physical_stable_residual_position_v3"`, `position_semantics_version="physical_ris_decoupled_v1"`, every resolved position flag/mode, legacy-alias use, deterministic nearest column expansion, independent shared-weight prior fusion, zero-initialized prior correction heads, attention scope, and residual style. Feature diagnostics report observation input, backbone output, raw/encoded prior, fused/refined feature, predicted delta, ideal residual, and their ratios.
