# V3.1 formal and ablation protocol

Formal work starts only after the seed-123 validation gate. It remains FP32, validation-selected, and freeze-before-test.

Spatial and temporal ablations are separate tables:

- Spatial scope is q0/q3 only: A physical grid, B dual Ridge, C coordinate encoding. Report q0, q3, and the energy-correct pilot-anchor aggregate.
- Temporal scope is q0–q5: static last-anchor, no-delta conditioning, trend-conditioned low rank, and Full with non-pilot correction. Report every query, q1/q2/q4/q5 aggregate, and overall.

No score column mixes a q0/q3-only model with q0–q5 overall. Any reference created under pre-fix q0/q1 semantics is invalid; only a new post-fix Full seed-123 reference may later be reused.

Targeted tuning and the three-seed runner remain available but are not launched by code delivery. Formal checkpoints, priors, protocols, and code commit must be frozen before the independent test gate can be opened.
