# V3.1 formal and ablation protocol

Formal work starts only after the seed-123 validation gate. It remains FP32, validation-selected, and freeze-before-test.

Spatial and temporal ablations are separate tables:

- Spatial scope is q0/q1 only: A physical grid, B dual Ridge, C coordinate encoding. Report q0, q1, and energy-correct q0/q1 aggregate.
- Temporal scope is q0–q5: static last-anchor, no-delta conditioning, trend-conditioned low rank, and Full with future correction. Report every query, q2–q5 aggregate, and overall.

No score column mixes a q0/q1-only model with q0–q5 overall. The Full seed-123 reference is reused, not silently retrained.

Targeted tuning and the three-seed runner remain available but are not launched by code delivery. Formal checkpoints, priors, protocols, and code commit must be frozen before the independent test gate can be opened.
