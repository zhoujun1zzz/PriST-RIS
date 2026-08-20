# V3.2 formal and ablation protocol

Formal work is not authorized by this repair. It may start only after the Mobility B seed-123 five-epoch validation gate demonstrates a useful learned correction, followed by an explicit decision. It remains FP32, validation-selected, and freeze-before-test.

Spatial and temporal ablations are separate tables:

- Spatial scope is q0/q3 only: A physical grid, B dual Ridge, C coordinates plus physical observed-to-dense residual attention. Report q0, q3, and the energy-correct pilot-anchor aggregate.
- Temporal scope is q0–q5: static last-anchor, no-delta conditioning, trend-conditioned low rank, and Full with non-pilot correction. Report every query, q1/q2/q4/q5 aggregate, and overall.

No score column mixes a q0/q3-only model with q0–q5 overall. Any V3.1 model reference is invalid; only a future `physical_stable_residual_v2` reference may be reused. Existing q0/q3 Ridge artifacts remain valid.

Targeted tuning and the three-seed runner remain available but are not launched by code delivery. Formal checkpoints, priors, protocols, and code commit must be frozen before the independent test gate can be opened.
