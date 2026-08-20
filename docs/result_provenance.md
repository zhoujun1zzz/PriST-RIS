# V3.1 result provenance

Every V3.1 metadata file, model config, checkpoint, profile, and report records `architecture_version="3.1"`. Post-fix Mobility checkpoints additionally record `mobility_contract_version="mobility_q0_q3_v1"`, the semantics hash, and q0/q3 data semantics. C/Full checkpoints record `spatial_protocol_version="physical_obsdense_attn_v1"`; resume, evaluation, ablation references, and freeze reject pre-attention C/Full checkpoints. The q0/q3 Ridge artifact remains reusable because it depends on data semantics rather than this spatial protocol. Quasi spatial transfer records every loaded/skipped/new key and only treats current-protocol C/Full as attention-bearing sources.

Run directories retain command, normalized config, metadata, data semantics, exact sample indices, Ridge path/hash, best/last checkpoints, optimizer, RNG and DataLoader-generator states, history, parameter accounting, and validation diagnostics. All smoke/dev results record `test_split_used=false`.

Mobility diagnostics contain q0–q5 linear NMSE/dB, q0/q3 pilot-anchor aggregate, q1/q2/q4/q5 non-pilot aggregate, and overall. Compact spatial diagnostics retain semantic labels q0 and q3. Overall remains the energy ratio over all selected tensors per sample; it is not the mean of per-query dB.

Profiles report architecture and spatial protocol versions, input/output shapes, all physical stage shapes, coordinate state, observed-dense attention state/heads, spatial residual style, prior-anchor count, temporal rank, parameters, trainable parameters, GMACs/GFLOPs, batch-1 latency, and CUDA peak allocated memory.

V3.0, pre-fix Mobility V3.1, and pre-attention q0/q3 C/Full rows remain preserved as legacy provenance and are not mixed into current-protocol validation tables. Synthetic tests establish contracts only and are not evidence of real-data NMSE improvement.
