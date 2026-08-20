# V3.2 result provenance

Every V3.2 metadata file, model config, checkpoint, profile, and report records `architecture_version="3.2"` and `spatial_protocol_version="physical_stable_residual_v2"`. Mobility also records the unchanged `mobility_contract_version="mobility_q0_q3_v1"`, semantics hash, and q0/q3 data semantics. Resume, evaluation, transfer, ablation, and freeze reject V3.1 model checkpoints. The q0/q3 Ridge artifact remains reusable because it depends on unchanged data semantics rather than the model architecture.

Run directories retain command, normalized config, metadata, data semantics, exact sample indices, Ridge path/hash, best/last checkpoints, optimizer, RNG and DataLoader-generator states, history, parameter accounting, and validation diagnostics. All smoke/dev results record `test_split_used=false`.

Mobility diagnostics contain q0–q5 linear NMSE/dB, q0/q3 pilot-anchor aggregate, q1/q2/q4/q5 non-pilot aggregate, and overall. Compact spatial diagnostics retain semantic labels q0 and q3. Overall remains the energy ratio over all selected tensors per sample; it is not the mean of per-query dB.

Profiles report architecture and spatial protocol versions, input/output shapes, all physical stage shapes, coordinate state, observed-dense attention state/heads, spatial residual style, prior-anchor count, temporal rank, parameters, trainable parameters, GMACs/GFLOPs, batch-1 latency, and CUDA peak allocated memory. Spatial diagnostics additionally expose the RMS scales along the repaired prior-guided path.

V3.0 and V3.1 rows remain preserved as legacy provenance and are not mixed into V3.2 validation tables. Synthetic tests establish functional contracts only and are not evidence of real-data NMSE improvement.
