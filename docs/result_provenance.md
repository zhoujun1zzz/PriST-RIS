# V3.1 result provenance

Every V3.1 metadata file, model config, checkpoint, profile, and report records `architecture_version="3.1"`. Post-fix Mobility checkpoints additionally record `mobility_contract_version="mobility_q0_q3_v1"`, the new semantics hash, and q0/q3 data semantics. Resume, evaluation, ablation references, and freeze reject pre-fix Mobility checkpoints. Quasi spatial transfer remains the only special loading path because the Quasi spatial contract did not change; it records every loaded/skipped/new key.

Run directories retain command, normalized config, metadata, data semantics, exact sample indices, Ridge path/hash, best/last checkpoints, optimizer, RNG and DataLoader-generator states, history, parameter accounting, and validation diagnostics. All smoke/dev results record `test_split_used=false`.

Mobility diagnostics contain q0–q5 linear NMSE/dB, q0/q3 pilot-anchor aggregate, q1/q2/q4/q5 non-pilot aggregate, and overall. Compact spatial diagnostics retain semantic labels q0 and q3. Overall remains the energy ratio over all selected tensors per sample; it is not the mean of per-query dB.

Profiles report architecture version, input/output shapes, all physical stage shapes, coordinate state, prior-anchor count, temporal rank, parameters, trainable parameters, GMACs/GFLOPs, batch-1 latency, and CUDA peak allocated memory.

V3.0 and pre-fix Mobility V3.1 rows remain preserved as legacy provenance and are not mixed into post-fix validation tables.
