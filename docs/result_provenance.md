# V3.1 result provenance

Every V3.1 metadata file, model config, checkpoint, profile, and report records `architecture_version="3.1"`. Resume and evaluation require an exact version match; V3.0 checkpoints are not silently loaded. Spatial transfer is the only special loading path and records every loaded/skipped/new key.

Run directories retain command, normalized config, metadata, data semantics, exact sample indices, Ridge path/hash, best/last checkpoints, optimizer, RNG and DataLoader-generator states, history, parameter accounting, and validation diagnostics. All smoke/dev results record `test_split_used=false`.

Mobility diagnostics contain q0–q5 linear NMSE/dB, q0/q1 observed-anchor aggregate, q2–q5 future aggregate, and overall. Overall remains the energy ratio over all selected tensors per sample; it is not the mean of per-query dB.

Profiles report architecture version, input/output shapes, all physical stage shapes, coordinate state, prior-anchor count, temporal rank, parameters, trainable parameters, GMACs/GFLOPs, batch-1 latency, and CUDA peak allocated memory.

V3.0 rows remain preserved as legacy provenance and are not mixed into V3.1 validation tables.
