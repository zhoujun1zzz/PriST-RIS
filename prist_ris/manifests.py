from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    DataSemantics,
    METRIC_CONTRACT,
)
from .engine import require_checkpoint_contract


METHOD_NAMES = {
    "interpolation": "Interpolation",
    "ridge": "Ridge",
    "edsr_lite": "EDSR-lite",
    "spatial_gcn": "Spatial GCN fixed",
    "cnn_gru": "CNN-GRU fixed",
    "gcn_gru": "GCN-GRU fixed",
    "lpan_l_progressive": "LPAN-L progressive",
    "lpan_progressive": "LPAN progressive",
    "phymeta_stgt": "PhyMeta-STGT V1",
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _checkpoint_semantics(path: Path, domain: str) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata missing: {path}")
    expected = DataSemantics.for_domain(domain)
    checks = {
        "domain": (metadata.get("domain"), domain),
        "complex_layout": (metadata.get("complex_layout"), expected.complex_layout),
        "obs_time_index": (metadata.get("obs_time_index"), list(expected.obs_time_index)),
        "obs_ris_index": (metadata.get("obs_ris_index"), list(expected.obs_ris_index)),
    }
    mismatches = {key: (actual, wanted) for key, (actual, wanted) in checks.items() if actual != wanted}
    if mismatches:
        raise ValueError(f"External checkpoint semantics mismatch: {mismatches}")
    return {"validation_source": "checkpoint_metadata", "semantics_hash": expected.stable_hash()}


def import_baseline_manifest(
    source: str | Path,
    output: str | Path,
    *,
    require_checkpoints: bool = False,
) -> dict[str, object]:
    payload = _read_json(source)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Source manifest must contain a results list.")
    if payload.get("protocol") != "v1_repair_compact":
        raise ValueError("Only the audited v1_repair_compact manifest is accepted.")
    imported = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        status = raw.get("status")
        if status not in {"reused", "rerun"}:
            continue
        domain = str(raw.get("domain"))
        if domain not in {"quasi", "mobility"}:
            raise ValueError(f"Invalid baseline domain: {domain}")
        model = str(raw.get("model"))
        if model not in METHOD_NAMES:
            continue
        checkpoint_value = raw.get("checkpoint")
        checkpoint = Path(str(checkpoint_value)).expanduser().resolve() if checkpoint_value else None
        semantics = DataSemantics.for_domain(domain)
        validation_source = "formal_manifest_contract"
        if checkpoint is not None:
            if not checkpoint.is_file():
                if require_checkpoints:
                    raise FileNotFoundError(f"External checkpoint missing: {checkpoint}")
            else:
                validated = _checkpoint_semantics(checkpoint, domain)
                validation_source = str(validated["validation_source"])
        metric = raw.get("validation_metric")
        if not isinstance(metric, dict):
            raise ValueError(f"Baseline {raw.get('id')} lacks a validation metric.")
        imported.append(
            {
                "id": raw.get("id"),
                "method": METHOD_NAMES[model],
                "model_key": model,
                "domain": domain,
                "seed": raw.get("seed"),
                "source_project": "PhyMeta-STGT-LPAN",
                "source_commit": raw.get("source_commit"),
                "source_run": raw.get("source_run"),
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "validation_nmse_linear": metric.get("sample_level_linear_nmse"),
                "validation_nmse_db": metric.get("nmse_db"),
                "metric_contract": METRIC_CONTRACT,
                "data_semantics_hash": semantics.stable_hash(),
                "input_shape": list(semantics.input_shape),
                "output_shape": list(semantics.target_shape),
                "obs_ris_index": list(semantics.obs_ris_index),
                "complex_layout": semantics.complex_layout,
                "training_reused": True,
                "test_artifact": None,
                "semantics_validation_source": validation_source,
            }
        )
    result = {
        "schema": "prist_ris.external_baselines.v1",
        "source_manifest": str(Path(source).resolve()),
        "source_manifest_sha256": sha256(source),
        "metric_contract": METRIC_CONTRACT,
        "test_information_imported": False,
        "results": imported,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def current_commit(project: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, check=True, capture_output=True, text=True
    ).stdout.strip()


def freeze_experiment(
    output: str | Path,
    *,
    project: Path,
    checkpoints: list[Path],
    prior_paths: list[Path],
    baseline_manifest: Path | None,
    unlock_test_after_freeze: bool,
) -> dict[str, object]:
    artifacts = []
    for kind, paths in (("checkpoint", checkpoints), ("prior", prior_paths)):
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Cannot freeze missing {kind}: {resolved}")
            if kind == "checkpoint":
                state = torch.load(resolved, map_location="cpu", weights_only=False)
                config = state.get("model_config")
                domain = config.get("domain") if isinstance(config, dict) else None
                require_checkpoint_contract(
                    state, "Freeze", expected_domain=str(domain)
                )
            artifacts.append({"kind": kind, "path": str(resolved), "sha256": sha256(resolved)})
    baseline = None
    if baseline_manifest is not None:
        resolved = baseline_manifest.expanduser().resolve()
        imported = _read_json(resolved)
        if imported.get("test_information_imported") is not False:
            raise ValueError("Baseline manifest must not carry test-selection information.")
        baseline = {"path": str(resolved), "sha256": sha256(resolved)}
    frozen = {
        "schema": "prist_ris.frozen_experiment.v1",
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
        "commit": current_commit(project),
        "artifacts": artifacts,
        "baseline_manifest": baseline,
        "architecture_frozen": True,
        "hyperparameters_frozen": True,
        "ablations_frozen": True,
        "peft_protocol_frozen": True,
        "test_unlocked": bool(unlock_test_after_freeze),
        "test_unlock_is_not_selection": True,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


def validate_test_unlock(freeze_manifest: str | Path, checkpoint: str | Path) -> dict[str, object]:
    frozen = _read_json(freeze_manifest)
    if (
        frozen.get("schema") != "prist_ris.frozen_experiment.v1"
        or frozen.get("architecture_version") != ARCHITECTURE_VERSION
        or frozen.get("mobility_contract_version") != MOBILITY_CONTRACT_VERSION
        or frozen.get("test_unlocked") is not True
    ):
        raise PermissionError("Independent test is locked until a valid frozen experiment manifest unlocks it.")
    resolved = str(Path(checkpoint).expanduser().resolve())
    expected_hash = None
    for artifact in frozen.get("artifacts", []):
        if isinstance(artifact, dict) and artifact.get("kind") == "checkpoint" and artifact.get("path") == resolved:
            expected_hash = artifact.get("sha256")
            break
    if expected_hash is None or expected_hash != sha256(resolved):
        raise PermissionError("Checkpoint is not the exact frozen artifact authorized for test.")
    return frozen
