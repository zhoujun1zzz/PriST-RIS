from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import h5py
import numpy as np
import torch

from .anchor_cache import read_anchor_cache_metadata
from .checkpoint import load_checkpoint
from .complexity import profile_model
from .contracts import DataSemantics
from .engine import (
    configure_adaptation,
    load_mobility_spatial_reference,
    require_checkpoint_contract,
)
from .models import PriSTRIS, build_model
from .paper_matrix import gpu_preflight, indices_hash, validate_prior_artifact
from .prior import file_sha256


TEMPORAL_LITE_SCHEMA = "prist_ris.temporal_lite.v1"
TEMPORAL_LITE_SEED = 123
TEMPORAL_LITE_FRACTION = 1.0
TEMPORAL_LITE_TRAIN_COUNT = 20000
TEMPORAL_LITE_VALIDATION_COUNT = 1800
LPAN_L_PARAMETERS = 1_112_904
LPAN_L_GMACS = 6.337769472
REFERENCE_NMSE_DB = {
    "LPAN-L_mean": -21.074314,
    "LPAN_mean": -21.768841,
    "Full_Direct_S3_T2_seed123": -21.992708,
}


@dataclass(frozen=True)
class TemporalLiteCandidate:
    candidate: str = "TL24"
    spatial_hidden: int = 32
    temporal_hidden: int = 24
    blocks_per_stage: tuple[int, int, int] = (2, 2, 1)
    final_refine_blocks: int = 1
    temporal_rank: int = 2
    temporal_residual: bool = False


TL24 = TemporalLiteCandidate()


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise FileExistsError(
                f"Existing Temporal-Lite reproducibility artifact differs: {path}"
            )
        return
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _git_head(project_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_full_sample_manifest(
    path: str | Path, *, seed: int = TEMPORAL_LITE_SEED
) -> dict[str, object]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("test_split_used") is not False:
        raise PermissionError("Temporal-Lite sample manifest must exclude TEST.")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("Temporal-Lite fixes seed=123.")
    if payload.get("nested") is not True:
        raise ValueError("Temporal-Lite requires the canonical nested manifest.")
    if int(payload.get("total_train_samples", -1)) != TEMPORAL_LITE_TRAIN_COUNT:
        raise ValueError("Temporal-Lite requires canonical 20,000-sample TRAIN.")
    fractions = payload.get("fractions")
    indices = fractions.get("1.00") if isinstance(fractions, dict) else None
    if not isinstance(indices, list) or len(indices) != TEMPORAL_LITE_TRAIN_COUNT:
        raise ValueError("Temporal-Lite manifest must contain 20,000 fraction 1.00 indices.")
    normalized = [int(value) for value in indices]
    if len(set(normalized)) != TEMPORAL_LITE_TRAIN_COUNT:
        raise ValueError("Temporal-Lite full TRAIN indices must be unique.")
    if set(normalized) != set(range(TEMPORAL_LITE_TRAIN_COUNT)):
        raise ValueError("Temporal-Lite fraction 1.00 must cover canonical TRAIN exactly.")
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "subset_indices_hash": indices_hash(normalized),
        "sample_count": TEMPORAL_LITE_TRAIN_COUNT,
        "fraction": TEMPORAL_LITE_FRACTION,
        "seed": seed,
        "nested": True,
        "test_split_used": False,
    }


def validate_full_prior(
    path: str | Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    source = Path(path).resolve()
    job = {
        "seed": TEMPORAL_LITE_SEED,
        "fraction": TEMPORAL_LITE_FRACTION,
        "sample_count": TEMPORAL_LITE_TRAIN_COUNT,
        "sample_manifest_sha256": manifest["sha256"],
        "indices_hash": manifest["subset_indices_hash"],
    }
    validate_prior_artifact(source, job)
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "seed": TEMPORAL_LITE_SEED,
        "fraction": TEMPORAL_LITE_FRACTION,
        "sample_count": TEMPORAL_LITE_TRAIN_COUNT,
        "sample_manifest_sha256": manifest["sha256"],
        "subset_indices_hash": manifest["subset_indices_hash"],
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "fit_split": "train",
        "selection_split": "validation",
        "test_split_used": False,
    }


def _require_lite_a_config(config: Mapping[str, object]) -> None:
    checks = {
        "model_key": "prist_ris_b",
        "domain": "mobility",
        "hidden": TL24.spatial_hidden,
        "blocks_per_stage": list(TL24.blocks_per_stage),
        "final_refine_blocks": TL24.final_refine_blocks,
        "backbone_ris_coordinate_enabled": True,
        "backbone_ris_coordinate_mode": "direct_add",
        "backbone_antenna_index_enabled": False,
        "attention_enabled": False,
        "attention_ris_coordinate_enabled": False,
        "attention_antenna_index_enabled": False,
        "spatial_multiscale_supervision": False,
        "spatial_channel_attention": "se",
        "spatial_residual_style": "scaled_true_residual",
    }
    for key, expected in checks.items():
        actual = config.get(key)
        if key == "blocks_per_stage" and isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise ValueError(
                f"Temporal-Lite spatial checkpoint is not exact Lite-A: "
                f"{key}={actual!r}, expected {expected!r}."
            )


def validate_lite_a_checkpoint(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    prior: Mapping[str, object],
) -> dict[str, object]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    state = load_checkpoint(source, torch.device("cpu"))
    require_checkpoint_contract(
        state, "Temporal-Lite spatial reference", expected_domain="mobility"
    )
    model_config = state.get("model_config")
    training_config = state.get("training_config")
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise ValueError("Lite-A checkpoint lacks model/training configuration.")
    _require_lite_a_config(model_config)
    if (
        int(training_config.get("seed", -1)) != TEMPORAL_LITE_SEED
        or training_config.get("mode") != "full"
        or training_config.get("test_split_used") is not False
    ):
        raise ValueError("Lite-A checkpoint is not the seed123 full TRAIN protocol.")
    prior_metadata = state.get("prior_metadata")
    if not isinstance(prior_metadata, dict):
        raise ValueError("Lite-A checkpoint lacks Ridge provenance.")
    required_prior = {
        "sha256": prior["sha256"],
        "sample_manifest_sha256": manifest["sha256"],
        "sample_count": TEMPORAL_LITE_TRAIN_COUNT,
        "fraction": TEMPORAL_LITE_FRACTION,
        "indices_hash": manifest["subset_indices_hash"],
        "test_split_used": False,
    }
    mismatched = {
        key: (prior_metadata.get(key), expected)
        for key, expected in required_prior.items()
        if prior_metadata.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Lite-A checkpoint Ridge provenance mismatch: {mismatched}")
    sample_indices_path = source.parent.parent / "manifests" / "sample_indices.json"
    if not sample_indices_path.is_file():
        raise FileNotFoundError(
            f"Lite-A checkpoint lacks sample-index provenance: {sample_indices_path}"
        )
    sample_payload = json.loads(sample_indices_path.read_text(encoding="utf-8"))
    sample_indices = sample_payload.get("indices")
    if not isinstance(sample_indices, list):
        raise ValueError("Lite-A checkpoint sample-index provenance is not explicit.")
    normalized = [int(value) for value in sample_indices]
    if (
        len(normalized) != TEMPORAL_LITE_TRAIN_COUNT
        or indices_hash(normalized) != manifest["subset_indices_hash"]
    ):
        raise ValueError("Lite-A checkpoint sample indices do not match full manifest.")
    spatial = build_model(**model_config)
    spatial.load_state_dict(state["model_state"])
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "model_config": json.loads(json.dumps(model_config)),
        "training_config": json.loads(json.dumps(training_config)),
        "sample_indices_path": str(sample_indices_path.resolve()),
        "sample_indices_sha256": file_sha256(sample_indices_path),
        "sample_count": len(normalized),
        "subset_indices_hash": indices_hash(normalized),
        "test_split_used": False,
    }


def tl24_model_config(*, learned: bool = True) -> dict[str, object]:
    return {
        "model_key": "prist_ris_full",
        "domain": "mobility",
        "hidden": TL24.spatial_hidden,
        "temporal_hidden": TL24.temporal_hidden if learned else None,
        "blocks_per_stage": TL24.blocks_per_stage,
        "final_refine_blocks": TL24.final_refine_blocks,
        "temporal_rank": TL24.temporal_rank,
        "temporal_residual": False,
        "spatial_multiscale_supervision": False,
        "spatial_channel_attention": "se",
        "backbone_ris_coordinate_enabled": True,
        "backbone_ris_coordinate_mode": "direct_add",
        "backbone_antenna_index_enabled": False,
        "attention_enabled": False,
        "attention_ris_coordinate_enabled": False,
        "attention_antenna_index_enabled": False,
        "spatial_residual_style": "scaled_true_residual",
        "temporal_base_mode": "linear_trend",
        "temporal_learned_residual_enabled": learned,
    }


def build_tl24_model(*, learned: bool = True) -> PriSTRIS:
    return build_model(**tl24_model_config(learned=learned))


def _load_spatial_reference(model: PriSTRIS, checkpoint_path: str | Path) -> None:
    state = load_checkpoint(checkpoint_path, torch.device("cpu"))
    load_mobility_spatial_reference(model, state)


def profile_temporal_lite(
    output_root: str | Path,
    *,
    spatial_checkpoint: str | Path,
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, object]]:
    if device.type != "cpu":
        raise ValueError("Temporal-Lite planning/profile must remain CPU-only.")
    root = Path(output_root).resolve()
    profiles: dict[str, dict[str, object]] = {}
    for name, learned in (("T1-Lite", False), ("TL24", True)):
        model = build_tl24_model(learned=learned).to(device)
        _load_spatial_reference(model, spatial_checkpoint)
        if learned:
            configure_adaptation(model, "temporal_only")
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        result = profile_model(
            model, domain="mobility", device=device, latency_runs=1
        )
        result.pop("latency_ms_batch1", None)
        result.pop("peak_gpu_memory_bytes", None)
        result.update(
            {
                "candidate": name,
                "batch_size": 1,
                "dtype": "FP32",
                "spatial_hidden": TL24.spatial_hidden,
                "temporal_hidden": TL24.temporal_hidden if learned else None,
                "temporal_rank": TL24.temporal_rank if learned else None,
                "temporal_residual_enabled": False,
                "target_scope": "mobility_q0_q5_end_to_end",
                "test_split_used": False,
            }
        )
        if learned:
            result["parameter_budget"] = LPAN_L_PARAMETERS
            result["gmac_budget"] = LPAN_L_GMACS
            result["parameter_budget_pass"] = (
                int(result["parameters"]) < LPAN_L_PARAMETERS
            )
            result["gmac_budget_pass"] = float(result["gmacs"]) < LPAN_L_GMACS
            result["budget_pass"] = bool(
                result["parameter_budget_pass"] and result["gmac_budget_pass"]
            )
        slug = name.lower().replace("-", "_")
        _write_json_exact(root / "profiles" / f"{slug}.json", result)
        profiles[name] = result
    return profiles


def fixed_tl24_training_config() -> dict[str, object]:
    return {
        "domain": "mobility",
        "model_key": "prist_ris_full",
        "mode": "full",
        "seed": TEMPORAL_LITE_SEED,
        "hidden": TL24.spatial_hidden,
        "temporal_hidden": TL24.temporal_hidden,
        "blocks_per_stage": list(TL24.blocks_per_stage),
        "final_refine_blocks": TL24.final_refine_blocks,
        "temporal_rank": TL24.temporal_rank,
        "temporal_base_mode": "linear_trend",
        "temporal_learned_residual_enabled": True,
        "temporal_residual": False,
        "temporal_delta_loss_weight": 0.0,
        "temporal_curvature_loss_weight": 0.0,
        "adaptation": "temporal_only",
        "sample_count": TEMPORAL_LITE_TRAIN_COUNT,
        "validation_count": TEMPORAL_LITE_VALIDATION_COUNT,
        "batch_size": 16,
        "eval_batch_size": 32,
        "workers": 8,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "scheduler": "fixed",
        "epochs": 30,
        "min_epochs": 31,
        "patience": 15,
        "amp": False,
        "formal_fp32": True,
        "test_split_used": False,
    }


def _command(values: list[object]) -> list[str]:
    return ["prist-ris", *(str(value) for value in values)]


def _tl24_training_arguments(
    spec: Mapping[str, object],
    *,
    data_root: str | Path,
    device: str,
    workers: int,
    spec_path: str | Path,
    resume: str | Path | None = None,
) -> list[object]:
    checkpoint = spec["spatial_checkpoint"]
    prior = spec["ridge_prior"]
    assert isinstance(checkpoint, Mapping) and isinstance(prior, Mapping)
    arguments: list[object] = [
        "train", "--domain", "mobility", "--model", "prist_ris_full",
        "--mode", "full", "--seed", TEMPORAL_LITE_SEED,
        "--prior", prior["path"],
        "--spatial-reference-checkpoint", checkpoint["path"],
        "--anchor-cache-root", spec["anchor_cache_root"],
        "--data-root", data_root, "--device", device, "--workers", workers,
        "--batch-size", 16, "--eval-batch-size", 32,
        "--hidden", TL24.spatial_hidden,
        "--temporal-hidden", TL24.temporal_hidden,
        "--blocks-per-stage", "2,2,1", "--final-refine-blocks", 1,
        "--temporal-rank", 2, "--no-temporal-residual",
        "--temporal-base-mode", "linear_trend",
        "--temporal-learned-residual-enabled",
        "--backbone-ris-coordinate-enabled",
        "--backbone-ris-coordinate-mode", "direct_add",
        "--no-backbone-antenna-index-enabled", "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled",
        "--no-attention-antenna-index-enabled",
        "--no-spatial-multiscale-supervision",
        "--spatial-channel-attention", "se",
        "--spatial-residual-style", "scaled_true_residual",
        "--temporal-delta-loss-weight", 0.0,
        "--temporal-curvature-loss-weight", 0.0,
        "--learning-rate", 5e-4, "--weight-decay", 1e-5,
        "--scheduler", "fixed", "--epochs", 30,
        "--min-epochs", 31, "--patience", 15,
        "--adaptation", "temporal_only", "--run-name", spec["run_name"],
        "--output-root", Path(str(spec["run_dir"])).parent,
        "--experiment-spec", spec_path,
    ]
    if resume is not None:
        arguments.extend(("--resume", resume))
    return arguments


def build_temporal_lite_plan(
    output_root: str | Path,
    *,
    spatial_checkpoint_path: str | Path,
    prior_path: str | Path,
    sample_manifest_path: str | Path,
    data_root: str | Path,
    workers: int = 8,
    seed: int = TEMPORAL_LITE_SEED,
    project_root: str | Path | None = None,
    head: str | None = None,
    run_profiles: bool = True,
) -> dict[str, object]:
    if seed != TEMPORAL_LITE_SEED:
        raise ValueError("Temporal-Lite V1 fixes seed=123; no seed search is allowed.")
    if workers != 8:
        raise ValueError("Temporal-Lite V1 fixes workers=8 for formal commands.")
    root = Path(output_root).resolve()
    project = Path(project_root).resolve() if project_root else Path.cwd()
    current_head = head or _git_head(project)
    manifest = validate_full_sample_manifest(sample_manifest_path, seed=seed)
    prior = validate_full_prior(prior_path, manifest)
    checkpoint = validate_lite_a_checkpoint(
        spatial_checkpoint_path, manifest=manifest, prior=prior
    )
    profiles = (
        profile_temporal_lite(
            root, spatial_checkpoint=spatial_checkpoint_path, device=torch.device("cpu")
        )
        if run_profiles
        else {}
    )
    cache_root = (root / "anchor_cache").resolve()
    t1_output = (root / "evaluations" / "t1_lite_validation.json").resolve()
    tl24_run = (root / "runs" / "tl24").resolve()
    best_output = (root / "evaluations" / "tl24_best_validation.json").resolve()
    common = {
        "schema": TEMPORAL_LITE_SCHEMA,
        "git_head": current_head,
        "canonical_mobility_semantics_hash": DataSemantics.for_domain(
            "mobility"
        ).stable_hash(),
        "sample_manifest": manifest,
        "ridge_prior": prior,
        "spatial_checkpoint": checkpoint,
        "test_split_used": False,
    }
    cache_spec = {
        **common,
        "artifact": "Lite-A anchor cache",
        "anchor_cache_root": str(cache_root),
        "train_count": TEMPORAL_LITE_TRAIN_COUNT,
        "validation_count": TEMPORAL_LITE_VALIDATION_COUNT,
        "target_cached": False,
    }
    t1_spec = {
        **common,
        "candidate": "T1-Lite",
        "anchor_cache_root": str(cache_root),
        "output": str(t1_output),
        "temporal_base_mode": "linear_trend",
        "temporal_learned_residual_enabled": False,
        "temporal_residual": False,
    }
    tl24_spec = {
        **common,
        "candidate": "TL24",
        "run_name": "tl24",
        "run_dir": str(tl24_run),
        "anchor_cache_root": str(cache_root),
        "training_config": fixed_tl24_training_config(),
        "test_split_used": False,
    }
    specs = root / "manifests" / "specs"
    cache_spec_path = specs / "anchor_cache.json"
    t1_spec_path = specs / "t1_lite.json"
    tl24_spec_path = specs / "tl24.json"
    for path, value in (
        (cache_spec_path, cache_spec),
        (t1_spec_path, t1_spec),
        (tl24_spec_path, tl24_spec),
    ):
        _write_json_exact(path, value)
    commands = {
        "cache": _command(
            [
                "cache-spatial-anchors", "--checkpoint", checkpoint["path"],
                "--prior", prior["path"], "--sample-index-manifest", manifest["path"],
                "--experiment-spec", cache_spec_path, "--seed", seed,
                "--batch-size", 16, "--max-train", TEMPORAL_LITE_TRAIN_COUNT,
                "--max-validation", TEMPORAL_LITE_VALIDATION_COUNT,
                "--data-root", data_root, "--device", "cuda:0", "--workers", workers,
                "--output-root", cache_root,
            ]
        ),
        "T1-Lite": _command(
            [
                "evaluate-temporal-cache", "--prior", prior["path"],
                "--spatial-checkpoint", checkpoint["path"],
                "--anchor-cache-root", cache_root, "--experiment-spec", t1_spec_path,
                "--seed", seed, "--batch-size", 32, "--data-root", data_root,
                "--device", "cuda:0", "--workers", workers, "--output", t1_output,
            ]
        ),
        "TL24": _command(
            _tl24_training_arguments(
                tl24_spec,
                data_root=data_root,
                device="cuda:0",
                workers=workers,
                spec_path=tl24_spec_path,
            )
        ),
        "TL24-best-validation": _command(
            [
                "evaluate", "--checkpoint", tl24_run / "checkpoints" / "best_checkpoint.pth",
                "--prior", prior["path"], "--split", "validation",
                "--experiment-spec", tl24_spec_path, "--batch-size", 32,
                "--data-root", data_root, "--device", "cuda:0", "--workers", workers,
                "--output", best_output,
            ]
        ),
    }
    tl_profile = profiles.get("TL24", {})
    budget_pass = tl_profile.get("budget_pass") if run_profiles else None
    plan = {
        "schema": TEMPORAL_LITE_SCHEMA,
        "workflow": "PriST-RIS Temporal-Lite V1",
        "candidate": asdict(TL24),
        "architecture_search": False,
        "rank_search": False,
        "loss_search": False,
        "planner_uses_gpu": False,
        "serial_execution": ["anchor-cache", "T1-Lite", "TL24", "TL24-best-validation"],
        "git_head": current_head,
        "output_root": str(root),
        "sample_manifest": manifest,
        "ridge_prior": prior,
        "spatial_checkpoint": checkpoint,
        "cache_spec": cache_spec,
        "t1_spec": t1_spec,
        "tl24_spec": tl24_spec,
        "profiles": profiles,
        "complexity_budget": {
            "reference": "LPAN-L",
            "parameters_less_than": LPAN_L_PARAMETERS,
            "gmacs_less_than": LPAN_L_GMACS,
            "budget_pass": budget_pass,
        },
        "commands": commands,
        "winner": None,
        "human_decision_only": True,
        "test_split_used": False,
    }
    _write_json_exact(root / "temporal_lite_plan.json", plan)
    return plan


def decide_temporal_run(
    spec: Mapping[str, object], *, resume_incomplete: bool
) -> dict[str, object]:
    run = Path(str(spec["run_dir"]))
    final = run / "results" / "final_result.json"
    recorded = run / "paper_experiment_spec.json"
    expected = dict(spec)
    if final.is_file():
        if not recorded.is_file() or json.loads(
            recorded.read_text(encoding="utf-8")
        ) != expected:
            raise ValueError(f"Completed Temporal-Lite run spec mismatch: {run}")
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("test_split_used") is not False:
            raise PermissionError("Completed Temporal-Lite run contains TEST evidence.")
        return {"action": "reuse", "run_dir": str(run)}
    if not run.exists():
        return {"action": "run", "run_dir": str(run)}
    if not resume_incomplete:
        raise FileExistsError(
            f"Incomplete Temporal-Lite run will not be overwritten: {run}"
        )
    if not recorded.is_file() or json.loads(
        recorded.read_text(encoding="utf-8")
    ) != expected:
        raise ValueError(f"Incomplete Temporal-Lite run spec mismatch: {run}")
    checkpoint = run / "checkpoints" / "last_checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Incomplete Temporal-Lite run has no resumable checkpoint: {run}"
        )
    return {"action": "resume", "run_dir": str(run), "checkpoint": str(checkpoint)}


def _validate_cache_file(
    path: Path, *, split: str, expected_count: int, spec: Mapping[str, object]
) -> None:
    metadata = read_anchor_cache_metadata(path)
    checkpoint = spec["spatial_checkpoint"]
    prior = spec["ridge_prior"]
    manifest = spec["sample_manifest"]
    assert isinstance(checkpoint, Mapping)
    assert isinstance(prior, Mapping)
    assert isinstance(manifest, Mapping)
    expected = {
        "split": split,
        "sample_count": expected_count,
        "checkpoint_sha256": checkpoint["sha256"],
        "prior_sha256": prior["sha256"],
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "target_cached": False,
        "test_split_used": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Temporal-Lite cache metadata mismatch: {key}.")
    with h5py.File(path, "r") as handle:
        cached_indices = [
            int(value) for value in np.asarray(handle["sample_index"], dtype=np.int64)
        ]
    if len(set(cached_indices)) != expected_count:
        raise ValueError("Temporal-Lite cache indices must be unique.")
    actual_hash = indices_hash(cached_indices)
    if metadata.get("sample_indices_hash") != actual_hash:
        raise ValueError("Temporal-Lite cache index hash mismatch.")
    if split == "train" and actual_hash != manifest["subset_indices_hash"]:
        raise ValueError("Temporal-Lite TRAIN cache does not match the full manifest.")
    if split == "validation" and set(cached_indices) != set(range(expected_count)):
        raise ValueError("Temporal-Lite VALIDATION cache is not the full split.")
    if metadata.get("sample_provenance") != dict(manifest):
        raise ValueError("Temporal-Lite cache sample provenance mismatch.")
    model_config = metadata.get("spatial_model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Temporal-Lite cache lacks spatial model configuration.")
    _require_lite_a_config(model_config)


def decide_anchor_cache(spec: Mapping[str, object]) -> dict[str, object]:
    root = Path(str(spec["anchor_cache_root"]))
    train = root / "train.h5"
    validation = root / "validation.h5"
    manifest_path = root / "cache_manifest.json"
    existing = [path.exists() for path in (train, validation, manifest_path)]
    if not any(existing):
        return {"action": "run", "root": str(root)}
    if not all(existing):
        raise FileExistsError(f"Incomplete Temporal-Lite anchor cache: {root}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("experiment_spec") != dict(spec) or payload.get(
        "test_split_used"
    ) is not False:
        raise ValueError("Completed Temporal-Lite cache spec mismatch.")
    _validate_cache_file(
        train,
        split="train",
        expected_count=TEMPORAL_LITE_TRAIN_COUNT,
        spec=spec,
    )
    _validate_cache_file(
        validation,
        split="validation",
        expected_count=TEMPORAL_LITE_VALIDATION_COUNT,
        spec=spec,
    )
    return {"action": "reuse", "root": str(root)}


def _decide_evaluation(
    path: str | Path,
    spec: Mapping[str, object],
    *,
    checkpoint_path: str | Path | None = None,
) -> str:
    output = Path(path)
    if not output.exists():
        return "run"
    payload = json.loads(output.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_spec") != dict(spec)
        or payload.get("test_split_used") is not False
    ):
        raise ValueError(f"Completed Temporal-Lite evaluation spec mismatch: {output}")
    if checkpoint_path is not None and payload.get("checkpoint_sha256") != file_sha256(
        checkpoint_path
    ):
        raise ValueError(
            f"Completed Temporal-Lite evaluation checkpoint mismatch: {output}"
        )
    spatial = spec.get("spatial_checkpoint")
    if isinstance(spatial, Mapping) and "spatial_checkpoint_sha256" in payload:
        if payload.get("spatial_checkpoint_sha256") != spatial.get("sha256"):
            raise ValueError(
                f"Completed Temporal-Lite spatial evaluation mismatch: {output}"
            )
    return "reuse"


Invoke = Callable[[list[object]], None]


def execute_temporal_lite_plan(
    plan: Mapping[str, object],
    *,
    data_root: str | Path,
    device: str,
    workers: int,
    physical_gpu_index: int,
    confirm_gpu_free: bool,
    resume_incomplete: bool,
    invoke: Invoke,
) -> dict[str, object]:
    if plan.get("test_split_used") is not False:
        raise PermissionError("Temporal-Lite runner rejects TEST evidence.")
    budget = plan.get("complexity_budget")
    if not isinstance(budget, Mapping) or budget.get("budget_pass") is not True:
        raise RuntimeError(
            "TL24 exceeds or lacks the predeclared LPAN-L complexity budget; "
            "human decision is required and GPU training is blocked."
        )
    profiles = plan.get("profiles")
    tl_profile = profiles.get("TL24") if isinstance(profiles, Mapping) else None
    if (
        not isinstance(tl_profile, Mapping)
        or int(tl_profile.get("parameters", LPAN_L_PARAMETERS)) >= LPAN_L_PARAMETERS
        or float(tl_profile.get("gmacs", LPAN_L_GMACS)) >= LPAN_L_GMACS
        or tl_profile.get("target_scope") != "mobility_q0_q5_end_to_end"
        or tl_profile.get("test_split_used") is not False
    ):
        raise RuntimeError("Temporal-Lite runner rejects invalid TL24 profile evidence.")
    if torch.device(device).type != "cuda":
        raise ValueError("Temporal-Lite formal run requires one CUDA GPU.")
    manifest = plan.get("sample_manifest")
    prior = plan.get("ridge_prior")
    checkpoint = plan.get("spatial_checkpoint")
    if not all(isinstance(value, Mapping) for value in (manifest, prior, checkpoint)):
        raise ValueError("Temporal-Lite plan lacks provenance.")
    current_manifest = validate_full_sample_manifest(manifest["path"])
    current_prior = validate_full_prior(prior["path"], current_manifest)
    current_checkpoint = validate_lite_a_checkpoint(
        checkpoint["path"], manifest=current_manifest, prior=current_prior
    )
    if (
        dict(manifest) != current_manifest
        or dict(prior) != current_prior
        or dict(checkpoint) != current_checkpoint
    ):
        raise ValueError("Temporal-Lite provenance changed after planning.")
    gpu_preflight(
        device=device,
        physical_gpu_index=physical_gpu_index,
        confirm_gpu_free=confirm_gpu_free,
    )
    commands = plan.get("commands")
    cache_spec = plan.get("cache_spec")
    t1_spec = plan.get("t1_spec")
    tl24_spec = plan.get("tl24_spec")
    if not isinstance(commands, Mapping) or not all(
        isinstance(value, Mapping) for value in (cache_spec, t1_spec, tl24_spec)
    ):
        raise ValueError("Temporal-Lite plan is incomplete.")
    statuses: dict[str, str] = {}
    cache_decision = decide_anchor_cache(cache_spec)
    if cache_decision["action"] == "run":
        invoke(list(commands["cache"])[1:])
        if decide_anchor_cache(cache_spec)["action"] != "reuse":
            raise RuntimeError("Temporal-Lite anchor cache did not complete exactly.")
    statuses["anchor-cache"] = str(cache_decision["action"])
    t1_action = _decide_evaluation(t1_spec["output"], t1_spec)
    if t1_action == "run":
        invoke(list(commands["T1-Lite"])[1:])
        if _decide_evaluation(t1_spec["output"], t1_spec) != "reuse":
            raise RuntimeError("T1-Lite evaluation did not complete exactly.")
    statuses["T1-Lite"] = t1_action
    decision = decide_temporal_run(
        tl24_spec, resume_incomplete=resume_incomplete
    )
    if decision["action"] != "reuse":
        spec_path = (
            Path(str(plan["output_root"])) / "manifests" / "specs" / "tl24.json"
        )
        invoke(
            _tl24_training_arguments(
                tl24_spec,
                data_root=data_root,
                device=device,
                workers=workers,
                spec_path=spec_path,
                resume=decision.get("checkpoint"),
            )
        )
        if decide_temporal_run(tl24_spec, resume_incomplete=False)["action"] != "reuse":
            raise RuntimeError("TL24 training did not complete exactly.")
    statuses["TL24"] = str(decision["action"])
    best_output = Path(str(plan["output_root"])) / "evaluations" / "tl24_best_validation.json"
    best_checkpoint = (
        Path(str(tl24_spec["run_dir"])) / "checkpoints" / "best_checkpoint.pth"
    )
    best_action = _decide_evaluation(
        best_output, tl24_spec, checkpoint_path=best_checkpoint
    )
    if best_action == "run":
        invoke(list(commands["TL24-best-validation"])[1:])
        if _decide_evaluation(
            best_output, tl24_spec, checkpoint_path=best_checkpoint
        ) != "reuse":
            raise RuntimeError("TL24 best-checkpoint evaluation did not complete exactly.")
    statuses["TL24-best-validation"] = best_action
    return {
        "workflow": "PriST-RIS Temporal-Lite V1",
        "serial_execution": True,
        "statuses": statuses,
        "budget_pass": True,
        "test_split_used": False,
    }


def _read_history(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _diagnostic_summary(payload: Mapping[str, object]) -> dict[str, object]:
    diagnostics = payload.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    per_query = diagnostics.get("per_query")
    per_query = per_query if isinstance(per_query, Mapping) else {}
    return {
        "overall_validation_nmse_db": payload.get("nmse_db"),
        "per_query": {
            key: value.get("nmse_db") if isinstance(value, Mapping) else None
            for key, value in per_query.items()
        },
        "anchor_q0_q3": diagnostics.get("pilot_anchor_aggregate"),
        "non_pilot": diagnostics.get("non_pilot_aggregate"),
        "interpolation_q1_q2": diagnostics.get("interpolation_q1_q2"),
        "extrapolation_q4_q5": diagnostics.get("extrapolation_q4_q5"),
        "delta_error": diagnostics.get("delta_error"),
        "curvature_error": diagnostics.get("curvature_error"),
    }


def _gaps(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    score = float(value)
    return {name: score - reference for name, reference in REFERENCE_NMSE_DB.items()}


def summarize_temporal_lite(
    output_root: str | Path, plan: Mapping[str, object]
) -> dict[str, object]:
    root = Path(output_root).resolve()
    profiles = plan.get("profiles")
    profiles = profiles if isinstance(profiles, Mapping) else {}
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    t1_path = root / "evaluations" / "t1_lite_validation.json"
    if t1_path.is_file():
        t1 = json.loads(t1_path.read_text(encoding="utf-8"))
        if t1.get("test_split_used") is not False:
            raise PermissionError("T1-Lite summary rejects TEST evidence.")
        profile = profiles.get("T1-Lite", {})
        profile = profile if isinstance(profile, Mapping) else {}
        diagnostics = _diagnostic_summary(t1)
        rows.append(
            {
                "candidate": "T1-Lite linear trend",
                **diagnostics,
                "best_epoch": None,
                "last_epoch": None,
                "last_validation_nmse_db": diagnostics["overall_validation_nmse_db"],
                "wall_clock_seconds": t1.get("wall_clock_seconds"),
                "parameters": profile.get("parameters"),
                "trainable_parameters": profile.get("trainable_parameters"),
                "gmacs": profile.get("gmacs"),
                "gflops": profile.get("gflops"),
                "performance_gap_db": _gaps(
                    diagnostics["overall_validation_nmse_db"]
                ),
                "test_split_used": False,
            }
        )
    else:
        missing.append("T1-Lite")
    best_path = root / "evaluations" / "tl24_best_validation.json"
    run = root / "runs" / "tl24"
    final_path = run / "results" / "final_result.json"
    history_path = run / "results" / "training_history.csv"
    if best_path.is_file() and final_path.is_file() and history_path.is_file():
        best_payload = json.loads(best_path.read_text(encoding="utf-8"))
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            best_payload.get("test_split_used") is not False
            or final.get("test_split_used") is not False
        ):
            raise PermissionError("TL24 summary rejects TEST evidence.")
        history = _read_history(history_path)
        best_row = min(history, key=lambda row: float(row["validation_nmse_linear"]))
        profile = profiles.get("TL24", {})
        profile = profile if isinstance(profile, Mapping) else {}
        diagnostics = _diagnostic_summary(best_payload)
        last = final.get("last_validation")
        rows.append(
            {
                "candidate": "TL24",
                **diagnostics,
                "best_epoch": int(best_row["epoch"]),
                "last_epoch": int(history[-1]["epoch"]),
                "last_validation_nmse_db": (
                    last.get("nmse_db") if isinstance(last, Mapping) else None
                ),
                "wall_clock_seconds": final.get("wall_clock_seconds"),
                "parameters": profile.get("parameters"),
                "trainable_parameters": profile.get("trainable_parameters"),
                "gmacs": profile.get("gmacs"),
                "gflops": profile.get("gflops"),
                "budget_pass": profile.get("budget_pass"),
                "performance_gap_db": _gaps(
                    diagnostics["overall_validation_nmse_db"]
                ),
                "test_split_used": False,
            }
        )
    else:
        missing.append("TL24")
    result = {
        "schema": TEMPORAL_LITE_SCHEMA,
        "workflow": "PriST-RIS Temporal-Lite V1",
        "results": rows,
        "missing_candidates": missing,
        "complexity_budget": plan.get("complexity_budget"),
        "reference_nmse_db": dict(REFERENCE_NMSE_DB),
        "winner": None,
        "human_decision_only": True,
        "test_split_used": False,
    }
    _write_json_atomic(root / "summaries" / "temporal_lite_summary.json", result)
    return result
