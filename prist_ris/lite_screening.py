from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch

from .complexity import profile_model
from .contracts import DataSemantics
from .data import EXPECTED_MOBILITY_COUNTS
from .models import build_model
from .paper_matrix import (
    CONVERGENCE_THRESHOLDS_DB,
    first_threshold_crossings,
    gpu_preflight,
    indices_hash,
    validate_prior_artifact,
)
from .prior import file_sha256


LITE_SCREENING_SCHEMA = "prist_ris.lite_screening.v1"
LITE_SEED = 123
LITE_FRACTION = 0.25
LITE_SAMPLE_COUNT = 5000
LITE_TARGET_BLOCKS = (0, 3)
LITE_SELECTION_RULES = (
    "minimize spatial parameters and GMAC",
    "maximize 25% canonical VALIDATION performance",
    "prefer faster convergence",
)
SPATIAL_SCOPE_CAVEAT = (
    "spatial-only GMAC cannot yet be claimed as an apples-to-apples "
    "end-to-end comparison with LPAN-L."
)


@dataclass(frozen=True)
class LiteCandidate:
    candidate: str
    hidden: int
    blocks_per_stage: tuple[int, int, int] = (2, 2, 1)
    final_refine_blocks: int = 1


LITE_CANDIDATES = (
    LiteCandidate("Lite-A", 32),
    LiteCandidate("Lite-B", 40),
)


def fixed_training_config(candidate: LiteCandidate) -> dict[str, object]:
    return {
        "domain": "mobility",
        "mode": "full",
        "model_key": "prist_ris_b",
        "seed": LITE_SEED,
        "fraction": LITE_FRACTION,
        "sample_count": LITE_SAMPLE_COUNT,
        "target_blocks": list(LITE_TARGET_BLOCKS),
        "hidden": candidate.hidden,
        "blocks_per_stage": list(candidate.blocks_per_stage),
        "final_refine_blocks": candidate.final_refine_blocks,
        "ridge_prior_enabled": True,
        "backbone_ris_coordinate_enabled": True,
        "backbone_ris_coordinate_mode": "direct_add",
        "backbone_antenna_index_enabled": False,
        "attention_enabled": False,
        "attention_ris_coordinate_enabled": False,
        "attention_antenna_index_enabled": False,
        "spatial_multiscale_supervision": False,
        "spatial_channel_attention": "se",
        "spatial_residual_style": "scaled_true_residual",
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "min_learning_rate": 5e-6,
        "epochs": 100,
        "min_epochs": 101,
        "patience": 15,
        "batch_size": 32,
        "eval_batch_size": 64,
        "amp": False,
        "formal_fp32": True,
        "adaptation": "full",
        "test_split_used": False,
    }


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise FileExistsError(f"Existing Lite reproducibility artifact differs: {path}")
        return
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def git_head(project_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_sample_manifest(
    path: str | Path, *, seed: int = LITE_SEED
) -> dict[str, object]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("test_split_used") is not False:
        raise PermissionError("Lite sample manifest must explicitly exclude TEST.")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("Lite sample manifest seed must be 123.")
    if payload.get("nested") is not True:
        raise ValueError("Lite requires the canonical nested sample manifest.")
    if int(payload.get("total_train_samples", -1)) != EXPECTED_MOBILITY_COUNTS["train"]:
        raise ValueError("Lite manifest total_train_samples must match canonical TRAIN.")
    fractions = payload.get("fractions")
    indices = fractions.get("0.25") if isinstance(fractions, dict) else None
    if not isinstance(indices, list) or len(indices) != LITE_SAMPLE_COUNT:
        raise ValueError("Lite manifest must contain exactly 5000 fraction 0.25 indices.")
    normalized = [int(value) for value in indices]
    if len(set(normalized)) != LITE_SAMPLE_COUNT:
        raise ValueError("Lite sample indices must be unique.")
    if min(normalized) < 0 or max(normalized) >= EXPECTED_MOBILITY_COUNTS["train"]:
        raise ValueError("Lite sample indices fall outside canonical TRAIN.")
    return {
        "path": str(manifest),
        "sha256": file_sha256(manifest),
        "subset_indices_hash": indices_hash(normalized),
        "sample_count": len(normalized),
        "fraction": LITE_FRACTION,
        "seed": seed,
        "nested": True,
        "test_split_used": False,
    }


def validate_fraction_prior(
    path: str | Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    prior = Path(path).resolve()
    if not prior.is_file():
        raise FileNotFoundError(prior)
    job = {
        "seed": LITE_SEED,
        "fraction": LITE_FRACTION,
        "sample_count": LITE_SAMPLE_COUNT,
        "sample_manifest_sha256": manifest["sha256"],
        "indices_hash": manifest["subset_indices_hash"],
    }
    validate_prior_artifact(prior, job)
    return {
        "path": str(prior),
        "sha256": file_sha256(prior),
        "seed": LITE_SEED,
        "fraction": LITE_FRACTION,
        "sample_count": LITE_SAMPLE_COUNT,
        "sample_manifest_sha256": manifest["sha256"],
        "subset_indices_hash": manifest["subset_indices_hash"],
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "fit_split": "train",
        "selection_split": "validation",
        "test_split_used": False,
    }


def build_candidate_model(candidate: LiteCandidate) -> torch.nn.Module:
    return build_model(
        "prist_ris_b",
        domain="mobility",
        hidden=candidate.hidden,
        blocks_per_stage=candidate.blocks_per_stage,
        final_refine_blocks=candidate.final_refine_blocks,
        backbone_ris_coordinate_enabled=True,
        backbone_ris_coordinate_mode="direct_add",
        backbone_antenna_index_enabled=False,
        attention_enabled=False,
        attention_ris_coordinate_enabled=False,
        attention_antenna_index_enabled=False,
        spatial_multiscale_supervision=False,
        spatial_channel_attention="se",
        spatial_residual_style="scaled_true_residual",
    )


def profile_candidates(
    output_root: str | Path,
    *,
    device: torch.device = torch.device("cpu"),
    latency_runs: int = 1,
) -> list[dict[str, object]]:
    if device.type != "cpu":
        raise ValueError("Lite planning/profile smoke is CPU-only.")
    root = Path(output_root).resolve()
    rows = []
    for candidate in LITE_CANDIDATES:
        model = build_candidate_model(candidate).to(device)
        result = profile_model(
            model,
            domain="mobility",
            device=device,
            latency_runs=latency_runs,
        )
        # Latency is intentionally excluded: the fixed screening evidence needs
        # deterministic architecture counts, and CPU wall time is host-specific.
        result.pop("latency_ms_batch1", None)
        result.pop("peak_gpu_memory_bytes", None)
        result.update(
            {
                "candidate": candidate.candidate,
                "hidden": candidate.hidden,
                "blocks_per_stage": list(candidate.blocks_per_stage),
                "final_refine_blocks": candidate.final_refine_blocks,
                "batch_size": 1,
                "dtype": "FP32",
                "target_scope": "mobility_q0_q3_spatial_only",
                "scope_caveat": SPATIAL_SCOPE_CAVEAT,
                "test_split_used": False,
            }
        )
        _write_json_exact(
            root / "profiles" / f"{candidate.candidate.lower().replace('-', '_')}.json",
            result,
        )
        rows.append(result)
    return rows


def experiment_spec(
    candidate: LiteCandidate,
    *,
    root: Path,
    head: str,
    manifest: Mapping[str, object],
    prior: Mapping[str, object],
) -> dict[str, object]:
    slug = candidate.candidate.lower().replace("-", "_")
    run_name = f"prist_ris_{slug}_seed123_fraction_0.25"
    return {
        "schema": LITE_SCREENING_SCHEMA,
        "candidate": candidate.candidate,
        "run_name": run_name,
        "run_dir": str((root / "runs" / run_name).resolve()),
        "git_head": head,
        "canonical_mobility_semantics_hash": DataSemantics.for_domain(
            "mobility"
        ).stable_hash(),
        "sample_manifest": dict(manifest),
        "ridge_prior": dict(prior),
        "training_config": fixed_training_config(candidate),
        "target_scope": "mobility_q0_q3",
        "test_split_used": False,
    }


def training_arguments(
    spec: Mapping[str, object],
    *,
    data_root: str | Path,
    device: str,
    workers: int,
    spec_path: str | Path,
    resume: str | Path | None = None,
) -> list[object]:
    config = spec["training_config"]
    assert isinstance(config, Mapping)
    manifest = spec["sample_manifest"]
    prior = spec["ridge_prior"]
    assert isinstance(manifest, Mapping) and isinstance(prior, Mapping)
    arguments: list[object] = [
        "train", "--domain", "mobility", "--model", "prist_ris_b",
        "--mode", "full", "--seed", LITE_SEED,
        "--fraction", LITE_FRACTION,
        "--sample-index-manifest", manifest["path"],
        "--target-blocks", "0,3", "--prior", prior["path"],
        "--data-root", data_root, "--device", device, "--workers", workers,
        "--batch-size", 32, "--eval-batch-size", 64,
        "--hidden", config["hidden"], "--blocks-per-stage", "2,2,1",
        "--final-refine-blocks", 1,
        "--backbone-ris-coordinate-enabled",
        "--backbone-ris-coordinate-mode", "direct_add",
        "--no-backbone-antenna-index-enabled", "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled",
        "--no-attention-antenna-index-enabled",
        "--no-spatial-multiscale-supervision",
        "--spatial-channel-attention", "se",
        "--spatial-residual-style", "scaled_true_residual",
        "--learning-rate", 5e-4, "--weight-decay", 1e-5,
        "--scheduler", "cosine", "--min-learning-rate", 5e-6,
        "--epochs", 100, "--min-epochs", 101, "--patience", 15,
        "--adaptation", "full", "--run-name", spec["run_name"],
        "--output-root", Path(str(spec["run_dir"])).parent,
        "--experiment-spec", spec_path,
    ]
    if resume is not None:
        arguments.extend(("--resume", resume))
    return arguments


def build_lite_plan(
    output_root: str | Path,
    *,
    prior_path: str | Path,
    sample_manifest_path: str | Path,
    data_root: str | Path,
    workers: int,
    seed: int = LITE_SEED,
    project_root: str | Path | None = None,
    head: str | None = None,
    run_profiles: bool = True,
) -> dict[str, object]:
    if seed != LITE_SEED:
        raise ValueError("PriST-RIS-Lite V1 fixes seed=123; no seed search is allowed.")
    root = Path(output_root).resolve()
    project = Path(project_root).resolve() if project_root else Path.cwd()
    current_head = head or git_head(project)
    manifest = validate_sample_manifest(sample_manifest_path, seed=seed)
    prior = validate_fraction_prior(prior_path, manifest)
    profiles = (
        profile_candidates(root, device=torch.device("cpu")) if run_profiles else []
    )
    specs = [
        experiment_spec(
            candidate,
            root=root,
            head=current_head,
            manifest=manifest,
            prior=prior,
        )
        for candidate in LITE_CANDIDATES
    ]
    commands = {}
    for spec in specs:
        spec_path = root / "manifests" / "specs" / f"{spec['candidate']}.json"
        command = training_arguments(
            spec,
            data_root=data_root,
            device="cuda:0",
            workers=workers,
            spec_path=spec_path,
        )
        commands[str(spec["candidate"])] = ["prist-ris", *(str(v) for v in command)]
        _write_json_exact(spec_path, spec)
    plan = {
        "schema": LITE_SCREENING_SCHEMA,
        "workflow": "PriST-RIS-Lite V1",
        "purpose": "fixed two-candidate spatial residual compression evidence",
        "architecture_search": False,
        "candidates": [asdict(value) for value in LITE_CANDIDATES],
        "experiments": specs,
        "commands": commands,
        "profiles": profiles,
        "output_root": str(root),
        "git_head": current_head,
        "canonical_mobility_semantics_hash": DataSemantics.for_domain(
            "mobility"
        ).stable_hash(),
        "sample_manifest": manifest,
        "ridge_prior": prior,
        "seed": LITE_SEED,
        "fraction": LITE_FRACTION,
        "sample_count": LITE_SAMPLE_COUNT,
        "target_scope": "mobility_q0_q3",
        "serial_execution": True,
        "planner_uses_gpu": False,
        "selection_rules": list(LITE_SELECTION_RULES),
        "if_both_fail": "stop; human decision required before adding Lite-C",
        "temporal_lite_implemented": False,
        "scope_caveat": SPATIAL_SCOPE_CAVEAT,
        "test_split_used": False,
    }
    _write_json_exact(root / "lite_screen_plan.json", plan)
    return plan


def decide_run(
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
            raise ValueError(f"Completed Lite run spec mismatch: {run}")
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("test_split_used") is not False:
            raise PermissionError("Completed Lite run contains TEST evidence.")
        return {"action": "reuse", "run_dir": str(run)}
    if not run.exists():
        return {"action": "run", "run_dir": str(run)}
    if not resume_incomplete:
        raise FileExistsError(f"Incomplete Lite run will not be overwritten: {run}")
    if not recorded.is_file() or json.loads(recorded.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"Incomplete Lite run spec mismatch: {run}")
    checkpoint = run / "checkpoints" / "last_checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Incomplete Lite run has no resumable checkpoint: {run}")
    return {"action": "resume", "run_dir": str(run), "checkpoint": str(checkpoint)}


Invoke = Callable[[list[object]], None]


def execute_lite_plan(
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
        raise PermissionError("Lite runner rejects plans using TEST.")
    if torch.device(device).type != "cuda":
        raise ValueError("Formal Lite run requires one CUDA GPU; plan/profile remain CPU-only.")
    planned_names = [
        value.get("candidate")
        for value in plan.get("experiments", [])
        if isinstance(value, Mapping)
    ]
    if planned_names != ["Lite-A", "Lite-B"]:
        raise ValueError("Lite runner accepts exactly the fixed serial Lite-A/Lite-B plan.")
    profiled_names = {
        value.get("candidate")
        for value in plan.get("profiles", [])
        if isinstance(value, Mapping) and value.get("test_split_used") is False
    }
    if profiled_names != {"Lite-A", "Lite-B"}:
        raise RuntimeError("Lite-A and Lite-B CPU profiles must complete before training.")
    manifest = plan.get("sample_manifest")
    prior = plan.get("ridge_prior")
    if not isinstance(manifest, Mapping) or not isinstance(prior, Mapping):
        raise ValueError("Lite plan lacks sample/prior provenance.")
    current_manifest = validate_sample_manifest(manifest["path"])
    current_prior = validate_fraction_prior(prior["path"], current_manifest)
    if dict(manifest) != current_manifest or dict(prior) != current_prior:
        raise ValueError("Lite plan provenance changed after planning.")
    rows = []
    for raw_spec in plan.get("experiments", []):
        if not isinstance(raw_spec, Mapping):
            raise ValueError("Invalid Lite experiment spec.")
        if raw_spec.get("test_split_used") is not False:
            raise PermissionError("Lite experiment spec uses TEST.")
        decision = decide_run(raw_spec, resume_incomplete=resume_incomplete)
        if decision["action"] == "reuse":
            rows.append({"candidate": raw_spec["candidate"], "status": "reused"})
            continue
        gpu_preflight(
            device=device,
            physical_gpu_index=physical_gpu_index,
            confirm_gpu_free=confirm_gpu_free,
        )
        spec_path = (
            Path(str(plan["output_root"]))
            / "manifests"
            / "specs"
            / f"{raw_spec['candidate']}.json"
        )
        invoke(
            training_arguments(
                raw_spec,
                data_root=data_root,
                device=device,
                workers=workers,
                spec_path=spec_path,
                resume=decision.get("checkpoint"),
            )
        )
        rows.append({"candidate": raw_spec["candidate"], "status": decision["action"]})
    return {
        "workflow": "PriST-RIS-Lite V1",
        "serial_execution": True,
        "results": rows,
        "test_split_used": False,
    }


def _read_history(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_values = (
        float(left["parameters"]),
        float(left["gmacs"]),
        float(left["best_validation_nmse_linear"]),
    )
    right_values = (
        float(right["parameters"]),
        float(right["gmacs"]),
        float(right["best_validation_nmse_linear"]),
    )
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def summarize_lite_plan(
    output_root: str | Path, plan: Mapping[str, object]
) -> dict[str, object]:
    root = Path(output_root).resolve()
    profiles = {
        str(row["candidate"]): row
        for row in plan.get("profiles", [])
        if isinstance(row, Mapping)
    }
    rows = []
    missing = []
    for spec in plan.get("experiments", []):
        if not isinstance(spec, Mapping):
            continue
        candidate = str(spec["candidate"])
        run = Path(str(spec["run_dir"]))
        final_path = run / "results" / "final_result.json"
        history_path = run / "results" / "training_history.csv"
        if not final_path.is_file() or not history_path.is_file():
            missing.append(candidate)
            continue
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("test_split_used") is not False:
            raise PermissionError("Lite summary rejects TEST evidence.")
        history = _read_history(history_path)
        best = min(history, key=lambda row: float(row["validation_nmse_linear"]))
        last = final.get("last_validation")
        profile = profiles.get(candidate, {})
        config = spec["training_config"]
        assert isinstance(config, Mapping)
        rows.append(
            {
                "candidate": candidate,
                "hidden": config["hidden"],
                "blocks_per_stage": config["blocks_per_stage"],
                "final_refine_blocks": config["final_refine_blocks"],
                "parameters": profile.get("parameters"),
                "trainable_parameters": profile.get("trainable_parameters"),
                "gmacs": profile.get("gmacs"),
                "gflops": profile.get("gflops"),
                "best_validation_nmse_linear": float(best["validation_nmse_linear"]),
                "best_validation_nmse_db": float(best["validation_nmse_db"]),
                "best_epoch": int(best["epoch"]),
                "last_validation_nmse_db": (
                    last.get("nmse_db") if isinstance(last, Mapping) else None
                ),
                "wall_clock_seconds": final.get("wall_clock_seconds"),
                "threshold_crossings": first_threshold_crossings(
                    history, CONVERGENCE_THRESHOLDS_DB
                ),
                "test_split_used": False,
            }
        )
    pareto = {}
    if len(rows) == 2 and all(
        row.get(key) is not None
        for row in rows
        for key in ("parameters", "gmacs", "best_validation_nmse_linear")
    ):
        by_name = {str(row["candidate"]): row for row in rows}
        pareto = {
            "Lite-A_dominated_by_Lite-B": _dominates(by_name["Lite-B"], by_name["Lite-A"]),
            "Lite-B_dominated_by_Lite-A": _dominates(by_name["Lite-A"], by_name["Lite-B"]),
        }
    result = {
        "schema": LITE_SCREENING_SCHEMA,
        "workflow": "PriST-RIS-Lite V1",
        "results": rows,
        "missing_candidates": missing,
        "pareto": pareto,
        "winner": None,
        "selection_requires_human_judgment": True,
        "selection_rules": list(LITE_SELECTION_RULES),
        "scope_caveat": SPATIAL_SCOPE_CAVEAT,
        "temporal_lite_implemented": False,
        "test_split_used": False,
    }
    _write_json_atomic(root / "summaries" / "lite_screen_summary.json", result)
    return result
