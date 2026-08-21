from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint
from .contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
)


@dataclass(frozen=True)
class SpatialScreeningCandidate:
    name: str
    hidden: int
    blocks_per_stage: tuple[int, int, int]
    final_refine_blocks: int


SPATIAL_SCREENING_CANDIDATES = (
    SpatialScreeningCandidate("B64", 64, (3, 3, 4), 4),
    SpatialScreeningCandidate("B48", 48, (3, 3, 4), 4),
    SpatialScreeningCandidate("D1", 80, (3, 3, 2), 1),
    SpatialScreeningCandidate("D2", 80, (2, 2, 2), 2),
)


@dataclass(frozen=True)
class PositionScreeningCandidate:
    name: str
    backbone_ris_coordinate_enabled: bool
    backbone_ris_coordinate_mode: str
    attention_enabled: bool
    attention_ris_coordinate_enabled: bool


POSITION_SCREENING_CANDIDATES = (
    PositionScreeningCandidate("P1_ris_direct", True, "direct_add", False, False),
    PositionScreeningCandidate("P2_ris_gated", True, "zero_init_gated", False, False),
    PositionScreeningCandidate("P3_attention_ris", False, "off", True, True),
)

MODULE_EXTENSION_RULE = {
    "initial_epoch": 30,
    "extended_epoch": 40,
    "late_window": [21, 30],
    "late_improvement_min_db": 0.05,
    "reference_margin_db": 0.30,
    "best_epoch_boundary": 26,
}
LONG_FOLLOWUP_RULE = {
    "target_epoch": 100,
    "best_epoch_boundary_at_40": 36,
    "close_margin_db": 0.10,
    "default_execute": False,
}
SPATIAL_MODULE_REFERENCE_DB = -20.25387859912702


@dataclass(frozen=True)
class SpatialModuleCandidate:
    name: str
    multiscale: bool
    channel_attention: str


SPATIAL_MODULE_CANDIDATES = (
    SpatialModuleCandidate("S2_multiscale", True, "off"),
    SpatialModuleCandidate("S3_se", False, "se"),
    SpatialModuleCandidate("S23_multiscale_se", True, "se"),
)


@dataclass(frozen=True)
class TemporalModuleCandidate:
    name: str
    learned_residual: bool
    delta_weight: float
    curvature_weight: float


TEMPORAL_MODULE_CANDIDATES = (
    TemporalModuleCandidate("T2_trend_residual", True, 0.0, 0.0),
    TemporalModuleCandidate("T3_delta", True, 0.1, 0.0),
    TemporalModuleCandidate("T4_curvature", True, 0.1, 0.05),
)


def should_extend_to_40(
    history: list[dict[str, object]], *, reference_db: float
) -> bool:
    by_epoch = {int(row["epoch"]): row for row in history}
    if 30 not in by_epoch:
        raise ValueError("Extension decision requires an exact epoch-30 history.")
    best = min(history, key=lambda row: float(row["validation_nmse_db"]))
    if int(best["epoch"]) >= int(MODULE_EXTENSION_RULE["best_epoch_boundary"]):
        return True
    if 21 not in by_epoch:
        return False
    late_improvement = float(by_epoch[21]["validation_nmse_db"]) - float(
        by_epoch[30]["validation_nmse_db"]
    )
    best_db = float(best["validation_nmse_db"])
    return (
        late_improvement >= float(MODULE_EXTENSION_RULE["late_improvement_min_db"])
        and best_db <= reference_db + float(MODULE_EXTENSION_RULE["reference_margin_db"])
    )


def recommend_long_followup(
    history: list[dict[str, object]], *, reference_db: float
) -> bool:
    if not history or int(history[-1]["epoch"]) < 40:
        return False
    best = min(history, key=lambda row: float(row["validation_nmse_db"]))
    return int(best["epoch"]) >= 36 or abs(
        float(best["validation_nmse_db"]) - reference_db
    ) <= float(LONG_FOLLOWUP_RULE["close_margin_db"])


def spatial_module_screening_plan(seed: int = 123) -> dict[str, object]:
    return {
        "method": "PriST-RIS",
        "phase": "S2_S3_spatial_modules",
        "reference": {
            "name": "S0_D1_RISCoord",
            "reuse_existing": True,
            "best_validation_nmse_db": SPATIAL_MODULE_REFERENCE_DB,
            "best_epoch": 74,
            "early_stop_epoch": 89,
        },
        "fixed_protocol": {
            "domain": "mobility", "model": "prist_ris_b", "seed": seed,
            "hidden": 80, "blocks_per_stage": [3, 3, 2],
            "final_refine_blocks": 1, "backbone_ris_coordinate_enabled": True,
            "backbone_ris_coordinate_mode": "direct_add",
            "backbone_antenna_index_enabled": False, "attention_enabled": False,
            "train_samples": 4096, "validation_samples": 1800,
            "learning_rate": 5e-4, "selection_split": "validation",
            "test_split_used": False,
            "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
            "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        },
        "candidates": [asdict(candidate) for candidate in SPATIAL_MODULE_CANDIDATES],
        "extension_rule": MODULE_EXTENSION_RULE,
        "long_followup_rule": LONG_FOLLOWUP_RULE,
        "canonical_changed": False,
    }


def spatial_module_training_arguments(
    candidate: SpatialModuleCandidate,
    *,
    prior: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device: str,
    workers: int,
    seed: int,
    stop_epoch: int = 30,
    epochs: int = 40,
    resume: str | Path | None = None,
) -> list[object]:
    arguments: list[object] = [
        "train", "--domain", "mobility", "--model", "prist_ris_b",
        "--mode", "dev", "--seed", seed, "--prior", prior,
        "--data-root", data_root, "--device", device, "--workers", workers,
        "--batch-size", 32, "--eval-batch-size", 64,
        "--hidden", 80, "--blocks-per-stage", "3,3,2",
        "--final-refine-blocks", 1, "--learning-rate", 5e-4,
        "--weight-decay", 1e-5, "--epochs", epochs, "--min-epochs", 41,
        "--patience", 15, "--target-blocks", "0,3",
        "--backbone-ris-coordinate-enabled", "--backbone-ris-coordinate-mode", "direct_add",
        "--no-backbone-antenna-index-enabled", "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled", "--no-attention-antenna-index-enabled",
        (
            "--spatial-multiscale-supervision"
            if candidate.multiscale else "--no-spatial-multiscale-supervision"
        ),
        "--spatial-channel-attention", candidate.channel_attention,
        "--stop-after-epoch", stop_epoch,
        "--run-name", candidate.name, "--output-root", output_root,
    ]
    if resume is not None:
        arguments.extend(("--resume", resume))
    return arguments


def temporal_module_screening_plan(
    seed: int = 123, *, include_curvature: bool = False
) -> dict[str, object]:
    candidates = [
        asdict(candidate)
        for candidate in TEMPORAL_MODULE_CANDIDATES
        if include_curvature or candidate.name != "T4_curvature"
    ]
    return {
        "method": "PriST-RIS",
        "phase": "T1_T4_temporal_modules",
        "reference": "fixed_spatial_anchor_cache",
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "t1": {"name": "T1_linear_trend", "training_required": False},
        "candidates": candidates,
        "extension_rule": MODULE_EXTENSION_RULE,
        "long_followup_rule": LONG_FOLLOWUP_RULE,
        "include_curvature": include_curvature,
        "selection_split": "validation",
        "test_split_used": False,
    }


def temporal_module_training_arguments(
    candidate: TemporalModuleCandidate,
    *,
    prior: str | Path,
    spatial_checkpoint: str | Path,
    anchor_cache_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device: str,
    workers: int,
    seed: int,
    stop_epoch: int = 30,
    epochs: int = 40,
    resume: str | Path | None = None,
) -> list[object]:
    arguments: list[object] = [
        "train", "--domain", "mobility", "--model", "prist_ris_full",
        "--mode", "dev", "--seed", seed, "--prior", prior,
        "--spatial-reference-checkpoint", spatial_checkpoint,
        "--anchor-cache-root", anchor_cache_root,
        "--data-root", data_root, "--device", device, "--workers", workers,
        "--batch-size", 16, "--eval-batch-size", 32,
        "--hidden", 80, "--blocks-per-stage", "3,3,2",
        "--final-refine-blocks", 1, "--learning-rate", 5e-4,
        "--weight-decay", 1e-5, "--epochs", epochs, "--min-epochs", 41,
        "--patience", 15, "--adaptation", "temporal_only",
        "--backbone-ris-coordinate-enabled", "--backbone-ris-coordinate-mode", "direct_add",
        "--no-backbone-antenna-index-enabled", "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled", "--no-attention-antenna-index-enabled",
        "--no-spatial-multiscale-supervision", "--spatial-channel-attention", "off",
        "--temporal-base-mode", "linear_trend",
        "--temporal-learned-residual-enabled",
        "--temporal-delta-loss-weight", candidate.delta_weight,
        "--temporal-curvature-loss-weight", candidate.curvature_weight,
        "--stop-after-epoch", stop_epoch,
        "--run-name", candidate.name, "--output-root", output_root,
    ]
    if resume is not None:
        arguments.extend(("--resume", resume))
    return arguments


def spatial_screening_plan(seed: int = 123) -> dict[str, object]:
    return {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "phase": "S1_capacity_depth_pareto",
        "research_question": (
            "Which width/depth candidate improves the validation-NMSE versus "
            "deployment-complexity Pareto frontier of Mobility PriST-RIS-B?"
        ),
        "reference": {
            "name": "B80",
            "reuse_existing": True,
            "hidden": 80,
            "blocks_per_stage": [3, 3, 4],
            "final_refine_blocks": 4,
        },
        "fixed_protocol": {
            "domain": "mobility",
            "model": "prist_ris_b",
            "mode": "dev",
            "seed": seed,
            "train_samples": 4096,
            "validation_samples": 1800,
            "batch_size": 32,
            "eval_batch_size": 64,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "epochs": 30,
            "stop_after_epoch": 30,
            "target_blocks": [0, 3],
            "backbone_ris_coordinate_enabled": False,
            "backbone_antenna_index_enabled": False,
            "backbone_ris_coordinate_mode": "off",
            "attention_enabled": False,
            "attention_ris_coordinate_enabled": False,
            "attention_antenna_index_enabled": False,
            "observed_dense_attention": False,
            "amp": False,
            "scheduler": None,
            "selection_split": "validation",
            "test_split_used": False,
        },
        "candidates": [asdict(candidate) for candidate in SPATIAL_SCREENING_CANDIDATES],
        "continuation_rule": (
            "Continue only candidates that remain clearly improving at epoch 30, "
            "or are within about 0.1-0.3 dB of B80 with a material complexity reduction."
        ),
        "canonical_changed": False,
    }


def spatial_candidate_training_arguments(
    candidate: SpatialScreeningCandidate,
    *,
    prior: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device: str = "cuda",
    workers: int = 8,
    seed: int = 123,
) -> list[object]:
    """Build the fixed 30-epoch, validation-only command for one S1 candidate."""

    return [
        "train",
        "--domain",
        "mobility",
        "--model",
        "prist_ris_b",
        "--mode",
        "dev",
        "--seed",
        seed,
        "--prior",
        prior,
        "--data-root",
        data_root,
        "--device",
        device,
        "--workers",
        workers,
        "--batch-size",
        32,
        "--eval-batch-size",
        64,
        "--hidden",
        candidate.hidden,
        "--blocks-per-stage",
        ",".join(str(value) for value in candidate.blocks_per_stage),
        "--final-refine-blocks",
        candidate.final_refine_blocks,
        "--learning-rate",
        5e-4,
        "--weight-decay",
        1e-5,
        "--epochs",
        30,
        "--min-epochs",
        31,
        "--patience",
        15,
        "--target-blocks",
        "0,3",
        "--no-backbone-ris-coordinate-enabled",
        "--no-backbone-antenna-index-enabled",
        "--backbone-ris-coordinate-mode",
        "off",
        "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled",
        "--no-attention-antenna-index-enabled",
        "--stop-after-epoch",
        30,
        "--run-name",
        candidate.name,
        "--output-root",
        output_root,
    ]


def position_screening_plan(seed: int = 123) -> dict[str, object]:
    """Return the fixed, factor-isolated P1-P3 Mobility validation plan."""

    return {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "phase": "position_semantics_repair_p1_p3",
        "fixed_protocol": {
            "domain": "mobility",
            "model": "prist_ris_b",
            "mode": "dev",
            "seed": seed,
            "train_samples": 4096,
            "validation_samples": 1800,
            "batch_size": 32,
            "eval_batch_size": 64,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "epochs": 30,
            "stop_after_epoch": 30,
            "target_blocks": [0, 3],
            "backbone_antenna_index_enabled": False,
            "attention_antenna_index_enabled": False,
            "selection_split": "validation",
            "test_split_used": False,
            "serial_execution": True,
        },
        "candidates": [asdict(candidate) for candidate in POSITION_SCREENING_CANDIDATES],
        "p4_scheduled": False,
        "canonical_changed": False,
    }


def position_candidate_training_arguments(
    candidate: PositionScreeningCandidate,
    *,
    prior: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device: str = "cuda",
    workers: int = 8,
    seed: int = 123,
) -> list[object]:
    """Build one fixed 30-epoch P1-P3 command without coupled position flags."""

    return [
        "train", "--domain", "mobility", "--model", "prist_ris_b",
        "--mode", "dev", "--seed", seed, "--prior", prior,
        "--data-root", data_root, "--device", device, "--workers", workers,
        "--batch-size", 32, "--eval-batch-size", 64,
        "--learning-rate", 5e-4, "--weight-decay", 1e-5,
        "--epochs", 30, "--min-epochs", 31, "--patience", 15,
        "--target-blocks", "0,3",
        (
            "--backbone-ris-coordinate-enabled"
            if candidate.backbone_ris_coordinate_enabled
            else "--no-backbone-ris-coordinate-enabled"
        ),
        "--no-backbone-antenna-index-enabled",
        "--backbone-ris-coordinate-mode", candidate.backbone_ris_coordinate_mode,
        "--attention-enabled" if candidate.attention_enabled else "--no-attention-enabled",
        (
            "--attention-ris-coordinate-enabled"
            if candidate.attention_ris_coordinate_enabled
            else "--no-attention-ris-coordinate-enabled"
        ),
        "--no-attention-antenna-index-enabled",
        "--stop-after-epoch", 30,
        "--run-name", candidate.name,
        "--output-root", output_root,
    ]


def accuracy_complexity_pareto(rows: list[dict[str, Any]]) -> list[str]:
    """Return candidates not dominated on best validation dB and GMACs."""

    comparable = [
        row
        for row in rows
        if row.get("best_validation_nmse_db") is not None and row.get("gmacs") is not None
    ]
    frontier: list[str] = []
    for candidate in comparable:
        candidate_nmse = float(candidate["best_validation_nmse_db"])
        candidate_gmacs = float(candidate["gmacs"])
        dominated = any(
            float(other["best_validation_nmse_db"]) <= candidate_nmse
            and float(other["gmacs"]) <= candidate_gmacs
            and (
                float(other["best_validation_nmse_db"]) < candidate_nmse
                or float(other["gmacs"]) < candidate_gmacs
            )
            for other in comparable
            if other is not candidate
        )
        if not dominated:
            frontier.append(str(candidate["candidate"]))
    return frontier


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _run_summary(
    candidate: str,
    run_dir: Path,
    profile_path: Path | None,
) -> dict[str, Any]:
    final_path = run_dir / "results" / "final_result.json"
    history_path = run_dir / "results" / "training_history.csv"
    best_path = run_dir / "checkpoints" / "best_checkpoint.pth"
    config_path = run_dir / "config.json"
    if not all(path.is_file() for path in (final_path, history_path, best_path, config_path)):
        raise FileNotFoundError(f"Incomplete screening run: {run_dir}")
    final = _read_json(final_path)
    config = _read_json(config_path)
    best = load_checkpoint(best_path, torch.device("cpu"))
    with history_path.open(newline="", encoding="utf-8-sig") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError(f"Empty training history: {history_path}")
    profile = _read_json(profile_path) if profile_path is not None and profile_path.is_file() else {}
    late = history[-min(10, len(history)) :]
    first_late_db = float(late[0]["validation_nmse_db"])
    last_late_db = float(late[-1]["validation_nmse_db"])
    validation = best.get("validation", {})
    diagnostics = validation.get("diagnostics", {}) if isinstance(validation, dict) else {}
    per_query = diagnostics.get("per_query", {}) if isinstance(diagnostics, dict) else {}

    def query_db(name: str) -> float | None:
        value = per_query.get(name, {}) if isinstance(per_query, dict) else {}
        return float(value["nmse_db"]) if isinstance(value, dict) and "nmse_db" in value else None

    epochs = {int(row["epoch"]): float(row["validation_nmse_db"]) for row in history}
    return {
        "candidate": candidate,
        "run_dir": str(run_dir.resolve()),
        "architecture_version": final.get("architecture_version"),
        "spatial_protocol_version": final.get("spatial_protocol_version"),
        "spatial_supervision_protocol_version": final.get(
            "spatial_supervision_protocol_version"
        ),
        "temporal_protocol_version": final.get("temporal_protocol_version"),
        "position_semantics_version": final.get("position_semantics_version"),
        "backbone_ris_coordinate_enabled": final.get(
            "backbone_ris_coordinate_enabled"
        ),
        "backbone_antenna_index_enabled": final.get(
            "backbone_antenna_index_enabled"
        ),
        "backbone_ris_coordinate_mode": final.get("backbone_ris_coordinate_mode"),
        "attention_enabled": final.get("attention_enabled"),
        "attention_ris_coordinate_enabled": final.get(
            "attention_ris_coordinate_enabled"
        ),
        "attention_antenna_index_enabled": final.get(
            "attention_antenna_index_enabled"
        ),
        "seed": config.get("seed"),
        "hidden": config.get("hidden"),
        "blocks_per_stage": config.get("blocks_per_stage"),
        "final_refine_blocks": config.get("final_refine_blocks"),
        "best_validation_nmse_db": float(final["best_validation_nmse_db"]),
        "best_epoch": int(best["epoch"]),
        "last_validation_nmse_db": float(final["last_validation"]["nmse_db"]),
        "epoch30_validation_nmse_db": epochs.get(30),
        "epoch40_validation_nmse_db": epochs.get(40),
        "q0_best_nmse_db": query_db("q0"),
        "q3_best_nmse_db": query_db("q3"),
        "late_window_start": int(late[0]["epoch"]),
        "late_window_end": int(late[-1]["epoch"]),
        "late_window_improvement_db": first_late_db - last_late_db,
        "parameters": profile.get("parameters"),
        "trainable_parameters": (
            final.get("metadata", {}).get("trainable_parameters")
            if isinstance(final.get("metadata"), dict)
            else profile.get("trainable_parameters")
        ),
        "gmacs": profile.get("gmacs"),
        "gflops": profile.get("gflops"),
        "latency_ms_batch1": profile.get("latency_ms_batch1"),
        "peak_gpu_memory_bytes": profile.get("peak_gpu_memory_bytes"),
        "spatial_multiscale_supervision": final.get(
            "spatial_multiscale_supervision"
        ),
        "spatial_channel_attention": final.get("spatial_channel_attention"),
        "temporal_base_mode": final.get("temporal_base_mode"),
        "temporal_learned_residual_enabled": final.get(
            "temporal_learned_residual_enabled"
        ),
        "temporal_delta_loss_weight": final.get("temporal_delta_loss_weight"),
        "temporal_curvature_loss_weight": final.get(
            "temporal_curvature_loss_weight"
        ),
        "inference_graph_changed": profile.get("inference_graph_changed"),
        "training_only_mechanism": profile.get("training_only_mechanism"),
        "training_wall_time_seconds": final.get("wall_clock_seconds"),
        "test_split_used": final.get("test_split_used"),
    }


def summarize_spatial_screening(
    study_root: str | Path,
    *,
    reference_run: str | Path | None = None,
    reference_profile: str | Path | None = None,
) -> dict[str, object]:
    root = Path(study_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if reference_run is not None:
        rows.append(
            _run_summary(
                "B80",
                Path(reference_run),
                Path(reference_profile) if reference_profile is not None else None,
            )
        )
    for candidate in SPATIAL_SCREENING_CANDIDATES:
        run_dir = root / "runs" / candidate.name
        profile = root / "profiles" / f"{candidate.name}.json"
        if (run_dir / "results" / "final_result.json").is_file():
            rows.append(_run_summary(candidate.name, run_dir, profile))
        else:
            missing.append(candidate.name)
    reference = next((row for row in rows if row["candidate"] == "B80"), None)
    if reference is not None:
        for row in rows:
            row["delta_nmse_vs_b80_db"] = (
                float(row["best_validation_nmse_db"])
                - float(reference["best_validation_nmse_db"])
            )
            for key, output in (
                ("parameters", "parameter_reduction_vs_b80_pct"),
                ("gmacs", "gmac_reduction_vs_b80_pct"),
                ("latency_ms_batch1", "latency_reduction_vs_b80_pct"),
            ):
                baseline = reference.get(key)
                value = row.get(key)
                row[output] = (
                    100.0 * (float(baseline) - float(value)) / float(baseline)
                    if baseline not in (None, 0) and value is not None
                    else None
                )
    return {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "phase": "S1_capacity_depth_pareto",
        "selection_split": "validation",
        "test_split_used": False,
        "rows": rows,
        "accuracy_gmac_pareto_frontier": accuracy_complexity_pareto(rows),
        "missing_candidates": missing,
        "canonical_changed": False,
    }


def summarize_position_screening(study_root: str | Path) -> dict[str, object]:
    root = Path(study_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate in POSITION_SCREENING_CANDIDATES:
        run_dir = root / "runs" / candidate.name
        profile = root / "profiles" / f"{candidate.name}.json"
        if (run_dir / "results" / "final_result.json").is_file():
            rows.append(_run_summary(candidate.name, run_dir, profile))
        else:
            missing.append(candidate.name)
    return {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "phase": "position_semantics_repair_p1_p3",
        "selection_split": "validation",
        "test_split_used": False,
        "rows": rows,
        "missing_candidates": missing,
        "p4_scheduled": False,
        "canonical_changed": False,
    }


def read_training_history(run_dir: str | Path) -> list[dict[str, object]]:
    path = Path(run_dir) / "results" / "training_history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing training history: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summarize_spatial_modules(
    study_root: str | Path, *, reference_db: float = SPATIAL_MODULE_REFERENCE_DB
) -> dict[str, object]:
    root = Path(study_root)
    rows = []
    missing = []
    recommended = []
    for candidate in SPATIAL_MODULE_CANDIDATES:
        run = root / "runs" / candidate.name
        if (run / "results" / "final_result.json").is_file():
            row = _run_summary(
                candidate.name, run, root / "profiles" / f"{candidate.name}.json"
            )
            row["delta_vs_s0_db"] = float(row["best_validation_nmse_db"]) - reference_db
            history = read_training_history(run)
            if recommend_long_followup(history, reference_db=reference_db):
                recommended.append(candidate.name)
            rows.append(row)
        else:
            missing.append(candidate.name)
    return {
        "method": "PriST-RIS",
        "phase": "S2_S3_spatial_modules",
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "reference_best_validation_nmse_db": reference_db,
        "rows": rows,
        "missing_candidates": missing,
        "extension_rule": MODULE_EXTENSION_RULE,
        "recommended_long_followups": recommended,
        "long_followups_executed": False,
        "test_split_used": False,
        "canonical_changed": False,
    }


def summarize_temporal_modules(study_root: str | Path) -> dict[str, object]:
    root = Path(study_root)
    rows: list[dict[str, object]] = []
    t1_path = root / "T1_linear_trend.json"
    if t1_path.is_file():
        t1 = _read_json(t1_path)
        diagnostics = t1.get("diagnostics", {})
        profile_path = root / "profiles" / "T1_linear_trend.json"
        profile = _read_json(profile_path) if profile_path.is_file() else {}
        rows.append(
            {
                "candidate": "T1_linear_trend",
                **t1,
                "per_query": diagnostics.get("per_query"),
                "interpolation_q1_q2": diagnostics.get("interpolation_q1_q2"),
                "extrapolation_q4_q5": diagnostics.get("extrapolation_q4_q5"),
                "non_pilot_aggregate": diagnostics.get("non_pilot_aggregate"),
                "overall_q0_q5": diagnostics.get("overall"),
                "delta_error": diagnostics.get("delta_error"),
                "curvature_error": diagnostics.get("curvature_error"),
                "parameters": profile.get("parameters"),
                "trainable_parameters": 0,
                "gmacs": profile.get("gmacs"),
                "gflops": profile.get("gflops"),
                "latency_ms_batch1": profile.get("latency_ms_batch1"),
                "peak_gpu_memory_bytes": profile.get("peak_gpu_memory_bytes"),
            }
        )
    reference_db = float(rows[0]["nmse_db"]) if rows else None
    recommended: list[str] = []
    for candidate in TEMPORAL_MODULE_CANDIDATES:
        run = root / "runs" / candidate.name
        if not (run / "results" / "final_result.json").is_file():
            continue
        base = _run_summary(
            candidate.name, run, root / "profiles" / f"{candidate.name}.json"
        )
        best = load_checkpoint(run / "checkpoints" / "best_checkpoint.pth", torch.device("cpu"))
        diagnostics = best.get("validation", {}).get("diagnostics", {})
        base.update(
            {
                "per_query": diagnostics.get("per_query"),
                "interpolation_q1_q2": diagnostics.get("interpolation_q1_q2"),
                "extrapolation_q4_q5": diagnostics.get("extrapolation_q4_q5"),
                "non_pilot_aggregate": diagnostics.get("non_pilot_aggregate"),
                "overall_q0_q5": diagnostics.get("overall"),
                "delta_error": diagnostics.get("delta_error"),
                "curvature_error": diagnostics.get("curvature_error"),
            }
        )
        if reference_db is not None and recommend_long_followup(
            read_training_history(run), reference_db=reference_db
        ):
            recommended.append(candidate.name)
        rows.append(base)
    return {
        "method": "PriST-RIS",
        "phase": "T1_T4_temporal_modules",
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "ranking_scope": "q0_q5_temporal_only",
        "rows": rows,
        "extension_rule": MODULE_EXTENSION_RULE,
        "long_followup_rule": LONG_FOLLOWUP_RULE,
        "recommended_long_followups": recommended,
        "long_followups_executed": False,
        "test_split_used": False,
    }
