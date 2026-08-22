from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint_atomic
from .contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
    DataSemantics,
    canonical_model_key,
)
from .metrics import MetricAccumulator, PerQueryMetricAccumulator
from .models import PriSTRIS, build_model, sample_ris_columns
from .objectives import prist_ris_loss, temporal_regularized_loss
from .prior import RidgePrior, file_sha256


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@dataclass(frozen=True)
class TrainingConfig:
    domain: str
    model_key: str
    mode: str = "dev"
    seed: int = 123
    hidden: int = 80
    blocks_per_stage: tuple[int, int, int] = (3, 3, 4)
    final_refine_blocks: int = 4
    temporal_rank: int = 2
    temporal_residual: bool = True
    spatial_multiscale_supervision: bool = False
    spatial_channel_attention: str = "off"
    coordinate_enabled: bool | None = None
    backbone_ris_coordinate_enabled: bool | None = None
    backbone_antenna_index_enabled: bool | None = None
    backbone_ris_coordinate_mode: str | None = None
    attention_enabled: bool | None = None
    attention_ris_coordinate_enabled: bool | None = None
    attention_antenna_index_enabled: bool | None = None
    observed_dense_attention_heads: int = 4
    spatial_residual_style: str = "scaled_true_residual"
    temporal_mode: str = "trend"
    temporal_base_mode: str | None = None
    temporal_learned_residual_enabled: bool | None = None
    temporal_delta_loss_weight: float = 0.0
    temporal_curvature_loss_weight: float = 0.0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    scheduler: str = "fixed"
    min_learning_rate: float = 5e-6
    epochs: int = 30
    min_epochs: int = 1
    patience: int = 15
    grad_clip: float = 1.0
    charbonnier_weight: float = 0.05
    amp: bool = False
    target_blocks: tuple[int, ...] | None = None
    adaptation: str = "full"
    architecture_version: str = ARCHITECTURE_VERSION
    spatial_protocol_version: str = SPATIAL_PROTOCOL_VERSION
    position_semantics_version: str = POSITION_SEMANTICS_VERSION
    spatial_supervision_protocol_version: str = SPATIAL_SUPERVISION_PROTOCOL_VERSION
    temporal_protocol_version: str = TEMPORAL_PROTOCOL_VERSION
    test_split_used: bool = False

    def normalized(self) -> "TrainingConfig":
        return TrainingConfig(**{**asdict(self), "model_key": canonical_model_key(self.model_key)})


def configure_adaptation(model: PriSTRIS, protocol: str) -> list[str]:
    valid = {
        "target_only_scratch", "full_finetune", "frozen_spatial",
        "selective", "temporal_only", "full",
    }
    if protocol not in valid:
        raise ValueError(f"Unknown adaptation protocol {protocol!r}.")
    for parameter in model.parameters():
        parameter.requires_grad_(protocol in {"target_only_scratch", "full_finetune", "full"})
    if protocol == "frozen_spatial":
        for module in (
            model.anchor_refiners,
            model.anchor_heads,
            model.temporal,
            model.temporal_correction,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    elif protocol == "selective":
        for module in (
            model.prior_encoder,
            model.observed_dense_attention,
            model.anchor_refiners,
            model.anchor_heads,
            model.temporal_correction,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if model.temporal is not None:
            for name, parameter in model.temporal.named_parameters():
                if name.startswith(("anchor_context", "time_encoder", "fusion", "coefficient_head", "alpha_head")):
                    parameter.requires_grad_(True)
    elif protocol == "temporal_only":
        for module in (model.temporal, model.temporal_correction):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError(f"Adaptation protocol {protocol!r} selected no trainable parameters.")
    return names


def _select(
    prediction: torch.Tensor,
    target: torch.Tensor,
    blocks: tuple[int, ...] | None,
    output_time_index: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Align compact prediction positions with semantic target query times."""
    if prediction.shape[1] != len(output_time_index):
        raise ValueError(
            "Prediction time dimension does not match model output_time_index."
        )
    selected_times = output_time_index if blocks is None else tuple(blocks)
    position_by_time = {
        semantic_time: position
        for position, semantic_time in enumerate(output_time_index)
    }
    missing = [value for value in selected_times if value not in position_by_time]
    if missing:
        raise ValueError(
            f"Requested target blocks {missing} are absent from model output times "
            f"{output_time_index}."
        )
    if any(value < 0 or value >= target.shape[1] for value in selected_times):
        raise ValueError("Requested semantic target time is out of range.")
    prediction_index = torch.tensor(
        [position_by_time[value] for value in selected_times],
        device=prediction.device,
    )
    target_index = torch.tensor(selected_times, device=target.device)
    return (
        prediction.index_select(1, prediction_index),
        target.index_select(1, target_index),
        selected_times,
    )


@torch.no_grad()
def evaluate(
    model: PriSTRIS,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    prior: RidgePrior | None,
    target_blocks: tuple[int, ...] | None,
) -> dict[str, object]:
    model.eval()
    metrics = MetricAccumulator()
    diagnostics: PerQueryMetricAccumulator | None = None
    for raw in loader:
        batch = move_batch(raw, device)
        prior_value = prior.predict(batch) if prior is not None else None
        prediction = model(batch, prior_value)
        prediction, target, selected_times = _select(
            prediction,
            batch["target_h"],
            target_blocks,
            tuple(model.output_time_index),
        )
        metrics.update(prediction, target)
        if diagnostics is None:
            diagnostics = PerQueryMetricAccumulator(selected_times)
        diagnostics.update(prediction, target)
    result: dict[str, object] = dict(metrics.compute())
    if diagnostics is not None:
        result["diagnostics"] = diagnostics.compute()
    return result


def _require_architecture_version(state: dict[str, object], purpose: str) -> None:
    if state.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            f"{purpose} requires architecture_version={ARCHITECTURE_VERSION}; "
            f"checkpoint has {state.get('architecture_version')!r}."
        )


def require_checkpoint_contract(
    state: dict[str, object], purpose: str, *, expected_domain: str | None = None
) -> None:
    """Enforce the V3.2 spatial protocol and Mobility time semantics."""
    _require_architecture_version(state, purpose)
    config = state.get("model_config")
    domain = config.get("domain") if isinstance(config, dict) else None
    if expected_domain is not None and domain != expected_domain:
        raise ValueError(
            f"{purpose} requires domain={expected_domain}; checkpoint has {domain!r}."
        )
    if state.get("spatial_protocol_version") != SPATIAL_PROTOCOL_VERSION:
        raise ValueError(
            f"{purpose} requires spatial_protocol_version={SPATIAL_PROTOCOL_VERSION}; "
            f"checkpoint has {state.get('spatial_protocol_version')!r}."
        )
    if state.get("position_semantics_version") != POSITION_SEMANTICS_VERSION:
        raise ValueError(
            f"{purpose} requires position_semantics_version={POSITION_SEMANTICS_VERSION}; "
            f"checkpoint has {state.get('position_semantics_version')!r}."
        )
    model_key = config.get("model_key") if isinstance(config, dict) else None
    if model_key == "prist_ris_full" and state.get(
        "temporal_protocol_version"
    ) != TEMPORAL_PROTOCOL_VERSION:
        raise ValueError(
            f"{purpose} requires temporal_protocol_version={TEMPORAL_PROTOCOL_VERSION} "
            "for Full checkpoints."
        )
    if domain != "mobility":
        return
    expected = DataSemantics.for_domain("mobility")
    semantics = state.get("data_semantics")
    if (
        state.get("mobility_contract_version") != MOBILITY_CONTRACT_VERSION
        or state.get("semantics_hash") != expected.stable_hash()
        or not isinstance(semantics, dict)
        or tuple(semantics.get("obs_time_index", ())) != expected.obs_time_index
    ):
        raise ValueError(
            f"{purpose} rejects pre-fix Mobility semantics; expected "
            f"contract={MOBILITY_CONTRACT_VERSION} and pilots q0/q3."
        )


def load_spatial_pretrained(model: PriSTRIS, state: dict[str, object]) -> dict[str, object]:
    """Load only structurally compatible Quasi spatial weights into Mobility."""

    require_checkpoint_contract(state, "Spatial transfer", expected_domain="quasi")
    source_config = state.get("model_config")
    if not isinstance(source_config, dict) or source_config.get("domain") != "quasi":
        raise ValueError("Spatial transfer source must be a Quasi PriST-RIS V3.2 checkpoint.")
    current = model.state_dict()
    source_state = state.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("Pretrained checkpoint lacks model_state.")
    allowed_prefixes = (
        "backbone.",
        "prior_encoder.",
        "observed_dense_attention.",
        "anchor_refiners.0.",
        "anchor_heads.0.",
    )
    loaded: list[str] = []
    skipped: list[str] = []
    update: dict[str, torch.Tensor] = {}
    for name, value in source_state.items():
        if not name.startswith(allowed_prefixes):
            skipped.append(name)
        elif name in current and isinstance(value, torch.Tensor) and value.shape == current[name].shape:
            update[name] = value
            loaded.append(name)
        else:
            skipped.append(name)
    current.update(update)
    model.load_state_dict(current)
    initialized = sorted(set(current) - set(loaded))
    if not loaded:
        raise ValueError("Spatial transfer loaded no compatible weights.")
    return {
        "mode": "spatial_only",
        "source_architecture_version": state.get("architecture_version"),
        "loaded_keys": sorted(loaded),
        "skipped_keys": sorted(skipped),
        "newly_initialized_keys": initialized,
    }


def load_mobility_spatial_reference(
    model: PriSTRIS, state: dict[str, object]
) -> dict[str, object]:
    """Load a compatible Mobility q0/q3 spatial reference into Full."""

    require_checkpoint_contract(
        state, "Mobility spatial reference", expected_domain="mobility"
    )
    source_config = state.get("model_config")
    if not isinstance(source_config, dict) or source_config.get("model_key") == "prist_ris_full":
        raise ValueError("Spatial reference must be a q0/q3 Mobility model checkpoint.")
    current = model.state_dict()
    source_state = state.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("Spatial reference checkpoint lacks model_state.")
    prefixes = (
        "backbone.", "prior_encoder.", "observed_dense_attention.",
        "anchor_refiners.", "anchor_heads.",
    )
    loaded = {
        name: value
        for name, value in source_state.items()
        if name.startswith(prefixes)
        and name in current
        and isinstance(value, torch.Tensor)
        and value.shape == current[name].shape
    }
    if not loaded:
        raise ValueError("Spatial reference loaded no compatible weights.")
    current.update(loaded)
    model.load_state_dict(current)
    return {
        "mode": "mobility_spatial_reference",
        "loaded_keys": sorted(loaded),
        "source_model_key": source_config.get("model_key"),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _loader_generator(loader: Iterable[dict[str, torch.Tensor]]) -> torch.Generator | None:
    generator = getattr(loader, "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


def restore_loader_generator_state(
    generator: torch.Generator, generator_state: torch.Tensor
) -> None:
    """Restore a DataLoader RNG state even when checkpoint mapping used CUDA."""
    generator.set_state(
        generator_state.detach().cpu().to(torch.uint8).contiguous()
    )


def train(
    config: TrainingConfig,
    train_loader: Iterable[dict[str, torch.Tensor]],
    validation_loader: Iterable[dict[str, torch.Tensor]],
    *,
    run_dir: Path,
    device: torch.device,
    prior_path: str | Path | None = None,
    resume: str | Path | None = None,
    pretrained: str | Path | None = None,
    spatial_reference: str | Path | None = None,
    command: str | None = None,
    sample_indices: list[int] | None = None,
    stop_after_epoch: int | None = None,
    experiment_spec: dict[str, object] | None = None,
) -> dict[str, object]:
    config = config.normalized()
    if config.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError("Training config architecture version mismatch.")
    if config.spatial_protocol_version != SPATIAL_PROTOCOL_VERSION:
        raise ValueError("Training config spatial protocol version mismatch.")
    if config.position_semantics_version != POSITION_SEMANTICS_VERSION:
        raise ValueError("Training config position semantics version mismatch.")
    if config.spatial_supervision_protocol_version != SPATIAL_SUPERVISION_PROTOCOL_VERSION:
        raise ValueError("Training config spatial supervision protocol mismatch.")
    if config.temporal_protocol_version != TEMPORAL_PROTOCOL_VERSION:
        raise ValueError("Training config temporal protocol mismatch.")
    if config.test_split_used:
        raise PermissionError("Training and screening must not use TEST.")
    if config.temporal_delta_loss_weight < 0 or config.temporal_curvature_loss_weight < 0:
        raise ValueError("Temporal loss weights must be non-negative.")
    if config.scheduler not in {"fixed", "cosine"}:
        raise ValueError("Training scheduler must be fixed or cosine.")
    if not 0 <= config.min_learning_rate <= config.learning_rate:
        raise ValueError("min_learning_rate must be in [0, learning_rate].")
    if config.spatial_multiscale_supervision and config.model_key == "prist_ris_full":
        raise ValueError("Multi-scale spatial supervision is q0/q3-only in this protocol.")
    if config.mode == "full" and config.amp:
        raise ValueError("Formal PriST-RIS training is FP32; AMP is development-only.")
    if run_dir.exists() and resume is None:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_spec_hash = None
    if experiment_spec is not None:
        if experiment_spec.get("test_split_used") is not False:
            raise PermissionError("Paper experiment specs must explicitly exclude TEST.")
        serialized_spec = json.dumps(
            experiment_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        experiment_spec_hash = hashlib.sha256(
            serialized_spec.encode("utf-8")
        ).hexdigest()
        recorded_spec_path = run_dir / "paper_experiment_spec.json"
        if recorded_spec_path.is_file():
            if json.loads(recorded_spec_path.read_text(encoding="utf-8")) != experiment_spec:
                raise ValueError("Paper experiment spec mismatch for existing run.")
        elif resume is not None:
            raise ValueError("Resuming a paper run requires its recorded exact spec.")
        else:
            _write_json(recorded_spec_path, experiment_spec)
    checkpoints = run_dir / "checkpoints"
    results = run_dir / "results"
    manifests = run_dir / "manifests"
    checkpoints.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    manifests.mkdir(exist_ok=True)
    semantics = DataSemantics.for_domain(config.domain)
    prior = RidgePrior.load(prior_path) if prior_path is not None else None
    if config.model_key != "prist_ris_a" and prior is None:
        raise ValueError(f"{config.model_key} requires --prior.")
    if prior is not None and prior.semantics_hash != semantics.stable_hash():
        raise ValueError("Ridge prior data semantics do not match this training run.")
    expected_prior_blocks = (0,) if config.domain == "quasi" else (0, 3)
    if prior is not None and prior.target_blocks != expected_prior_blocks:
        raise ValueError(
            f"PriST-RIS V3.2 {config.domain} requires Ridge target_blocks={expected_prior_blocks}, "
            f"got {prior.target_blocks}."
        )
    prior_metadata = (
        {**prior.metadata(), "path": str(Path(prior_path).resolve()), "sha256": file_sha256(prior_path)}
        if prior is not None and prior_path is not None
        else None
    )
    seed_everything(config.seed)
    model = build_model(
        config.model_key,
        domain=config.domain,
        hidden=config.hidden,
        blocks_per_stage=config.blocks_per_stage,
        final_refine_blocks=config.final_refine_blocks,
        temporal_rank=config.temporal_rank,
        temporal_residual=config.temporal_residual,
        spatial_multiscale_supervision=config.spatial_multiscale_supervision,
        spatial_channel_attention=config.spatial_channel_attention,
        coordinate_enabled=config.coordinate_enabled,
        backbone_ris_coordinate_enabled=config.backbone_ris_coordinate_enabled,
        backbone_antenna_index_enabled=config.backbone_antenna_index_enabled,
        backbone_ris_coordinate_mode=config.backbone_ris_coordinate_mode,
        attention_enabled=config.attention_enabled,
        attention_ris_coordinate_enabled=config.attention_ris_coordinate_enabled,
        attention_antenna_index_enabled=config.attention_antenna_index_enabled,
        observed_dense_attention_heads=config.observed_dense_attention_heads,
        spatial_residual_style=config.spatial_residual_style,
        temporal_mode=config.temporal_mode,
        temporal_base_mode=config.temporal_base_mode,
        temporal_learned_residual_enabled=config.temporal_learned_residual_enabled,
        architecture_version=config.architecture_version,
        spatial_protocol_version=config.spatial_protocol_version,
        position_semantics_version=config.position_semantics_version,
        spatial_supervision_protocol_version=config.spatial_supervision_protocol_version,
        temporal_protocol_version=config.temporal_protocol_version,
    ).to(device)
    config = TrainingConfig(
        **{
            **asdict(config),
            "coordinate_enabled": None,
            "backbone_ris_coordinate_enabled": model.config.backbone_ris_coordinate_enabled,
            "backbone_antenna_index_enabled": model.config.backbone_antenna_index_enabled,
            "backbone_ris_coordinate_mode": model.config.backbone_ris_coordinate_mode,
            "attention_enabled": model.config.attention_enabled,
            "attention_ris_coordinate_enabled": model.config.attention_ris_coordinate_enabled,
            "attention_antenna_index_enabled": model.config.attention_antenna_index_enabled,
            "temporal_base_mode": model.config.temporal_base_mode,
            "temporal_learned_residual_enabled": model.config.temporal_learned_residual_enabled,
        }
    )
    pretrained_metadata = None
    if pretrained is not None and spatial_reference is not None:
        raise ValueError("Use either pretrained or spatial_reference, not both.")
    if pretrained is not None:
        source = load_checkpoint(pretrained, device)
        pretrained_metadata = load_spatial_pretrained(model, source)
        print(
            f"spatial transfer loaded={len(pretrained_metadata['loaded_keys'])} "
            f"skipped={len(pretrained_metadata['skipped_keys'])} "
            f"initialized={len(pretrained_metadata['newly_initialized_keys'])}",
            flush=True,
        )
    if spatial_reference is not None:
        reference_state = load_checkpoint(spatial_reference, device)
        pretrained_metadata = load_mobility_spatial_reference(model, reference_state)
    trainable_names = configure_adaptation(model, config.adaptation)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_learning_rate
        )
        if config.scheduler == "cosine"
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    start_epoch, best_nmse, stale = 1, float("inf"), 0
    accumulated_wall_clock_seconds = 0.0
    history: list[dict[str, object]] = []
    validation: dict[str, object] | None = None
    if resume is not None:
        state = load_checkpoint(resume, device)
        require_checkpoint_contract(state, "Resume", expected_domain=config.domain)
        stored_config = state.get("training_config")
        current_config = asdict(config)
        compatible_config = isinstance(stored_config, dict)
        if compatible_config:
            stored_compare = dict(stored_config)
            current_compare = dict(current_config)
            # Checkpoints written before scheduler/timing support are fixed-LR
            # runs. Preserve their default resume compatibility.
            stored_compare.setdefault("scheduler", "fixed")
            stored_compare.setdefault("min_learning_rate", 5e-6)
            stored_compare.setdefault("test_split_used", False)
            stored_epochs = int(stored_compare.pop("epochs"))
            current_epochs = int(current_compare.pop("epochs"))
            compatible_config = (
                stored_compare == current_compare
                and current_epochs >= stored_epochs
                and (config.scheduler == "fixed" or current_epochs == stored_epochs)
            )
        if not compatible_config or state.get("semantics_hash") != semantics.stable_hash() or state.get("prior_metadata") != prior_metadata:
            raise ValueError("Resume configuration, semantics, or prior metadata mismatch.")
        if state.get("experiment_spec_hash") != experiment_spec_hash:
            raise ValueError("Resume paper experiment spec hash mismatch.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None:
            scheduler_state = state.get("scheduler_state")
            if not isinstance(scheduler_state, dict):
                raise ValueError("Cosine resume checkpoint lacks scheduler_state.")
            scheduler.load_state_dict(scheduler_state)
        restore_rng_state(state["rng_state"])
        generator = _loader_generator(train_loader)
        generator_state = state.get("train_loader_generator_state")
        if generator is not None and generator_state is not None:
            restore_loader_generator_state(generator, generator_state)
        start_epoch = int(state["epoch"]) + 1
        best_nmse = float(state["best_validation_nmse_linear"])
        stale = int(state.get("stale_epochs", 0))
        history = list(state.get("history", []))
        stored_validation = state.get("validation")
        validation = stored_validation if isinstance(stored_validation, dict) else None
        accumulated_wall_clock_seconds = float(
            state.get("wall_clock_seconds", 0.0)
        )
    model_protocol = model.protocol_metadata()
    position_metadata = {
        key: model_protocol[key]
        for key in (
            "coordinate_enabled",
            "legacy_coordinate_alias_used",
            "backbone_ris_coordinate_enabled",
            "backbone_antenna_index_enabled",
            "backbone_ris_coordinate_mode",
            "attention_enabled",
            "attention_ris_coordinate_enabled",
            "attention_antenna_index_enabled",
            "position_semantics_version",
            "antenna_encoding_semantics",
            "spatial_multiscale_supervision",
            "spatial_channel_attention",
            "spatial_supervision_protocol_version",
            "temporal_base_mode",
            "temporal_learned_residual_enabled",
            "temporal_protocol_version",
        )
    }
    position_metadata.update(
        {
            "temporal_delta_loss_weight": config.temporal_delta_loss_weight,
            "temporal_curvature_loss_weight": config.temporal_curvature_loss_weight,
        }
    )
    resolved_training_config = asdict(config)
    metadata = {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": (
            MOBILITY_CONTRACT_VERSION if config.domain == "mobility" else None
        ),
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        **position_metadata,
        "model_key": config.model_key,
        "domain": config.domain,
        "seed": config.seed,
        "mode": config.mode,
        "semantics_hash": semantics.stable_hash(),
        "prior": prior_metadata,
        "amp": config.amp,
        "formal_fp32": config.mode == "full",
        "adaptation": config.adaptation,
        "pretrained_transfer": pretrained_metadata,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_parameter_names": trainable_names,
        "top_level_trainable_modules": sorted({name.split(".")[0] for name in trainable_names}),
        "model_protocol": model_protocol,
        "test_split_used": False,
        "experiment_spec_hash": experiment_spec_hash,
    }
    (run_dir / "command.txt").write_text(command or shlex.join(sys.argv), encoding="utf-8")
    _write_json(run_dir / "config.json", resolved_training_config)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(manifests / "data_semantics.json", semantics.to_dict())
    _write_json(manifests / "sample_indices.json", {"indices": sample_indices})
    _write_json(manifests / "prior.json", prior_metadata)
    max_epochs = config.epochs
    call_max_epochs = min(max_epochs, stop_after_epoch) if stop_after_epoch is not None else max_epochs
    started = time.perf_counter()
    epoch = start_epoch
    while epoch <= call_max_epochs:
        epoch_started = time.perf_counter()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        train_total, batches = 0.0, 0
        component_totals: dict[str, float] = {}
        for raw in train_loader:
            batch = move_batch(raw, device)
            prior_value = prior.predict(batch) if prior is not None else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                if config.spatial_multiscale_supervision:
                    predictions = model.spatial_multiscale_anchors(batch, prior_value)
                    _, target_anchors, _ = _select(
                        predictions[16],
                        batch["target_h"],
                        config.target_blocks,
                        tuple(model.output_time_index),
                    )
                    scale_losses = []
                    components: dict[str, float] = {}
                    for width in (4, 8, 16):
                        scale_loss, _ = prist_ris_loss(
                            predictions[width],
                            sample_ris_columns(target_anchors, width),
                            charbonnier_weight=config.charbonnier_weight,
                        )
                        scale_losses.append(scale_loss)
                        components[f"scale_width{width}"] = float(scale_loss.detach())
                    loss = torch.stack(scale_losses).mean()
                    components["total"] = float(loss.detach())
                else:
                    prediction = model(batch, prior_value)
                    prediction, target, selected_times = _select(
                        prediction,
                        batch["target_h"],
                        config.target_blocks,
                        tuple(model.output_time_index),
                    )
                    if selected_times == tuple(range(6)):
                        loss, components = temporal_regularized_loss(
                            prediction,
                            target,
                            charbonnier_weight=config.charbonnier_weight,
                            delta_weight=config.temporal_delta_loss_weight,
                            curvature_weight=config.temporal_curvature_loss_weight,
                        )
                    else:
                        loss, components = prist_ris_loss(
                            prediction,
                            target,
                            charbonnier_weight=config.charbonnier_weight,
                        )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach())
            for name, value_component in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + value_component
            batches += 1
        validation = evaluate(
            model,
            validation_loader,
            device,
            prior=prior,
            target_blocks=config.target_blocks,
        )
        value = float(validation["nmse_linear"])
        epoch_seconds = time.perf_counter() - epoch_started
        wall_clock_seconds = (
            accumulated_wall_clock_seconds + time.perf_counter() - started
        )
        improved = value < best_nmse
        if improved:
            best_nmse, stale = value, 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": train_total / max(1, batches),
            "validation_nmse_linear": value,
            "validation_nmse_db": validation["nmse_db"],
            "improved": improved,
            "stale_epochs": stale,
            "learning_rate": epoch_learning_rate,
            "epoch_seconds": epoch_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            **{
                f"train_{name}": total / max(1, batches)
                for name, total in component_totals.items()
            },
        }
        history.append(row)
        if scheduler is not None:
            scheduler.step()
        state = {
            "method": "PriST-RIS",
            "architecture_version": ARCHITECTURE_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_validation_nmse_linear": best_nmse,
            "training_config": resolved_training_config,
            "model_config": asdict(model.config),
            "semantics_hash": semantics.stable_hash(),
            "mobility_contract_version": (
                MOBILITY_CONTRACT_VERSION if config.domain == "mobility" else None
            ),
            "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
            **position_metadata,
            "data_semantics": semantics.to_dict(),
            "prior_metadata": prior_metadata,
            "rng_state": capture_rng_state(),
            "train_loader_generator_state": (
                _loader_generator(train_loader).get_state()
                if _loader_generator(train_loader) is not None
                else None
            ),
            "stale_epochs": stale,
            "history": history,
            "validation": validation,
            "experiment_spec_hash": experiment_spec_hash,
            "wall_clock_seconds": wall_clock_seconds,
        }
        save_checkpoint_atomic(checkpoints / "last_checkpoint.pth", state)
        if improved:
            save_checkpoint_atomic(checkpoints / "best_checkpoint.pth", state)
        _write_history(results / "training_history.csv", history)
        print(
            f"epoch={epoch} train={row['train_loss']:.6g} "
            f"val={float(validation['nmse_db']):.4f} dB",
            flush=True,
        )
        if config.mode == "dev" and epoch == 30:
            best_epoch = min(history, key=lambda item: float(item["validation_nmse_linear"]))["epoch"]
            if int(best_epoch) >= 26 and max_epochs == 30:
                max_epochs = 45
                call_max_epochs = min(max_epochs, stop_after_epoch) if stop_after_epoch is not None else max_epochs
        if epoch >= config.min_epochs and stale >= config.patience:
            break
        epoch += 1
    if validation is None:
        validation = evaluate(
            model,
            validation_loader,
            device,
            prior=prior,
            target_blocks=config.target_blocks,
        )
    final = {
        "status": "smoke_test" if config.mode == "smoke" else "validation",
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": (
            MOBILITY_CONTRACT_VERSION if config.domain == "mobility" else None
        ),
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        **position_metadata,
        "model_key": config.model_key,
        "best_validation_nmse_linear": best_nmse,
        "best_validation_nmse_db": 10 * math.log10(max(best_nmse, 1e-12)),
        "epochs_completed": int(history[-1]["epoch"]),
        "wall_clock_seconds": accumulated_wall_clock_seconds
        + time.perf_counter()
        - started,
        "last_validation": validation,
        "metadata": metadata,
        "test_split_used": False,
    }
    _write_json(results / "final_result.json", final)
    return {"run_dir": str(run_dir), "result": final}
