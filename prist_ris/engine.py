from __future__ import annotations

import csv
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
from .contracts import ARCHITECTURE_VERSION, DataSemantics, canonical_model_key
from .metrics import MetricAccumulator, PerQueryMetricAccumulator
from .models import PriSTRIS, build_model
from .objectives import prist_ris_loss
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
    coordinate_enabled: bool | None = None
    temporal_mode: str = "trend"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    min_epochs: int = 1
    patience: int = 15
    grad_clip: float = 1.0
    charbonnier_weight: float = 0.05
    amp: bool = False
    target_blocks: tuple[int, ...] | None = None
    adaptation: str = "full"
    architecture_version: str = ARCHITECTURE_VERSION

    def normalized(self) -> "TrainingConfig":
        return TrainingConfig(**{**asdict(self), "model_key": canonical_model_key(self.model_key)})


def configure_adaptation(model: PriSTRIS, protocol: str) -> list[str]:
    valid = {"target_only_scratch", "full_finetune", "frozen_spatial", "selective", "full"}
    if protocol not in valid:
        raise ValueError(f"Unknown adaptation protocol {protocol!r}.")
    for parameter in model.parameters():
        parameter.requires_grad_(protocol in {"target_only_scratch", "full_finetune", "full"})
    if protocol == "frozen_spatial":
        for module in (model.anchor_feature, model.anchor_heads, model.temporal, model.temporal_correction):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    elif protocol == "selective":
        for module in (
            model.prior_encoder,
            model.anchor_feature,
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
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError(f"Adaptation protocol {protocol!r} selected no trainable parameters.")
    return names


def _select(prediction: torch.Tensor, target: torch.Tensor, blocks: tuple[int, ...] | None) -> tuple[torch.Tensor, torch.Tensor]:
    if blocks is None:
        return prediction, target
    index = torch.tensor(blocks, device=prediction.device)
    return prediction.index_select(1, index), target.index_select(1, index)


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
        prediction, target = _select(prediction, batch["target_h"], target_blocks)
        metrics.update(prediction, target)
        if diagnostics is None:
            diagnostics = PerQueryMetricAccumulator(prediction.shape[1])
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


def load_spatial_pretrained(model: PriSTRIS, state: dict[str, object]) -> dict[str, object]:
    """Load only structurally compatible Quasi spatial weights into Mobility."""

    _require_architecture_version(state, "Spatial transfer")
    source_config = state.get("model_config")
    if not isinstance(source_config, dict) or source_config.get("domain") != "quasi":
        raise ValueError("Spatial transfer source must be a Quasi PriST-RIS V3.1 checkpoint.")
    current = model.state_dict()
    source_state = state.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("Pretrained checkpoint lacks model_state.")
    allowed_prefixes = (
        "backbone.",
        "prior_encoder.",
        "anchor_feature.",
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
    command: str | None = None,
    sample_indices: list[int] | None = None,
    stop_after_epoch: int | None = None,
) -> dict[str, object]:
    config = config.normalized()
    if config.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError("Training config architecture version mismatch.")
    if config.mode == "full" and config.amp:
        raise ValueError("Formal PriST-RIS training is FP32; AMP is development-only.")
    if run_dir.exists() and resume is None:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
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
    expected_prior_blocks = (0,) if config.domain == "quasi" else (0, 1)
    if prior is not None and prior.target_blocks != expected_prior_blocks:
        raise ValueError(
            f"PriST-RIS V3.1 {config.domain} requires Ridge target_blocks={expected_prior_blocks}, "
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
        coordinate_enabled=config.coordinate_enabled,
        temporal_mode=config.temporal_mode,
        architecture_version=config.architecture_version,
    ).to(device)
    pretrained_metadata = None
    if pretrained is not None:
        source = load_checkpoint(pretrained, device)
        pretrained_metadata = load_spatial_pretrained(model, source)
        print(
            f"spatial transfer loaded={len(pretrained_metadata['loaded_keys'])} "
            f"skipped={len(pretrained_metadata['skipped_keys'])} "
            f"initialized={len(pretrained_metadata['newly_initialized_keys'])}",
            flush=True,
        )
    trainable_names = configure_adaptation(model, config.adaptation)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    start_epoch, best_nmse, stale = 1, float("inf"), 0
    history: list[dict[str, object]] = []
    validation: dict[str, object] | None = None
    if resume is not None:
        state = load_checkpoint(resume, device)
        _require_architecture_version(state, "Resume")
        if state.get("training_config") != asdict(config) or state.get("semantics_hash") != semantics.stable_hash() or state.get("prior_metadata") != prior_metadata:
            raise ValueError("Resume configuration, semantics, or prior metadata mismatch.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        restore_rng_state(state["rng_state"])
        generator = _loader_generator(train_loader)
        generator_state = state.get("train_loader_generator_state")
        if generator is not None and generator_state is not None:
            generator.set_state(generator_state)
        start_epoch = int(state["epoch"]) + 1
        best_nmse = float(state["best_validation_nmse_linear"])
        stale = int(state.get("stale_epochs", 0))
        history = list(state.get("history", []))
        stored_validation = state.get("validation")
        validation = stored_validation if isinstance(stored_validation, dict) else None
    metadata = {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
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
        "model_protocol": model.protocol_metadata(),
        "test_split_used": False,
    }
    (run_dir / "command.txt").write_text(command or shlex.join(sys.argv), encoding="utf-8")
    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(manifests / "data_semantics.json", semantics.to_dict())
    _write_json(manifests / "sample_indices.json", {"indices": sample_indices})
    _write_json(manifests / "prior.json", prior_metadata)
    max_epochs = config.epochs
    call_max_epochs = min(max_epochs, stop_after_epoch) if stop_after_epoch is not None else max_epochs
    started = time.perf_counter()
    epoch = start_epoch
    while epoch <= call_max_epochs:
        model.train()
        train_total, batches = 0.0, 0
        for raw in train_loader:
            batch = move_batch(raw, device)
            prior_value = prior.predict(batch) if prior is not None else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                prediction = model(batch, prior_value)
                prediction, target = _select(prediction, batch["target_h"], config.target_blocks)
                loss, _ = prist_ris_loss(
                    prediction, target, charbonnier_weight=config.charbonnier_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach())
            batches += 1
        validation = evaluate(
            model,
            validation_loader,
            device,
            prior=prior,
            target_blocks=config.target_blocks,
        )
        value = float(validation["nmse_linear"])
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
        }
        history.append(row)
        state = {
            "method": "PriST-RIS",
            "architecture_version": ARCHITECTURE_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_nmse_linear": best_nmse,
            "training_config": asdict(config),
            "model_config": asdict(model.config),
            "semantics_hash": semantics.stable_hash(),
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
        "model_key": config.model_key,
        "best_validation_nmse_linear": best_nmse,
        "best_validation_nmse_db": 10 * math.log10(max(best_nmse, 1e-12)),
        "epochs_completed": int(history[-1]["epoch"]),
        "wall_clock_seconds": time.perf_counter() - started,
        "last_validation": validation,
        "metadata": metadata,
        "test_split_used": False,
    }
    _write_json(results / "final_result.json", final)
    return {"run_dir": str(run_dir), "result": final}
