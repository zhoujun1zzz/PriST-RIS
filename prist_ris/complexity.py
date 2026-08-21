from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from .models import PriSTRIS, canonical_batch
from .contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
)


@torch.no_grad()
def profile_model(
    model: PriSTRIS,
    *,
    domain: str,
    device: torch.device,
    prior: torch.Tensor | None = None,
    latency_runs: int = 20,
) -> dict[str, Any]:
    batch = canonical_batch(domain, device=device)
    if model.uses_prior and prior is None:
        anchors = 1 if domain == "quasi" else 2
        prior = torch.zeros(1, anchors, 256, 64, 2, device=device)
    macs = 0
    hooks = []

    def count(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            kernel = 1
            for width in module.kernel_size:
                kernel *= width
            macs += int(output.numel() * (module.in_channels // module.groups) * kernel)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            hooks.append(module.register_forward_hook(count))
    output = model(batch, prior)
    for hook in hooks:
        hook.remove()
    batch_size = int(batch["obs_h"].shape[0])
    if model.uses_observed_dense_attention:
        # Per-antenna QK^T and attention-value products; antennas are not flattened.
        macs += 2 * batch_size * 64 * 256 * 32 * model.config.hidden
    if model.temporal is not None:
        queries = 4
        # One complex multiply-accumulate is counted as four real MACs.
        macs += 4 * batch_size * queries * model.config.temporal_rank * 256 * 64
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        model(batch, prior)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(latency_runs):
        model(batch, prior)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = 1000 * (time.perf_counter() - started) / latency_runs
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": (
            MOBILITY_CONTRACT_VERSION if domain == "mobility" else None
        ),
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "model_key": model.config.model_key,
        "domain": domain,
        "input_shape": list(batch["obs_h"].shape),
        "output_shape": list(output.shape),
        "stage_shapes": [list(shape) for shape in model.last_stage_shapes],
        "coordinate_enabled": model.legacy_coordinate_alias_value,
        "legacy_coordinate_alias_used": model.legacy_coordinate_alias_used,
        "backbone_ris_coordinate_enabled": model.config.backbone_ris_coordinate_enabled,
        "backbone_antenna_index_enabled": model.config.backbone_antenna_index_enabled,
        "backbone_ris_coordinate_mode": model.config.backbone_ris_coordinate_mode,
        "attention_enabled": model.config.attention_enabled,
        "attention_ris_coordinate_enabled": model.config.attention_ris_coordinate_enabled,
        "attention_antenna_index_enabled": model.config.attention_antenna_index_enabled,
        "observed_dense_attention": model.uses_observed_dense_attention,
        "observed_dense_attention_heads": (
            model.config.observed_dense_attention_heads
            if model.uses_observed_dense_attention
            else 0
        ),
        "spatial_residual_style": model.config.spatial_residual_style,
        "spatial_multiscale_supervision": model.config.spatial_multiscale_supervision,
        "spatial_channel_attention": model.config.spatial_channel_attention,
        "inference_graph_changed": model.config.spatial_channel_attention != "off",
        "training_only_mechanism": model.config.spatial_multiscale_supervision,
        "prior_anchors": model.anchor_count if model.uses_prior else 0,
        "spatial_anchor_time_index": list(model.spatial_anchor_time_index),
        "output_time_index": list(model.output_time_index),
        "temporal_rank": model.config.temporal_rank if model.temporal is not None else None,
        "temporal_base_mode": model.config.temporal_base_mode,
        "temporal_learned_residual_enabled": model.config.temporal_learned_residual_enabled,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "macs": macs,
        "gmacs": macs / 1e9,
        "flops": 2 * macs,
        "gflops": 2 * macs / 1e9,
        "latency_ms_batch1": latency_ms,
        "peak_gpu_memory_bytes": peak,
        "convention": (
            "batch1 FP32 single forward; Conv2d/Conv3d, linear, "
            "per-antenna observed-to-dense attention, and complex low-rank "
            "reconstruction; 1 real MAC=2 FLOPs"
        ),
    }
