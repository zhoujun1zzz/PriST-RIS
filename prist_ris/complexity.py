from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from .models import PriSTRIS, canonical_batch


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
        prior = torch.zeros(1, 1, 256, 64, 2, device=device)
    macs = 0
    hooks = []

    def count(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs += int(output.numel() * (module.in_channels // module.groups) * kernel)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count))
    output = model(batch, prior)
    for hook in hooks:
        hook.remove()
    batch_size = int(batch["obs_h"].shape[0])
    if model.cross_attention is not None:
        hidden = model.config.hidden
        antennas, queries, keys = 64, 256, 32
        # Q/K/V/out projections plus QK^T and attention-value products.
        macs += batch_size * antennas * (
            queries * hidden * hidden
            + 2 * keys * hidden * hidden
            + queries * hidden * hidden
            + 2 * queries * keys * hidden
        )
    if model.temporal is not None:
        queries = 1 if domain == "quasi" else 6
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
        "model_key": model.config.model_key,
        "domain": domain,
        "input_shape": list(batch["obs_h"].shape),
        "output_shape": list(output.shape),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "macs": macs,
        "gmacs": macs / 1e9,
        "flops": 2 * macs,
        "gflops": 2 * macs / 1e9,
        "latency_ms_batch1": latency_ms,
        "peak_gpu_memory_bytes": peak,
        "convention": (
            "batch1 FP32 single forward; convolution, linear, attention contractions, "
            "and complex low-rank reconstruction; 1 real MAC=2 FLOPs"
        ),
    }
