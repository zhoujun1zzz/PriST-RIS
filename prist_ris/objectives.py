from __future__ import annotations

import torch

from .metrics import sample_linear_nmse


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon).mean()


def prist_ris_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    charbonnier_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    nmse = sample_linear_nmse(prediction, target).mean()
    residual = charbonnier(prediction, target)
    total = nmse + charbonnier_weight * residual
    return total, {
        "nmse": float(nmse.detach()),
        "charbonnier": float(residual.detach()),
        "total": float(total.detach()),
    }


def temporal_deltas(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5 or value.shape[1] < 2:
        raise ValueError("Temporal deltas require [B,T,N,A,2] with T>=2.")
    return value[:, 1:] - value[:, :-1]


def temporal_curvatures(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5 or value.shape[1] < 3:
        raise ValueError("Temporal curvature requires [B,T,N,A,2] with T>=3.")
    return value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]


def temporal_regularized_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    charbonnier_weight: float,
    delta_weight: float = 0.0,
    curvature_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if delta_weight < 0 or curvature_weight < 0:
        raise ValueError("Temporal loss weights must be non-negative.")
    total, components = prist_ris_loss(
        prediction, target, charbonnier_weight=charbonnier_weight
    )
    result = {f"reconstruction_{key}": value for key, value in components.items()}
    if delta_weight > 0:
        delta, _ = prist_ris_loss(
            temporal_deltas(prediction),
            temporal_deltas(target),
            charbonnier_weight=charbonnier_weight,
        )
        total = total + delta_weight * delta
        result["delta"] = float(delta.detach())
    else:
        result["delta"] = 0.0
    if curvature_weight > 0:
        curvature, _ = prist_ris_loss(
            temporal_curvatures(prediction),
            temporal_curvatures(target),
            charbonnier_weight=charbonnier_weight,
        )
        total = total + curvature_weight * curvature
        result["curvature"] = float(curvature.detach())
    else:
        result["curvature"] = 0.0
    result["total"] = float(total.detach())
    return total, result
