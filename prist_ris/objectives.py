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
