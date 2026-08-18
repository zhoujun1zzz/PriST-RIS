from __future__ import annotations

import math

import torch


def sample_linear_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Prediction and target must be finite.")
    reduce = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target).square(), dim=reduce)
    denominator = torch.sum(target.square(), dim=reduce).clamp_min(1e-12)
    value = numerator / denominator
    if not torch.isfinite(value).all():
        raise FloatingPointError("Sample NMSE is non-finite.")
    return value


class MetricAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        values = sample_linear_nmse(prediction, target).detach().double().cpu()
        self.total += float(values.sum())
        self.count += int(values.numel())

    def compute(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("No samples accumulated.")
        linear = self.total / self.count
        return {
            "sample_count": self.count,
            "nmse_linear": linear,
            "nmse_db": 10 * math.log10(max(linear, 1e-12)),
        }
