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


class PerQueryMetricAccumulator:
    """Diagnostic query metrics plus energy-correct aggregate ratios."""

    def __init__(self, query_count: int) -> None:
        self.query_count = query_count
        self.per_query = [MetricAccumulator() for _ in range(query_count)]
        self.observed = MetricAccumulator() if query_count == 6 else None
        self.future = MetricAccumulator() if query_count == 6 else None
        self.overall = MetricAccumulator()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape[1] != self.query_count or target.shape[1] != self.query_count:
            raise ValueError("Per-query accumulator received an unexpected query count.")
        self.overall.update(prediction, target)
        for query in range(self.query_count):
            self.per_query[query].update(prediction[:, query : query + 1], target[:, query : query + 1])
        if self.observed is not None and self.future is not None:
            self.observed.update(prediction[:, :2], target[:, :2])
            self.future.update(prediction[:, 2:], target[:, 2:])

    def compute(self) -> dict[str, object]:
        result: dict[str, object] = {
            "per_query": {f"q{index}": metric.compute() for index, metric in enumerate(self.per_query)},
            "overall": self.overall.compute(),
        }
        if self.query_count == 2:
            result["observed_anchor_aggregate"] = self.overall.compute()
        if self.observed is not None and self.future is not None:
            result["observed_anchor_aggregate"] = self.observed.compute()
            result["future_aggregate"] = self.future.compute()
        return result
