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

    def __init__(self, query_time_index: int | tuple[int, ...]) -> None:
        self.query_time_index = (
            tuple(range(query_time_index))
            if isinstance(query_time_index, int)
            else tuple(query_time_index)
        )
        self.query_count = len(self.query_time_index)
        self.per_query = [MetricAccumulator() for _ in self.query_time_index]
        self.pilot_positions = tuple(
            position
            for position, semantic_time in enumerate(self.query_time_index)
            if semantic_time in {0, 3}
        )
        self.non_pilot_positions = tuple(
            position
            for position, semantic_time in enumerate(self.query_time_index)
            if semantic_time not in {0, 3}
        )
        self.pilots = MetricAccumulator() if self.pilot_positions else None
        self.non_pilots = MetricAccumulator() if self.non_pilot_positions else None
        self.overall = MetricAccumulator()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape[1] != self.query_count or target.shape[1] != self.query_count:
            raise ValueError("Per-query accumulator received an unexpected query count.")
        self.overall.update(prediction, target)
        for query in range(self.query_count):
            self.per_query[query].update(prediction[:, query : query + 1], target[:, query : query + 1])
        if self.pilots is not None:
            index = torch.tensor(self.pilot_positions, device=prediction.device)
            self.pilots.update(
                prediction.index_select(1, index), target.index_select(1, index)
            )
        if self.non_pilots is not None:
            index = torch.tensor(self.non_pilot_positions, device=prediction.device)
            self.non_pilots.update(
                prediction.index_select(1, index), target.index_select(1, index)
            )

    def compute(self) -> dict[str, object]:
        result: dict[str, object] = {
            "per_query": {
                f"q{semantic_time}": metric.compute()
                for semantic_time, metric in zip(
                    self.query_time_index, self.per_query, strict=True
                )
            },
            "overall": self.overall.compute(),
        }
        if self.pilots is not None:
            result["pilot_anchor_aggregate"] = self.pilots.compute()
            result["observed_anchor_aggregate"] = self.pilots.compute()
        if self.non_pilots is not None:
            result["non_pilot_aggregate"] = self.non_pilots.compute()
        return result
