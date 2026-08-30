from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from ..contracts import DataSemantics
from ..metrics import MetricAccumulator, PerQueryMetricAccumulator
from ..prior import RidgePrior, RidgeStatistics


@dataclass
class RidgeLinearBaseline:
    """q0/q3 spatial Ridge anchors followed by deterministic linear time recovery."""

    prior: RidgePrior

    def __post_init__(self) -> None:
        if tuple(self.prior.target_blocks) != (0, 3):
            raise ValueError(
                "Ridge-Linear requires RidgePrior target_blocks=(0, 3)."
            )

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        anchors = self.prior.predict(batch)

        if anchors.ndim != 5 or anchors.shape[1] != 2:
            raise ValueError(
                "Ridge anchor prediction must have shape [B,2,256,64,2]."
            )

        a0 = anchors[:, 0]
        a3 = anchors[:, 1]
        difference = a3 - a0

        query_time = batch["query_time"]
        if query_time.ndim > 1:
            query_time = query_time[0]

        if tuple(int(v) for v in query_time.tolist()) != tuple(range(6)):
            raise ValueError(
                "Ridge-Linear requires canonical queries q0,...,q5."
            )

        alpha = (
            query_time.to(
                device=anchors.device,
                dtype=anchors.dtype,
            )
            / 3.0
        )

        return (
            a0[:, None]
            + alpha[None, :, None, None, None]
            * difference[:, None]
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[dict[str, torch.Tensor]],
    ) -> dict[str, object]:
        metrics = PerQueryMetricAccumulator(tuple(range(6)))

        for batch in loader:
            prediction = self.predict(batch)
            metrics.update(prediction, batch["target_h"])

        result = metrics.compute()
        result["regularization"] = float(self.prior.regularization)
        result["ridge_target_blocks"] = [0, 3]
        result["temporal_rule"] = "A0 + (t/3) * (A3-A0)"
        return result


@torch.no_grad()
def evaluate_ridge_anchors(
    prior: RidgePrior,
    loader: Iterable[dict[str, torch.Tensor]],
) -> dict[str, object]:
    """Canonical PriST-RIS Ridge selection metric: q0/q3 anchors only."""

    if tuple(prior.target_blocks) != (0, 3):
        raise ValueError(
            "Anchor evaluation requires target_blocks=(0, 3)."
        )

    overall = MetricAccumulator()
    diagnostics = PerQueryMetricAccumulator((0, 3))
    blocks = torch.tensor((0, 3), dtype=torch.long)

    for batch in loader:
        prediction = prior.predict(batch)
        target = batch["target_h"].index_select(1, blocks)

        overall.update(prediction, target)
        diagnostics.update(prediction, target)

    return {
        **overall.compute(),
        "diagnostics": diagnostics.compute(),
    }


def fit_ridge_linear(
    train_loader: Iterable[dict[str, torch.Tensor]],
    validation_loader: Iterable[dict[str, torch.Tensor]],
    *,
    lambdas: Sequence[float],
    semantics: DataSemantics,
) -> tuple[RidgeLinearBaseline, dict[str, object]]:
    """Fit TRAIN Ridge and select lambda by q0/q3 VALIDATION anchor NMSE."""

    candidates = tuple(float(value) for value in lambdas)

    if not candidates:
        raise ValueError("At least one Ridge lambda is required.")
    if any(value < 0.0 for value in candidates):
        raise ValueError("Ridge lambdas must be non-negative.")

    statistics = RidgeStatistics.accumulate(
        train_loader,
        target_blocks=(0, 3),
    )

    rows: list[dict[str, object]] = []
    best_prior: RidgePrior | None = None
    best_linear = float("inf")

    for regularization in candidates:
        prior = statistics.solve(
            regularization,
            semantics,
        )

        anchor_validation = evaluate_ridge_anchors(
            prior,
            validation_loader,
        )

        linear_nmse = float(
            anchor_validation["nmse_linear"]
        )

        rows.append(
            {
                "regularization": regularization,
                "anchor_validation_nmse_linear": linear_nmse,
                "anchor_validation_nmse_db": float(
                    anchor_validation["nmse_db"]
                ),
                "anchor_validation": anchor_validation,
            }
        )

        if linear_nmse < best_linear:
            best_linear = linear_nmse
            best_prior = prior

    if best_prior is None:
        raise RuntimeError(
            "Ridge lambda selection produced no model."
        )

    model = RidgeLinearBaseline(best_prior)

    report: dict[str, object] = {
        "baseline": "Ridge-Linear",
        "fit_split": "train",
        "selection_split": "validation",
        "selection_target_blocks": [0, 3],
        "selection_metric": "q0/q3 anchor linear NMSE",
        "ridge_target_blocks": [0, 3],
        "spatial_mapping": "2P -> 2N complex Ridge",
        "temporal_rule": "A0 + (t/3) * (A3-A0)",
        "candidates": rows,
        "selected_regularization": float(
            best_prior.regularization
        ),
        "selected_anchor_validation_nmse_linear": best_linear,
    }

    return model, report
