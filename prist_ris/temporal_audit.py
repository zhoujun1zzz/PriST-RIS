from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import torch


def _energy(value: torch.Tensor) -> torch.Tensor:
    return value.square().sum(dim=(1, 2, 3))


def _complex_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_complex = torch.complex(left[..., 0].double(), left[..., 1].double())
    right_complex = torch.complex(right[..., 0].double(), right[..., 1].double())
    inner = (left_complex.conj() * right_complex).sum(dim=(1, 2)).abs()
    denominator = (
        left_complex.abs().square().sum(dim=(1, 2))
        * right_complex.abs().square().sum(dim=(1, 2))
    ).sqrt().clamp_min(1e-12)
    return inner / denominator


class TemporalAuditAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def _add(self, name: str, value: torch.Tensor) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value.double().sum())

    def update(self, target: torch.Tensor) -> None:
        if target.ndim != 5 or target.shape[1:] != (6, 256, 64, 2):
            raise ValueError("Temporal audit requires Mobility q0..q5 targets.")
        batch = target.shape[0]
        self.count += batch
        delta = target[:, 1:] - target[:, :-1]
        curvature = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        for time in range(5):
            self._add(
                f"adjacent_q{time}_q{time+1}_normalized_mse",
                _energy(delta[:, time]) / _energy(target[:, time]).clamp_min(1e-12),
            )
            self._add(
                f"adjacent_q{time}_q{time+1}_correlation",
                _complex_correlation(target[:, time], target[:, time + 1]),
            )
            self._add(f"delta_q{time}_q{time+1}_norm", _energy(delta[:, time]).sqrt())
        for time in range(6):
            difference = target[:, time] - target[:, 0]
            self._add(
                f"relative_q0_q{time}_normalized_difference",
                _energy(difference) / _energy(target[:, 0]).clamp_min(1e-12),
            )
            self._add(
                f"relative_q0_q{time}_correlation",
                _complex_correlation(target[:, 0], target[:, time]),
            )
        for time in range(1, 5):
            curvature_value = curvature[:, time - 1]
            adjacent_delta = delta[:, time - 1]
            self._add(f"curvature_q{time}_norm", _energy(curvature_value).sqrt())
            self._add(
                f"curvature_q{time}_squared_over_delta_squared",
                _energy(curvature_value) / _energy(adjacent_delta).clamp_min(1e-12),
            )

    def compute(self, split: str) -> dict[str, object]:
        if not self.count:
            raise RuntimeError("Temporal audit received no samples.")
        return {
            "split": split,
            "sample_count": self.count,
            "metrics": {
                name: total / self.count for name, total in sorted(self.sums.items())
            },
            "test_split_used": False,
        }


@torch.no_grad()
def audit_temporal_loaders(
    loaders: dict[str, Iterable[dict[str, torch.Tensor]]]
) -> dict[str, object]:
    if set(loaders) - {"train", "validation"}:
        raise PermissionError("Temporal audit is restricted to TRAIN/VALIDATION.")
    results = []
    for split in ("train", "validation"):
        accumulator = TemporalAuditAccumulator()
        for batch in loaders[split]:
            accumulator.update(batch["target_h"])
        results.append(accumulator.compute(split))
    return {
        "method": "PriST-RIS",
        "phase": "T0_temporal_statistics",
        "splits": results,
        "test_split_used": False,
    }


def write_temporal_audit(
    result: dict[str, object], json_path: str | Path, csv_path: str | Path
) -> None:
    destination = Path(json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for split in result["splits"]:  # type: ignore[index]
        for metric, value in split["metrics"].items():
            rows.append(
                {
                    "split": split["split"],
                    "sample_count": split["sample_count"],
                    "metric": metric,
                    "value": value,
                }
            )
    csv_destination = Path(csv_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "sample_count", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(rows)
