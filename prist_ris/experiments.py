from __future__ import annotations

import json
import statistics
from pathlib import Path


CAPACITY_HIDDEN = (64, 80, 96)
LEARNING_RATES = (2e-4, 5e-4, 1e-3)
TEMPORAL_RANKS = (2, 3)
MECHANISM_ABLATIONS = (
    "physical_grid_spatial",
    "prior_guided_dual_anchor",
    "coordinate_encoding",
    "trend_conditioned_temporal",
    "temporal_residual",
    "full",
)
TRANSFER_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 1.0)
TRANSFER_PROTOCOLS = (
    "target_only_scratch",
    "full_finetune",
    "frozen_spatial",
    "selective",
)

SPATIAL_ABLATION_TARGET_BLOCKS = (0, 3)
TEMPORAL_ABLATION_TARGET_BLOCKS = (0, 1, 2, 3, 4, 5)


def late_window_score(history_path: str | Path, start: int, end: int) -> dict[str, float | int]:
    import csv

    with Path(history_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["validation_nmse_linear"])
        for row in rows
        if start <= int(row["epoch"]) <= end
    ]
    if not values:
        raise ValueError(f"No validation rows in epoch window {start}-{end}.")
    return {
        "window_start": start,
        "window_end": end,
        "samples": len(values),
        "median_linear_nmse": statistics.median(values),
        "mean_linear_nmse": statistics.fmean(values),
        "best_linear_nmse": min(values),
    }


def targeted_tuning_plan(domain: str, seed: int = 123) -> dict[str, object]:
    return {
        "method": "PriST-RIS",
        "domain": domain,
        "selection_split": "validation",
        "test_split_used": False,
        "seed": seed,
        "fixed": {
            "architecture_version": "3.1",
            "blocks_per_stage": [3, 3, 4],
            "final_refine_blocks": 4,
            "physical_grid": True,
            "weight_decay": 1e-5,
        },
        "phase_a": {"hidden": list(CAPACITY_HIDDEN), "lr": 5e-4, "epochs": 30, "window": [25, 30]},
        "phase_b": {"learning_rate": list(LEARNING_RATES), "epochs": 40, "window": [31, 40]},
        "phase_c": {"temporal_rank": list(TEMPORAL_RANKS), "epochs": 40, "domain": "mobility_only"},
        "final": {"max_epochs": 100, "min_epochs": 40, "patience": 15, "seeds": [123, 456, 789]},
    }


def write_plan(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
