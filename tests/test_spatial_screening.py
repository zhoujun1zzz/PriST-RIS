from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from prist_ris.cli import parser
from prist_ris.screening import (
    SPATIAL_SCREENING_CANDIDATES,
    accuracy_complexity_pareto,
    spatial_candidate_training_arguments,
    spatial_screening_plan,
    summarize_spatial_screening,
)


def test_s1_plan_has_only_the_fixed_width_depth_candidates() -> None:
    plan = spatial_screening_plan()
    assert [candidate["name"] for candidate in plan["candidates"]] == [
        "B64",
        "B48",
        "D1",
        "D2",
    ]
    assert plan["reference"]["name"] == "B80"
    assert plan["reference"]["reuse_existing"] is True
    assert plan["fixed_protocol"]["test_split_used"] is False
    assert plan["fixed_protocol"]["epochs"] == 30
    assert plan["canonical_changed"] is False


def test_s1_training_command_is_fixed_validation_only_and_stops_at_30() -> None:
    arguments = spatial_candidate_training_arguments(
        SPATIAL_SCREENING_CANDIDATES[0],
        prior="ridge.npz",
        data_root="data",
        output_root="runs",
    )
    rendered = [str(value) for value in arguments]
    assert rendered[rendered.index("--model") + 1] == "prist_ris_b"
    assert rendered[rendered.index("--hidden") + 1] == "64"
    assert rendered[rendered.index("--epochs") + 1] == "30"
    assert rendered[rendered.index("--stop-after-epoch") + 1] == "30"
    assert rendered[rendered.index("--target-blocks") + 1] == "0,3"
    assert "--no-coordinate-enabled" in rendered
    assert "test" not in rendered
    assert "--amp" not in rendered


def test_accuracy_gmac_pareto_rejects_dominated_candidates() -> None:
    rows = [
        {"candidate": "accurate", "best_validation_nmse_db": -19.5, "gmacs": 10.0},
        {"candidate": "light", "best_validation_nmse_db": -19.3, "gmacs": 5.0},
        {"candidate": "dominated", "best_validation_nmse_db": -19.0, "gmacs": 12.0},
    ]
    assert accuracy_complexity_pareto(rows) == ["accurate", "light"]


def test_screening_summary_reads_best_epoch_late_trend_and_profile(tmp_path: Path) -> None:
    root = tmp_path / "study"
    run = root / "runs" / "B64"
    results = run / "results"
    checkpoints = run / "checkpoints"
    profiles = root / "profiles"
    results.mkdir(parents=True)
    checkpoints.mkdir()
    profiles.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "seed": 123,
                "hidden": 64,
                "blocks_per_stage": [3, 3, 4],
                "final_refine_blocks": 4,
            }
        ),
        encoding="utf-8",
    )
    (results / "final_result.json").write_text(
        json.dumps(
            {
                "architecture_version": "3.2",
                "spatial_protocol_version": "physical_stable_residual_v2",
                "best_validation_nmse_db": -19.4,
                "last_validation": {"nmse_db": -19.35},
                "wall_clock_seconds": 120.0,
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    with (results / "training_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("epoch", "validation_nmse_linear", "validation_nmse_db")
        )
        writer.writeheader()
        for epoch in range(21, 31):
            writer.writerow(
                {
                    "epoch": epoch,
                    "validation_nmse_linear": 0.02,
                    "validation_nmse_db": -18.0 - 0.1 * (epoch - 20),
                }
            )
    torch.save(
        {
            "epoch": 29,
            "validation": {
                "diagnostics": {
                    "per_query": {
                        "q0": {"nmse_db": -19.3},
                        "q3": {"nmse_db": -19.5},
                    }
                }
            },
        },
        checkpoints / "best_checkpoint.pth",
    )
    (profiles / "B64.json").write_text(
        json.dumps(
            {
                "parameters": 100,
                "trainable_parameters": 100,
                "gmacs": 4.0,
                "gflops": 8.0,
                "latency_ms_batch1": 2.0,
                "peak_gpu_memory_bytes": 1024,
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_spatial_screening(root)
    row = summary["rows"][0]
    assert row["candidate"] == "B64"
    assert row["best_epoch"] == 29
    assert row["epoch30_validation_nmse_db"] == -19.0
    assert row["late_window_improvement_db"] == pytest.approx(0.9)
    assert row["q0_best_nmse_db"] == -19.3
    assert row["gmacs"] == 4.0
    assert summary["accuracy_gmac_pareto_frontier"] == ["B64"]
    assert summary["missing_candidates"] == ["B48", "D1", "D2"]
    assert summary["test_split_used"] is False


def test_cli_exposes_spatial_screen_without_enabling_execution() -> None:
    args = parser().parse_args(["screen-spatial", "--data-root", "data"])
    assert args.execute is False
    assert args.summarize_only is False
    assert args.physical_gpu_index == 3
