from __future__ import annotations

import json
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prist_ris.contracts import DataSemantics
from prist_ris.paper_matrix import (
    CONVERGENCE_THRESHOLDS_DB,
    PAPER_DATA_FRACTIONS,
    PAPER_SEEDS,
    PAPER_TRANSFER_FRACTIONS,
    ExperimentSpec,
    build_paper_matrix_plan,
    decide_run,
    first_threshold_crossings,
    frozen_protocol_manifest,
    indices_hash,
    require_temporal_binding,
    summarize_paper_matrix,
    training_arguments,
    validate_prior_artifact,
)
from prist_ris.prior import RidgePrior


def _plan(tmp_path: Path, phase: str = "data-efficiency") -> dict[str, object]:
    return build_paper_matrix_plan(
        tmp_path,
        phase=phase,
        seeds=(123,),
        project_root=tmp_path,
        git_head="abc123",
    )


def _spec(plan: dict[str, object], method: str, fraction: float) -> ExperimentSpec:
    return next(
        ExperimentSpec.from_dict(value)
        for value in plan["experiments"]  # type: ignore[index]
        if value["method_variant"] == method and value["fraction"] == fraction
    )


def test_paper_constants_and_seed123_data_plan_counts(tmp_path: Path) -> None:
    assert PAPER_SEEDS == (123, 456, 789)
    assert PAPER_DATA_FRACTIONS == (0.10, 0.25, 0.50, 1.00)
    plan = _plan(tmp_path)
    assert plan["gpu_runs"] == 8
    assert plan["planner_uses_gpu"] is False
    assert plan["test_split_used"] is False
    counts = {
        value["fraction"]: value["sample_count"]
        for value in plan["prior_jobs"]  # type: ignore[index]
    }
    assert counts == {0.10: 2000, 0.25: 5000, 0.50: 10000, 1.00: 20000}
    assert all(value["test_split_used"] is False for value in plan["experiments"])


def test_data_subsets_are_deterministic_nested_and_per_seed(tmp_path: Path) -> None:
    _plan(tmp_path)
    payload = json.loads(
        (tmp_path / "manifests" / "data_efficiency_seed123.json").read_text(
            encoding="utf-8"
        )
    )
    fractions = payload["fractions"]
    assert set(fractions["0.10"]).issubset(fractions["0.25"])
    assert set(fractions["0.25"]).issubset(fractions["0.50"])
    assert set(fractions["0.50"]).issubset(fractions["1.00"])
    second = build_paper_matrix_plan(
        tmp_path / "other",
        phase="data-efficiency",
        seeds=(456,),
        git_head="abc123",
    )
    assert second["seeds"] == [456]
    other = json.loads(
        (tmp_path / "other" / "manifests" / "data_efficiency_seed456.json").read_text()
    )
    assert fractions["0.10"] != other["fractions"]["0.10"]


def test_prior_fairness_and_direct_prior_configs_match(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    direct = _spec(plan, "direct_s3", 0.10)
    prior10 = _spec(plan, "prior_s3", 0.10)
    prior25 = _spec(plan, "prior_s3", 0.25)
    assert prior10.prior_path != prior25.prior_path
    assert "fraction_0.10.npz" in str(prior10.prior_path)
    assert "fraction_0.25.npz" in str(prior25.prior_path)
    assert direct.sample_manifest == prior10.sample_manifest
    assert direct.fraction == prior10.fraction
    assert direct.training_config == prior10.training_config
    assert direct.model_key == "prist_ris_a" and direct.prior_path is None
    assert prior10.model_key == "prist_ris_b" and prior10.prior_path is not None


def test_transfer_plan_is_three_fractions_by_three_protocols(tmp_path: Path) -> None:
    assert PAPER_TRANSFER_FRACTIONS == (0.05, 0.10, 0.25)
    plan = _plan(tmp_path, "transfer")
    assert plan["gpu_runs"] == 9
    experiments = plan["experiments"]
    assert {value["fraction"] for value in experiments} == {0.05, 0.10, 0.25}
    assert {value["method_variant"] for value in experiments} == {
        "scratch", "full_finetune", "selective"
    }
    assert all(value["target_scope"] == "mobility_q0_q3" for value in experiments)
    assert all(value["test_split_used"] is False for value in experiments)


def test_data_and_transfer_plans_can_share_one_root(tmp_path: Path) -> None:
    _plan(tmp_path, "data-efficiency")
    _plan(tmp_path, "transfer")
    assert (tmp_path / "paper_matrix_plan_data_efficiency.json").is_file()
    assert (tmp_path / "paper_matrix_plan_transfer.json").is_file()


def test_convergence_threshold_direction_and_first_crossing() -> None:
    history = [
        {
            "epoch": epoch,
            "validation_nmse_db": value,
            "wall_clock_seconds": float(epoch * 100),
        }
        for epoch, value in enumerate((-17.0, -18.1, -18.9, -19.2, -20.1), 1)
    ]
    result = first_threshold_crossings(history)
    assert CONVERGENCE_THRESHOLDS_DB == (-18.0, -19.0, -20.0)
    assert result["-18.0"] == {"epoch": 2, "wall_clock_seconds": 200.0}
    assert result["-19.0"] == {"epoch": 4, "wall_clock_seconds": 400.0}
    assert result["-20.0"] == {"epoch": 5, "wall_clock_seconds": 500.0}


def test_completed_reuse_incomplete_refusal_and_spec_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    spec = _spec(plan, "direct_s3", 0.10)
    run = Path(spec.run_dir)
    assert decide_run(spec, resume_incomplete=False)["action"] == "run"
    run.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Incomplete"):
        decide_run(spec, resume_incomplete=False)
    (run / "paper_experiment_spec.json").write_text(
        json.dumps(spec.to_dict()), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="no resumable checkpoint"):
        decide_run(spec, resume_incomplete=True)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "last_checkpoint.pth").write_bytes(b"checkpoint")
    assert decide_run(spec, resume_incomplete=True)["action"] == "resume"
    (run / "results").mkdir()
    (run / "results" / "final_result.json").write_text("{}")
    assert decide_run(spec, resume_incomplete=False)["action"] == "reuse"
    with pytest.raises(ValueError, match="spec mismatch"):
        decide_run(replace(spec, seed=456), resume_incomplete=False)


def test_frozen_manifest_can_plan_with_pending_temporal_but_run_guard_rejects() -> None:
    manifest = frozen_protocol_manifest("abc123")
    assert manifest["test_split_used"] is False
    assert manifest["temporal"]["final_checkpoint"] is None  # type: ignore[index]
    with pytest.raises(RuntimeError, match="not bound"):
        require_temporal_binding(manifest)
    manifest["temporal"] = {
        "final_checkpoint": "t2.pth", "anchor_cache": "cache"
    }
    assert require_temporal_binding(manifest) == ("t2.pth", "cache")


def test_fraction_prior_metadata_roundtrip_and_mismatch_rejection(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    job = plan["prior_jobs"][0]  # type: ignore[index]
    coefficients = np.zeros((64, 512), dtype=np.complex128)
    prior = RidgePrior(
        coefficients=coefficients,
        regularization=1e-4,
        rows=64,
        target_blocks=(0, 3),
        semantics_hash=DataSemantics.for_domain("mobility").stable_hash(),
        provenance={
            "seed": job["seed"],
            "fraction": job["fraction"],
            "sample_count": job["sample_count"],
            "sample_manifest_sha256": job["sample_manifest_sha256"],
            "indices_hash": job["indices_hash"],
            "selection_split": "validation",
            "test_split_used": False,
        },
    )
    artifact = tmp_path / "prior.npz"
    prior.save(artifact)
    validate_prior_artifact(artifact, job)
    bad = dict(job)
    bad["fraction"] = 0.25
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_prior_artifact(artifact, bad)


def test_training_command_freezes_cosine_and_no_test(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    spec = _spec(plan, "prior_s3", 0.10)
    arguments = [
        str(value)
        for value in training_arguments(
            spec,
            data_root="data",
            device="cpu",
            workers=0,
            spec_path="spec.json",
        )
    ]
    assert arguments[arguments.index("--scheduler") + 1] == "cosine"
    assert arguments[arguments.index("--min-learning-rate") + 1] == "5e-06"
    assert arguments[arguments.index("--epochs") + 1] == "100"
    assert "test" not in arguments


def test_summary_writes_paper_csvs_and_convergence_rows(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    spec = _spec(plan, "direct_s3", 0.10)
    run = Path(spec.run_dir)
    (run / "results").mkdir(parents=True)
    (run / "results" / "final_result.json").write_text(
        json.dumps(
            {
                "best_validation_nmse_db": -19.2,
                "last_validation": {"nmse_db": -19.1},
                "wall_clock_seconds": 300.0,
                "metadata": {
                    "total_parameters": 10,
                    "trainable_parameters": 10,
                },
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    with (run / "results" / "training_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch", "validation_nmse_linear", "validation_nmse_db",
                "wall_clock_seconds",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "epoch": 1,
                    "validation_nmse_linear": 0.02,
                    "validation_nmse_db": -17.0,
                    "wall_clock_seconds": 100.0,
                },
                {
                    "epoch": 2,
                    "validation_nmse_linear": 0.012,
                    "validation_nmse_db": -19.2,
                    "wall_clock_seconds": 200.0,
                },
            ]
        )
    summary = summarize_paper_matrix(tmp_path, plan)
    assert len(summary["data_efficiency"]) == 1
    assert summary["test_split_used"] is False
    names = {path.name for path in (tmp_path / "summaries").iterdir()}
    assert names == {
        "summary.json", "data_efficiency.csv", "convergence_efficiency.csv",
        "transfer.csv", "complexity.csv",
    }
    convergence = summary["convergence_efficiency"]
    assert next(row for row in convergence if row["threshold_db"] == "-18.0")["epoch"] == 2
    assert next(row for row in convergence if row["threshold_db"] == "-20.0")["epoch"] is None
