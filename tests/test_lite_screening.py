from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from prist_ris.contracts import DataSemantics
from prist_ris.lite_screening import (
    LITE_CANDIDATES,
    LITE_FRACTION,
    LITE_SAMPLE_COUNT,
    SPATIAL_SCOPE_CAVEAT,
    build_candidate_model,
    build_lite_plan,
    decide_run,
    execute_lite_plan,
    fixed_training_config,
    summarize_lite_plan,
    validate_fraction_prior,
    validate_sample_manifest,
)
from prist_ris.paper_matrix import (
    PAPER_OPTIMIZER_CONFIG,
    PAPER_SPATIAL_CONFIG,
    indices_hash,
)
from prist_ris.prior import RidgePrior, file_sha256


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    indices = list(range(LITE_SAMPLE_COUNT))
    manifest = tmp_path / "seed123.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "prist_ris.paper_matrix.v1",
                "seed": 123,
                "nested": True,
                "total_train_samples": 20000,
                "fractions": {"0.25": indices},
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    prior = RidgePrior(
        coefficients=np.zeros((64, 512), dtype=np.complex128),
        regularization=1e-4,
        rows=64,
        target_blocks=(0, 3),
        semantics_hash=DataSemantics.for_domain("mobility").stable_hash(),
        provenance={
            "seed": 123,
            "fraction": 0.25,
            "sample_count": 5000,
            "sample_manifest_sha256": file_sha256(manifest),
            "indices_hash": indices_hash(indices),
            "selection_split": "validation",
            "test_split_used": False,
        },
    )
    prior_path = tmp_path / "prior.npz"
    prior.save(prior_path)
    return manifest, prior_path


def _plan(tmp_path: Path) -> dict[str, object]:
    manifest, prior = _artifacts(tmp_path)
    return build_lite_plan(
        tmp_path / "lite",
        prior_path=prior,
        sample_manifest_path=manifest,
        data_root="data",
        workers=0,
        project_root=tmp_path,
        head="abc123",
        run_profiles=False,
    )


def test_lite_candidates_are_fixed_to_only_a_and_b() -> None:
    assert [(value.candidate, value.hidden) for value in LITE_CANDIDATES] == [
        ("Lite-A", 32),
        ("Lite-B", 40),
    ]
    assert all(value.blocks_per_stage == (2, 2, 1) for value in LITE_CANDIDATES)
    assert all(value.final_refine_blocks == 1 for value in LITE_CANDIDATES)


@pytest.mark.parametrize("candidate", LITE_CANDIDATES)
def test_lite_model_configuration_is_exact(candidate) -> None:
    config = fixed_training_config(candidate)
    assert config["model_key"] == "prist_ris_b"
    assert config["spatial_channel_attention"] == "se"
    assert config["backbone_ris_coordinate_enabled"] is True
    assert config["backbone_ris_coordinate_mode"] == "direct_add"
    assert config["backbone_antenna_index_enabled"] is False
    assert config["attention_enabled"] is False
    assert config["spatial_multiscale_supervision"] is False
    assert config["spatial_residual_style"] == "scaled_true_residual"
    assert config["epochs"] == 100 and config["min_epochs"] == 101
    assert config["amp"] is False and config["test_split_used"] is False
    model = build_candidate_model(candidate)
    protocol = model.protocol_metadata()
    assert model.config.hidden == candidate.hidden
    assert model.config.blocks_per_stage == (2, 2, 1)
    assert protocol["spatial_channel_attention"] == "se"
    assert protocol["backbone_ris_coordinate_mode"] == "direct_add"
    assert protocol["attention_enabled"] is False


def test_plan_is_cpu_only_exact_two_candidates_and_no_test(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan["planner_uses_gpu"] is False
    assert plan["architecture_search"] is False
    assert plan["test_split_used"] is False
    assert plan["seed"] == 123
    assert plan["fraction"] == LITE_FRACTION
    assert plan["sample_count"] == LITE_SAMPLE_COUNT
    assert plan["target_scope"] == "mobility_q0_q3"
    assert [value["candidate"] for value in plan["experiments"]] == [
        "Lite-A",
        "Lite-B",
    ]
    for command in plan["commands"].values():
        joined = " ".join(command)
        assert "--epochs 100" in joined
        assert "--min-epochs 101" in joined
        assert "--target-blocks 0,3" in joined
        assert "--amp" not in command
        assert "--split" not in command and "--include-test" not in command
    assert plan["scope_caveat"] == SPATIAL_SCOPE_CAVEAT
    assert plan["temporal_lite_implemented"] is False


def test_manifest_and_fraction_prior_provenance_are_locked(tmp_path: Path) -> None:
    manifest_path, prior_path = _artifacts(tmp_path)
    manifest = validate_sample_manifest(manifest_path)
    prior = validate_fraction_prior(prior_path, manifest)
    assert manifest["sample_count"] == 5000
    assert manifest["fraction"] == 0.25
    assert prior["sample_manifest_sha256"] == manifest["sha256"]
    assert prior["subset_indices_hash"] == manifest["subset_indices_hash"]
    assert prior["test_split_used"] is False
    changed = dict(manifest)
    changed["subset_indices_hash"] = "wrong"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_fraction_prior(prior_path, changed)


def test_completed_exact_spec_reuses_and_incomplete_refuses(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    spec = plan["experiments"][0]
    run = Path(spec["run_dir"])
    assert decide_run(spec, resume_incomplete=False)["action"] == "run"
    run.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Incomplete Lite"):
        decide_run(spec, resume_incomplete=False)
    (run / "paper_experiment_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "last_checkpoint.pth").write_bytes(b"checkpoint")
    assert decide_run(spec, resume_incomplete=True)["action"] == "resume"
    (run / "results").mkdir()
    (run / "results" / "final_result.json").write_text(
        json.dumps({"test_split_used": False}), encoding="utf-8"
    )
    assert decide_run(spec, resume_incomplete=False)["action"] == "reuse"


def test_summary_best_epoch_thresholds_and_pareto_without_winner(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["profiles"] = [
        {
            "candidate": "Lite-A",
            "parameters": 10,
            "trainable_parameters": 10,
            "gmacs": 2.0,
            "gflops": 4.0,
        },
        {
            "candidate": "Lite-B",
            "parameters": 20,
            "trainable_parameters": 20,
            "gmacs": 3.0,
            "gflops": 6.0,
        },
    ]
    values = {
        "Lite-A": (-17.5, -18.2, -19.1),
        "Lite-B": (-17.8, -18.5, -18.4),
    }
    for spec in plan["experiments"]:
        run = Path(spec["run_dir"])
        (run / "results").mkdir(parents=True)
        history = [
            {
                "epoch": index,
                "validation_nmse_linear": 10 ** (db / 10),
                "validation_nmse_db": db,
                "wall_clock_seconds": float(index * 100),
            }
            for index, db in enumerate(values[spec["candidate"]], 1)
        ]
        with (run / "results" / "training_history.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0])
            writer.writeheader()
            writer.writerows(history)
        (run / "results" / "final_result.json").write_text(
            json.dumps(
                {
                    "last_validation": {"nmse_db": values[spec["candidate"]][-1]},
                    "wall_clock_seconds": 300.0,
                    "test_split_used": False,
                }
            ),
            encoding="utf-8",
        )
    summary = summarize_lite_plan(tmp_path / "lite", plan)
    a = next(row for row in summary["results"] if row["candidate"] == "Lite-A")
    assert a["best_epoch"] == 3
    assert a["threshold_crossings"]["-18.0"]["epoch"] == 2
    assert a["threshold_crossings"]["-20.0"] is None
    assert summary["pareto"] == {
        "Lite-A_dominated_by_Lite-B": False,
        "Lite-B_dominated_by_Lite-A": True,
    }
    assert summary["winner"] is None
    assert summary["selection_requires_human_judgment"] is True


def test_gpu_preflight_rejects_wrong_visible_device(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    plan["profiles"] = [
        {"candidate": "Lite-A", "test_split_used": False},
        {"candidate": "Lite-B", "test_split_used": False},
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(PermissionError, match="CUDA_VISIBLE_DEVICES=0"):
        execute_lite_plan(
            plan,
            data_root="data",
            device="cuda:0",
            workers=0,
            physical_gpu_index=0,
            confirm_gpu_free=True,
            resume_incomplete=False,
            invoke=lambda _: None,
        )


def test_lite_planning_does_not_mutate_frozen_paper_matrix(tmp_path: Path) -> None:
    spatial = deepcopy(PAPER_SPATIAL_CONFIG)
    optimizer = deepcopy(PAPER_OPTIMIZER_CONFIG)
    _plan(tmp_path)
    assert PAPER_SPATIAL_CONFIG == spatial
    assert PAPER_OPTIMIZER_CONFIG == optimizer
