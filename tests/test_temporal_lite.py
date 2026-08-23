from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from prist_ris.contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    DataSemantics,
)
from prist_ris.engine import load_mobility_spatial_reference
from prist_ris.complexity import profile_model
from prist_ris.models import build_model, canonical_batch
from prist_ris.paper_matrix import indices_hash
from prist_ris.prior import RidgePrior, file_sha256
from prist_ris.temporal_lite import (
    LPAN_L_GMACS,
    LPAN_L_PARAMETERS,
    REFERENCE_METADATA,
    REFERENCE_NMSE_DB,
    TEMPORAL_LITE_TRAIN_COUNT,
    TL24,
    build_temporal_lite_plan,
    build_tl24_model,
    decide_anchor_cache,
    decide_temporal_run,
    execute_temporal_lite_plan,
    fixed_tl24_training_config,
    profile_temporal_lite,
    profile_ridge_complexity,
    summarize_temporal_lite,
    validate_full_prior,
    validate_full_sample_manifest,
    validate_lite_a_checkpoint,
)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    indices = list(range(TEMPORAL_LITE_TRAIN_COUNT))
    manifest = tmp_path / "seed123_full.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "prist_ris.paper_matrix.v1",
                "seed": 123,
                "nested": True,
                "total_train_samples": TEMPORAL_LITE_TRAIN_COUNT,
                "fractions": {"1.00": indices},
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    provenance = {
        "seed": 123,
        "fraction": 1.0,
        "sample_count": TEMPORAL_LITE_TRAIN_COUNT,
        "sample_manifest_sha256": file_sha256(manifest),
        "indices_hash": indices_hash(indices),
        "selection_split": "validation",
        "test_split_used": False,
    }
    prior = RidgePrior(
        coefficients=np.zeros((64, 512), dtype=np.complex128),
        regularization=1e-4,
        rows=TEMPORAL_LITE_TRAIN_COUNT * 64,
        target_blocks=(0, 3),
        semantics_hash=DataSemantics.for_domain("mobility").stable_hash(),
        provenance=provenance,
    )
    prior_path = tmp_path / "ridge_full.npz"
    prior.save(prior_path)
    spatial = build_model(
        "prist_ris_b",
        domain="mobility",
        hidden=32,
        blocks_per_stage=(2, 2, 1),
        final_refine_blocks=1,
        backbone_ris_coordinate_enabled=True,
        backbone_ris_coordinate_mode="direct_add",
        backbone_antenna_index_enabled=False,
        attention_enabled=False,
        attention_ris_coordinate_enabled=False,
        attention_antenna_index_enabled=False,
        spatial_multiscale_supervision=False,
        spatial_channel_attention="se",
        spatial_residual_style="scaled_true_residual",
    )
    run = tmp_path / "lite_a_full"
    checkpoint = run / "checkpoints" / "best_checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    (run / "manifests").mkdir()
    (run / "manifests" / "sample_indices.json").write_text(
        json.dumps({"indices": indices}), encoding="utf-8"
    )
    prior_metadata = {
        **prior.metadata(),
        "path": str(prior_path.resolve()),
        "sha256": file_sha256(prior_path),
    }
    semantics = DataSemantics.for_domain("mobility")
    torch.save(
        {
            "method": "PriST-RIS",
            "architecture_version": ARCHITECTURE_VERSION,
            "model_state": spatial.state_dict(),
            "model_config": asdict(spatial.config),
            "training_config": {
                "mode": "full",
                "seed": 123,
                "test_split_used": False,
            },
            "prior_metadata": prior_metadata,
            "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
            "position_semantics_version": POSITION_SEMANTICS_VERSION,
            "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
            "semantics_hash": semantics.stable_hash(),
            "data_semantics": semantics.to_dict(),
        },
        checkpoint,
    )
    return manifest, prior_path, checkpoint


def _plan(tmp_path: Path, *, profiles: bool = False) -> dict[str, object]:
    manifest, prior, checkpoint = _artifacts(tmp_path)
    return build_temporal_lite_plan(
        tmp_path / "temporal_lite",
        spatial_checkpoint_path=checkpoint,
        prior_path=prior,
        sample_manifest_path=manifest,
        data_root="data",
        workers=8,
        project_root=tmp_path,
        head="abc123",
        run_profiles=profiles,
    )


def test_old_config_and_state_dict_keep_spatial_temporal_width_coupled() -> None:
    original = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=8,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        temporal_residual=False,
    )
    old_config = asdict(original.config)
    old_config.pop("temporal_hidden")
    restored = build_model(**old_config)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert restored.config.temporal_hidden is None
    assert restored.config.effective_temporal_hidden == 8
    assert restored.temporal is not None
    assert restored.temporal.spatial_encoder[0].out_channels == 8


def test_tl24_independent_width_cached_forward_and_no_future_correction() -> None:
    model = build_tl24_model()
    assert model.config.hidden == 32
    assert model.config.effective_temporal_hidden == 24
    assert model.backbone.input.out_channels == 32
    assert model.temporal is not None
    assert model.temporal.spatial_encoder[0].out_channels == 24
    assert model.temporal.rank == 2
    assert model.temporal_correction is None
    batch = canonical_batch("mobility")
    batch["spatial_anchors"] = torch.randn(1, 2, 256, 64, 2)
    output = model(batch)
    assert output.shape == (1, 6, 256, 64, 2)
    output.square().mean().backward()
    assert model.temporal.coefficient_head.weight.grad is not None


def test_uncached_end_to_end_matches_cached_anchor_path(tmp_path: Path) -> None:
    _, prior_path, checkpoint = _artifacts(tmp_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spatial = build_model(**state["model_config"]).eval()
    spatial.load_state_dict(state["model_state"])
    full = build_tl24_model().eval()
    load_mobility_spatial_reference(full, state)
    prior = RidgePrior.load(prior_path)
    batch = canonical_batch("mobility")
    prior_value = prior.predict(batch)
    with torch.no_grad():
        anchors = spatial(batch, prior_value)
        torch.testing.assert_close(full.spatial_anchors(batch, prior_value), anchors)
        uncached = full(batch, prior_value)
        cached = full({**batch, "spatial_anchors": anchors})
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    assert uncached.shape == (1, 6, 256, 64, 2)


def test_full_manifest_prior_and_checkpoint_provenance_are_strict(
    tmp_path: Path,
) -> None:
    manifest_path, prior_path, checkpoint_path = _artifacts(tmp_path)
    manifest = validate_full_sample_manifest(manifest_path)
    prior = validate_full_prior(prior_path, manifest)
    checkpoint = validate_lite_a_checkpoint(
        checkpoint_path, manifest=manifest, prior=prior
    )
    assert checkpoint["sample_count"] == 20000
    assert checkpoint["model_config"]["hidden"] == 32
    assert checkpoint["test_split_used"] is False
    changed = dict(manifest)
    changed["subset_indices_hash"] = "wrong"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_full_prior(prior_path, changed)


def test_tl24_cpu_profile_is_end_to_end_and_budgeted(tmp_path: Path) -> None:
    manifest_path, prior, checkpoint = _artifacts(tmp_path)
    manifest = validate_full_sample_manifest(manifest_path)
    validated_prior = validate_full_prior(prior, manifest)
    profiles = profile_temporal_lite(
        tmp_path / "profile",
        spatial_checkpoint=checkpoint,
        prior_path=prior,
        validated_prior=validated_prior,
    )
    t1 = profiles["T1-Lite"]
    tl24 = profiles["TL24"]
    assert tl24["target_scope"] == "mobility_q0_q5_end_to_end"
    assert tl24["output_shape"] == [1, 6, 256, 64, 2]
    assert tl24["spatial_hidden"] == 32
    assert tl24["temporal_hidden"] == 24
    assert tl24["temporal_rank"] == 2
    assert tl24["temporal_residual_enabled"] is False
    assert tl24["parameters"] == tl24["neural_parameters"] == 524301
    assert tl24["gmacs"] == tl24["neural_gmacs"] == pytest.approx(6.133700016)
    assert tl24["prior_coefficient_shape"] == [64, 512]
    assert tl24["prior_complex_parameters"] == 32768
    assert tl24["prior_parameter_real_equivalents"] == 65536
    assert tl24["prior_real_macs"] == 8388608
    assert tl24["prior_gmacs"] == pytest.approx(0.008388608)
    assert tl24["total_parameter_real_equivalents"] == 589837
    assert tl24["total_real_macs"] == 6142088624
    assert tl24["total_gmacs"] == pytest.approx(6.142088624)
    assert tl24["total_gflops"] == pytest.approx(12.284177248)
    assert tl24["total_parameter_real_equivalents"] < LPAN_L_PARAMETERS
    assert tl24["total_gmacs"] < LPAN_L_GMACS
    assert LPAN_L_PARAMETERS - tl24["total_parameter_real_equivalents"] == 523067
    assert LPAN_L_GMACS - tl24["total_gmacs"] == pytest.approx(0.195680848)
    assert tl24["budget_pass"] is True
    assert tl24["test_split_used"] is False
    assert t1["neural_parameters"] == 452364
    assert t1["prior_parameter_real_equivalents"] == 65536
    assert t1["total_parameter_real_equivalents"] == 517900
    assert t1["neural_gmacs"] == pytest.approx(5.048664064)
    assert t1["prior_gmacs"] == pytest.approx(0.008388608)
    assert t1["total_gmacs"] == pytest.approx(5.057052672)
    assert t1["total_gflops"] == pytest.approx(10.114105344)


def test_ridge_complexity_fails_closed_on_noncanonical_coefficients() -> None:
    prior = RidgePrior(
        coefficients=np.zeros((63, 512), dtype=np.complex128),
        regularization=1e-4,
        rows=1,
        target_blocks=(0, 3),
        semantics_hash=DataSemantics.for_domain("mobility").stable_hash(),
    )
    with pytest.raises(ValueError, match="coefficient shape mismatch"):
        profile_ridge_complexity(prior, [1, 2, 32, 64, 2])


def test_generic_profile_model_keeps_neural_only_semantics() -> None:
    result = profile_model(
        build_tl24_model(),
        domain="mobility",
        device=torch.device("cpu"),
        latency_runs=1,
    )
    assert result["parameters"] == 524301
    assert result["gmacs"] == pytest.approx(6.133700016)
    for total_only in (
        "neural_parameters",
        "prior_complex_parameters",
        "prior_parameter_real_equivalents",
        "prior_gmacs",
        "total_parameter_real_equivalents",
        "total_gmacs",
    ):
        assert total_only not in result


def test_plan_is_fixed_cpu_only_and_uses_canonical_run_directory(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert plan["candidate"] == asdict(TL24)
    assert plan["architecture_search"] is False
    assert plan["rank_search"] is False
    assert plan["planner_uses_gpu"] is False
    assert plan["test_split_used"] is False
    spec = plan["tl24_spec"]
    assert Path(spec["run_dir"]).name == spec["run_name"] == "tl24"
    command = plan["commands"]["TL24"]
    joined = " ".join(command)
    for fragment in (
        "--hidden 32",
        "--temporal-hidden 24",
        "--temporal-rank 2",
        "--no-temporal-residual",
        "--scheduler fixed",
        "--epochs 30",
        "--min-epochs 31",
        "--adaptation temporal_only",
    ):
        assert fragment in joined
    assert "--amp" not in command
    assert fixed_tl24_training_config()["test_split_used"] is False


def test_budget_failure_blocks_before_gpu_or_invocation() -> None:
    invoked: list[list[object]] = []
    with pytest.raises(RuntimeError, match="complexity budget"):
        execute_temporal_lite_plan(
            {
                "test_split_used": False,
                "complexity_budget": {"budget_pass": False},
            },
            data_root="data",
            device="cuda:0",
            workers=8,
            physical_gpu_index=0,
            confirm_gpu_free=True,
            resume_incomplete=False,
            invoke=invoked.append,
        )
    assert invoked == []


def test_neural_pass_but_total_fail_blocks_gpu_training() -> None:
    invoked: list[list[object]] = []
    with pytest.raises(RuntimeError, match="invalid TL24 profile evidence"):
        execute_temporal_lite_plan(
            {
                "test_split_used": False,
                "complexity_budget": {"budget_pass": True},
                "profiles": {
                    "TL24": {
                        "parameters": 524301,
                        "gmacs": 6.133700016,
                        "neural_parameters": 524301,
                        "prior_parameter_real_equivalents": 588604,
                        "total_parameter_real_equivalents": LPAN_L_PARAMETERS + 1,
                        "macs": 6133700016,
                        "neural_macs": 6133700016,
                        "prior_real_macs": 8388608,
                        "prior_gmacs": 0.008388608,
                        "prior_complex_parameters": 294302,
                        "total_real_macs": 6142088624,
                        "total_gmacs": 6.142088624,
                        "total_real_flops": 12284177248,
                        "total_gflops": 12.284177248,
                        "parameter_budget_pass": False,
                        "gmac_budget_pass": True,
                        "budget_pass": False,
                        "target_scope": "mobility_q0_q5_end_to_end",
                        "test_split_used": False,
                    }
                },
            },
            data_root="data",
            device="cuda:0",
            workers=8,
            physical_gpu_index=0,
            confirm_gpu_free=True,
            resume_incomplete=False,
            invoke=invoked.append,
        )
    assert invoked == []


def test_gpu_preflight_rejects_wrong_visible_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan["complexity_budget"]["budget_pass"] = True
    plan["profiles"] = {
        "TL24": {
            "parameters": 524301,
            "gmacs": 6.133700016,
            "neural_parameters": 524301,
            "prior_parameter_real_equivalents": 65536,
            "total_parameter_real_equivalents": 589837,
            "macs": 6133700016,
            "neural_macs": 6133700016,
            "prior_real_macs": 8388608,
            "prior_gmacs": 0.008388608,
            "prior_complex_parameters": 32768,
            "total_real_macs": 6142088624,
            "total_gmacs": 6.142088624,
            "total_real_flops": 12284177248,
            "total_gflops": 12.284177248,
            "parameter_budget_pass": True,
            "gmac_budget_pass": True,
            "budget_pass": True,
            "target_scope": "mobility_q0_q5_end_to_end",
            "test_split_used": False,
        }
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(PermissionError, match="CUDA_VISIBLE_DEVICES=0"):
        execute_temporal_lite_plan(
            plan,
            data_root="data",
            device="cuda:0",
            workers=8,
            physical_gpu_index=0,
            confirm_gpu_free=True,
            resume_incomplete=False,
            invoke=lambda _: None,
        )


def test_incomplete_anchor_cache_is_never_overwritten(tmp_path: Path) -> None:
    spec = _plan(tmp_path)["cache_spec"]
    root = Path(spec["anchor_cache_root"])
    root.mkdir()
    (root / "train.h5").write_bytes(b"incomplete")
    with pytest.raises(FileExistsError, match="Incomplete Temporal-Lite anchor cache"):
        decide_anchor_cache(spec)


def test_exact_completed_run_reuses_and_incomplete_refuses(tmp_path: Path) -> None:
    spec = _plan(tmp_path)["tl24_spec"]
    run = Path(spec["run_dir"])
    assert decide_temporal_run(spec, resume_incomplete=False)["action"] == "run"
    run.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Incomplete Temporal-Lite"):
        decide_temporal_run(spec, resume_incomplete=False)
    (run / "paper_experiment_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "last_checkpoint.pth").write_bytes(b"checkpoint")
    assert decide_temporal_run(spec, resume_incomplete=True)["action"] == "resume"
    (run / "results").mkdir()
    (run / "results" / "final_result.json").write_text(
        json.dumps({"test_split_used": False}), encoding="utf-8"
    )
    assert decide_temporal_run(spec, resume_incomplete=False)["action"] == "reuse"


def _diagnostics(db: float) -> dict[str, object]:
    metric = {"nmse_linear": 10 ** (db / 10), "nmse_db": db, "sample_count": 2}
    return {
        "per_query": {f"q{index}": metric for index in range(6)},
        "overall": metric,
        "pilot_anchor_aggregate": metric,
        "non_pilot_aggregate": metric,
        "interpolation_q1_q2": metric,
        "extrapolation_q4_q5": metric,
        "delta_error": metric,
        "curvature_error": metric,
    }


def test_summary_reports_t1_tl24_gaps_and_no_winner(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["profiles"] = {
        "T1-Lite": {
            "parameters": 452364,
            "trainable_parameters": 0,
            "gmacs": 5.0,
            "gflops": 10.0,
            "neural_parameters": 452364,
            "neural_gmacs": 5.0,
            "prior_complex_parameters": 32768,
            "prior_parameter_real_equivalents": 65536,
            "prior_gmacs": 0.008388608,
            "total_parameter_real_equivalents": 517900,
            "total_gmacs": 5.008388608,
            "total_gflops": 10.016777216,
        },
        "TL24": {
            "parameters": 520000,
            "trainable_parameters": 68000,
            "gmacs": 6.1,
            "gflops": 12.2,
            "neural_parameters": 520000,
            "neural_gmacs": 6.1,
            "prior_complex_parameters": 32768,
            "prior_parameter_real_equivalents": 65536,
            "prior_gmacs": 0.008388608,
            "total_parameter_real_equivalents": 585536,
            "total_gmacs": 6.108388608,
            "total_gflops": 12.216777216,
            "budget_pass": True,
        },
    }
    root = Path(plan["output_root"])
    evaluations = root / "evaluations"
    evaluations.mkdir()
    (evaluations / "t1_lite_validation.json").write_text(
        json.dumps(
            {
                "nmse_db": -20.0,
                "diagnostics": _diagnostics(-20.0),
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    (evaluations / "tl24_best_validation.json").write_text(
        json.dumps(
            {
                "nmse_db": -20.5,
                "diagnostics": _diagnostics(-20.5),
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    results = root / "runs" / "tl24" / "results"
    results.mkdir(parents=True)
    history = [
        {
            "epoch": 1,
            "validation_nmse_linear": 0.01,
            "validation_nmse_db": -20.0,
        },
        {
            "epoch": 2,
            "validation_nmse_linear": 0.008,
            "validation_nmse_db": -20.9691,
        },
    ]
    with (results / "training_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (results / "final_result.json").write_text(
        json.dumps(
            {
                "last_validation": {"nmse_db": -20.4},
                "wall_clock_seconds": 120.0,
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_temporal_lite(root, plan)
    assert [row["candidate"] for row in summary["results"]] == [
        "T1-Lite linear trend",
        "TL24",
    ]
    assert summary["results"][1]["best_epoch"] == 2
    assert summary["results"][1]["performance_gap_db"]["LPAN-L_mean"] > 0
    assert summary["winner"] is None
    assert summary["human_decision_only"] is True
    assert summary["test_split_used"] is False
    assert "Full_Direct_S3_T2_seed123" not in REFERENCE_NMSE_DB
    reference = "Direct_S3_T2_cache_composed_validation_seed123"
    assert REFERENCE_NMSE_DB[reference] == -21.992708
    assert REFERENCE_METADATA[reference]["split"] == "validation"
    assert "cached Direct-S3" in REFERENCE_METADATA[reference]["composition"]
    assert REFERENCE_METADATA[reference]["purpose"] == "performance gap reference only"
    assert REFERENCE_METADATA[reference][
        "standard_end_to_end_deployable_checkpoint_validated"
    ] is False


def test_temporal_lite_terminology_and_formula_are_unambiguous() -> None:
    project = Path(__file__).resolve().parents[1]
    documentation = (project / "docs" / "temporal_lite_v1.md").read_text(
        encoding="utf-8"
    )
    readme = (project / "README.md").read_text(encoding="utf-8")
    combined = documentation + readme
    assert "full-data Prior-S3" not in combined
    assert "full-data prior-guided Lite-A spatial model" in documentation
    assert "bounded trend-coefficient correction" in combined
    assert "Direct_S3_T2_cache_composed_validation_seed123" in documentation
