from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
import torch

from prist_ris.anchor_cache import (
    ANCHOR_CACHE_SCHEMA,
    SpatialAnchorCacheDataset,
)
from prist_ris.contracts import (
    DataSemantics,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
)
from prist_ris.complexity import profile_model
from prist_ris.cli import _completed_epoch
from prist_ris.data import DatasetSource
from prist_ris.models import (
    PHYSICAL_STAGE_COLUMNS,
    GatedSE3D,
    PhysicalGridBackbone,
    TrendConditionedTemporal,
    build_model,
    canonical_batch,
    linear_trend_non_pilot,
    sample_ris_columns,
)
from prist_ris.objectives import temporal_curvatures, temporal_deltas
from prist_ris.screening import (
    SPATIAL_MODULE_CANDIDATES,
    TEMPORAL_MODULE_CANDIDATES,
    should_extend_to_40,
    spatial_module_screening_plan,
    spatial_module_training_arguments,
    temporal_module_screening_plan,
    temporal_module_training_arguments,
)
from prist_ris.temporal_audit import TemporalAuditAccumulator


SMALL = {"hidden": 4, "blocks_per_stage": (1, 1, 1), "final_refine_blocks": 1}


def _spatial_model(*, multiscale: bool, channel_attention: str = "off"):
    return build_model(
        "prist_ris_b",
        domain="mobility",
        spatial_multiscale_supervision=multiscale,
        spatial_channel_attention=channel_attention,
        backbone_ris_coordinate_enabled=True,
        backbone_antenna_index_enabled=False,
        backbone_ris_coordinate_mode="direct_add",
        attention_enabled=False,
        attention_ris_coordinate_enabled=False,
        attention_antenna_index_enabled=False,
        **SMALL,
    )


def test_multiscale_physical_columns_shapes_and_initial_ridge_equality() -> None:
    assert PHYSICAL_STAGE_COLUMNS[4] == (0, 4, 8, 12)
    assert PHYSICAL_STAGE_COLUMNS[8] == tuple(range(0, 16, 2))
    model = _spatial_model(multiscale=True)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    predictions = model.spatial_multiscale_anchors(batch, prior)
    assert {width: tuple(value.shape) for width, value in predictions.items()} == {
        4: (1, 2, 64, 64, 2),
        8: (1, 2, 128, 64, 2),
        16: (1, 2, 256, 64, 2),
    }
    for width in (4, 8, 16):
        torch.testing.assert_close(
            predictions[width], sample_ris_columns(prior, width), rtol=0, atol=0
        )


def test_multiscale_is_training_only_same_parameters_and_inference_graph() -> None:
    torch.manual_seed(5)
    disabled = _spatial_model(multiscale=False)
    torch.manual_seed(5)
    enabled = _spatial_model(multiscale=True)
    assert sum(p.numel() for p in disabled.parameters()) == sum(
        p.numel() for p in enabled.parameters()
    )
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    torch.testing.assert_close(disabled(batch, prior), enabled(batch, prior), rtol=0, atol=0)


def test_multiscale_intermediate_backbone_gradients_open_after_shared_head() -> None:
    model = _spatial_model(multiscale=True)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    sum(value.square().mean() for value in model.spatial_multiscale_anchors(batch, prior).values()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sum(value.square().mean() for value in model.spatial_multiscale_anchors(batch, prior).values()).backward()
    gradient = model.backbone.stages[0].blocks[0].body[0].weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0


def test_multiscale_forward_never_reads_target() -> None:
    model = _spatial_model(multiscale=True).eval()
    batch = canonical_batch("mobility")
    prior = torch.randn(1, 2, 256, 64, 2)
    changed = {**batch, "target_h": torch.randn_like(batch["target_h"])}
    for width, prediction in model.spatial_multiscale_anchors(batch, prior).items():
        torch.testing.assert_close(
            prediction,
            model.spatial_multiscale_anchors(changed, prior)[width],
            rtol=0,
            atol=0,
        )


def test_gated_se_preserves_baseline_then_opens_gradients() -> None:
    torch.manual_seed(9)
    baseline = PhysicalGridBackbone(
        hidden=4, blocks_per_stage=(1, 1, 1), final_refine_blocks=1
    )
    torch.manual_seed(9)
    se_model = PhysicalGridBackbone(
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        channel_attention="se",
    )
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    torch.testing.assert_close(
        baseline(batch["obs_h"], batch["obs_ris_index"])[0],
        se_model(batch["obs_h"], batch["obs_ris_index"])[0],
        rtol=0,
        atol=0,
    )
    se = next(module for module in se_model.modules() if isinstance(module, GatedSE3D))
    optimizer = torch.optim.SGD(se_model.parameters(), lr=1e-2)
    output = se_model(batch["obs_h"], batch["obs_ris_index"])[0]
    output.square().mean().backward()
    assert se.gate.grad is not None and float(se.gate.grad.abs()) > 0
    assert se.reduce.weight.grad is not None and float(se.reduce.weight.grad.abs().sum()) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    se_model(batch["obs_h"], batch["obs_ris_index"])[0].square().mean().backward()
    assert se.reduce.weight.grad is not None and float(se.reduce.weight.grad.abs().sum()) > 0


def test_se_profile_records_parameter_and_compute_overhead() -> None:
    baseline = _spatial_model(multiscale=False)
    se_model = _spatial_model(multiscale=False, channel_attention="se")
    baseline_profile = profile_model(
        baseline, domain="mobility", device=torch.device("cpu"), latency_runs=1
    )
    se_profile = profile_model(
        se_model, domain="mobility", device=torch.device("cpu"), latency_runs=1
    )
    assert se_profile["parameters"] > baseline_profile["parameters"]
    assert se_profile["macs"] > baseline_profile["macs"]
    assert se_profile["inference_graph_changed"] is True
    assert se_profile["training_only_mechanism"] is False


def test_t1_linear_trend_and_full_anchor_scatter_are_exact() -> None:
    anchors = torch.randn(1, 2, 256, 64, 2)
    query = torch.arange(6)
    nonpilot = linear_trend_non_pilot(anchors, query)
    delta = anchors[:, 1] - anchors[:, 0]
    for position, alpha in enumerate((1 / 3, 2 / 3, 4 / 3, 5 / 3)):
        torch.testing.assert_close(
            nonpilot[:, position], anchors[:, 0] + alpha * delta
        )
    model = build_model(
        "prist_ris_full",
        domain="mobility",
        temporal_base_mode="linear_trend",
        temporal_learned_residual_enabled=False,
        temporal_residual=False,
        **SMALL,
    )
    batch = canonical_batch("mobility")
    batch["spatial_anchors"] = anchors
    output = model(batch)
    torch.testing.assert_close(output[:, 0], anchors[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 3], anchors[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(output[:, (1, 2, 4, 5)], nonpilot, rtol=0, atol=0)


def test_t2_exact_init_and_coefficient_then_basis_gradient_opening() -> None:
    module = TrendConditionedTemporal(hidden=4, rank=2)
    anchors = torch.randn(1, 2, 256, 64, 2)
    query = torch.arange(6)
    expected = linear_trend_non_pilot(anchors, query)
    torch.testing.assert_close(module(anchors, query, (0, 3)), expected, rtol=0, atol=0)
    assert torch.count_nonzero(module.alpha_head.weight) == 0
    assert torch.count_nonzero(module.coefficient_head.weight) == 0
    assert torch.count_nonzero(module.basis_head.weight) > 0
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    module(anchors, query, (0, 3)).square().mean().backward()
    assert float(module.coefficient_head.weight.grad.abs().sum()) > 0
    assert float(module.alpha_head.weight.grad.abs().sum()) > 0
    assert float(module.basis_head.weight.grad.abs().sum()) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(anchors, query, (0, 3)).square().mean().backward()
    assert float(module.basis_head.weight.grad.abs().sum()) > 0


def test_temporal_forward_does_not_read_target_and_future_terminal_is_zero() -> None:
    model = build_model("prist_ris_full", domain="mobility", **SMALL).eval()
    assert model.temporal_correction is not None
    assert torch.count_nonzero(model.temporal_correction.output.weight) == 0
    batch = canonical_batch("mobility")
    batch["spatial_anchors"] = torch.randn(1, 2, 256, 64, 2)
    changed = {**batch, "target_h": torch.randn_like(batch["target_h"])}
    torch.testing.assert_close(model(batch), model(changed), rtol=0, atol=0)
    model(batch).square().mean().backward()
    assert model.temporal_correction.output.weight.grad is not None
    assert float(model.temporal_correction.output.weight.grad.abs().sum()) > 0


def test_temporal_delta_and_curvature_formulas() -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1, 1).expand(
        1, 6, 1, 1, 2
    )
    torch.testing.assert_close(temporal_deltas(value), torch.ones_like(value[:, 1:]))
    torch.testing.assert_close(temporal_curvatures(value), torch.zeros_like(value[:, 2:]))


def test_t0_statistics_report_split_count_and_zero_linear_curvature() -> None:
    target = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1, 1).expand(
        2, 6, 256, 64, 2
    )
    accumulator = TemporalAuditAccumulator()
    accumulator.update(target)
    result = accumulator.compute("validation")
    assert result["sample_count"] == 2
    assert result["metrics"]["curvature_q2_norm"] == 0


def test_module_plans_commands_and_extension_rule_are_deterministic() -> None:
    assert [row["name"] for row in spatial_module_screening_plan()["candidates"]] == [
        "S2_multiscale", "S3_se", "S23_multiscale_se"
    ]
    spatial_command = [
        str(value)
        for value in spatial_module_training_arguments(
            SPATIAL_MODULE_CANDIDATES[0],
            prior="ridge.npz",
            data_root="data",
            output_root="runs",
            device="cpu",
            workers=0,
            seed=123,
        )
    ]
    assert spatial_command[spatial_command.index("--stop-after-epoch") + 1] == "30"
    assert "--spatial-multiscale-supervision" in spatial_command
    assert temporal_module_screening_plan()["include_curvature"] is False
    temporal_command = [
        str(value)
        for value in temporal_module_training_arguments(
            TEMPORAL_MODULE_CANDIDATES[1],
            prior="ridge.npz",
            spatial_checkpoint="spatial.pth",
            anchor_cache_root="cache",
            data_root="data",
            output_root="runs",
            device="cpu",
            workers=0,
            seed=123,
        )
    ]
    assert temporal_command[temporal_command.index("--temporal-delta-loss-weight") + 1] == "0.1"
    flat = [
        {"epoch": str(epoch), "validation_nmse_db": str(-19.0)}
        for epoch in range(1, 31)
    ]
    assert should_extend_to_40(flat, reference_db=-20.0) is False
    flat[-1]["validation_nmse_db"] = "-19.1"
    assert should_extend_to_40(flat, reference_db=-19.3) is True


def _sparse_mobility_source(tmp_path: Path) -> DatasetSource:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "mobility.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "Yd", shape=(4, 32, 64, 20000), dtype="f4", chunks=(4, 32, 64, 1)
        )
        handle.create_dataset(
            "Hd", shape=(12, 256, 64, 20000), dtype="f4", chunks=(12, 256, 64, 1)
        )
    return DatasetSource("mobility", "train", path, "Yd", "Hd", "synthetic")


def test_anchor_cache_rejects_checkpoint_hash_mismatch_and_never_stores_target(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "spatial.pth"
    checkpoint.write_bytes(b"spatial")
    prior = tmp_path / "ridge.npz"
    prior.write_bytes(b"ridge")
    cache = tmp_path / "train.h5"
    metadata = {
        "schema": ANCHOR_CACHE_SCHEMA,
        "split": "train",
        "checkpoint_sha256": __import__("hashlib").sha256(b"spatial").hexdigest(),
        "prior_sha256": __import__("hashlib").sha256(b"ridge").hexdigest(),
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "target_cached": False,
    }
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("spatial_anchors", data=torch.zeros(1, 2, 256, 64, 2).numpy())
        handle.create_dataset("sample_index", data=[0])
        handle.attrs["metadata_json"] = json.dumps(metadata)
    dataset = SpatialAnchorCacheDataset(
        cache,
        _sparse_mobility_source(tmp_path),
        expected_checkpoint=checkpoint,
        expected_prior=prior,
    )
    assert dataset[0]["spatial_anchors"].shape == (2, 256, 64, 2)
    assert "target_h" in dataset[0]
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        SpatialAnchorCacheDataset(
            cache,
            _sparse_mobility_source(tmp_path / "second"),
            expected_checkpoint=checkpoint,
            expected_prior=prior,
        )


def test_anchor_cache_and_runner_no_test_and_incomplete_guards(tmp_path: Path) -> None:
    from prist_ris.anchor_cache import write_spatial_anchor_cache

    with pytest.raises(PermissionError, match="TRAIN/VALIDATION"):
        write_spatial_anchor_cache(
            tmp_path / "test.h5",
            model=_spatial_model(multiscale=False),
            prior=None,  # type: ignore[arg-type]
            loader=[],
            device=torch.device("cpu"),
            split="test",
            checkpoint_path=tmp_path / "missing.pth",
            prior_path=tmp_path / "missing.npz",
        )
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileExistsError, match="Incomplete run"):
        _completed_epoch(incomplete)
