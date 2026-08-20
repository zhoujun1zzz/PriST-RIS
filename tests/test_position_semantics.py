from __future__ import annotations

import pytest
import torch

from prist_ris.contracts import (
    ARCHITECTURE_VERSION,
    OBSERVED_RIS_INDICES,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    ris_index_to_grid,
)
from prist_ris.engine import require_checkpoint_contract
from prist_ris.models import (
    PhysicalGridBackbone,
    PhysicalObservedDenseResidualAttention,
    build_model,
    canonical_batch,
)
from prist_ris.screening import (
    POSITION_SCREENING_CANDIDATES,
    position_candidate_training_arguments,
    position_screening_plan,
)


SMALL = {"hidden": 4, "blocks_per_stage": (1, 1, 1), "final_refine_blocks": 1}


def _position_model(**overrides: object):
    settings = {
        "backbone_ris_coordinate_enabled": False,
        "backbone_antenna_index_enabled": False,
        "backbone_ris_coordinate_mode": "off",
        "attention_enabled": False,
        "attention_ris_coordinate_enabled": False,
        "attention_antenna_index_enabled": False,
    }
    settings.update(overrides)
    return build_model("prist_ris_b", domain="mobility", **SMALL, **settings)


def test_ris_index_mapping_contract_is_unchanged() -> None:
    assert [ris_index_to_grid(index) for index in (0, 15, 16, 255)] == [
        (0, 0), (0, 15), (1, 0), (15, 15)
    ]
    assert [ris_index_to_grid(index) for index in OBSERVED_RIS_INDICES] == [
        (row, column) for row in range(16) for column in (0, 8)
    ]


def test_backbone_ris_and_antenna_paths_are_independently_constructed() -> None:
    ris_only = _position_model(
        backbone_ris_coordinate_enabled=True,
        backbone_ris_coordinate_mode="direct_add",
    )
    assert ris_only.backbone.ris_coordinate_encoder is not None
    assert ris_only.backbone.antenna_index_encoder is None
    antenna_only = _position_model(backbone_antenna_index_enabled=True)
    assert antenna_only.backbone.ris_coordinate_encoder is None
    assert antenna_only.backbone.antenna_index_encoder is not None


def test_attention_ris_coordinates_are_independent_of_backbone_positions() -> None:
    model = _position_model(
        attention_enabled=True,
        attention_ris_coordinate_enabled=True,
    )
    assert model.backbone.ris_coordinate_encoder is None
    assert model.backbone.antenna_index_encoder is None
    assert model.observed_dense_attention is not None
    assert model.observed_dense_attention.observed_coordinate_projection is not None
    assert model.observed_dense_attention.dense_coordinate_projection is not None
    assert model.observed_dense_attention.antenna_projection is None


def test_observed_and_dense_attention_coordinates_follow_row_major_ris_mapping() -> None:
    module = PhysicalObservedDenseResidualAttention(
        4, heads=4, ris_coordinate_enabled=True, antenna_index_enabled=False
    )
    observed = module.observed_coordinates(
        torch.tensor(OBSERVED_RIS_INDICES), 1, dtype=torch.float32
    )
    torch.testing.assert_close(observed[0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(observed[1], torch.tensor([-1.0, 1.0 / 15.0]))
    torch.testing.assert_close(observed[-1], torch.tensor([1.0, 1.0 / 15.0]))
    dense = module.dense_coordinates(device=torch.device("cpu"), dtype=torch.float32)
    expected = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [-13.0 / 15.0, -1.0], [1.0, 1.0]]
    )
    torch.testing.assert_close(dense[[0, 15, 16, 255]], expected)


def test_zero_init_gated_ris_path_is_exact_initial_backbone_baseline() -> None:
    torch.manual_seed(13)
    baseline = PhysicalGridBackbone(hidden=4, blocks_per_stage=(1, 1, 1), final_refine_blocks=1)
    torch.manual_seed(13)
    gated = PhysicalGridBackbone(
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        ris_coordinate_enabled=True,
        ris_coordinate_mode="zero_init_gated",
    )
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    baseline_output, _ = baseline(batch["obs_h"], batch["obs_ris_index"])
    gated_output, _ = gated(batch["obs_h"], batch["obs_ris_index"])
    torch.testing.assert_close(gated.ris_coordinate_gates, torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(gated_output, baseline_output, rtol=0, atol=0)


def test_zero_init_gate_opens_before_coordinate_projection_gradients() -> None:
    torch.manual_seed(17)
    backbone = PhysicalGridBackbone(
        hidden=2,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        ris_coordinate_enabled=True,
        ris_coordinate_mode="zero_init_gated",
    )
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    optimizer = torch.optim.SGD(backbone.parameters(), lr=1e-2)
    output, _ = backbone(batch["obs_h"], batch["obs_ris_index"])
    output.square().mean().backward()
    assert backbone.ris_coordinate_gates.grad is not None
    assert float(backbone.ris_coordinate_gates.grad.abs().sum()) > 0
    projection = backbone.ris_coordinate_encoder.projection  # type: ignore[union-attr]
    assert projection.weight.grad is not None
    assert float(projection.weight.grad.abs().sum()) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output, _ = backbone(batch["obs_h"], batch["obs_ris_index"])
    output.square().mean().backward()
    assert projection.weight.grad is not None
    assert float(projection.weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("candidate", POSITION_SCREENING_CANDIDATES)
def test_p1_p3_models_never_read_target(candidate) -> None:
    model = _position_model(
        backbone_ris_coordinate_enabled=candidate.backbone_ris_coordinate_enabled,
        backbone_ris_coordinate_mode=candidate.backbone_ris_coordinate_mode,
        attention_enabled=candidate.attention_enabled,
        attention_ris_coordinate_enabled=candidate.attention_ris_coordinate_enabled,
    ).eval()
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    changed = {**batch, "target_h": torch.randn_like(batch["target_h"])}
    prior = torch.randn(1, 2, 256, 64, 2)
    torch.testing.assert_close(model(batch, prior), model(changed, prior), rtol=0, atol=0)


def test_p1_p3_plan_and_commands_are_fixed_explicit_and_have_no_p4() -> None:
    plan = position_screening_plan()
    assert [candidate["name"] for candidate in plan["candidates"]] == [
        "P1_ris_direct", "P2_ris_gated", "P3_attention_ris"
    ]
    assert plan["p4_scheduled"] is False
    for candidate in POSITION_SCREENING_CANDIDATES:
        command = [
            str(value)
            for value in position_candidate_training_arguments(
                candidate, prior="ridge.npz", data_root="data", output_root="runs"
            )
        ]
        assert command[command.index("--epochs") + 1] == "30"
        assert command[command.index("--stop-after-epoch") + 1] == "30"
        assert "--no-backbone-antenna-index-enabled" in command
        assert "--no-attention-antenna-index-enabled" in command
        assert "--coordinate-enabled" not in command
        assert "test" not in command


def test_position_metadata_is_explicit_and_legacy_alias_conflicts_are_rejected() -> None:
    model = _position_model(
        attention_enabled=True,
        attention_ris_coordinate_enabled=True,
    )
    metadata = model.protocol_metadata()
    assert metadata["position_semantics_version"] == POSITION_SEMANTICS_VERSION
    assert metadata["backbone_ris_coordinate_mode"] == "off"
    assert metadata["attention_enabled"] is True
    assert metadata["attention_ris_coordinate_enabled"] is True
    assert metadata["attention_antenna_index_enabled"] is False
    assert metadata["antenna_encoding_semantics"] == "antenna_index_encoding"
    with pytest.raises(ValueError, match="legacy coordinate_enabled"):
        build_model(
            "prist_ris_b",
            domain="mobility",
            coordinate_enabled=True,
            backbone_ris_coordinate_enabled=True,
        )


def test_pre_position_checkpoint_is_rejected_even_with_current_spatial_marker() -> None:
    with pytest.raises(ValueError, match="position_semantics_version"):
        require_checkpoint_contract(
            {
                "architecture_version": ARCHITECTURE_VERSION,
                "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
                "model_config": {"domain": "quasi"},
            },
            "Resume",
            expected_domain="quasi",
        )
