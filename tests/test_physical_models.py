from __future__ import annotations

import torch

from prist_ris.contracts import ARCHITECTURE_VERSION, OBSERVED_RIS_INDICES, ris_index_to_grid
from prist_ris.models import (
    AntennaIndexEncoder,
    PHYSICAL_STAGE_COLUMNS,
    RISCoordinateEncoder,
    build_model,
    canonical_batch,
    complex_factorized_reconstruction,
    observations_to_physical_grid,
    physical_grid_to_observations,
)


SMALL = {"hidden": 4, "blocks_per_stage": (1, 1, 1), "final_refine_blocks": 1}


def test_observed_indices_form_exact_16x2_grid() -> None:
    positions = [ris_index_to_grid(index) for index in OBSERVED_RIS_INDICES]
    assert positions == [(row, column) for row in range(16) for column in (0, 8)]


def test_flat_index_to_grid_row_major() -> None:
    assert ris_index_to_grid(0) == (0, 0)
    assert ris_index_to_grid(15) == (0, 15)
    assert ris_index_to_grid(16) == (1, 0)
    assert ris_index_to_grid(255) == (15, 15)


def test_observation_grid_round_trip() -> None:
    batch = canonical_batch("mobility", batch_size=2)
    batch["obs_h"].normal_()
    grid = observations_to_physical_grid(batch["obs_h"], batch["obs_ris_index"])
    assert grid.shape == (2, 4, 64, 16, 2)
    torch.testing.assert_close(physical_grid_to_observations(grid, 2), batch["obs_h"])


def test_stage_physical_columns() -> None:
    assert PHYSICAL_STAGE_COLUMNS == {
        2: (0, 8),
        4: (0, 4, 8, 12),
        8: (0, 2, 4, 6, 8, 10, 12, 14),
        16: tuple(range(16)),
    }


def test_progressive_shapes_are_64x16x2_4_8_16() -> None:
    model = build_model("prist_ris_a", domain="mobility", **SMALL)
    output = model(canonical_batch("mobility"))
    assert output.shape == (1, 2, 256, 64, 2)
    assert [shape[2:] for shape in model.last_stage_shapes] == [
        (64, 16, 2),
        (64, 16, 4),
        (64, 16, 8),
        (64, 16, 16),
    ]


def test_ris_coordinate_encoder_uses_row_and_physical_column() -> None:
    coordinates = RISCoordinateEncoder.coordinates(
        4, device=torch.device("cpu"), dtype=torch.float32
    )
    expected_columns = torch.tensor([0, 4, 8, 12], dtype=torch.float32) * (2 / 15) - 1
    torch.testing.assert_close(coordinates[0, 1, 0, 0], expected_columns)
    assert coordinates[0, 0, 0, 0, 0] == -1
    assert coordinates[0, 0, 0, 15, 0] == 1


def test_antenna_encoding_is_index_not_physical_coordinate() -> None:
    assert AntennaIndexEncoder.semantics == "antenna_index_encoding"


def test_coordinate_enabled_model_output_changes_when_embedding_disabled() -> None:
    torch.manual_seed(7)
    model = build_model("prist_ris_c", domain="mobility", **SMALL)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.zeros(1, 2, 256, 64, 2)
    enabled = model(batch, prior)
    model.backbone.coordinate_enabled = False
    disabled = model(batch, prior)
    assert not torch.equal(enabled, disabled)


def test_canonical_ladder_and_dual_spatial_outputs() -> None:
    batch = canonical_batch("mobility")
    for key in ("prist_ris_a", "prist_ris_b", "prist_ris_c"):
        model = build_model(key, domain="mobility", **SMALL)
        prior = None if key == "prist_ris_a" else torch.zeros(1, 2, 256, 64, 2)
        assert model(batch, prior).shape == (1, 2, 256, 64, 2)
        assert model.protocol_metadata()["cross_attention_layers"] == 0
    full = build_model("prist_ris_full", domain="mobility", **SMALL)
    assert full(batch, torch.zeros(1, 2, 256, 64, 2)).shape == (1, 6, 256, 64, 2)
    assert full.config.architecture_version == ARCHITECTURE_VERSION


def test_full_q0_q1_equal_spatial_anchors() -> None:
    torch.manual_seed(5)
    model = build_model("prist_ris_full", domain="mobility", **SMALL)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    anchors = model.spatial_anchors(batch, prior)
    output = model(batch, prior)
    torch.testing.assert_close(output[:, :2], anchors)


def test_future_prediction_uses_no_target_and_receives_delta() -> None:
    torch.manual_seed(2)
    model = build_model("prist_ris_full", domain="mobility", **SMALL)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    first = model(batch, prior)
    changed = {**batch, "target_h": torch.randn_like(batch["target_h"])}
    second = model(changed, prior)
    torch.testing.assert_close(first, second)
    assert model.temporal is not None and model.temporal.last_delta_norm is not None
    assert model.temporal.last_delta_norm > 0


def test_rank_2_and_3_shapes() -> None:
    batch = canonical_batch("mobility")
    prior = torch.zeros(1, 2, 256, 64, 2)
    for rank in (2, 3):
        model = build_model(
            "prist_ris_full", domain="mobility", temporal_rank=rank, **SMALL
        )
        assert model(batch, prior).shape == (1, 6, 256, 64, 2)


def test_temporal_residual_only_changes_q2_to_q5() -> None:
    torch.manual_seed(12)
    corrected = build_model("prist_ris_full", domain="mobility", **SMALL)
    plain = build_model(
        "prist_ris_full", domain="mobility", temporal_residual=False, **SMALL
    )
    common = {
        name: value
        for name, value in corrected.state_dict().items()
        if name in plain.state_dict() and value.shape == plain.state_dict()[name].shape
    }
    plain.load_state_dict(common, strict=False)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    with_correction = corrected(batch, prior)
    without_correction = plain(batch, prior)
    torch.testing.assert_close(with_correction[:, :2], without_correction[:, :2])
    assert not torch.equal(with_correction[:, 2:], without_correction[:, 2:])


def test_complex_factorization_is_float32_island() -> None:
    bases = torch.randn(1, 2, 256, 64, 2, dtype=torch.float16)
    coefficients = torch.randn(1, 4, 2, 2, dtype=torch.float16)
    output = complex_factorized_reconstruction(bases, coefficients)
    assert output.dtype == torch.float32
    assert output.shape == (1, 4, 256, 64, 2)


def test_quasi_uses_one_anchor_and_no_fake_future() -> None:
    model = build_model("prist_ris_full", domain="quasi", **SMALL)
    batch = canonical_batch("quasi")
    prior = torch.zeros(1, 1, 256, 64, 2)
    assert model(batch, prior).shape == (1, 1, 256, 64, 2)
    assert model.temporal is None
    assert "observation_mask" not in batch
