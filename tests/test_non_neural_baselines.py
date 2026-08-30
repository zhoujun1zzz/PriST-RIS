from __future__ import annotations

import torch

from prist_ris.baselines.interpolation import (
    grid_aware_spatial_weights,
    interpolation_baseline,
)
from prist_ris.baselines.ridge_linear import RidgeLinearBaseline


def canonical_metadata(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "obs_ris_index": torch.arange(
            0, 256, 8, dtype=torch.long
        ).repeat(batch_size, 1),
        "obs_time_index": torch.tensor(
            [0, 3], dtype=torch.long
        ).repeat(batch_size, 1),
        "query_time": torch.arange(
            6, dtype=torch.long
        ).repeat(batch_size, 1),
    }


def test_spatial_interpolation_preserves_observed_nodes() -> None:
    index = torch.arange(0, 256, 8)
    weights = grid_aware_spatial_weights(index)

    assert weights.shape == (256, 32)

    for position, node in enumerate(index.tolist()):
        expected = torch.zeros(32)
        expected[position] = 1.0
        assert torch.allclose(weights[node], expected)


def test_spatial_interpolation_does_not_cross_ris_rows() -> None:
    index = torch.arange(0, 256, 8)
    weights = grid_aware_spatial_weights(index)

    # Node 4 belongs to row 0 and lies halfway between columns 0 and 8.
    assert torch.isclose(weights[4, 0], torch.tensor(0.5))
    assert torch.isclose(weights[4, 1], torch.tensor(0.5))
    assert torch.count_nonzero(weights[4]) == 2

    # Node 12 is outside row-0 observed interval and therefore extends
    # from column 8 only. It must not use the next row's column-0 sample.
    assert torch.isclose(weights[12, 1], torch.tensor(1.0))
    assert torch.count_nonzero(weights[12]) == 1

    # First point of row 1 is observation index 16, packed position 2.
    assert torch.isclose(weights[16, 2], torch.tensor(1.0))
    assert torch.count_nonzero(weights[16]) == 1


def test_interpolation_uses_nearest_temporal_extension() -> None:
    batch = canonical_metadata()
    obs = torch.zeros(1, 2, 32, 64, 2)

    # Artificial q0=0, q3=3.
    obs[:, 1] = 3.0
    batch["obs_h"] = obs

    prediction = interpolation_baseline(batch)

    assert prediction.shape == (1, 6, 256, 64, 2)

    for q in range(4):
        expected = torch.full_like(
            prediction[:, q], float(q)
        )
        assert torch.allclose(prediction[:, q], expected)

    # Historical baseline: nearest-value temporal extrapolation.
    assert torch.allclose(prediction[:, 4], prediction[:, 3])
    assert torch.allclose(prediction[:, 5], prediction[:, 3])


class FakeAnchorPrior:
    target_blocks = (0, 3)
    regularization = 1e-3

    def predict(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = batch["obs_h"].shape[0]
        output = torch.zeros(
            batch_size,
            2,
            256,
            64,
            2,
            dtype=batch["obs_h"].dtype,
        )
        output[:, 1] = 3.0
        return output


def test_ridge_linear_uses_true_linear_extrapolation() -> None:
    batch = canonical_metadata()
    batch["obs_h"] = torch.zeros(1, 2, 32, 64, 2)

    model = RidgeLinearBaseline(
        FakeAnchorPrior()  # type: ignore[arg-type]
    )
    prediction = model.predict(batch)

    assert prediction.shape == (1, 6, 256, 64, 2)

    for q in range(6):
        expected = torch.full_like(
            prediction[:, q], float(q)
        )
        assert torch.allclose(prediction[:, q], expected)


class WrongPrior:
    target_blocks = (0, 1)
    regularization = 1e-3


def test_ridge_linear_rejects_wrong_anchor_semantics() -> None:
    try:
        RidgeLinearBaseline(
            WrongPrior()  # type: ignore[arg-type]
        )
    except ValueError:
        return

    raise AssertionError(
        "RidgeLinearBaseline accepted non-q0/q3 target blocks."
    )
