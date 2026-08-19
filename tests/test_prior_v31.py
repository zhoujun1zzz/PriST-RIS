from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from prist_ris.cli import parser
from prist_ris.contracts import DataSemantics
from prist_ris.prior import RidgePrior, RidgeStatistics


def _mobility_batch(samples: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    return {
        "obs_h": torch.randn(samples, 2, 32, 64, 2),
        "target_h": torch.randn(samples, 6, 256, 64, 2),
    }


def test_mobility_prior_outputs_two_anchors_and_round_trips(tmp_path: Path) -> None:
    statistics = RidgeStatistics.accumulate([_mobility_batch()], (0, 1))
    prior = statistics.solve(1e-3, DataSemantics.for_domain("mobility"))
    assert prior.target_blocks == (0, 1)
    assert prior.predict(_mobility_batch()).shape == (2, 2, 256, 64, 2)
    path = tmp_path / "dual_ridge.npz"
    metadata = prior.save(path)
    loaded = RidgePrior.load(path)
    np.testing.assert_array_equal(loaded.coefficients, prior.coefficients)
    assert loaded.metadata() == prior.metadata()
    assert metadata["fit_split"] == "train"


def test_fit_prior_cli_uses_domain_aware_default() -> None:
    quasi = parser().parse_args(["fit-prior", "--domain", "quasi", "--output", "q.npz"])
    mobility = parser().parse_args(["fit-prior", "--domain", "mobility", "--output", "m.npz"])
    assert quasi.target_blocks is None
    assert mobility.target_blocks is None


def test_ridge_rejects_non_train_loader() -> None:
    class ValidationDataset(Dataset):
        split = "validation"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {key: value[0] for key, value in _mobility_batch().items()}

    with pytest.raises(PermissionError):
        RidgeStatistics.accumulate(DataLoader(ValidationDataset(), batch_size=1), (0, 1))
