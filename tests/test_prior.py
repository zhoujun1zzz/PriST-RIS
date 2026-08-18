from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader, Dataset

from prist_ris.contracts import DataSemantics
from prist_ris.prior import RidgePrior, RidgeStatistics


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    return {
        "obs_h": torch.randn(2, 1, 32, 64, 2),
        "target_h": torch.randn(2, 1, 256, 64, 2),
    }


def test_ridge_fit_metadata_round_trip_and_prediction_shape(tmp_path: Path) -> None:
    statistics = RidgeStatistics.accumulate([_batch()], (0,))
    prior = statistics.solve(1e-3, DataSemantics.for_domain("quasi"))
    assert prior.fit_split == "train"
    assert prior.predict(_batch()).shape == (2, 1, 256, 64, 2)
    path = tmp_path / "ridge.npz"
    metadata = prior.save(path)
    loaded = RidgePrior.load(path)
    np.testing.assert_array_equal(loaded.coefficients, prior.coefficients)
    assert loaded.metadata() == prior.metadata()
    assert metadata["fit_split"] == "train"
    assert "sha256" in metadata


def test_ridge_rejects_non_train_loader() -> None:
    class ValidationDataset(Dataset):
        split = "validation"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {key: value[0] for key, value in _batch().items()}

    with pytest.raises(PermissionError):
        RidgeStatistics.accumulate(DataLoader(ValidationDataset(), batch_size=1), (0,))
