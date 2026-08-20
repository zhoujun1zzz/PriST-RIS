from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from prist_ris.checkpoint import load_checkpoint
from prist_ris.cli import parser
from prist_ris.contracts import SPATIAL_PROTOCOL_VERSION, DataSemantics
from prist_ris.engine import TrainingConfig, train
from prist_ris.models import canonical_batch
from prist_ris.prior import RidgePrior, RidgeStatistics


def _mobility_batch(samples: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    return {
        "obs_h": torch.randn(samples, 2, 32, 64, 2),
        "target_h": torch.randn(samples, 6, 256, 64, 2),
    }


def test_mobility_prior_outputs_two_anchors_and_round_trips(tmp_path: Path) -> None:
    statistics = RidgeStatistics.accumulate([_mobility_batch()], (0, 3))
    prior = statistics.solve(1e-3, DataSemantics.for_domain("mobility"))
    assert prior.target_blocks == (0, 3)
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
        RidgeStatistics.accumulate(DataLoader(ValidationDataset(), batch_size=1), (0, 3))


def test_prefix_q0_q1_mobility_prior_is_rejected(tmp_path: Path) -> None:
    current = DataSemantics.for_domain("mobility")
    old = replace(current, obs_time_index=(0, 1))
    assert old.stable_hash() != current.stable_hash()
    coefficients = np.zeros((64, 2 * 256), dtype=np.complex128)
    prior = RidgePrior(
        coefficients=coefficients,
        regularization=1e-3,
        rows=1,
        target_blocks=(0, 1),
        semantics_hash=old.stable_hash(),
    )
    path = tmp_path / "prefix_prior.npz"
    prior.save(path)
    config = TrainingConfig(
        domain="mobility",
        model_key="prist_ris_b",
        mode="smoke",
        hidden=2,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        epochs=1,
    )
    batch = canonical_batch("mobility")
    with pytest.raises(ValueError, match="semantics do not match"):
        train(
            config,
            [batch],
            [batch],
            run_dir=tmp_path / "rejected_run",
            device=torch.device("cpu"),
            prior_path=path,
        )


def test_q0q3_ridge_remains_reusable_for_attention_c(tmp_path: Path) -> None:
    semantics = DataSemantics.for_domain("mobility")
    prior = RidgePrior(
        coefficients=np.zeros((64, 2 * 256), dtype=np.complex128),
        regularization=1e-3,
        rows=1,
        target_blocks=(0, 3),
        semantics_hash=semantics.stable_hash(),
    )
    path = tmp_path / "q0q3_prior.npz"
    prior.save(path)
    config = TrainingConfig(
        domain="mobility",
        model_key="prist_ris_c",
        mode="smoke",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        epochs=1,
    )
    batch = canonical_batch("mobility")
    run = tmp_path / "attention_c"
    train(
        config,
        [batch],
        [batch],
        run_dir=run,
        device=torch.device("cpu"),
        prior_path=path,
    )
    state = load_checkpoint(run / "checkpoints" / "last_checkpoint.pth", torch.device("cpu"))
    assert state["spatial_protocol_version"] == SPATIAL_PROTOCOL_VERSION
    assert state["prior_metadata"]["target_blocks"] == [0, 3]
