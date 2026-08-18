from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from argparse import Namespace

from prist_ris.data import PriSTRISDataset, nested_fraction_indices
from prist_ris.manifests import validate_test_unlock
from prist_ris.metrics import sample_linear_nmse
from prist_ris.cli import _allowed_test


def _quasi_file(path: Path, input_key: str, target_key: str) -> None:
    real_obs = np.ones((1, 32, 64, 2), dtype=np.float32)
    imag_obs = np.full((1, 32, 64, 2), 2.0, dtype=np.float32)
    real_target = np.full((1, 256, 64, 2), 3.0, dtype=np.float32)
    imag_target = np.full((1, 256, 64, 2), 4.0, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle[input_key] = np.concatenate((real_obs, imag_obs), axis=0)
        handle[target_key] = np.concatenate((real_target, imag_target), axis=0)


def test_grouped_complex_loader_and_test_lock(tmp_path: Path) -> None:
    train_path = tmp_path / "train.mat"
    _quasi_file(train_path, "input_da", "output_da")
    item = PriSTRISDataset(train_path, "quasi", "train")[0]
    assert item["obs_h"].shape == (1, 32, 64, 2)
    assert item["target_h"].shape == (1, 256, 64, 2)
    assert torch.all(item["obs_h"][..., 0] == 1)
    assert torch.all(item["obs_h"][..., 1] == 2)
    test_path = tmp_path / "test.mat"
    _quasi_file(test_path, "Yd", "Hd")
    with pytest.raises(PermissionError):
        PriSTRISDataset(test_path, "quasi", "test")


def test_metric_is_mean_of_sample_level_linear_nmse() -> None:
    target = torch.ones(2, 1, 1, 1, 2)
    prediction = target.clone()
    prediction[0] = 0
    prediction[1] = 3
    values = sample_linear_nmse(prediction, target)
    torch.testing.assert_close(values, torch.tensor([1.0, 4.0]))
    assert float(values.mean()) == 2.5


def test_nested_transfer_subsets() -> None:
    values = nested_fraction_indices(20_000, (0.01, 0.05, 0.1, 0.2, 1.0), seed=123)
    assert [len(values[key]) for key in ("0.01", "0.05", "0.10", "0.20", "1.00")] == [200, 1000, 2000, 4000, 20000]
    assert set(values["0.01"]) <= set(values["0.05"]) <= set(values["1.00"])


def test_test_unlock_requires_exact_frozen_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"frozen")
    payload = {
        "schema": "prist_ris.frozen_experiment.v1",
        "test_unlocked": True,
        "artifacts": [
            {
                "kind": "checkpoint",
                "path": str(checkpoint.resolve()),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_test_unlock(manifest, checkpoint)["test_unlocked"] is True
    checkpoint.write_bytes(b"changed")
    with pytest.raises(PermissionError):
        validate_test_unlock(manifest, checkpoint)


def test_cli_test_evaluation_is_locked_without_freeze() -> None:
    with pytest.raises(PermissionError):
        _allowed_test(Namespace(split="test", freeze_manifest=None, checkpoint=Path("x.pth")))
