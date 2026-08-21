from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from prist_ris.cli import _allowed_test
from prist_ris.contracts import (
    ARCHITECTURE_VERSION,
    COMPLEX_LAYOUT,
    MOBILITY_CONTRACT_VERSION,
    OBSERVED_RIS_INDICES,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
    DataSemantics,
)
from prist_ris.data import (
    DATASET_FILENAMES,
    DatasetSource,
    PriSTRISDataset,
    resolve_dataset_source,
    validate_dataset_source,
)
from prist_ris.manifests import validate_test_unlock
from prist_ris.metrics import MetricAccumulator, PerQueryMetricAccumulator


def _write_quasi(
    path: Path,
    input_key: str,
    target_key: str,
    *,
    samples: int = 2,
    input_shape: tuple[int, int, int] = (2, 32, 64),
    target_shape: tuple[int, int, int] = (2, 256, 64),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle[input_key] = np.zeros((*input_shape, samples), dtype=np.float32)
        handle[target_key] = np.zeros((*target_shape, samples), dtype=np.float32)


def test_quasi_validation_prefers_separate_yd_hd(tmp_path: Path) -> None:
    validation = tmp_path / DATASET_FILENAMES[("quasi", "validation")]
    train = tmp_path / DATASET_FILENAMES[("quasi", "train")]
    _write_quasi(validation, "Yd", "Hd")
    with h5py.File(train, "w") as handle:
        handle["input_da"] = np.zeros((2, 32, 64, 2), dtype=np.float32)
        handle["output_da"] = np.zeros((2, 256, 64, 2), dtype=np.float32)
        handle["input_da_test"] = np.zeros((2, 32, 64, 2), dtype=np.float32)
        handle["output_da_test"] = np.zeros((2, 256, 64, 2), dtype=np.float32)
    source = resolve_dataset_source(tmp_path, "quasi", "validation")
    assert source.path == validation.resolve()
    assert (source.input_key, source.target_key) == ("Yd", "Hd")
    assert source.provenance == "separate_validation_yd_hd"


def test_quasi_validation_falls_back_to_train_test_keys(tmp_path: Path) -> None:
    train = tmp_path / DATASET_FILENAMES[("quasi", "train")]
    with h5py.File(train, "w") as handle:
        handle["input_da"] = np.zeros((2, 32, 64, 3), dtype=np.float32)
        handle["output_da"] = np.zeros((2, 256, 64, 3), dtype=np.float32)
        handle["input_da_test"] = np.zeros((2, 32, 64, 2), dtype=np.float32)
        handle["output_da_test"] = np.zeros((2, 256, 64, 2), dtype=np.float32)
    source = resolve_dataset_source(tmp_path, "quasi", "validation")
    assert source.path == train.resolve()
    assert (source.input_key, source.target_key) == ("input_da_test", "output_da_test")
    assert source.provenance == "train_file_validation_fallback"


def test_audit_and_dataset_share_same_source_resolver(tmp_path: Path) -> None:
    validation = tmp_path / DATASET_FILENAMES[("quasi", "validation")]
    _write_quasi(validation, "Yd", "Hd")
    source = resolve_dataset_source(tmp_path, "quasi", "validation")
    audited = validate_dataset_source(source)
    dataset = PriSTRISDataset(source)
    assert dataset.source == source
    assert audited["path"] == str(dataset.path)
    assert audited["input_key"] == dataset.input_key
    assert audited["source_provenance"] == source.provenance


@pytest.mark.parametrize(
    ("input_shape", "target_shape", "message"),
    [
        ((2, 31, 64), (2, 256, 64), "raw input shape"),
        ((2, 32, 63), (2, 256, 64), "raw input shape"),
    ],
)
def test_reject_wrong_raw_input_dimensions(
    tmp_path: Path,
    input_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    message: str,
) -> None:
    path = tmp_path / "bad.mat"
    _write_quasi(path, "input_da", "output_da", input_shape=input_shape, target_shape=target_shape)
    source = DatasetSource("quasi", "train", path, "input_da", "output_da", "test")
    with pytest.raises(ValueError, match=message):
        PriSTRISDataset(source)


def test_reject_mismatched_sample_counts(tmp_path: Path) -> None:
    path = tmp_path / "bad.mat"
    with h5py.File(path, "w") as handle:
        handle["input_da"] = np.zeros((2, 32, 64, 2), dtype=np.float32)
        handle["output_da"] = np.zeros((2, 256, 64, 3), dtype=np.float32)
    source = DatasetSource("quasi", "train", path, "input_da", "output_da", "test")
    with pytest.raises(ValueError, match="sample count mismatch"):
        PriSTRISDataset(source)


def test_grouped_complex_mapping_and_nonfinite_guard(tmp_path: Path) -> None:
    path = tmp_path / "values.mat"
    with h5py.File(path, "w") as handle:
        observed = np.zeros((2, 32, 64, 1), dtype=np.float32)
        observed[0] = 1
        observed[1] = 2
        handle["input_da"] = observed
        target = np.zeros((2, 256, 64, 1), dtype=np.float32)
        target[0] = 3
        target[1] = 4
        handle["output_da"] = target
    source = DatasetSource("quasi", "train", path, "input_da", "output_da", "test")
    item = PriSTRISDataset(source)[0]
    assert torch.all(item["obs_h"][..., 0] == 1)
    assert torch.all(item["obs_h"][..., 1] == 2)
    assert "observation_mask" not in item

    with h5py.File(path, "r+") as handle:
        handle["input_da"][0, 0, 0, 0] = np.nan
    nonfinite = PriSTRISDataset(source)
    with pytest.raises(FloatingPointError):
        nonfinite[0]


def test_test_dataset_and_cli_remain_locked(tmp_path: Path) -> None:
    path = tmp_path / "test.mat"
    _write_quasi(path, "Yd", "Hd")
    source = DatasetSource("quasi", "test", path, "Yd", "Hd", "test")
    with pytest.raises(PermissionError):
        PriSTRISDataset(source)
    with pytest.raises(PermissionError):
        _allowed_test(Namespace(split="test", freeze_manifest=None, checkpoint=Path("x.pth")))


def test_exact_frozen_checkpoint_hash_gate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"frozen")
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
                {
                    "schema": "prist_ris.frozen_experiment.v1",
                    "architecture_version": ARCHITECTURE_VERSION,
                    "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
                    "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
                    "position_semantics_version": POSITION_SEMANTICS_VERSION,
                    "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
                    "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
                "test_unlocked": True,
                "artifacts": [
                    {
                        "kind": "checkpoint",
                        "path": str(checkpoint.resolve()),
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    validate_test_unlock(manifest, checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(PermissionError):
        validate_test_unlock(manifest, checkpoint)


def test_per_query_diagnostics_and_overall_contract() -> None:
    target = torch.ones(2, 6, 1, 1, 2)
    target[:, 5] *= 10
    prediction = target.clone()
    prediction[:, 0] = 0
    prediction[:, 5] = 0
    diagnostics = PerQueryMetricAccumulator(6)
    diagnostics.update(prediction, target)
    result = diagnostics.compute()
    assert set(result["per_query"]) == {f"q{index}" for index in range(6)}
    assert "observed_anchor_aggregate" in result
    assert "pilot_anchor_aggregate" in result
    assert "non_pilot_aggregate" in result
    assert "future_aggregate" not in result
    overall = MetricAccumulator()
    overall.update(prediction, target)
    assert result["overall"] == overall.compute()
    query_db_mean = sum(value["nmse_db"] for value in result["per_query"].values()) / 6
    assert result["overall"]["nmse_db"] != query_db_mean


def test_mobility_q0_q3_data_semantics_contract() -> None:
    semantics = DataSemantics.for_domain("mobility")
    assert semantics.obs_time_index == (0, 3)
    assert semantics.query_time == (0, 1, 2, 3, 4, 5)
    assert semantics.complex_layout == COMPLEX_LAYOUT == "grouped"
    assert semantics.obs_ris_index == OBSERVED_RIS_INDICES == tuple(range(0, 256, 8))


def test_compact_q0_q3_diagnostics_keep_semantic_labels() -> None:
    target = torch.ones(1, 2, 1, 1, 2)
    diagnostics = PerQueryMetricAccumulator((0, 3))
    diagnostics.update(target.clone(), target)
    result = diagnostics.compute()
    assert set(result["per_query"]) == {"q0", "q3"}
    assert "pilot_anchor_aggregate" in result
