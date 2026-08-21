from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .contracts import (
    DataSemantics,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
)
from .data import DatasetSource, PriSTRISDataset
from .engine import move_batch
from .models import PriSTRIS
from .prior import RidgePrior, file_sha256


ANCHOR_CACHE_SCHEMA = "prist_ris.spatial_anchor_cache.v1"


def read_anchor_cache_metadata(path: str | Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        raw = handle.attrs.get("metadata_json")
    if not isinstance(raw, str):
        raise ValueError("Anchor cache is missing metadata_json.")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Anchor cache metadata must be an object.")
    return value


@torch.no_grad()
def write_spatial_anchor_cache(
    path: str | Path,
    *,
    model: PriSTRIS,
    prior: RidgePrior,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    split: str,
    checkpoint_path: str | Path,
    prior_path: str | Path,
) -> dict[str, object]:
    if split not in {"train", "validation"}:
        raise PermissionError("Spatial anchor cache is restricted to TRAIN/VALIDATION.")
    if model.config.domain != "mobility" or tuple(model.output_time_index) != (0, 3):
        raise ValueError("Anchor cache source must be a Mobility q0/q3 spatial model.")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Anchor cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "schema": ANCHOR_CACHE_SCHEMA,
        "split": split,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "prior_path": str(Path(prior_path).resolve()),
        "prior_sha256": file_sha256(prior_path),
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "spatial_multiscale_supervision": model.config.spatial_multiscale_supervision,
        "spatial_channel_attention": model.config.spatial_channel_attention,
        "output_time_index": [0, 3],
        "target_cached": False,
        "test_split_used": False,
    }
    count = 0
    model.eval()
    with h5py.File(destination, "x") as handle:
        anchors_ds = handle.create_dataset(
            "spatial_anchors",
            shape=(0, 2, 256, 64, 2),
            maxshape=(None, 2, 256, 64, 2),
            chunks=(1, 2, 256, 64, 2),
            dtype=np.float32,
        )
        indices_ds = handle.create_dataset(
            "sample_index", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int64
        )
        for raw in loader:
            model_batch = move_batch(
                {key: value for key, value in raw.items() if key != "target_h"}, device
            )
            prior_value = prior.predict(model_batch)
            anchors = model.spatial_anchors(model_batch, prior_value).float().cpu().numpy()
            indices = raw["sample_index"].detach().cpu().numpy().astype(np.int64)
            next_count = count + int(anchors.shape[0])
            anchors_ds.resize(next_count, axis=0)
            indices_ds.resize(next_count, axis=0)
            anchors_ds[count:next_count] = anchors
            indices_ds[count:next_count] = indices
            count = next_count
        metadata["sample_count"] = count
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return {**metadata, "path": str(destination.resolve())}


class SpatialAnchorCacheDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache_path: str | Path,
        source: DatasetSource,
        *,
        expected_checkpoint: str | Path,
        expected_prior: str | Path,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.metadata = read_anchor_cache_metadata(self.cache_path)
        if self.metadata.get("schema") != ANCHOR_CACHE_SCHEMA:
            raise ValueError("Unsupported anchor cache schema.")
        split = self.metadata.get("split")
        if split not in {"train", "validation"} or source.split != split:
            raise PermissionError("Anchor cache/source split mismatch or TEST cache request.")
        expected = DataSemantics.for_domain("mobility").stable_hash()
        if (
            self.metadata.get("semantics_hash") != expected
            or self.metadata.get("spatial_protocol_version") != SPATIAL_PROTOCOL_VERSION
            or self.metadata.get("spatial_supervision_protocol_version")
            != SPATIAL_SUPERVISION_PROTOCOL_VERSION
            or self.metadata.get("position_semantics_version") != POSITION_SEMANTICS_VERSION
            or self.metadata.get("temporal_protocol_version")
            != TEMPORAL_PROTOCOL_VERSION
            or self.metadata.get("checkpoint_sha256") != file_sha256(expected_checkpoint)
            or self.metadata.get("prior_sha256") != file_sha256(expected_prior)
            or self.metadata.get("target_cached") is not False
        ):
            raise ValueError("Anchor cache checkpoint/prior/protocol/hash mismatch.")
        with h5py.File(self.cache_path, "r") as handle:
            indices = np.asarray(handle["sample_index"], dtype=np.int64)
            anchors = handle["spatial_anchors"]
            if anchors.shape != (len(indices), 2, 256, 64, 2):
                raise ValueError("Anchor cache tensor shape mismatch.")
        self.source_dataset = PriSTRISDataset(source, indices=indices.tolist())
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.source_dataset)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.cache_path, "r")
        return self._handle

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.source_dataset[item]
        handle = self._file()
        cached_index = int(handle["sample_index"][item])
        if cached_index != int(sample["sample_index"]):
            raise ValueError("Anchor cache sample-index alignment mismatch.")
        sample["spatial_anchors"] = torch.from_numpy(
            np.asarray(handle["spatial_anchors"][item], dtype=np.float32)
        )
        return sample

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


def make_anchor_cache_loader(
    cache_path: str | Path,
    source: DatasetSource,
    *,
    expected_checkpoint: str | Path,
    expected_prior: str | Path,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SpatialAnchorCacheDataset(
        cache_path,
        source,
        expected_checkpoint=expected_checkpoint,
        expected_prior=expected_prior,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )
