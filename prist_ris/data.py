from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .contracts import DataSemantics


DATASET_FILENAMES = {
    ("quasi", "train"): "indoorH_LS_Data6users_1B32pilot.mat",
    ("quasi", "validation"): "indoorH_LSval_Data6users_1B32pilot.mat",
    ("quasi", "test"): "indoorH_LStest_Data6users_1B32pilot.mat",
    ("mobility", "train"): "OutdoorH_LS_Data6users_60B32pilot.mat",
    ("mobility", "validation"): "OutdoorH_LSval_Data6users_60B32pilot.mat",
    ("mobility", "test"): "OutdoorH_LStest_Data6users_60B32pilot.mat",
}
EXPECTED_MOBILITY_COUNTS = {"train": 20000, "validation": 1800, "test": 9000}


@dataclass(frozen=True)
class DatasetSource:
    domain: str
    split: str
    path: Path
    input_key: str
    target_key: str
    provenance: str


def dataset_candidates(root: str | Path, domain: str, split: str) -> list[Path]:
    key = (domain, split)
    if key not in DATASET_FILENAMES:
        raise ValueError(f"Unsupported dataset selection: {domain}/{split}")
    base = Path(root).expanduser().resolve()
    filename = DATASET_FILENAMES[key]
    folder = Path(filename).stem
    legacy = "risce" if domain == "quasi" else "risce-0"
    values = [
        base / domain / folder / filename,
        base / domain / filename,
        base / folder / filename,
        base / filename,
        base / legacy / folder / filename,
    ]
    return list(dict.fromkeys(values))


def _has_keys(path: Path, input_key: str, target_key: str) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as handle:
        return input_key in handle and target_key in handle


def resolve_dataset_source(root: str | Path, domain: str, split: str) -> DatasetSource:
    """Resolve the file and keys once for Dataset, audit, and DataLoader."""

    if domain not in {"quasi", "mobility"} or split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported dataset selection: {domain}/{split}")
    attempted: list[str] = []
    if domain == "quasi" and split == "validation":
        for path in dataset_candidates(root, domain, split):
            attempted.append(f"{path} [Yd/Hd]")
            if _has_keys(path, "Yd", "Hd"):
                return DatasetSource(domain, split, path, "Yd", "Hd", "separate_validation_yd_hd")
        for path in dataset_candidates(root, domain, "train"):
            attempted.append(f"{path} [input_da_test/output_da_test]")
            if _has_keys(path, "input_da_test", "output_da_test"):
                return DatasetSource(
                    domain,
                    split,
                    path,
                    "input_da_test",
                    "output_da_test",
                    "train_file_validation_fallback",
                )
    else:
        input_key, target_key = (
            ("input_da", "output_da") if domain == "quasi" and split == "train" else ("Yd", "Hd")
        )
        for path in dataset_candidates(root, domain, split):
            attempted.append(f"{path} [{input_key}/{target_key}]")
            if _has_keys(path, input_key, target_key):
                provenance = "quasi_train" if domain == "quasi" and split == "train" else "separate_split_yd_hd"
                return DatasetSource(domain, split, path, input_key, target_key, provenance)
    rendered = "\n".join(f"  - {value}" for value in attempted)
    raise FileNotFoundError(f"Could not resolve {domain}/{split}. Attempted:\n{rendered}")


def resolve_dataset_path(root: str | Path, domain: str, split: str) -> Path:
    """Compatibility wrapper; new code should retain the complete DatasetSource."""

    return resolve_dataset_source(root, domain, split).path


def _validate_raw_shapes(
    handle: h5py.File, source: DatasetSource, semantics: DataSemantics
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if source.input_key not in handle or source.target_key not in handle:
        raise KeyError(f"Missing {source.input_key}/{source.target_key} in {source.path}")
    input_shape = tuple(int(value) for value in handle[source.input_key].shape)
    target_shape = tuple(int(value) for value in handle[source.target_key].shape)
    expected_input_channels = 2 if source.domain == "quasi" else 4
    expected_target_channels = 2 if source.domain == "quasi" else 12
    if len(input_shape) != 4 or input_shape[:3] != (expected_input_channels, 32, 64):
        raise ValueError(
            f"Invalid {source.domain}/{source.split} raw input shape {input_shape}; "
            f"expected [{expected_input_channels},32,64,N]."
        )
    if len(target_shape) != 4 or target_shape[:3] != (expected_target_channels, 256, 64):
        raise ValueError(
            f"Invalid {source.domain}/{source.split} raw target shape {target_shape}; "
            f"expected [{expected_target_channels},256,64,N]."
        )
    if input_shape[-1] != target_shape[-1]:
        raise ValueError(f"Input/target sample count mismatch: {input_shape[-1]} vs {target_shape[-1]}.")
    total = input_shape[-1]
    if source.domain == "mobility" and total != EXPECTED_MOBILITY_COUNTS[source.split]:
        raise ValueError(
            f"Mobility {source.split} must contain {EXPECTED_MOBILITY_COUNTS[source.split]} samples, got {total}."
        )
    if tuple(semantics.obs_ris_index) != tuple(range(0, 256, 8)):
        raise ValueError("Data semantics no longer match the frozen observed RIS indices.")
    return input_shape, target_shape, total


def validate_dataset_source(source: DatasetSource) -> dict[str, object]:
    semantics = DataSemantics.for_domain(source.domain)
    with h5py.File(source.path, "r") as handle:
        input_shape, target_shape, total = _validate_raw_shapes(handle, source, semantics)
    return {
        "domain": source.domain,
        "split": source.split,
        "path": str(source.path),
        "input_key": source.input_key,
        "target_key": source.target_key,
        "raw_input_shape": list(input_shape),
        "raw_target_shape": list(target_shape),
        "samples": total,
        "source_provenance": source.provenance,
        "semantics_hash": semantics.stable_hash(),
    }


def _grouped_complex(raw: np.ndarray, blocks: int) -> np.ndarray:
    if raw.shape[0] != 2 * blocks:
        raise ValueError(f"Expected {2 * blocks} grouped complex channels, got {raw.shape[0]}.")
    value = np.stack((raw[:blocks], raw[blocks:]), axis=-1)
    if not np.isfinite(value).all():
        raise FloatingPointError("Raw channel sample contains non-finite values.")
    return value


class PriSTRISDataset(Dataset[dict[str, torch.Tensor]]):
    """Standalone HDF5 loader with a validated, provenance-bearing source."""

    def __init__(
        self,
        source: DatasetSource,
        *,
        indices: Sequence[int] | None = None,
        allow_test: bool = False,
    ) -> None:
        if source.split == "test" and not allow_test:
            raise PermissionError("The test split is locked until a freeze manifest unlocks it.")
        self.source = source
        self.path = source.path
        self.domain = source.domain
        self.split = source.split
        self.input_key = source.input_key
        self.target_key = source.target_key
        self.semantics = DataSemantics.for_domain(source.domain)
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            _, _, total = _validate_raw_shapes(handle, source, self.semantics)
        values = np.arange(total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional sequence.")
        if values.min() < 0 or values.max() >= total or np.unique(values).size != values.size:
            raise ValueError("indices must be unique and within the dataset.")
        self.indices = values

    def __len__(self) -> int:
        return int(self.indices.size)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample_index = int(self.indices[item])
        handle = self._file()
        obs_blocks, query_blocks = ((1, 1) if self.domain == "quasi" else (2, 6))
        observed = np.asarray(handle[self.input_key][..., sample_index], dtype=np.float32)
        target = np.asarray(handle[self.target_key][..., sample_index], dtype=np.float32)
        observed = _grouped_complex(observed, obs_blocks)
        target = _grouped_complex(target, query_blocks)
        return {
            "obs_h": torch.from_numpy(observed.copy()),
            "target_h": torch.from_numpy(target.copy()),
            "obs_ris_index": torch.tensor(self.semantics.obs_ris_index, dtype=torch.long),
            "obs_time_index": torch.tensor(self.semantics.obs_time_index, dtype=torch.long),
            "query_time": torch.tensor(self.semantics.query_time, dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


def make_loader(
    data_root: str | Path,
    domain: str,
    split: str,
    *,
    batch_size: int,
    workers: int = 0,
    seed: int = 123,
    max_samples: int | None = None,
    indices: Sequence[int] | None = None,
    shuffle: bool | None = None,
    allow_test: bool = False,
) -> DataLoader:
    source = resolve_dataset_source(data_root, domain, split)
    if indices is None and max_samples is not None:
        total = int(validate_dataset_source(source)["samples"])
        generator = np.random.default_rng(seed)
        indices = generator.permutation(total)[: min(total, max_samples)].tolist()
    dataset = PriSTRISDataset(source, indices=indices, allow_test=allow_test)
    torch_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=workers,
        generator=torch_generator,
        pin_memory=torch.cuda.is_available(),
    )


def nested_fraction_indices(total: int, fractions: Iterable[float], seed: int) -> dict[str, list[int]]:
    ordered = np.random.default_rng(seed).permutation(total)
    result: dict[str, list[int]] = {}
    for fraction in sorted(set(float(value) for value in fractions)):
        if not 0 < fraction <= 1:
            raise ValueError("fractions must be in (0, 1].")
        count = max(1, int(round(total * fraction)))
        result[f"{fraction:.2f}"] = ordered[:count].tolist()
    return result


def write_index_manifest(path: Path, values: dict[str, list[int]], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": seed, "nested": True, "fractions": values}, indent=2),
        encoding="utf-8",
    )
