from __future__ import annotations

import json
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


def resolve_dataset_path(root: str | Path, domain: str, split: str) -> Path:
    for path in dataset_candidates(root, domain, split):
        if path.is_file():
            return path
    attempted = "\n".join(f"  - {path}" for path in dataset_candidates(root, domain, split))
    raise FileNotFoundError(f"Could not find {domain}/{split}. Attempted:\n{attempted}")


def _grouped_complex(raw: np.ndarray, blocks: int) -> np.ndarray:
    if raw.shape[0] != 2 * blocks:
        raise ValueError(f"Expected {2 * blocks} grouped complex channels, got {raw.shape[0]}.")
    real = raw[:blocks]
    imag = raw[blocks:]
    return np.stack((real, imag), axis=-1)


class PriSTRISDataset(Dataset[dict[str, torch.Tensor]]):
    """Standalone HDF5 loader with frozen grouped-complex semantics."""

    def __init__(
        self,
        path: str | Path,
        domain: str,
        split: str,
        *,
        indices: Sequence[int] | None = None,
        allow_test: bool = False,
    ) -> None:
        if split == "test" and not allow_test:
            raise PermissionError("The test split is locked until a freeze manifest unlocks it.")
        self.path = Path(path).expanduser().resolve()
        self.domain = domain
        self.split = split
        self.semantics = DataSemantics.for_domain(domain)
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            if domain == "quasi" and split == "train":
                input_key, target_key = "input_da", "output_da"
            else:
                input_key, target_key = "Yd", "Hd"
            if input_key not in handle or target_key not in handle:
                raise KeyError(f"Missing {input_key}/{target_key} in {self.path}")
            self.input_key, self.target_key = input_key, target_key
            total = int(handle[input_key].shape[-1])
        if domain == "mobility" and total != EXPECTED_MOBILITY_COUNTS[split]:
            raise ValueError(
                f"Mobility {split} must contain {EXPECTED_MOBILITY_COUNTS[split]} samples, got {total}."
            )
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
            "observation_mask": torch.ones(obs_blocks, 32, dtype=torch.bool),
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
    if indices is None and max_samples is not None:
        total_path = resolve_dataset_path(data_root, domain, split)
        with h5py.File(total_path, "r") as handle:
            key = "input_da" if domain == "quasi" and split == "train" else "Yd"
            total = int(handle[key].shape[-1])
        generator = np.random.default_rng(seed)
        indices = generator.permutation(total)[: min(total, max_samples)].tolist()
    dataset = PriSTRISDataset(
        resolve_dataset_path(data_root, domain, split),
        domain,
        split,
        indices=indices,
        allow_test=allow_test,
    )
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
