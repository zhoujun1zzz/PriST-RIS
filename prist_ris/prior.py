from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .contracts import DataSemantics


def _complex(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(value[..., 0], value[..., 1])


@dataclass
class RidgeStatistics:
    xhx: np.ndarray
    xhy: np.ndarray
    rows: int
    target_blocks: tuple[int, ...]
    fit_split: str = "train"

    @classmethod
    def empty(cls, features: int, outputs: int, target_blocks: tuple[int, ...]) -> "RidgeStatistics":
        return cls(
            np.zeros((features, features), dtype=np.complex128),
            np.zeros((features, outputs), dtype=np.complex128),
            0,
            target_blocks,
        )

    @classmethod
    def accumulate(
        cls, loader: Iterable[dict[str, torch.Tensor]], target_blocks: tuple[int, ...]
    ) -> "RidgeStatistics":
        loader_split = getattr(getattr(loader, "dataset", None), "split", None)
        if loader_split is not None and loader_split != "train":
            raise PermissionError(f"Ridge statistics must use train only, got {loader_split!r}.")
        statistics: RidgeStatistics | None = None
        for batch in loader:
            observed = _complex(batch["obs_h"]).permute(0, 3, 1, 2).reshape(-1, batch["obs_h"].shape[1] * 32)
            selected = _complex(batch["target_h"][:, target_blocks])
            target = selected.permute(0, 3, 1, 2).reshape(-1, len(target_blocks) * 256)
            x = observed.numpy().astype(np.complex128, copy=False)
            y = target.numpy().astype(np.complex128, copy=False)
            if statistics is None:
                statistics = cls.empty(x.shape[1], y.shape[1], target_blocks)
            # Real-valued GEMMs avoid a known complex-BLAS abort on some
            # Windows MKL builds while remaining algebraically identical.
            xr, xi = x.real, x.imag
            yr, yi = y.real, y.imag
            statistics.xhx += (xr.T @ xr + xi.T @ xi) + 1j * (xr.T @ xi - xi.T @ xr)
            statistics.xhy += (xr.T @ yr + xi.T @ yi) + 1j * (xr.T @ yi - xi.T @ yr)
            statistics.rows += x.shape[0]
        if statistics is None:
            raise RuntimeError("Cannot fit Ridge prior from an empty loader.")
        return statistics

    def solve(self, regularization: float, semantics: DataSemantics) -> "RidgePrior":
        if regularization < 0:
            raise ValueError("regularization must be non-negative.")
        scale = max(1, self.rows)
        system = self.xhx / scale + regularization * np.eye(self.xhx.shape[0])
        right = self.xhy / scale
        real_system = np.block([[system.real, -system.imag], [system.imag, system.real]])
        real_right = np.concatenate((right.real, right.imag), axis=0)
        # torch.linalg uses the PyTorch-shipped LAPACK path and is reliable on
        # Windows environments where NumPy's MKL solve may abort the process.
        real_solution = torch.linalg.solve(
            torch.from_numpy(np.ascontiguousarray(real_system)),
            torch.from_numpy(np.ascontiguousarray(real_right)),
        ).numpy()
        features = system.shape[0]
        coefficients = real_solution[:features] + 1j * real_solution[features:]
        return RidgePrior(
            coefficients=coefficients,
            regularization=regularization,
            rows=self.rows,
            target_blocks=self.target_blocks,
            semantics_hash=semantics.stable_hash(),
            fit_split=self.fit_split,
        )


@dataclass
class RidgePrior:
    coefficients: np.ndarray
    regularization: float
    rows: int
    target_blocks: tuple[int, ...]
    semantics_hash: str
    fit_split: str = "train"
    provenance: dict[str, object] | None = None

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        observed = _complex(batch["obs_h"])
        b, t, _, antennas = observed.shape
        x = observed.permute(0, 3, 1, 2).reshape(b * antennas, t * 32)
        weights = torch.from_numpy(self.coefficients).to(device=x.device, dtype=x.dtype)
        output = x @ weights
        output = output.reshape(b, antennas, len(self.target_blocks), 256)
        output = output.permute(0, 2, 3, 1)
        return torch.stack((output.real, output.imag), dim=-1).to(batch["obs_h"].dtype)

    def metadata(self) -> dict[str, object]:
        metadata = {
            "regularization": self.regularization,
            "fit_rows": self.rows,
            "fit_split": self.fit_split,
            "target_blocks": list(self.target_blocks),
            "semantics_hash": self.semantics_hash,
            "coefficient_shape": list(self.coefficients.shape),
        }
        if self.provenance:
            metadata.update(self.provenance)
        return metadata

    def save(self, path: str | Path) -> dict[str, object]:
        destination = Path(path).expanduser().resolve()
        if destination.suffix.lower() != ".npz":
            destination = destination.with_suffix(".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            coefficients=self.coefficients,
            metadata=np.asarray(json.dumps(self.metadata(), sort_keys=True)),
        )
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return {**self.metadata(), "path": str(destination), "sha256": digest}

    @classmethod
    def load(cls, path: str | Path) -> "RidgePrior":
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as artifact:
            coefficients = artifact["coefficients"]
            metadata = json.loads(str(artifact["metadata"].item()))
        base_keys = {
            "regularization", "fit_rows", "fit_split", "target_blocks",
            "semantics_hash", "coefficient_shape",
        }
        return cls(
            coefficients=coefficients,
            regularization=float(metadata["regularization"]),
            rows=int(metadata["fit_rows"]),
            target_blocks=tuple(int(v) for v in metadata["target_blocks"]),
            semantics_hash=str(metadata["semantics_hash"]),
            fit_split=str(metadata["fit_split"]),
            provenance={key: value for key, value in metadata.items() if key not in base_keys},
        )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
