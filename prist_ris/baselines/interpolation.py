from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def _shared_vector(value: torch.Tensor) -> torch.Tensor:
    """Return the common semantic vector from batched or unbatched metadata."""
    return value[0] if value.ndim > 1 else value


def linear_query_weights(
    obs_time: torch.Tensor,
    query_time: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear temporal weights with nearest-value extension."""
    obs = obs_time.to(dtype=torch.float32)
    query = query_time.to(dtype=torch.float32)

    if obs.numel() == 1:
        return torch.ones(
            query.numel(), 1, device=query.device, dtype=torch.float32
        )

    order = torch.argsort(obs)
    obs_sorted = obs[order]

    weights = torch.zeros(
        query.numel(),
        obs.numel(),
        device=query.device,
        dtype=torch.float32,
    )

    for qi, value in enumerate(query):
        if value <= obs_sorted[0]:
            weights[qi, 0] = 1.0
        elif value >= obs_sorted[-1]:
            weights[qi, -1] = 1.0
        else:
            right = int(torch.searchsorted(obs_sorted, value).item())
            left = right - 1
            alpha = (
                (value - obs_sorted[left])
                / (obs_sorted[right] - obs_sorted[left])
            )
            weights[qi, left] = 1.0 - alpha
            weights[qi, right] = alpha

    restored = torch.zeros_like(weights)
    restored[:, order] = weights
    return restored


def grid_aware_spatial_weights(
    obs_index: torch.Tensor,
    *,
    nodes: int = 256,
    grid_width: int = 16,
    nearest: bool = False,
) -> torch.Tensor:
    """Row-wise interpolation weights on the physical 16x16 RIS grid."""
    index = obs_index.to(dtype=torch.long)

    if nodes <= 0 or grid_width <= 0 or nodes % grid_width:
        raise ValueError("nodes must be divisible by grid_width.")
    if index.ndim != 1 or index.numel() == 0:
        raise ValueError("obs_index must be a non-empty 1-D tensor.")
    if int(index.min()) < 0 or int(index.max()) >= nodes:
        raise ValueError("Observed RIS index is outside the dense grid.")
    if torch.unique(index).numel() != index.numel():
        raise ValueError("Observed RIS indices must be unique.")

    weights = torch.zeros(
        nodes,
        index.numel(),
        device=index.device,
        dtype=torch.float32,
    )

    rows = torch.div(index, grid_width, rounding_mode="floor")
    cols = index % grid_width

    for row in range(nodes // grid_width):
        row_positions = torch.where(rows == row)[0]
        if row_positions.numel() == 0:
            raise ValueError(
                f"RIS row {row} contains no observed element."
            )

        order = torch.argsort(cols[row_positions])
        positions = row_positions[order]
        columns = cols[positions].to(torch.float32)

        for column in range(grid_width):
            node = row * grid_width + column
            value = torch.tensor(
                float(column),
                device=index.device,
                dtype=torch.float32,
            )

            if nearest or positions.numel() == 1:
                distances = torch.abs(columns - value)
                weights[
                    node,
                    positions[torch.argmin(distances)],
                ] = 1.0
            elif value <= columns[0]:
                weights[node, positions[0]] = 1.0
            elif value >= columns[-1]:
                weights[node, positions[-1]] = 1.0
            else:
                right = int(torch.searchsorted(columns, value).item())
                if columns[right] == value:
                    weights[node, positions[right]] = 1.0
                else:
                    left = right - 1
                    alpha = (
                        (value - columns[left])
                        / (columns[right] - columns[left])
                    )
                    weights[node, positions[left]] = 1.0 - alpha
                    weights[node, positions[right]] = alpha

    return weights


def expand_observations_to_grid(
    batch: Mapping[str, torch.Tensor],
    *,
    nearest: bool = False,
) -> torch.Tensor:
    obs = batch["obs_h"]

    if obs.ndim != 5 or obs.shape[-1] != 2:
        raise ValueError(
            "obs_h must have shape [B,T,P,M,2]."
        )

    obs_index = _shared_vector(
        batch["obs_ris_index"]
    ).to(obs.device)

    weights = grid_aware_spatial_weights(
        obs_index,
        nearest=nearest,
    ).to(obs.dtype)

    return torch.einsum(
        "np,btpmc->btnmc",
        weights,
        obs,
    )


@torch.no_grad()
def interpolation_baseline(
    batch: Mapping[str, torch.Tensor],
    *,
    spatial: str = "linear",
    temporal: str = "linear",
) -> torch.Tensor:
    """Training-free spatial/temporal interpolation baseline."""

    if spatial not in {"linear", "nearest"}:
        raise ValueError(
            "spatial must be 'linear' or 'nearest'."
        )
    if temporal not in {"linear", "nearest"}:
        raise ValueError(
            "temporal must be 'linear' or 'nearest'."
        )

    obs = batch["obs_h"]
    obs_time = _shared_vector(
        batch["obs_time_index"]
    ).to(obs.device)
    query_time = _shared_vector(
        batch["query_time"]
    ).to(obs.device)

    spatial_full = expand_observations_to_grid(
        batch,
        nearest=(spatial == "nearest"),
    )

    if temporal == "nearest":
        distances = torch.abs(
            query_time[:, None].float()
            - obs_time[None, :].float()
        )
        temporal_weights = F.one_hot(
            distances.argmin(dim=1),
            num_classes=obs_time.numel(),
        ).to(obs.dtype)
    else:
        temporal_weights = linear_query_weights(
            obs_time,
            query_time,
        ).to(obs.dtype)

    return torch.einsum(
        "qt,btnmc->bqnmc",
        temporal_weights,
        spatial_full,
    )
