from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (
    ANTENNAS,
    ARCHITECTURE_VERSION,
    DataSemantics,
    MOBILITY_CONTRACT_VERSION,
    MODEL_DISPLAY_NAME,
    OBSERVED_RIS_INDICES,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    canonical_model_key,
)


PHYSICAL_STAGE_COLUMNS: dict[int, tuple[int, ...]] = {
    2: (0, 8),
    4: (0, 4, 8, 12),
    8: tuple(range(0, 16, 2)),
    16: tuple(range(16)),
}
SPATIAL_RESIDUAL_STYLES = ("post_activation", "scaled_true_residual")
BACKBONE_RIS_COORDINATE_MODES = ("off", "direct_add", "zero_init_gated")
OBSERVED_DENSE_ATTENTION_HEADS = 4
OBSERVED_DENSE_ATTENTION_RESIDUAL_SCALE = 0.1


def _tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _canonical_indices(obs_ris_index: torch.Tensor, batch_size: int) -> torch.Tensor:
    if obs_ris_index.ndim == 2:
        if obs_ris_index.shape != (batch_size, 32):
            raise ValueError("batched obs_ris_index must have shape [B,32].")
        expected = obs_ris_index[0]
        if not torch.equal(obs_ris_index, expected.expand_as(obs_ris_index)):
            raise ValueError("Every sample must use the same frozen observed RIS indices.")
    elif obs_ris_index.ndim == 1 and obs_ris_index.shape == (32,):
        expected = obs_ris_index
    else:
        raise ValueError("obs_ris_index must have shape [32] or [B,32].")
    canonical = torch.tensor(OBSERVED_RIS_INDICES, device=expected.device, dtype=expected.dtype)
    if not torch.equal(expected, canonical):
        raise ValueError("Observed RIS indices must be exactly [0,8,...,248].")
    rows = torch.div(expected, 16, rounding_mode="floor")
    columns = expected.remainder(16)
    if not torch.equal(rows, torch.arange(16, device=rows.device).repeat_interleave(2)):
        raise ValueError("Observed RIS indices do not cover rows 0..15 twice in row-major order.")
    if not torch.equal(columns, torch.tensor([0, 8], device=columns.device).repeat(16)):
        raise ValueError("Observed RIS columns must be {0,8} for every row.")
    return expected


def observations_to_physical_grid(
    observations: torch.Tensor, obs_ris_index: torch.Tensor
) -> torch.Tensor:
    """Map [B,T,32,64,2] to [B,4,64,16,2] using validated row-major indices."""

    if observations.ndim != 5 or observations.shape[2:] != (32, ANTENNAS, 2):
        raise ValueError("observations must have shape [B,T,32,64,2].")
    b, times = observations.shape[:2]
    if times not in {1, 2}:
        raise ValueError("PriST-RIS supports one or two observed time blocks.")
    _canonical_indices(obs_ris_index, b)
    grid = observations.reshape(b, times, 16, 2, ANTENNAS, 2)
    grid = grid.permute(0, 1, 5, 4, 2, 3).reshape(b, 2 * times, ANTENNAS, 16, 2)
    if grid.shape[1] < 4:
        grid = F.pad(grid, (0, 0, 0, 0, 0, 0, 0, 4 - grid.shape[1]))
    return grid.contiguous()


def physical_grid_to_observations(grid: torch.Tensor, observed_times: int) -> torch.Tensor:
    """Inverse of observations_to_physical_grid for contract tests."""

    if grid.ndim != 5 or grid.shape[1:] != (4, ANTENNAS, 16, 2):
        raise ValueError("grid must have shape [B,4,64,16,2].")
    if observed_times not in {1, 2}:
        raise ValueError("observed_times must be 1 or 2.")
    b = grid.shape[0]
    value = grid[:, : 2 * observed_times].reshape(b, observed_times, 2, ANTENNAS, 16, 2)
    return value.permute(0, 1, 4, 5, 3, 2).reshape(b, observed_times, 32, ANTENNAS, 2)


def prior_to_physical_grid(prior: torch.Tensor) -> torch.Tensor:
    """Map K complex anchors to [B,4,64,16,16], padding Quasi's second anchor."""

    if prior.ndim != 5 or prior.shape[1] not in {1, 2} or prior.shape[2:] != (256, ANTENNAS, 2):
        raise ValueError("prior must have shape [B,1|2,256,64,2].")
    b, anchors = prior.shape[:2]
    grid = prior.reshape(b, anchors, 16, 16, ANTENNAS, 2)
    grid = grid.permute(0, 1, 5, 4, 2, 3).reshape(b, 2 * anchors, ANTENNAS, 16, 16)
    if anchors == 1:
        grid = F.pad(grid, (0, 0, 0, 0, 0, 0, 0, 2))
    return grid.contiguous()


def prior_anchor_to_physical_grid(prior: torch.Tensor) -> torch.Tensor:
    """Map one complex anchor to an independent [B,2,64,16,16] grid."""

    if prior.ndim != 5 or prior.shape[1:] != (1, 256, ANTENNAS, 2):
        raise ValueError("prior anchor must have shape [B,1,256,64,2].")
    batch_size = prior.shape[0]
    grid = prior.reshape(batch_size, 1, 16, 16, ANTENNAS, 2)
    return grid.permute(0, 1, 5, 4, 2, 3).reshape(
        batch_size, 2, ANTENNAS, 16, 16
    ).contiguous()


def physical_grid_to_anchors(grid: torch.Tensor, anchors: int) -> torch.Tensor:
    if grid.ndim != 5 or grid.shape[1:] != (4, ANTENNAS, 16, 16):
        raise ValueError("anchor grid must have shape [B,4,64,16,16].")
    if anchors not in {1, 2}:
        raise ValueError("anchors must be 1 or 2.")
    b = grid.shape[0]
    value = grid[:, : 2 * anchors].reshape(b, anchors, 2, ANTENNAS, 16, 16)
    return value.permute(0, 1, 4, 5, 3, 2).reshape(b, anchors, 256, ANTENNAS, 2).contiguous()


class RISCoordinateEncoder(nn.Module):
    """Encode physical RIS row/column coordinates for each progressive stage."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(2, hidden, 1)

    @staticmethod
    def coordinates(width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if width not in PHYSICAL_STAGE_COLUMNS:
            raise ValueError(f"Unsupported physical stage width {width}.")
        rows = torch.arange(16, device=device, dtype=dtype) * (2.0 / 15.0) - 1.0
        columns = torch.tensor(PHYSICAL_STAGE_COLUMNS[width], device=device, dtype=dtype)
        columns = columns * (2.0 / 15.0) - 1.0
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        return torch.stack((row_grid, column_grid), dim=0).reshape(1, 2, 1, 16, width)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        coordinates = self.coordinates(reference.shape[-1], device=reference.device, dtype=reference.dtype)
        return self.projection(coordinates).expand(-1, -1, ANTENNAS, -1, -1)


class AntennaIndexEncoder(nn.Module):
    """Encode antenna index only; this is not a physical BS-coordinate claim."""

    semantics = "antenna_index_encoding"

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(1, hidden, 1)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        values = torch.arange(ANTENNAS, device=reference.device, dtype=reference.dtype)
        values = values * (2.0 / (ANTENNAS - 1)) - 1.0
        return self.projection(values.reshape(1, 1, ANTENNAS, 1, 1)).expand(
            -1, -1, -1, 16, reference.shape[-1]
        )


class StrongSpatioRISResidualBlock(nn.Module):
    def __init__(self, hidden: int, residual_style: str = "scaled_true_residual") -> None:
        super().__init__()
        if residual_style not in SPATIAL_RESIDUAL_STYLES:
            raise ValueError(
                f"residual_style must be one of {SPATIAL_RESIDUAL_STYLES}."
            )
        self.residual_style = residual_style
        self.body = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.residual_style == "scaled_true_residual":
            return value + 0.1 * F.gelu(self.body(value))
        return self.activation(value + self.body(value))


class PhysicalColumnUpsample(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        before = value.shape
        if value.ndim != 5 or value.shape[1] != self.hidden:
            raise ValueError("PhysicalColumnUpsample expects [B,H,A,16,W].")
        result = F.interpolate(value, scale_factor=(1, 1, 2), mode="nearest")
        if result.shape[2:4] != before[2:4] or result.shape[-1] != 2 * before[-1]:
            raise RuntimeError("Physical column upsampling changed a non-column axis.")
        return result


class PhysicalProgressiveStage(nn.Module):
    def __init__(self, hidden: int, blocks: int, residual_style: str) -> None:
        super().__init__()
        self.upsample = PhysicalColumnUpsample(hidden)
        self.blocks = nn.Sequential(
            *(StrongSpatioRISResidualBlock(hidden, residual_style) for _ in range(blocks))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.upsample(value))


class PhysicalGridBackbone(nn.Module):
    def __init__(
        self,
        hidden: int = 80,
        blocks_per_stage: tuple[int, int, int] = (3, 3, 4),
        final_refine_blocks: int = 4,
        *,
        ris_coordinate_enabled: bool = False,
        antenna_index_enabled: bool = False,
        ris_coordinate_mode: str = "off",
        residual_style: str = "scaled_true_residual",
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 3 or final_refine_blocks < 1:
            raise ValueError("PriST-RIS V3.2 requires three progressive stages and final refinement.")
        if ris_coordinate_mode not in BACKBONE_RIS_COORDINATE_MODES:
            raise ValueError(
                f"ris_coordinate_mode must be one of {BACKBONE_RIS_COORDINATE_MODES}."
            )
        if ris_coordinate_enabled != (ris_coordinate_mode != "off"):
            raise ValueError(
                "RIS coordinate enablement and injection mode must agree."
            )
        self.hidden = hidden
        self.ris_coordinate_enabled = ris_coordinate_enabled
        self.antenna_index_enabled = antenna_index_enabled
        self.ris_coordinate_mode = ris_coordinate_mode
        self.residual_style = residual_style
        self.input = nn.Conv3d(4, hidden, 3, padding=1)
        # Optional position modules must not perturb the initialization of the
        # stable B path when a zero-init gate is added for an ablation.
        with torch.random.fork_rng(devices=[]):
            self.ris_coordinate_encoder = (
                RISCoordinateEncoder(hidden) if ris_coordinate_enabled else None
            )
            self.antenna_index_encoder = (
                AntennaIndexEncoder(hidden) if antenna_index_enabled else None
            )
        self.ris_coordinate_gates = (
            nn.Parameter(torch.zeros(4))
            if ris_coordinate_mode == "zero_init_gated"
            else None
        )
        self.stages = nn.ModuleList(
            PhysicalProgressiveStage(hidden, blocks, residual_style)
            for blocks in blocks_per_stage
        )
        self.final_refine = nn.Sequential(
            *(
                StrongSpatioRISResidualBlock(hidden, residual_style)
                for _ in range(final_refine_blocks)
            )
        )

    def _position_features(self, value: torch.Tensor, stage_index: int) -> torch.Tensor:
        result = torch.zeros_like(value)
        if self.ris_coordinate_encoder is not None:
            ris = self.ris_coordinate_encoder(value)
            if self.ris_coordinate_gates is not None:
                ris = self.ris_coordinate_gates[stage_index] * ris
            result = result + ris
        if self.antenna_index_encoder is not None:
            result = result + self.antenna_index_encoder(value)
        return result

    def forward(
        self, observations: torch.Tensor, obs_ris_index: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        grid = observations_to_physical_grid(observations, obs_ris_index)
        value = self.input(grid)
        if self.ris_coordinate_enabled or self.antenna_index_enabled:
            value = value + self._position_features(value, 0)
        shapes = [tuple(value.shape)]
        for stage_index, stage in enumerate(self.stages, start=1):
            value = stage(value)
            if self.ris_coordinate_enabled or self.antenna_index_enabled:
                value = value + self._position_features(value, stage_index)
            shapes.append(tuple(value.shape))
        value = self.final_refine(value)
        if value.shape[2:] != (ANTENNAS, 16, 16):
            raise RuntimeError(f"Physical backbone ended at {value.shape[2:]}, expected (64,16,16).")
        return value, shapes


class PhysicalObservedDenseResidualAttention(nn.Module):
    """Per-antenna 32-to-256 cross-attention over physical RIS positions."""

    def __init__(
        self,
        hidden: int,
        heads: int = OBSERVED_DENSE_ATTENTION_HEADS,
        *,
        residual_scale: float = OBSERVED_DENSE_ATTENTION_RESIDUAL_SCALE,
        ris_coordinate_enabled: bool = True,
        antenna_index_enabled: bool = True,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("Attention hidden width must be divisible by heads.")
        if residual_scale <= 0:
            raise ValueError("Attention residual_scale must be positive.")
        self.hidden = hidden
        self.heads = heads
        self.head_width = hidden // heads
        self.residual_scale = float(residual_scale)
        self.ris_coordinate_enabled = ris_coordinate_enabled
        self.antenna_index_enabled = antenna_index_enabled
        self.pilot_token_projection = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.observed_coordinate_projection = (
            nn.Linear(2, hidden) if ris_coordinate_enabled else None
        )
        self.dense_coordinate_projection = (
            nn.Linear(2, hidden) if ris_coordinate_enabled else None
        )
        self.antenna_projection = (
            nn.Linear(1, hidden) if antenna_index_enabled else None
        )
        self.observed_norm = nn.LayerNorm(hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.query_projection = nn.Linear(hidden, hidden)
        self.key_projection = nn.Linear(hidden, hidden)
        self.value_projection = nn.Linear(hidden, hidden)
        self.output_projection = nn.Linear(hidden, hidden)

    @staticmethod
    def pilot_descriptors(
        obs_time_index: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        if obs_time_index.ndim == 1:
            obs_time_index = obs_time_index.unsqueeze(0)
        if obs_time_index.ndim != 2 or obs_time_index.shape[1] not in {1, 2}:
            raise ValueError("obs_time_index must have shape [T] or [B,T] for T=1|2.")
        times = obs_time_index.to(dtype=dtype) / 3.0
        count = obs_time_index.shape[1]
        slots = (
            torch.zeros(1, 1, device=obs_time_index.device, dtype=dtype)
            if count == 1
            else torch.tensor([-1.0, 1.0], device=obs_time_index.device, dtype=dtype).reshape(1, 2)
        )
        return torch.stack((times, slots.expand_as(times)), dim=-1)

    @staticmethod
    def observed_coordinates(
        obs_ris_index: torch.Tensor, batch_size: int, *, dtype: torch.dtype
    ) -> torch.Tensor:
        indices = _canonical_indices(obs_ris_index, batch_size)
        rows = torch.div(indices, 16, rounding_mode="floor").to(dtype)
        columns = indices.remainder(16).to(dtype)
        return torch.stack((rows * (2.0 / 15.0) - 1.0, columns * (2.0 / 15.0) - 1.0), dim=-1)

    @staticmethod
    def dense_coordinates(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = RISCoordinateEncoder.coordinates(16, device=device, dtype=dtype)
        return value.reshape(2, 256).transpose(0, 1).contiguous()

    @staticmethod
    def antenna_coordinates(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = torch.arange(ANTENNAS, device=device, dtype=dtype)
        return (values * (2.0 / (ANTENNAS - 1)) - 1.0).reshape(ANTENNAS, 1)

    def _observed_tokens(
        self,
        observations: torch.Tensor,
        obs_ris_index: torch.Tensor,
        obs_time_index: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim != 5 or observations.shape[2:] != (32, ANTENNAS, 2):
            raise ValueError("observations must have shape [B,T,32,64,2].")
        batch_size, times = observations.shape[:2]
        descriptors = self.pilot_descriptors(obs_time_index, dtype=observations.dtype)
        if descriptors.shape[0] == 1 and batch_size > 1:
            descriptors = descriptors.expand(batch_size, -1, -1)
        if descriptors.shape[:2] != (batch_size, times):
            raise ValueError("obs_time_index does not match observation batch/time dimensions.")
        values = observations.permute(0, 3, 2, 1, 4)
        descriptors = descriptors[:, None, None].expand(-1, ANTENNAS, 32, -1, -1)
        pilot_inputs = torch.cat((values, descriptors), dim=-1)
        tokens = self.pilot_token_projection(pilot_inputs).sum(dim=3) / (times ** 0.5)
        if self.observed_coordinate_projection is not None:
            observed_coordinates = self.observed_coordinates(
                obs_ris_index, batch_size, dtype=observations.dtype
            )
            tokens = tokens + self.observed_coordinate_projection(
                observed_coordinates
            )[None, None]
        if self.antenna_projection is not None:
            antenna_coordinates = self.antenna_coordinates(
                device=observations.device, dtype=observations.dtype
            )
            tokens = tokens + self.antenna_projection(antenna_coordinates)[None, :, None]
        return self.observed_norm(tokens).reshape(batch_size * ANTENNAS, 32, self.hidden)

    def _dense_queries(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        queries = features.permute(0, 2, 3, 4, 1)
        if self.dense_coordinate_projection is not None:
            dense_coordinates = self.dense_coordinates(
                device=features.device, dtype=features.dtype
            )
            queries = queries + self.dense_coordinate_projection(dense_coordinates)[
                None, None
            ].reshape(1, 1, 16, 16, self.hidden)
        if self.antenna_projection is not None:
            antenna_coordinates = self.antenna_coordinates(
                device=features.device, dtype=features.dtype
            )
            queries = queries + self.antenna_projection(antenna_coordinates)[
                None, :, None, None
            ]
        return self.query_norm(queries).reshape(batch_size * ANTENNAS, 256, self.hidden)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_width).transpose(1, 2)

    def forward(
        self,
        features: torch.Tensor,
        observations: torch.Tensor,
        obs_ris_index: torch.Tensor,
        obs_time_index: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 5 or features.shape[1:] != (self.hidden, ANTENNAS, 16, 16):
            raise ValueError("features must have shape [B,H,64,16,16].")
        batch_size = features.shape[0]
        queries = self._dense_queries(features)
        observed = self._observed_tokens(observations, obs_ris_index, obs_time_index)
        query = self._heads(self.query_projection(queries))
        key = self._heads(self.key_projection(observed))
        value = self._heads(self.value_projection(observed))
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_width ** -0.5)
        attention = torch.softmax(scores, dim=-1)
        delta = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch_size * ANTENNAS, 256, self.hidden
        )
        delta = self.output_projection(delta).reshape(
            batch_size, ANTENNAS, 16, 16, self.hidden
        ).permute(0, 4, 1, 2, 3)
        return features + self.residual_scale * delta


def complex_factorized_reconstruction(
    bases: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """FP32 island for complex bases [B,R,N,M,2] and coefficients [B,Q,R,2]."""

    if bases.ndim != 5 or coefficients.ndim != 4 or bases.shape[-1] != 2 or coefficients.shape[-1] != 2:
        raise ValueError("Invalid complex basis/coefficient shapes.")
    if bases.shape[0] != coefficients.shape[0] or bases.shape[1] != coefficients.shape[2]:
        raise ValueError("Basis and coefficient batch/rank dimensions must match.")
    with torch.amp.autocast(device_type=bases.device.type, enabled=False):
        bases32, coefficients32 = bases.float(), coefficients.float()
        basis = torch.complex(bases32[..., 0], bases32[..., 1])
        coefficient = torch.complex(coefficients32[..., 0], coefficients32[..., 1])
        output = torch.einsum("bqr,brnm->bqnm", coefficient, basis)
        return torch.stack((output.real, output.imag), dim=-1)


class TrendConditionedTemporal(nn.Module):
    def __init__(self, hidden: int, rank: int, *, use_delta: bool = True) -> None:
        super().__init__()
        if rank not in {2, 3}:
            raise ValueError("PriST-RIS temporal rank must be 2 or 3.")
        self.rank = rank
        self.use_delta = use_delta
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(6, hidden, 3, padding=1),
            nn.GELU(),
            StrongSpatioRISResidualBlock(hidden),
            StrongSpatioRISResidualBlock(hidden),
        )
        self.basis_head = nn.Conv3d(hidden, 2 * rank, 1)
        self.anchor_context = nn.Linear(2, hidden)
        self.time_encoder = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.fusion = nn.Sequential(nn.Linear(4 * hidden, 2 * hidden), nn.GELU())
        self.coefficient_head = nn.Linear(2 * hidden, 2 * rank)
        self.alpha_head = nn.Linear(2 * hidden, 1)
        self.last_delta_norm: float | None = None
        self.last_complex_dtype: torch.dtype | None = None
        self.last_missing_time_index: tuple[int, ...] | None = None
        self.last_base_alpha: torch.Tensor | None = None

    def forward(
        self,
        anchors: torch.Tensor,
        query_time: torch.Tensor,
        anchor_time_index: tuple[int, int],
    ) -> torch.Tensor:
        if anchors.shape[1:] != (2, 256, ANTENNAS, 2):
            raise ValueError("Mobility temporal input must contain A0/A3 dual anchors.")
        query_values = tuple(int(value) for value in query_time.detach().cpu().tolist())
        missing_time_index = tuple(
            value for value in query_values if value not in anchor_time_index
        )
        if missing_time_index != (1, 2, 4, 5):
            raise ValueError("Mobility non-pilot queries must be q1,q2,q4,q5.")
        missing_time = torch.tensor(
            missing_time_index, device=anchors.device, dtype=anchors.dtype
        )
        a0, a3 = anchors[:, 0], anchors[:, 1]
        delta = a3 - a0 if self.use_delta else torch.zeros_like(a3)
        self.last_delta_norm = float(delta.detach().norm())
        self.last_missing_time_index = missing_time_index
        temporal_input = torch.cat((a0, a3, delta), dim=-1)
        temporal_grid = temporal_input.reshape(anchors.shape[0], 256, ANTENNAS, 6)
        temporal_grid = temporal_grid.reshape(anchors.shape[0], 16, 16, ANTENNAS, 6)
        temporal_grid = temporal_grid.permute(0, 4, 3, 1, 2).contiguous()
        features = self.spatial_encoder(temporal_grid)
        raw_bases = self.basis_head(features)
        bases = raw_bases.reshape(anchors.shape[0], self.rank, 2, ANTENNAS, 256)
        bases = bases.permute(0, 1, 4, 3, 2).contiguous()
        pooled = (a0.mean(dim=(1, 2)), a3.mean(dim=(1, 2)), delta.mean(dim=(1, 2)))
        contexts = [self.anchor_context(value) for value in pooled]
        base_alpha = normalized_time_coordinates(
            missing_time, anchor_time_index
        ).reshape(1, 4, 1, 1, 1)
        self.last_base_alpha = base_alpha.detach().cpu()
        time_context = self.time_encoder(base_alpha.reshape(-1, 1))
        fused = []
        for position in range(4):
            fused.append(
                self.fusion(
                    torch.cat(
                        (
                            contexts[0],
                            contexts[1],
                            contexts[2],
                            time_context[position].expand(anchors.shape[0], -1),
                        ),
                        dim=-1,
                    )
                )
            )
        fused_context = torch.stack(fused, dim=1)
        coefficients = self.coefficient_head(fused_context).reshape(
            anchors.shape[0], 4, self.rank, 2
        )
        residual = complex_factorized_reconstruction(bases, coefficients)
        self.last_complex_dtype = torch.complex64
        learned_alpha = 0.25 * torch.tanh(self.alpha_head(fused_context)).reshape(
            anchors.shape[0], 4, 1, 1, 1
        )
        return a0.unsqueeze(1) + (base_alpha + learned_alpha) * delta.unsqueeze(1) + residual


def normalized_time_coordinates(
    query_time: torch.Tensor, anchor_time_index: tuple[int, int]
) -> torch.Tensor:
    """Normalize semantic query times by the actual pilot spacing."""
    t0, t1 = anchor_time_index
    if t1 <= t0:
        raise ValueError("anchor_time_index must contain two increasing times.")
    return (query_time - float(t0)) / float(t1 - t0)


class FutureResidualCorrection(nn.Module):
    def __init__(self, hidden: int = 24) -> None:
        super().__init__()
        self.input = nn.Conv3d(2, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            StrongSpatioRISResidualBlock(hidden), StrongSpatioRISResidualBlock(hidden)
        )
        self.output = nn.Conv3d(hidden, 2, 3, padding=1)

    def forward(self, non_pilot: torch.Tensor) -> torch.Tensor:
        if non_pilot.shape[1:] != (4, 256, ANTENNAS, 2):
            raise ValueError("Temporal correction only accepts q1,q2,q4,q5.")
        b = non_pilot.shape[0]
        grid = non_pilot.reshape(b * 4, 16, 16, ANTENNAS, 2).permute(0, 4, 3, 1, 2)
        correction = self.output(self.blocks(F.gelu(self.input(grid))))
        correction = correction.permute(0, 3, 4, 2, 1).reshape(b, 4, 256, ANTENNAS, 2)
        return non_pilot + correction


@dataclass(frozen=True)
class PriSTRISConfig:
    model_key: str
    domain: str
    hidden: int = 80
    blocks_per_stage: tuple[int, int, int] = (3, 3, 4)
    final_refine_blocks: int = 4
    temporal_rank: int = 2
    temporal_residual: bool = True
    # Deprecated compatibility alias. New experiments must use the explicit
    # position flags below.
    coordinate_enabled: bool | None = None
    backbone_ris_coordinate_enabled: bool | None = None
    backbone_antenna_index_enabled: bool | None = None
    backbone_ris_coordinate_mode: str | None = None
    attention_enabled: bool | None = None
    attention_ris_coordinate_enabled: bool | None = None
    attention_antenna_index_enabled: bool | None = None
    observed_dense_attention_heads: int = OBSERVED_DENSE_ATTENTION_HEADS
    spatial_residual_style: str = "scaled_true_residual"
    temporal_mode: str = "trend"
    architecture_version: str = ARCHITECTURE_VERSION
    spatial_protocol_version: str = SPATIAL_PROTOCOL_VERSION
    position_semantics_version: str = POSITION_SEMANTICS_VERSION


class PriSTRIS(nn.Module):
    """Canonical PriST-RIS V3.2 stable prior-guided physical-grid model."""

    def __init__(self, config: PriSTRISConfig) -> None:
        super().__init__()
        key = canonical_model_key(config.model_key)
        if config.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError("PriST-RIS model config architecture version mismatch.")
        if config.domain not in {"quasi", "mobility"}:
            raise ValueError("domain must be quasi or mobility.")
        if config.spatial_protocol_version != SPATIAL_PROTOCOL_VERSION:
            raise ValueError("PriST-RIS spatial protocol version mismatch.")
        if config.position_semantics_version != POSITION_SEMANTICS_VERSION:
            raise ValueError("PriST-RIS position semantics version mismatch.")
        explicit_position = (
            config.backbone_ris_coordinate_enabled,
            config.backbone_antenna_index_enabled,
            config.backbone_ris_coordinate_mode,
            config.attention_ris_coordinate_enabled,
            config.attention_antenna_index_enabled,
        )
        if config.coordinate_enabled is not None and any(
            value is not None for value in explicit_position
        ):
            raise ValueError(
                "The legacy coordinate_enabled alias cannot be combined with explicit position flags."
            )
        legacy_coordinate_alias_used = config.coordinate_enabled is not None
        default_coupled = key in {"prist_ris_c", "prist_ris_full"}
        attention_enabled = (
            default_coupled
            if config.attention_enabled is None
            else config.attention_enabled
        )
        if legacy_coordinate_alias_used:
            backbone_ris_coordinate_enabled = bool(config.coordinate_enabled)
            backbone_antenna_index_enabled = bool(config.coordinate_enabled)
            attention_ris_coordinate_enabled = bool(config.coordinate_enabled) and attention_enabled
            attention_antenna_index_enabled = bool(config.coordinate_enabled) and attention_enabled
            backbone_ris_coordinate_mode = (
                "direct_add" if backbone_ris_coordinate_enabled else "off"
            )
        else:
            backbone_ris_coordinate_enabled = (
                default_coupled
                if config.backbone_ris_coordinate_enabled is None
                else config.backbone_ris_coordinate_enabled
            )
            backbone_antenna_index_enabled = (
                default_coupled
                if config.backbone_antenna_index_enabled is None
                else config.backbone_antenna_index_enabled
            )
            attention_default = default_coupled and attention_enabled
            attention_ris_coordinate_enabled = (
                attention_default
                if config.attention_ris_coordinate_enabled is None
                else config.attention_ris_coordinate_enabled
            )
            attention_antenna_index_enabled = (
                attention_default
                if config.attention_antenna_index_enabled is None
                else config.attention_antenna_index_enabled
            )
            backbone_ris_coordinate_mode = config.backbone_ris_coordinate_mode or (
                "direct_add" if backbone_ris_coordinate_enabled else "off"
            )
        if backbone_ris_coordinate_enabled != (
            backbone_ris_coordinate_mode != "off"
        ):
            raise ValueError(
                "backbone RIS coordinate enablement and mode must agree."
            )
        if not attention_enabled and (
            attention_ris_coordinate_enabled or attention_antenna_index_enabled
        ):
            raise ValueError(
                "Attention position features require attention_enabled=True."
            )
        self.config = PriSTRISConfig(
            **{
                **asdict(config),
                "model_key": key,
                # Persist a reload-safe resolved config. Alias provenance is
                # recorded separately in protocol/checkpoint metadata.
                "coordinate_enabled": None,
                "backbone_ris_coordinate_enabled": backbone_ris_coordinate_enabled,
                "backbone_antenna_index_enabled": backbone_antenna_index_enabled,
                "backbone_ris_coordinate_mode": backbone_ris_coordinate_mode,
                "attention_enabled": attention_enabled,
                "attention_ris_coordinate_enabled": attention_ris_coordinate_enabled,
                "attention_antenna_index_enabled": attention_antenna_index_enabled,
            }
        )
        self.legacy_coordinate_alias_used = legacy_coordinate_alias_used
        self.legacy_coordinate_alias_value = config.coordinate_enabled
        self.anchor_count = 1 if config.domain == "quasi" else 2
        semantics = DataSemantics.for_domain(config.domain)
        self.spatial_anchor_time_index = semantics.obs_time_index
        self.output_time_index = (
            semantics.query_time
            if key == "prist_ris_full" and config.domain == "mobility"
            else semantics.obs_time_index
        )
        self.uses_prior = key in {"prist_ris_b", "prist_ris_c", "prist_ris_full"}
        self.uses_observed_dense_attention = attention_enabled
        self.backbone = PhysicalGridBackbone(
            config.hidden,
            config.blocks_per_stage,
            config.final_refine_blocks,
            ris_coordinate_enabled=backbone_ris_coordinate_enabled,
            antenna_index_enabled=backbone_antenna_index_enabled,
            ris_coordinate_mode=backbone_ris_coordinate_mode,
            residual_style=config.spatial_residual_style,
        )
        self.prior_encoder = nn.Conv3d(2, config.hidden, 1) if self.uses_prior else None
        self.observed_dense_attention = (
            PhysicalObservedDenseResidualAttention(
                config.hidden,
                config.observed_dense_attention_heads,
                ris_coordinate_enabled=attention_ris_coordinate_enabled,
                antenna_index_enabled=attention_antenna_index_enabled,
            )
            if self.uses_observed_dense_attention
            else None
        )
        self.anchor_refiners = nn.ModuleList(
            StrongSpatioRISResidualBlock(
                config.hidden, config.spatial_residual_style
            )
            for _ in range(self.anchor_count)
        )
        self.anchor_heads = nn.ModuleList(
            nn.Conv3d(config.hidden, 2, 3, padding=1) for _ in range(self.anchor_count)
        )
        if self.uses_prior:
            for head in self.anchor_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        has_temporal = key == "prist_ris_full" and config.domain == "mobility"
        if config.temporal_mode not in {"trend", "no_delta", "static"}:
            raise ValueError("temporal_mode must be trend, no_delta, or static.")
        self.temporal = (
            TrendConditionedTemporal(
                config.hidden,
                config.temporal_rank,
                use_delta=config.temporal_mode != "no_delta",
            )
            if has_temporal and config.temporal_mode != "static"
            else None
        )
        self.temporal_correction = (
            FutureResidualCorrection()
            if self.temporal is not None and config.temporal_residual
            else None
        )
        self.last_stage_shapes: list[tuple[int, ...]] = []
        self.last_spatial_feature_scales: dict[str, float] = {}

    def spatial_anchors(
        self, batch: Mapping[str, torch.Tensor], prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        features, shapes = self.backbone(batch["obs_h"], batch["obs_ris_index"])
        self.last_stage_shapes = shapes
        scales = {
            "obs_input_rms": _tensor_rms(batch["obs_h"]),
            "backbone_output_rms": _tensor_rms(features),
        }
        if self.uses_prior:
            if prior is None:
                raise ValueError(f"{self.config.model_key} requires an explicit Ridge prior artifact.")
            if prior.shape[1] != self.anchor_count:
                raise ValueError(
                    f"{self.config.domain} V3.2 prior must contain {self.anchor_count} anchor(s)."
                )
        if self.observed_dense_attention is not None:
            features = self.observed_dense_attention(
                features,
                batch["obs_h"],
                batch["obs_ris_index"],
                batch["obs_time_index"],
            )
            scales["attention_output_rms"] = _tensor_rms(features)
        prior_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []
        refined_features: list[torch.Tensor] = []
        anchor_grids_list: list[torch.Tensor] = []
        for anchor_index, (refiner, head) in enumerate(
            zip(self.anchor_refiners, self.anchor_heads)
        ):
            fused = features
            if self.uses_prior and prior is not None:
                encoded = self.prior_encoder(  # type: ignore[operator]
                    prior_anchor_to_physical_grid(prior[:, anchor_index : anchor_index + 1])
                )
                prior_features.append(encoded)
                fused = fused + encoded
            refined = refiner(fused)
            fused_features.append(fused)
            refined_features.append(refined)
            anchor_grids_list.append(head(refined))
        anchor_grids = torch.cat(anchor_grids_list, dim=1)
        if self.anchor_count == 1:
            anchor_grids = F.pad(anchor_grids, (0, 0, 0, 0, 0, 0, 0, 2))
        delta = physical_grid_to_anchors(anchor_grids, self.anchor_count)
        if prior is not None:
            scales["prior_raw_rms"] = _tensor_rms(prior)
        if prior_features:
            scales["prior_encoder_rms"] = _tensor_rms(torch.stack(prior_features, dim=1))
        scales["fused_feature_rms"] = _tensor_rms(torch.stack(fused_features, dim=1))
        scales["refined_feature_rms"] = _tensor_rms(torch.stack(refined_features, dim=1))
        scales["delta_rms"] = _tensor_rms(delta)
        if scales["backbone_output_rms"] > 0 and "prior_encoder_rms" in scales:
            scales["prior_to_backbone_rms_ratio"] = (
                scales["prior_encoder_rms"] / scales["backbone_output_rms"]
            )
        self.last_spatial_feature_scales = scales
        return delta + prior if self.uses_prior and prior is not None else delta

    def forward(
        self, batch: Mapping[str, torch.Tensor], prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        anchors = self.spatial_anchors(batch, prior)
        if self.config.domain == "quasi" or self.config.model_key != "prist_ris_full":
            return anchors
        query_time = batch["query_time"][0] if batch["query_time"].ndim > 1 else batch["query_time"]
        if self.config.temporal_mode == "static":
            non_pilot = anchors[:, 1:2].expand(-1, 4, -1, -1, -1)
        else:
            non_pilot = self.temporal(  # type: ignore[operator]
                anchors,
                query_time,
                tuple(self.spatial_anchor_time_index),
            )
        if self.temporal_correction is not None:
            non_pilot = self.temporal_correction(non_pilot)
        query_values = tuple(int(value) for value in query_time.detach().cpu().tolist())
        non_pilot_time_index = tuple(
            value for value in query_values if value not in self.spatial_anchor_time_index
        )
        positions = {value: position for position, value in enumerate(query_values)}
        output = anchors.new_empty(
            anchors.shape[0], len(query_values), *anchors.shape[2:]
        )
        for anchor_position, semantic_time in enumerate(self.spatial_anchor_time_index):
            output[:, positions[semantic_time]] = anchors[:, anchor_position]
        for compact_position, semantic_time in enumerate(non_pilot_time_index):
            output[:, positions[semantic_time]] = non_pilot[:, compact_position]
        return output

    def protocol_metadata(self) -> dict[str, object]:
        return {
            "method": MODEL_DISPLAY_NAME,
            "architecture_version": ARCHITECTURE_VERSION,
            "mobility_contract_version": (
                MOBILITY_CONTRACT_VERSION
                if self.config.domain == "mobility"
                else None
            ),
            "canonical_model_key": self.config.model_key,
            "physical_grid": True,
            "physical_ris_shapes": ["16x2", "16x4", "16x8", "16x16"],
            "strong_spatio_ris_conv3d": True,
            "prior_guided": self.uses_prior,
            "prior_anchors": self.anchor_count if self.uses_prior else 0,
            "spatial_anchor_time_index": list(self.spatial_anchor_time_index),
            "output_time_index": list(self.output_time_index),
            "coordinate_enabled": self.legacy_coordinate_alias_value,
            "legacy_coordinate_alias_used": self.legacy_coordinate_alias_used,
            "backbone_ris_coordinate_enabled": self.config.backbone_ris_coordinate_enabled,
            "backbone_antenna_index_enabled": self.config.backbone_antenna_index_enabled,
            "backbone_ris_coordinate_mode": self.config.backbone_ris_coordinate_mode,
            "attention_enabled": self.config.attention_enabled,
            "attention_ris_coordinate_enabled": self.config.attention_ris_coordinate_enabled,
            "attention_antenna_index_enabled": self.config.attention_antenna_index_enabled,
            "position_semantics_version": POSITION_SEMANTICS_VERSION,
            "antenna_encoding_semantics": "antenna_index_encoding",
            "spatial_residual_style": self.config.spatial_residual_style,
            "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
            "deterministic_physical_upsampling": "nearest_column_only",
            "per_anchor_prior_fusion": self.uses_prior,
            "prior_encoder_input_channels": 2 if self.uses_prior else 0,
            "zero_initialized_delta_heads": self.uses_prior,
            "observed_dense_attention": self.uses_observed_dense_attention,
            "observed_dense_attention_layers": 1 if self.uses_observed_dense_attention else 0,
            "observed_dense_attention_heads": (
                self.config.observed_dense_attention_heads
                if self.uses_observed_dense_attention
                else 0
            ),
            "observed_dense_attention_scope": (
                "per_antenna_32_to_256" if self.uses_observed_dense_attention else None
            ),
            "observed_dense_attention_residual_scale": (
                OBSERVED_DENSE_ATTENTION_RESIDUAL_SCALE
                if self.uses_observed_dense_attention
                else None
            ),
            "observed_dense_attention_uses_target": False,
            "temporal_mode": (
                self.config.temporal_mode
                if self.config.domain == "mobility" and self.config.model_key == "prist_ris_full"
                else None
            ),
            "temporal_rank": self.config.temporal_rank if self.temporal is not None else None,
            "temporal_prediction_scope": (
                "non_pilot_q1_q2_q4_q5" if self.temporal is not None else None
            ),
            "target_h_forward_inputs": False,
            "observation_mask_used": False,
        }


def build_model(
    model_key: str,
    *,
    domain: str,
    hidden: int = 80,
    blocks_per_stage: tuple[int, int, int] = (3, 3, 4),
    final_refine_blocks: int = 4,
    temporal_rank: int = 2,
    temporal_residual: bool = True,
    coordinate_enabled: bool | None = None,
    backbone_ris_coordinate_enabled: bool | None = None,
    backbone_antenna_index_enabled: bool | None = None,
    backbone_ris_coordinate_mode: str | None = None,
    attention_enabled: bool | None = None,
    attention_ris_coordinate_enabled: bool | None = None,
    attention_antenna_index_enabled: bool | None = None,
    observed_dense_attention_heads: int = OBSERVED_DENSE_ATTENTION_HEADS,
    spatial_residual_style: str = "scaled_true_residual",
    temporal_mode: str = "trend",
    architecture_version: str = ARCHITECTURE_VERSION,
    spatial_protocol_version: str = SPATIAL_PROTOCOL_VERSION,
    position_semantics_version: str = POSITION_SEMANTICS_VERSION,
) -> PriSTRIS:
    return PriSTRIS(
        PriSTRISConfig(
            model_key=canonical_model_key(model_key),
            domain=domain,
            hidden=hidden,
            blocks_per_stage=blocks_per_stage,
            final_refine_blocks=final_refine_blocks,
            temporal_rank=temporal_rank,
            temporal_residual=temporal_residual,
            coordinate_enabled=coordinate_enabled,
            backbone_ris_coordinate_enabled=backbone_ris_coordinate_enabled,
            backbone_antenna_index_enabled=backbone_antenna_index_enabled,
            backbone_ris_coordinate_mode=backbone_ris_coordinate_mode,
            attention_enabled=attention_enabled,
            attention_ris_coordinate_enabled=attention_ris_coordinate_enabled,
            attention_antenna_index_enabled=attention_antenna_index_enabled,
            observed_dense_attention_heads=observed_dense_attention_heads,
            spatial_residual_style=spatial_residual_style,
            temporal_mode=temporal_mode,
            architecture_version=architecture_version,
            spatial_protocol_version=spatial_protocol_version,
            position_semantics_version=position_semantics_version,
        )
    )


def canonical_batch(
    domain: str, batch_size: int = 1, device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    device = device or torch.device("cpu")
    semantics = DataSemantics.for_domain(domain)
    times, queries = len(semantics.obs_time_index), len(semantics.query_time)
    return {
        "obs_h": torch.zeros(batch_size, times, 32, ANTENNAS, 2, device=device),
        "target_h": torch.zeros(batch_size, queries, 256, ANTENNAS, 2, device=device),
        "obs_ris_index": torch.tensor(OBSERVED_RIS_INDICES, device=device).expand(batch_size, -1),
        "obs_time_index": torch.tensor(
            semantics.obs_time_index, device=device
        ).expand(batch_size, -1),
        "query_time": torch.tensor(
            semantics.query_time, device=device
        ).expand(batch_size, -1),
        "sample_index": torch.arange(batch_size, device=device),
    }
