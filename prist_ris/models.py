from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (
    ANTENNAS,
    ARCHITECTURE_VERSION,
    MODEL_DISPLAY_NAME,
    OBSERVED_RIS_INDICES,
    canonical_model_key,
)


PHYSICAL_STAGE_COLUMNS: dict[int, tuple[int, ...]] = {
    2: (0, 8),
    4: (0, 4, 8, 12),
    8: tuple(range(0, 16, 2)),
    16: tuple(range(16)),
}


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
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.body(value))


class PhysicalColumnUpsample(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            hidden, hidden, kernel_size=(1, 1, 2), stride=(1, 1, 2)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        before = value.shape
        result = self.upsample(value)
        if result.shape[2:4] != before[2:4] or result.shape[-1] != 2 * before[-1]:
            raise RuntimeError("Physical column upsampling changed a non-column axis.")
        return result


class PhysicalProgressiveStage(nn.Module):
    def __init__(self, hidden: int, blocks: int) -> None:
        super().__init__()
        self.upsample = PhysicalColumnUpsample(hidden)
        self.blocks = nn.Sequential(*(StrongSpatioRISResidualBlock(hidden) for _ in range(blocks)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.upsample(value))


class PhysicalGridBackbone(nn.Module):
    def __init__(
        self,
        hidden: int = 80,
        blocks_per_stage: tuple[int, int, int] = (3, 3, 4),
        final_refine_blocks: int = 4,
        *,
        coordinate_enabled: bool,
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 3 or final_refine_blocks < 1:
            raise ValueError("V3.1 requires three progressive stages and final refinement.")
        self.hidden = hidden
        self.coordinate_enabled = coordinate_enabled
        self.input = nn.Conv3d(4, hidden, 3, padding=1)
        self.ris_coordinate_encoder = RISCoordinateEncoder(hidden) if coordinate_enabled else None
        self.antenna_index_encoder = AntennaIndexEncoder(hidden) if coordinate_enabled else None
        self.stages = nn.ModuleList(
            PhysicalProgressiveStage(hidden, blocks) for blocks in blocks_per_stage
        )
        self.final_refine = nn.Sequential(
            *(StrongSpatioRISResidualBlock(hidden) for _ in range(final_refine_blocks))
        )

    def _coordinates(self, value: torch.Tensor) -> torch.Tensor:
        if self.ris_coordinate_encoder is None or self.antenna_index_encoder is None:
            return torch.zeros_like(value)
        return self.ris_coordinate_encoder(value) + self.antenna_index_encoder(value)

    def forward(
        self, observations: torch.Tensor, obs_ris_index: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        grid = observations_to_physical_grid(observations, obs_ris_index)
        value = self.input(grid)
        if self.coordinate_enabled:
            value = value + self._coordinates(value)
        shapes = [tuple(value.shape)]
        for stage in self.stages:
            value = stage(value)
            if self.coordinate_enabled:
                value = value + self._coordinates(value)
            shapes.append(tuple(value.shape))
        value = self.final_refine(value)
        if value.shape[2:] != (ANTENNAS, 16, 16):
            raise RuntimeError(f"Physical backbone ended at {value.shape[2:]}, expected (64,16,16).")
        return value, shapes


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

    def forward(self, anchors: torch.Tensor, query_time: torch.Tensor) -> torch.Tensor:
        if anchors.shape[1:] != (2, 256, ANTENNAS, 2):
            raise ValueError("Mobility temporal input must contain A0/A1 dual anchors.")
        future_time = query_time[2:]
        if future_time.numel() != 4:
            raise ValueError("Mobility future queries must be q2..q5.")
        a0, a1 = anchors[:, 0], anchors[:, 1]
        delta = a1 - a0 if self.use_delta else torch.zeros_like(a1)
        self.last_delta_norm = float(delta.detach().norm())
        temporal_input = torch.cat((a0, a1, delta), dim=-1)
        temporal_grid = temporal_input.reshape(anchors.shape[0], 256, ANTENNAS, 6)
        temporal_grid = temporal_grid.reshape(anchors.shape[0], 16, 16, ANTENNAS, 6)
        temporal_grid = temporal_grid.permute(0, 4, 3, 1, 2).contiguous()
        features = self.spatial_encoder(temporal_grid)
        raw_bases = self.basis_head(features)
        bases = raw_bases.reshape(anchors.shape[0], self.rank, 2, ANTENNAS, 256)
        bases = bases.permute(0, 1, 4, 3, 2).contiguous()
        pooled = (a0.mean(dim=(1, 2)), a1.mean(dim=(1, 2)), delta.mean(dim=(1, 2)))
        contexts = [self.anchor_context(value) for value in pooled]
        time = future_time.to(anchors).reshape(-1, 1) / 5.0
        time_context = self.time_encoder(time)
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
        base_alpha = (future_time.to(anchors) - 1.0).reshape(1, 4, 1, 1, 1)
        learned_alpha = 0.25 * torch.tanh(self.alpha_head(fused_context)).reshape(
            anchors.shape[0], 4, 1, 1, 1
        )
        return a1.unsqueeze(1) + (base_alpha + learned_alpha) * delta.unsqueeze(1) + residual


class FutureResidualCorrection(nn.Module):
    def __init__(self, hidden: int = 24) -> None:
        super().__init__()
        self.input = nn.Conv3d(2, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            StrongSpatioRISResidualBlock(hidden), StrongSpatioRISResidualBlock(hidden)
        )
        self.output = nn.Conv3d(hidden, 2, 3, padding=1)

    def forward(self, future: torch.Tensor) -> torch.Tensor:
        if future.shape[1:] != (4, 256, ANTENNAS, 2):
            raise ValueError("Temporal correction only accepts q2..q5.")
        b = future.shape[0]
        grid = future.reshape(b * 4, 16, 16, ANTENNAS, 2).permute(0, 4, 3, 1, 2)
        correction = self.output(self.blocks(F.gelu(self.input(grid))))
        correction = correction.permute(0, 3, 4, 2, 1).reshape(b, 4, 256, ANTENNAS, 2)
        return future + correction


@dataclass(frozen=True)
class PriSTRISConfig:
    model_key: str
    domain: str
    hidden: int = 80
    blocks_per_stage: tuple[int, int, int] = (3, 3, 4)
    final_refine_blocks: int = 4
    temporal_rank: int = 2
    temporal_residual: bool = True
    coordinate_enabled: bool | None = None
    temporal_mode: str = "trend"
    architecture_version: str = ARCHITECTURE_VERSION


class PriSTRIS(nn.Module):
    """Canonical PriST-RIS V3.1 physical-grid implementation."""

    def __init__(self, config: PriSTRISConfig) -> None:
        super().__init__()
        key = canonical_model_key(config.model_key)
        if config.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError("PriST-RIS model config architecture version mismatch.")
        if config.domain not in {"quasi", "mobility"}:
            raise ValueError("domain must be quasi or mobility.")
        coordinate_enabled = (
            key in {"prist_ris_c", "prist_ris_full"}
            if config.coordinate_enabled is None
            else config.coordinate_enabled
        )
        self.config = PriSTRISConfig(
            **{**asdict(config), "model_key": key, "coordinate_enabled": coordinate_enabled}
        )
        self.anchor_count = 1 if config.domain == "quasi" else 2
        self.uses_prior = key in {"prist_ris_b", "prist_ris_c", "prist_ris_full"}
        self.backbone = PhysicalGridBackbone(
            config.hidden,
            config.blocks_per_stage,
            config.final_refine_blocks,
            coordinate_enabled=coordinate_enabled,
        )
        self.prior_encoder = nn.Conv3d(4, config.hidden, 1) if self.uses_prior else None
        self.anchor_feature = nn.Sequential(
            nn.Conv3d(config.hidden, config.hidden, 3, padding=1), nn.GELU()
        )
        self.anchor_heads = nn.ModuleList(
            nn.Conv3d(config.hidden, 2, 3, padding=1) for _ in range(self.anchor_count)
        )
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

    def spatial_anchors(
        self, batch: Mapping[str, torch.Tensor], prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        features, shapes = self.backbone(batch["obs_h"], batch["obs_ris_index"])
        self.last_stage_shapes = shapes
        if self.uses_prior:
            if prior is None:
                raise ValueError(f"{self.config.model_key} requires an explicit Ridge prior artifact.")
            if prior.shape[1] != self.anchor_count:
                raise ValueError(
                    f"{self.config.domain} V3.1 prior must contain {self.anchor_count} anchor(s)."
                )
            features = features + self.prior_encoder(prior_to_physical_grid(prior))  # type: ignore[operator]
        anchor_features = self.anchor_feature(features)
        anchor_grids = torch.cat([head(anchor_features) for head in self.anchor_heads], dim=1)
        if self.anchor_count == 1:
            anchor_grids = F.pad(anchor_grids, (0, 0, 0, 0, 0, 0, 0, 2))
        delta = physical_grid_to_anchors(anchor_grids, self.anchor_count)
        return delta + prior if self.uses_prior and prior is not None else delta

    def forward(
        self, batch: Mapping[str, torch.Tensor], prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        anchors = self.spatial_anchors(batch, prior)
        if self.config.domain == "quasi" or self.config.model_key != "prist_ris_full":
            return anchors
        if self.config.temporal_mode == "static":
            future = anchors[:, 1:2].expand(-1, 4, -1, -1, -1)
        else:
            query_time = batch["query_time"][0] if batch["query_time"].ndim > 1 else batch["query_time"]
            future = self.temporal(anchors, query_time)  # type: ignore[operator]
        if self.temporal_correction is not None:
            future = self.temporal_correction(future)
        return torch.cat((anchors, future), dim=1)

    def protocol_metadata(self) -> dict[str, object]:
        return {
            "method": MODEL_DISPLAY_NAME,
            "architecture_version": ARCHITECTURE_VERSION,
            "canonical_model_key": self.config.model_key,
            "physical_grid": True,
            "physical_ris_shapes": ["16x2", "16x4", "16x8", "16x16"],
            "strong_spatio_ris_conv3d": True,
            "prior_guided": self.uses_prior,
            "prior_anchors": self.anchor_count if self.uses_prior else 0,
            "coordinate_enabled": self.config.coordinate_enabled,
            "antenna_encoding_semantics": "antenna_index_encoding",
            "cross_attention_layers": 0,
            "temporal_mode": (
                self.config.temporal_mode
                if self.config.domain == "mobility" and self.config.model_key == "prist_ris_full"
                else None
            ),
            "temporal_rank": self.config.temporal_rank if self.temporal is not None else None,
            "future_target_inputs": False,
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
    temporal_mode: str = "trend",
    architecture_version: str = ARCHITECTURE_VERSION,
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
            temporal_mode=temporal_mode,
            architecture_version=architecture_version,
        )
    )


def canonical_batch(
    domain: str, batch_size: int = 1, device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    device = device or torch.device("cpu")
    times, queries = ((1, 1) if domain == "quasi" else (2, 6))
    return {
        "obs_h": torch.zeros(batch_size, times, 32, ANTENNAS, 2, device=device),
        "target_h": torch.zeros(batch_size, queries, 256, ANTENNAS, 2, device=device),
        "obs_ris_index": torch.tensor(OBSERVED_RIS_INDICES, device=device).expand(batch_size, -1),
        "obs_time_index": torch.arange(times, device=device).expand(batch_size, -1),
        "query_time": torch.arange(queries, device=device).expand(batch_size, -1),
        "sample_index": torch.arange(batch_size, device=device),
    }
