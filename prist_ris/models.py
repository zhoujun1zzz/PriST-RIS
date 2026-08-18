from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import MODEL_DISPLAY_NAME, canonical_model_key


def observations_to_image(observations: torch.Tensor) -> torch.Tensor:
    """[B,T,32,64,2] -> [B,2T,64,32] with grouped real/imag channels."""

    if observations.ndim != 5 or observations.shape[2:] != (32, 64, 2):
        raise ValueError("observations must have shape [B,T,32,64,2].")
    b, t, _, antennas, _ = observations.shape
    image = observations.permute(0, 1, 4, 3, 2).reshape(b, 2 * t, antennas, 32)
    if image.shape[1] < 4:
        image = F.pad(image, (0, 0, 0, 0, 0, 4 - image.shape[1]))
    return image


def channel_to_image(channel: torch.Tensor) -> torch.Tensor:
    """[B,1,256,64,2] -> [B,2,64,256]."""

    if channel.ndim != 5 or channel.shape[1:] != (1, 256, 64, 2):
        raise ValueError("spatial prior must have shape [B,1,256,64,2].")
    return channel[:, 0].permute(0, 3, 2, 1).contiguous()


def image_to_anchor(image: torch.Tensor) -> torch.Tensor:
    """[B,2,64,256] -> [B,1,256,64,2]."""

    if image.ndim != 4 or image.shape[1:] != (2, 64, 256):
        raise ValueError("anchor image must have shape [B,2,64,256].")
    return image.permute(0, 3, 2, 1).unsqueeze(1).contiguous()


class FactorizedAntennaRISBlock(nn.Module):
    """Separate local RIS (1x3) and antenna (3x1) modeling with channel mixing."""

    def __init__(self, hidden: int, *, antenna_branch: bool = True) -> None:
        super().__init__()
        self.antenna_branch_enabled = antenna_branch
        self.ris_depthwise = nn.Conv2d(
            hidden, hidden, (1, 3), padding=(0, 1), groups=hidden
        )
        self.ris_mix = nn.Sequential(
            nn.Conv2d(hidden, 2 * hidden, 1),
            nn.GELU(),
            nn.Conv2d(2 * hidden, hidden, 1),
        )
        if antenna_branch:
            self.antenna_depthwise: nn.Module | None = nn.Conv2d(
                hidden, hidden, (3, 1), padding=(1, 0), groups=hidden
            )
            self.antenna_mix: nn.Module | None = nn.Sequential(
                nn.Conv2d(hidden, 2 * hidden, 1),
                nn.GELU(),
                nn.Conv2d(2 * hidden, hidden, 1),
            )
        else:
            self.antenna_depthwise = None
            self.antenna_mix = None
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.ris_mix(self.ris_depthwise(value))
        if self.antenna_depthwise is not None and self.antenna_mix is not None:
            residual = residual + self.antenna_mix(self.antenna_depthwise(value))
        return value + 0.1 * self.activation(residual)


class ProgressiveStage(nn.Module):
    def __init__(self, hidden: int, blocks: int, *, antenna_branch: bool) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                FactorizedAntennaRISBlock(hidden, antenna_branch=antenna_branch)
                for _ in range(blocks)
            )
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.blocks(value)
        value = F.interpolate(value, scale_factor=(1, 2), mode="nearest")
        return self.refine(value)


class StructuredProgressiveBackbone(nn.Module):
    def __init__(
        self,
        hidden: int = 80,
        blocks_per_stage: tuple[int, int, int] = (2, 2, 3),
        *,
        antenna_branch: bool = True,
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 3:
            raise ValueError("PriST-RIS requires exactly three progressive stages.")
        self.hidden = hidden
        self.input = nn.Conv2d(4, hidden, 3, padding=1)
        self.stages = nn.ModuleList(
            ProgressiveStage(hidden, blocks, antenna_branch=antenna_branch)
            for blocks in blocks_per_stage
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        value = self.input(observations_to_image(observations))
        shapes = [tuple(value.shape)]
        for stage in self.stages:
            value = stage(value)
            shapes.append(tuple(value.shape))
        if value.shape[-2:] != (64, 256):
            raise RuntimeError(f"Progressive backbone ended at {value.shape[-2:]}, expected (64,256).")
        return value, shapes


class ResidualObservedCrossAttention(nn.Module):
    """One observed-to-dense residual attention layer; K/V never use targets."""

    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by attention heads.")
        self.observed_encoder = nn.Linear(4, hidden)
        self.coordinate_encoder = nn.Linear(2, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        columns = torch.linspace(-1, 1, 256)
        rows = torch.linspace(-1, 1, 64)
        self.register_buffer(
            "dense_coordinates",
            torch.stack(torch.meshgrid(rows, columns, indexing="ij"), dim=-1),
            persistent=False,
        )
        self.last_query_tokens = 0
        self.last_key_tokens = 0

    def forward(self, dense: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        b, hidden, antennas, nodes = dense.shape
        obs = observed.permute(0, 3, 2, 1, 4).reshape(b * antennas, 32, -1)
        if obs.shape[-1] < 4:
            obs = F.pad(obs, (0, 4 - obs.shape[-1]))
        memory = self.observed_encoder(obs)
        query = dense.permute(0, 2, 3, 1).reshape(b * antennas, nodes, hidden)
        coordinates = self.dense_coordinates.to(dense)
        coordinate_tokens = self.coordinate_encoder(coordinates).reshape(antennas, nodes, hidden)
        query = query + coordinate_tokens.repeat(b, 1, 1)
        delta, _ = self.attention(query, memory, memory, need_weights=False)
        self.last_query_tokens = int(query.shape[1])
        self.last_key_tokens = int(memory.shape[1])
        return (query + delta).reshape(b, antennas, nodes, hidden).permute(0, 3, 1, 2)


def complex_factorized_reconstruction(
    bases: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """Multiply complex bases [B,R,N,M,2] by coefficients [B,Q,R,2]."""

    if bases.ndim != 5 or coefficients.ndim != 4 or bases.shape[-1] != 2 or coefficients.shape[-1] != 2:
        raise ValueError("Invalid complex basis/coefficient shapes.")
    if bases.shape[0] != coefficients.shape[0] or bases.shape[1] != coefficients.shape[2]:
        raise ValueError("Basis and coefficient batch/rank dimensions must match.")
    basis = torch.complex(bases[..., 0], bases[..., 1])
    coeff = torch.complex(coefficients[..., 0], coefficients[..., 1])
    output = torch.einsum("bqr,brnm->bqnm", coeff, basis)
    return torch.stack((output.real, output.imag), dim=-1)


class LowRankTemporalFactorization(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        if rank not in {2, 3}:
            raise ValueError("PriST-RIS temporal rank must be 2 or 3.")
        self.rank = rank
        self.basis_head = nn.Conv2d(hidden, 2 * rank, 1)
        self.observed_context = nn.Linear(2, hidden)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.future_fusion = nn.Linear(2 * hidden, hidden)
        self.coefficient_head = nn.Linear(hidden, 2 * rank)

    def aligned_query_context(
        self,
        observations: torch.Tensor,
        obs_time: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        b, observed_count = observations.shape[:2]
        if observed_count != obs_time.numel():
            raise ValueError("Observed tensor/time count mismatch.")
        pooled = observations.mean(dim=(2, 3))
        contexts = self.observed_context(pooled)
        scale = max(1, int(query_time.max().item()))
        time = self.time_encoder(query_time.to(observations).reshape(-1, 1) / scale)
        pooled_context = contexts.mean(dim=1)
        outputs = []
        for position, query in enumerate(query_time):
            matches = torch.where(obs_time == query)[0]
            if matches.numel():
                outputs.append(contexts[:, int(matches[0])])
            else:
                outputs.append(
                    self.future_fusion(
                        torch.cat((pooled_context, time[position].expand(b, -1)), dim=-1)
                    )
                )
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        observations: torch.Tensor,
        obs_time: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        b, _, antennas, nodes = features.shape
        raw_bases = self.basis_head(features).reshape(b, self.rank, 2, antennas, nodes)
        bases = raw_bases.permute(0, 1, 4, 3, 2).contiguous()
        context = self.aligned_query_context(observations, obs_time, query_time)
        coefficients = self.coefficient_head(context).reshape(b, query_time.numel(), self.rank, 2)
        return complex_factorized_reconstruction(bases, coefficients)


class TemporalResidualCorrection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(2, 2, 3, padding=1, groups=2)
        self.pointwise = nn.Conv2d(2, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        b, q, nodes, antennas, _ = value.shape
        image = value.permute(0, 1, 4, 3, 2).reshape(b * q, 2, antennas, nodes)
        correction = self.pointwise(F.gelu(self.depthwise(image)))
        correction = correction.reshape(b, q, 2, antennas, nodes).permute(0, 1, 4, 3, 2)
        return value + 0.1 * correction


@dataclass(frozen=True)
class PriSTRISConfig:
    model_key: str
    domain: str
    hidden: int = 80
    blocks_per_stage: tuple[int, int, int] = (2, 2, 3)
    heads: int = 4
    dropout: float = 0.0
    temporal_rank: int = 2
    antenna_branch: bool = True
    temporal_residual: bool = True


class PriSTRIS(nn.Module):
    """Canonical PriST-RIS implementation for controlled A/B/C/Full variants."""

    def __init__(self, config: PriSTRISConfig) -> None:
        super().__init__()
        key = canonical_model_key(config.model_key)
        if config.dropout != 0:
            raise ValueError("The initial PriST-RIS protocol fixes dropout=0.")
        self.config = PriSTRISConfig(**{**asdict(config), "model_key": key})
        obs_blocks = 1 if config.domain == "quasi" else 2
        query_blocks = 1 if config.domain == "quasi" else 6
        self.obs_blocks = obs_blocks
        self.query_blocks = query_blocks
        self.backbone = StructuredProgressiveBackbone(
            config.hidden,
            config.blocks_per_stage,
            antenna_branch=config.antenna_branch,
        )
        self.uses_prior = key in {"prist_ris_b", "prist_ris_c", "prist_ris_full"}
        self.prior_encoder = nn.Conv2d(2, config.hidden, 1) if self.uses_prior else None
        self.cross_attention = (
            ResidualObservedCrossAttention(config.hidden, config.heads)
            if key in {"prist_ris_c", "prist_ris_full"}
            else None
        )
        self.anchor_head = nn.Conv2d(config.hidden, 2, 3, padding=1)
        self.temporal = (
            LowRankTemporalFactorization(config.hidden, config.temporal_rank)
            if key == "prist_ris_full"
            else None
        )
        self.temporal_correction = (
            TemporalResidualCorrection()
            if self.temporal is not None and config.temporal_residual
            else None
        )

    def protocol_metadata(self) -> dict[str, object]:
        return {
            "method": MODEL_DISPLAY_NAME,
            "canonical_model_key": self.config.model_key,
            "prior_guided": self.uses_prior,
            "progressive_ris_widths": [32, 64, 128, 256],
            "factorized_antenna_ris": self.config.antenna_branch,
            "cross_attention_layers": 1 if self.cross_attention is not None else 0,
            "attention_heads": self.config.heads,
            "temporal_rank": self.config.temporal_rank if self.temporal is not None else None,
            "future_target_inputs": False,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observations = batch["obs_h"]
        features, _ = self.backbone(observations)
        if self.uses_prior:
            if prior is None:
                raise ValueError(f"{self.config.model_key} requires an explicit Ridge prior artifact.")
            features = features + self.prior_encoder(channel_to_image(prior))  # type: ignore[operator]
        if self.cross_attention is not None:
            features = self.cross_attention(features, observations)
        delta_anchor = image_to_anchor(self.anchor_head(features))
        anchor = delta_anchor + prior if prior is not None and self.uses_prior else delta_anchor
        if self.temporal is None:
            return anchor.expand(-1, self.query_blocks, -1, -1, -1).contiguous()
        obs_time = batch["obs_time_index"][0] if batch["obs_time_index"].ndim > 1 else batch["obs_time_index"]
        query_time = batch["query_time"][0] if batch["query_time"].ndim > 1 else batch["query_time"]
        factorized = self.temporal(features, observations, obs_time, query_time)
        # The learned dense spatial anchor is the reference surface for every
        # queried block; the low-rank temporal factors model only its residual.
        output = anchor.expand(-1, self.query_blocks, -1, -1, -1) + factorized
        if self.temporal_correction is not None:
            output = self.temporal_correction(output)
        return output


def build_model(
    model_key: str,
    *,
    domain: str,
    hidden: int = 80,
    blocks_per_stage: tuple[int, int, int] = (2, 2, 3),
    heads: int = 4,
    dropout: float = 0.0,
    temporal_rank: int = 2,
    antenna_branch: bool = True,
    temporal_residual: bool = True,
) -> PriSTRIS:
    return PriSTRIS(
        PriSTRISConfig(
            model_key=canonical_model_key(model_key),
            domain=domain,
            hidden=hidden,
            blocks_per_stage=blocks_per_stage,
            heads=heads,
            dropout=dropout,
            temporal_rank=temporal_rank,
            antenna_branch=antenna_branch,
            temporal_residual=temporal_residual,
        )
    )


def canonical_batch(domain: str, batch_size: int = 1, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    obs_blocks, query_blocks = ((1, 1) if domain == "quasi" else (2, 6))
    return {
        "obs_h": torch.randn(batch_size, obs_blocks, 32, 64, 2, device=device),
        "target_h": torch.randn(batch_size, query_blocks, 256, 64, 2, device=device),
        "obs_ris_index": torch.arange(0, 256, 8, device=device).expand(batch_size, -1),
        "obs_time_index": torch.arange(obs_blocks, device=device).expand(batch_size, -1),
        "query_time": torch.arange(query_blocks, device=device).expand(batch_size, -1),
        "observation_mask": torch.ones(batch_size, obs_blocks, 32, dtype=torch.bool, device=device),
        "sample_index": torch.arange(batch_size, device=device),
    }
