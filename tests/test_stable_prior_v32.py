from __future__ import annotations

import torch
from torch.nn import functional as F

from prist_ris.diagnostics import spatial_feature_scale_report
from prist_ris.models import (
    PhysicalColumnUpsample,
    StrongSpatioRISResidualBlock,
    build_model,
    canonical_batch,
    observations_to_physical_grid,
    physical_grid_to_anchors,
)


SMALL = {"hidden": 4, "blocks_per_stage": (1, 1, 1), "final_refine_blocks": 1}


def _mobility_problem(batch_size: int = 2) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(41)
    batch = canonical_batch("mobility", batch_size=batch_size)
    batch["obs_h"].normal_()
    prior = 0.25 * torch.randn(batch_size, 2, 256, 64, 2)
    observed_grid = observations_to_physical_grid(
        batch["obs_h"], batch["obs_ris_index"]
    )
    dense_observation = F.interpolate(
        observed_grid, scale_factor=(1, 1, 8), mode="nearest"
    )
    residual = 0.1 * physical_grid_to_anchors(dense_observation, anchors=2)
    return batch, prior, prior + residual


def _gradient_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    ) ** 0.5


def test_physical_column_upsample_is_deterministic_nearest() -> None:
    module = PhysicalColumnUpsample(2)
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    value = torch.arange(4.0).reshape(1, 2, 1, 1, 2)
    expected = F.interpolate(value, scale_factor=(1, 1, 2), mode="nearest")
    torch.testing.assert_close(module(value), expected, rtol=0, atol=0)


def test_v32_default_spatial_block_is_true_residual() -> None:
    block = StrongSpatioRISResidualBlock(2)
    assert block.residual_style == "scaled_true_residual"
    with torch.no_grad():
        for parameter in block.body.parameters():
            parameter.zero_()
    value = torch.randn(1, 2, 3, 4, 5)
    torch.testing.assert_close(block(value), value, rtol=0, atol=0)


def test_b_initialization_is_exactly_the_ridge_prior() -> None:
    model = build_model("prist_ris_b", domain="mobility", **SMALL)
    batch, prior, _ = _mobility_problem()
    prediction = model(batch, prior)
    torch.testing.assert_close(prediction, prior, rtol=0, atol=0)


def test_zero_init_head_opens_upstream_gradients_after_first_update() -> None:
    model = build_model("prist_ris_b", domain="mobility", **SMALL)
    batch, prior, target = _mobility_problem()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    optimizer.zero_grad(set_to_none=True)
    first = F.mse_loss(model(batch, prior), target)
    first.backward()
    assert _gradient_norm(model.anchor_heads) > 0
    assert _gradient_norm(model.prior_encoder) == 0
    assert _gradient_norm(model.backbone.input) == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = F.mse_loss(model(batch, prior), target)
    second.backward()
    assert _gradient_norm(model.anchor_heads) > 0
    assert _gradient_norm(model.prior_encoder) > 0
    assert _gradient_norm(model.backbone.input) > 0


def test_b_learns_a_synthetic_residual_on_top_of_strong_prior() -> None:
    model = build_model("prist_ris_b", domain="mobility", **SMALL)
    batch, prior, target = _mobility_problem()
    ideal = target - prior
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses: list[float] = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, prior)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    correction = model(batch, prior).detach() - prior
    cosine = F.cosine_similarity(correction.reshape(1, -1), ideal.reshape(1, -1))[0]
    assert losses[-1] < 0.25 * losses[0]
    assert float(correction.square().mean().sqrt()) > 0
    assert float(cosine) > 0.5


def test_per_anchor_prior_fusion_is_isolated() -> None:
    torch.manual_seed(44)
    model = build_model("prist_ris_b", domain="mobility", **SMALL).eval()
    with torch.no_grad():
        for head in model.anchor_heads:
            head.weight.normal_(std=0.05)
            head.bias.normal_(std=0.01)
    batch, prior, _ = _mobility_problem(batch_size=1)
    first = model(batch, prior)
    q3_changed = prior.clone()
    q3_changed[:, 1] += torch.randn_like(q3_changed[:, 1])
    second = model(batch, q3_changed)
    torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
    assert not torch.equal(first[:, 1], second[:, 1])


def test_c_initialization_also_preserves_the_ridge_prior() -> None:
    model = build_model("prist_ris_c", domain="mobility", **SMALL)
    batch, prior, _ = _mobility_problem(batch_size=1)
    torch.testing.assert_close(model(batch, prior), prior, rtol=0, atol=0)


def test_spatial_feature_scale_report_covers_the_repaired_path() -> None:
    model = build_model("prist_ris_b", domain="mobility", **SMALL)
    batch, prior, target = _mobility_problem(batch_size=1)
    model.spatial_anchors(batch, prior)
    report = spatial_feature_scale_report(model, ideal_residual=target - prior)
    assert {
        "obs_input_rms",
        "backbone_output_rms",
        "prior_raw_rms",
        "prior_encoder_rms",
        "fused_feature_rms",
        "refined_feature_rms",
        "delta_rms",
        "ideal_residual_rms",
        "delta_to_ideal_ratio",
    } <= report.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())
