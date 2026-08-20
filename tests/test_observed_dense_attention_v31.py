from __future__ import annotations

import torch

from prist_ris.contracts import OBSERVED_RIS_INDICES, SPATIAL_PROTOCOL_VERSION
from prist_ris.diagnostics import parameter_group_gradient_norms
from prist_ris.models import (
    PhysicalObservedDenseResidualAttention,
    build_model,
    canonical_batch,
)


SMALL = {"hidden": 4, "blocks_per_stage": (1, 1, 1), "final_refine_blocks": 1}


def _attention(hidden: int = 4) -> PhysicalObservedDenseResidualAttention:
    return PhysicalObservedDenseResidualAttention(hidden, heads=4)


def _module_inputs(hidden: int = 4, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    features = torch.randn(batch_size, hidden, 64, 16, 16)
    observations = torch.randn(batch_size, 2, 32, 64, 2)
    indices = torch.tensor(OBSERVED_RIS_INDICES).expand(batch_size, -1)
    times = torch.tensor([0, 3]).expand(batch_size, -1)
    return features, observations, indices, times


def test_attention_shape_contract_and_residual_scale() -> None:
    module = _attention()
    inputs = _module_inputs()
    output = module(*inputs)
    assert output.shape == inputs[0].shape
    assert module.residual_scale == 0.1


def test_attention_is_per_antenna_isolated() -> None:
    torch.manual_seed(3)
    module = _attention().eval()
    features, observations, indices, times = _module_inputs()
    first = module(features, observations, indices, times)
    changed = observations.clone()
    changed[:, :, :, 0] += 10.0
    second = module(features, changed, indices, times)
    assert not torch.equal(first[:, :, 0], second[:, :, 0])
    torch.testing.assert_close(first[:, :, 1], second[:, :, 1], rtol=0, atol=0)


def test_observed_token_is_physical_coordinate_aware() -> None:
    torch.manual_seed(4)
    module = _attention().eval()
    features = torch.zeros(1, 4, 64, 16, 16)
    observations = torch.zeros(1, 2, 32, 64, 2)
    observations[:, :, 0] = 1.0
    observations[:, :, 1] = -2.0
    indices = torch.tensor(OBSERVED_RIS_INDICES).reshape(1, 32)
    times = torch.tensor([[0, 3]])
    first = module(features, observations, indices, times)
    swapped = observations.clone()
    swapped[:, :, 0], swapped[:, :, 1] = observations[:, :, 1], observations[:, :, 0]
    second = module(features, swapped, indices, times)
    assert not torch.equal(first, second)
    coordinates = module.observed_coordinates(indices, 1, dtype=torch.float32)
    assert coordinates.shape == (32, 2)
    assert not torch.equal(coordinates[0], coordinates[1])


def test_pilot_descriptors_distinguish_q0_and_q3() -> None:
    descriptors = _attention().pilot_descriptors(
        torch.tensor([[0, 3]]), dtype=torch.float32
    )
    torch.testing.assert_close(descriptors[0, 0], torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(descriptors[0, 1], torch.tensor([1.0, 1.0]))


def test_attention_backward_reaches_query_key_and_value_projections() -> None:
    torch.manual_seed(5)
    module = _attention()
    inputs = _module_inputs()
    output = module(*inputs)
    output.square().mean().backward()
    for projection in (
        module.query_projection,
        module.key_projection,
        module.value_projection,
    ):
        assert projection.weight.grad is not None
        assert float(projection.weight.grad.abs().sum()) > 0


def test_realistic_attention_shape_is_memory_safe() -> None:
    module = _attention(hidden=80)
    output = module(*_module_inputs(hidden=80))
    assert output.shape == (1, 80, 64, 16, 16)


def test_quasi_single_pilot_is_supported() -> None:
    module = _attention()
    features = torch.randn(1, 4, 64, 16, 16)
    observations = torch.randn(1, 1, 32, 64, 2)
    indices = torch.tensor(OBSERVED_RIS_INDICES).reshape(1, 32)
    output = module(features, observations, indices, torch.tensor([[0]]))
    assert output.shape == features.shape


def test_b_has_no_attention_while_c_and_full_share_the_spatial_ladder() -> None:
    b_model = build_model("prist_ris_b", domain="mobility", **SMALL)
    c_model = build_model("prist_ris_c", domain="mobility", **SMALL)
    full_model = build_model("prist_ris_full", domain="mobility", **SMALL)
    assert b_model.observed_dense_attention is None
    assert c_model.observed_dense_attention is not None
    assert full_model.observed_dense_attention is not None
    assert sum(parameter.numel() for parameter in c_model.parameters()) > sum(
        parameter.numel() for parameter in b_model.parameters()
    )
    metadata = c_model.protocol_metadata()
    assert metadata["spatial_protocol_version"] == SPATIAL_PROTOCOL_VERSION
    assert metadata["observed_dense_attention_scope"] == "per_antenna_32_to_256"
    assert metadata["observed_dense_attention_uses_target"] is False


def test_full_reuses_c_spatial_anchor_path() -> None:
    torch.manual_seed(7)
    c_model = build_model("prist_ris_c", domain="mobility", **SMALL)
    full_model = build_model("prist_ris_full", domain="mobility", **SMALL)
    common = {
        name: value
        for name, value in c_model.state_dict().items()
        if name in full_model.state_dict() and value.shape == full_model.state_dict()[name].shape
    }
    full_model.load_state_dict(common, strict=False)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    torch.testing.assert_close(
        c_model.spatial_anchors(batch, prior),
        full_model.spatial_anchors(batch, prior),
    )


def test_c_and_full_do_not_read_target_h() -> None:
    torch.manual_seed(8)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    changed = {**batch, "target_h": torch.randn_like(batch["target_h"])}
    for key in ("prist_ris_c", "prist_ris_full"):
        model = build_model(key, domain="mobility", **SMALL).eval()
        torch.testing.assert_close(model(batch, prior), model(changed, prior), rtol=0, atol=0)


def test_gradient_summary_reports_observed_attention_path() -> None:
    model = build_model("prist_ris_c", domain="mobility", **SMALL)
    batch = canonical_batch("mobility")
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model(batch, prior).square().mean().backward()
    first = parameter_group_gradient_norms(model)
    assert first["observed_dense_attention"]["l2_norm"] == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(batch, prior).square().mean().backward()
    summary = parameter_group_gradient_norms(model)
    assert summary["observed_dense_attention"]["l2_norm"] > 0
    assert summary["backbone.input"]["gradient_tensors"] > 0
    assert summary["anchor_heads"]["gradient_tensors"] > 0
    attention = model.observed_dense_attention
    assert attention is not None
    for projection in (
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
        attention.output_projection,
    ):
        assert projection.weight.grad is not None
        assert float(projection.weight.grad.abs().sum()) > 0
