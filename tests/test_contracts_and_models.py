from __future__ import annotations

import torch

from prist_ris.contracts import (
    MODEL_DISPLAY_NAME,
    OBSERVED_RIS_INDICES,
    DataSemantics,
    canonical_model_key,
    ris_index_to_grid,
)
from prist_ris.models import (
    FactorizedAntennaRISBlock,
    StructuredProgressiveBackbone,
    build_model,
    canonical_batch,
    complex_factorized_reconstruction,
)


def test_canonical_name_and_semantics() -> None:
    assert MODEL_DISPLAY_NAME == "PriST-RIS"
    assert canonical_model_key("v3_full") == "prist_ris_full"
    assert OBSERVED_RIS_INDICES == tuple(range(0, 256, 8))
    assert ris_index_to_grid(255) == (15, 15)
    assert DataSemantics.for_domain("quasi").target_shape == ("B", 1, 256, 64, 2)
    assert DataSemantics.for_domain("mobility").target_shape == ("B", 6, 256, 64, 2)


@torch.no_grad()
def test_every_variant_has_frozen_output_shape_and_uses_no_target() -> None:
    for domain, query_blocks in (("quasi", 1), ("mobility", 6)):
        batch = canonical_batch(domain, batch_size=1, device=torch.device("cpu"))
        batch_with_changed_target = {**batch, "target_h": torch.randn_like(batch["target_h"])}
        for key in ("prist_ris_a", "prist_ris_b", "prist_ris_c", "prist_ris_full"):
            model = build_model(key, domain=domain, hidden=8)
            prior = torch.zeros(1, 1, 256, 64, 2) if key != "prist_ris_a" else None
            first = model(batch, prior)
            second = model(batch_with_changed_target, prior)
            assert first.shape == (1, query_blocks, 256, 64, 2)
            torch.testing.assert_close(first, second)


def test_cross_attention_keys_are_observed_tokens_only() -> None:
    model = build_model("prist_ris_c", domain="mobility", hidden=8)
    batch = canonical_batch("mobility", batch_size=1, device=torch.device("cpu"))
    model(batch, torch.zeros(1, 1, 256, 64, 2))
    assert model.cross_attention is not None
    assert model.cross_attention.last_query_tokens == 256
    assert model.cross_attention.last_key_tokens == 32


def test_progressive_width_only_and_factorized_kernel_contract() -> None:
    backbone = StructuredProgressiveBackbone(hidden=8)
    batch = canonical_batch("mobility", batch_size=1, device=torch.device("cpu"))
    _, shapes = backbone(batch["obs_h"])
    assert [shape[-2:] for shape in shapes] == [(64, 32), (64, 64), (64, 128), (64, 256)]
    block = FactorizedAntennaRISBlock(8)
    assert block.ris_depthwise.kernel_size == (1, 3)
    assert block.antenna_depthwise is not None
    assert block.antenna_depthwise.kernel_size == (3, 1)


def test_each_factorized_branch_changes_output() -> None:
    torch.manual_seed(3)
    value = torch.randn(1, 8, 64, 32)
    block = FactorizedAntennaRISBlock(8).eval()
    baseline = block(value)
    with torch.no_grad():
        block.ris_depthwise.weight.zero_()
    without_ris = block(value)
    assert not torch.equal(baseline, without_ris)
    block = FactorizedAntennaRISBlock(8).eval()
    baseline = block(value)
    assert block.antenna_depthwise is not None
    with torch.no_grad():
        block.antenna_depthwise.weight.zero_()
    without_antenna = block(value)
    assert not torch.equal(baseline, without_antenna)


def test_complex_low_rank_reconstruction() -> None:
    bases = torch.randn(2, 3, 256, 64, 2)
    coefficients = torch.randn(2, 6, 3, 2)
    assert complex_factorized_reconstruction(bases, coefficients).shape == (2, 6, 256, 64, 2)


def test_temporal_observed_queries_are_aligned_and_ranks_work() -> None:
    batch = canonical_batch("mobility", batch_size=1, device=torch.device("cpu"))
    for rank in (2, 3):
        model = build_model("prist_ris_full", domain="mobility", hidden=8, temporal_rank=rank)
        temporal = model.temporal
        assert temporal is not None
        contexts = temporal.aligned_query_context(
            batch["obs_h"], batch["obs_time_index"][0], batch["query_time"][0]
        )
        pooled = batch["obs_h"].mean(dim=(2, 3))
        observed_contexts = temporal.observed_context(pooled)
        torch.testing.assert_close(contexts[:, :2], observed_contexts)
        output = model(batch, torch.zeros(1, 1, 256, 64, 2))
        assert output.shape == (1, 6, 256, 64, 2)


def test_parameter_ceiling_at_largest_search_width() -> None:
    model = build_model("prist_ris_full", domain="mobility", hidden=96, temporal_rank=3)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_prior_is_explicit_for_guided_variants() -> None:
    model = build_model("prist_ris_b", domain="quasi", hidden=8)
    batch = canonical_batch("quasi", batch_size=1, device=torch.device("cpu"))
    try:
        model(batch)
    except ValueError as error:
        assert "Ridge prior" in str(error)
    else:
        raise AssertionError("Prior-guided model accepted a missing prior.")
