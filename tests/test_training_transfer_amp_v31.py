from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from prist_ris.checkpoint import load_checkpoint
from prist_ris.contracts import (
    ARCHITECTURE_VERSION,
    OBSERVED_RIS_INDICES,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    DataSemantics,
)
from prist_ris.engine import (
    TrainingConfig,
    _select,
    _require_architecture_version,
    configure_adaptation,
    load_spatial_pretrained,
    require_checkpoint_contract,
    restore_loader_generator_state,
    train,
)
from prist_ris.experiments import (
    SPATIAL_ABLATION_TARGET_BLOCKS,
    TEMPORAL_ABLATION_TARGET_BLOCKS,
    TRANSFER_PROTOCOLS,
)
from prist_ris.models import build_model, canonical_batch, complex_factorized_reconstruction


class OneQuasiSample(Dataset[dict[str, torch.Tensor]]):
    split = "train"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs_h": torch.zeros(1, 32, 64, 2),
            "target_h": torch.zeros(1, 256, 64, 2),
            "obs_ris_index": torch.tensor(OBSERVED_RIS_INDICES),
            "obs_time_index": torch.tensor([0]),
            "query_time": torch.tensor([0]),
            "sample_index": torch.tensor(index),
        }


def _loader(seed: int = 123) -> DataLoader:
    return DataLoader(
        OneQuasiSample(), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )


def _config(epochs: int = 1) -> TrainingConfig:
    return TrainingConfig(
        domain="quasi",
        model_key="prist_ris_a",
        mode="smoke",
        hidden=2,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
        epochs=epochs,
        min_epochs=epochs + 1,
    )


def test_checkpoint_contains_v32_reproducibility_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    train(_config(), _loader(), _loader(), run_dir=run, device=torch.device("cpu"))
    state = load_checkpoint(run / "checkpoints" / "last_checkpoint.pth", torch.device("cpu"))
    assert state["architecture_version"] == ARCHITECTURE_VERSION
    assert state["model_config"]["architecture_version"] == ARCHITECTURE_VERSION
    assert state["position_semantics_version"] == POSITION_SEMANTICS_VERSION
    for key in (
        "backbone_ris_coordinate_enabled",
        "backbone_antenna_index_enabled",
        "backbone_ris_coordinate_mode",
        "attention_enabled",
        "attention_ris_coordinate_enabled",
        "attention_antenna_index_enabled",
    ):
        assert state[key] is not None
        assert state["training_config"][key] is not None
        assert state["model_config"][key] is not None
    assert state["rng_state"] and state["train_loader_generator_state"] is not None
    metadata = (run / "metadata.json").read_text(encoding="utf-8")
    assert f'"architecture_version": "{ARCHITECTURE_VERSION}"' in metadata
    assert f'"position_semantics_version": "{POSITION_SEMANTICS_VERSION}"' in metadata
    assert '"test_split_used": false' in metadata
    final = (run / "results" / "final_result.json").read_text(encoding="utf-8")
    assert f'"position_semantics_version": "{POSITION_SEMANTICS_VERSION}"' in final


def test_resume_is_bitwise_deterministic(tmp_path: Path) -> None:
    config = _config(epochs=2)
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    train(config, _loader(), _loader(), run_dir=uninterrupted, device=torch.device("cpu"))
    train(
        config,
        _loader(),
        _loader(),
        run_dir=resumed,
        device=torch.device("cpu"),
        stop_after_epoch=1,
    )
    checkpoint = resumed / "checkpoints" / "last_checkpoint.pth"
    train(
        config,
        _loader(),
        _loader(),
        run_dir=resumed,
        device=torch.device("cpu"),
        resume=checkpoint,
    )
    first = load_checkpoint(uninterrupted / "checkpoints" / "last_checkpoint.pth", torch.device("cpu"))
    second = load_checkpoint(resumed / "checkpoints" / "last_checkpoint.pth", torch.device("cpu"))
    for name, value in first["model_state"].items():
        torch.testing.assert_close(value, second["model_state"][name], rtol=0, atol=0)


def test_legacy_checkpoint_is_rejected_by_version_guard() -> None:
    with pytest.raises(ValueError, match=f"architecture_version={ARCHITECTURE_VERSION}"):
        _require_architecture_version(
            {"method": "PriST-RIS", "architecture_version": "3.1"}, "Resume"
        )


def test_tiny_overfit_loss_decreases() -> None:
    torch.manual_seed(9)
    model = build_model(
        "prist_ris_a",
        domain="quasi",
        hidden=2,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    batch = canonical_batch("quasi")
    batch["obs_h"].normal_()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_spatial_transfer_loads_only_compatible_weights() -> None:
    torch.manual_seed(1)
    source = build_model(
        "prist_ris_c",
        domain="quasi",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    target = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    before_second_head = target.anchor_heads[1].weight.detach().clone()
    state = {
        "architecture_version": ARCHITECTURE_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "model_config": {"domain": "quasi", "model_key": "prist_ris_c"},
        "model_state": source.state_dict(),
    }
    metadata = load_spatial_pretrained(target, state)
    torch.testing.assert_close(target.backbone.input.weight, source.backbone.input.weight)
    torch.testing.assert_close(target.anchor_heads[0].weight, source.anchor_heads[0].weight)
    torch.testing.assert_close(
        target.observed_dense_attention.query_projection.weight,
        source.observed_dense_attention.query_projection.weight,
    )
    torch.testing.assert_close(target.anchor_heads[1].weight, before_second_head)
    assert all(not name.startswith("temporal.") for name in metadata["loaded_keys"])
    assert any(name.startswith("temporal.") for name in metadata["newly_initialized_keys"])


def test_pre_attention_quasi_c_transfer_is_rejected() -> None:
    source = build_model(
        "prist_ris_c",
        domain="quasi",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    target = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    state = {
        "architecture_version": ARCHITECTURE_VERSION,
        "model_config": {"domain": "quasi", "model_key": "prist_ris_c"},
        "model_state": source.state_dict(),
    }
    with pytest.raises(ValueError, match="spatial_protocol_version"):
        load_spatial_pretrained(target, state)


def test_quasi_a_backbone_transfer_remains_compatible() -> None:
    source = build_model(
        "prist_ris_a",
        domain="quasi",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    target = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    metadata = load_spatial_pretrained(
        target,
        {
            "architecture_version": ARCHITECTURE_VERSION,
            "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
            "position_semantics_version": POSITION_SEMANTICS_VERSION,
            "model_config": {"domain": "quasi", "model_key": "prist_ris_a"},
            "model_state": source.state_dict(),
        },
    )
    assert "backbone.input.weight" in metadata["loaded_keys"]
    assert any(
        name.startswith("observed_dense_attention.")
        for name in metadata["newly_initialized_keys"]
    )


def test_transfer_protocols_have_no_fake_adapter_and_distinct_sets() -> None:
    assert "adapter_only" not in TRANSFER_PROTOCOLS
    assert len(TRANSFER_PROTOCOLS) == 4
    frozen = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    selective = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    )
    frozen_names = set(configure_adaptation(frozen, "frozen_spatial"))
    selective_names = set(configure_adaptation(selective, "selective"))
    assert frozen_names != selective_names
    assert any(name.startswith("temporal.spatial_encoder") for name in frozen_names)
    assert all(not name.startswith("temporal.spatial_encoder") for name in selective_names)


def test_ablation_scopes_are_not_mixed() -> None:
    assert SPATIAL_ABLATION_TARGET_BLOCKS == (0, 3)
    assert TEMPORAL_ABLATION_TARGET_BLOCKS == (0, 1, 2, 3, 4, 5)


@pytest.mark.parametrize("blocks", [(0, 3), (0,), (3,)])
def test_compact_spatial_predictions_align_by_semantic_time(
    blocks: tuple[int, ...],
) -> None:
    prediction = torch.tensor([[[10.0], [30.0]]])
    target = torch.arange(6.0).reshape(1, 6, 1)
    selected_prediction, selected_target, selected_times = _select(
        prediction, target, blocks, (0, 3)
    )
    assert selected_times == blocks
    expected = torch.tensor(
        [[[10.0 if value == 0 else 30.0]] for value in blocks]
    ).reshape(1, len(blocks), 1)
    torch.testing.assert_close(selected_prediction, expected)
    torch.testing.assert_close(
        selected_target, torch.tensor(blocks, dtype=torch.float32).reshape(1, -1, 1)
    )


def test_prefix_mobility_checkpoint_is_rejected_by_contract_guard() -> None:
    semantics = DataSemantics.for_domain("mobility")
    old_state = {
        "architecture_version": ARCHITECTURE_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "model_config": {"domain": "mobility"},
        "semantics_hash": semantics.stable_hash(),
        "data_semantics": semantics.to_dict(),
    }
    with pytest.raises(ValueError, match="pre-fix Mobility"):
        require_checkpoint_contract(old_state, "Resume", expected_domain="mobility")


@pytest.mark.parametrize("model_key", ["prist_ris_c", "prist_ris_full"])
def test_old_q0q3_spatial_checkpoint_is_rejected(model_key: str) -> None:
    semantics = DataSemantics.for_domain("mobility")
    old_state = {
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": "mobility_q0_q3_v1",
        "model_config": {"domain": "mobility", "model_key": model_key},
        "semantics_hash": semantics.stable_hash(),
        "data_semantics": semantics.to_dict(),
    }
    with pytest.raises(ValueError, match="spatial_protocol_version"):
        require_checkpoint_contract(old_state, "Resume", expected_domain="mobility")


def test_current_q0q3_attention_checkpoint_contract_is_accepted() -> None:
    semantics = DataSemantics.for_domain("mobility")
    require_checkpoint_contract(
        {
            "architecture_version": ARCHITECTURE_VERSION,
            "mobility_contract_version": "mobility_q0_q3_v1",
            "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
            "position_semantics_version": POSITION_SEMANTICS_VERSION,
            "model_config": {"domain": "mobility", "model_key": "prist_ris_c"},
            "semantics_hash": semantics.stable_hash(),
            "data_semantics": semantics.to_dict(),
        },
        "Resume",
        expected_domain="mobility",
    )


def test_spatial_protocol_does_not_invalidate_ridge_only_model_checkpoint() -> None:
    semantics = DataSemantics.for_domain("mobility")
    state = {
        "architecture_version": ARCHITECTURE_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "mobility_contract_version": "mobility_q0_q3_v1",
        "model_config": {"domain": "mobility", "model_key": "prist_ris_b"},
        "semantics_hash": semantics.stable_hash(),
        "data_semantics": semantics.to_dict(),
    }
    require_checkpoint_contract(state, "Resume", expected_domain="mobility")


def test_loader_generator_state_is_normalized_to_cpu_uint8() -> None:
    source = torch.Generator().manual_seed(987)
    restored = torch.Generator()
    state = source.get_state().to(torch.int16).contiguous()
    restore_loader_generator_state(restored, state)
    torch.testing.assert_close(restored.get_state(), source.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_loader_generator_state_mapped_to_cuda_is_restored() -> None:
    source = torch.Generator().manual_seed(654)
    restored = torch.Generator()
    restore_loader_generator_state(restored, source.get_state().cuda())
    torch.testing.assert_close(restored.get_state(), source.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_amp_complex_factorization_uses_complex64() -> None:
    device = torch.device("cuda")
    with torch.amp.autocast("cuda"):
        bases = torch.randn(1, 2, 256, 64, 2, device=device)
        coefficients = torch.randn(1, 4, 2, 2, device=device)
        output = complex_factorized_reconstruction(bases, coefficients)
    assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_amp_mobility_full_forward_backward() -> None:
    device = torch.device("cuda")
    model = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=4,
        blocks_per_stage=(1, 1, 1),
        final_refine_blocks=1,
    ).to(device)
    batch = canonical_batch("mobility", device=device)
    batch["obs_h"].normal_()
    prior = torch.randn(1, 2, 256, 64, 2, device=device)
    with torch.amp.autocast("cuda"):
        output = model(batch, prior)
        loss = output.square().mean()
    loss.backward()
    assert output.shape == (1, 6, 256, 64, 2)
    assert model.temporal is not None and model.temporal.last_complex_dtype == torch.complex64
