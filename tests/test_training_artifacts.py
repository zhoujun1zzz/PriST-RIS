from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from prist_ris.checkpoint import load_checkpoint
from prist_ris.engine import TrainingConfig, train


class _OneSample(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs_h": torch.zeros(1, 32, 64, 2),
            "target_h": torch.zeros(1, 256, 64, 2),
            "obs_ris_index": torch.arange(0, 256, 8),
            "obs_time_index": torch.tensor([0]),
            "query_time": torch.tensor([0]),
            "observation_mask": torch.ones(1, 32, dtype=torch.bool),
            "sample_index": torch.tensor(index),
        }


def test_checkpoint_contains_reproducibility_contract(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(123)
    loader = DataLoader(_OneSample(), batch_size=1, shuffle=True, generator=generator)
    config = TrainingConfig(
        domain="quasi",
        model_key="prist_ris_a",
        mode="smoke",
        hidden=4,
        epochs=1,
        min_epochs=2,
        target_blocks=(0,),
    )
    run = tmp_path / "run"
    train(config, loader, loader, run_dir=run, device=torch.device("cpu"))
    state = load_checkpoint(run / "checkpoints" / "last_checkpoint.pth", torch.device("cpu"))
    assert state["method"] == "PriST-RIS"
    assert state["semantics_hash"]
    assert state["rng_state"]
    assert state["train_loader_generator_state"] is not None
    assert (run / "metadata.json").is_file()
    assert (run / "manifests" / "data_semantics.json").is_file()


def _loader(seed: int = 123) -> DataLoader:
    return DataLoader(
        _OneSample(), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )


def test_resume_is_bitwise_deterministic(tmp_path: Path) -> None:
    config = TrainingConfig(
        domain="quasi",
        model_key="prist_ris_a",
        mode="smoke",
        seed=17,
        hidden=4,
        epochs=2,
        min_epochs=3,
        target_blocks=(0,),
    )
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
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


def test_tiny_overfit_loss_decreases() -> None:
    from prist_ris.models import build_model, canonical_batch

    torch.manual_seed(9)
    model = build_model("prist_ris_a", domain="quasi", hidden=4)
    batch = canonical_batch("quasi", batch_size=1, device=torch.device("cpu"))
    batch["obs_h"].normal_()
    batch["target_h"].zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
