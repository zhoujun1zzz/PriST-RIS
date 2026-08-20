from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from prist_ris.contracts import DataSemantics
from prist_ris.data import make_loader
from prist_ris.diagnostics import parameter_group_gradient_norms
from prist_ris.engine import move_batch, seed_everything
from prist_ris.metrics import sample_linear_nmse
from prist_ris.models import build_model
from prist_ris.objectives import prist_ris_loss
from prist_ris.prior import RidgePrior


def _diagnostics(
    prediction: torch.Tensor,
    prior: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    correction = prediction - prior
    ideal = target - prior
    correction_rms = float(correction.square().mean().sqrt().detach())
    ideal_rms = float(ideal.square().mean().sqrt().detach())
    cosine = float(
        F.cosine_similarity(
            correction.detach().reshape(1, -1),
            ideal.detach().reshape(1, -1),
            dim=1,
            eps=1e-12,
        )[0]
    )
    nmse = float(sample_linear_nmse(prediction, target).mean().detach())
    return {
        "nmse_linear": nmse,
        "nmse_db": 10.0 * math.log10(max(nmse, 1e-12)),
        "correction_rms": correction_rms,
        "ideal_residual_rms": ideal_rms,
        "correction_ideal_ratio": correction_rms / max(ideal_rms, 1e-12),
        "correction_vs_ideal_cosine": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRAIN-only spatial micro-overfit and gradient diagnostic"
    )
    parser.add_argument("--domain", choices=("quasi", "mobility"), default="mobility")
    parser.add_argument("--model", choices=("prist_ris_c", "prist_ris_full"), default="prist_ris_c")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden", type=int, default=80)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 1 or args.log_every < 1:
        raise ValueError("samples, steps, and log-every must be positive.")

    seed_everything(args.seed)
    device = torch.device(args.device)
    loader = make_loader(
        args.data_root,
        args.domain,
        "train",
        batch_size=args.samples,
        workers=0,
        seed=args.seed,
        max_samples=args.samples,
        shuffle=False,
    )
    if getattr(loader.dataset, "split", None) != "train":
        raise PermissionError("Spatial micro-overfit is restricted to the TRAIN split.")
    batch = move_batch(next(iter(loader)), device)
    semantics = DataSemantics.for_domain(args.domain)
    prior_model = RidgePrior.load(args.prior)
    expected_blocks = (0,) if args.domain == "quasi" else (0, 3)
    if prior_model.semantics_hash != semantics.stable_hash():
        raise ValueError("Ridge prior semantics do not match this micro-overfit run.")
    if prior_model.target_blocks != expected_blocks:
        raise ValueError(f"Expected Ridge target blocks {expected_blocks}.")

    model = build_model(args.model, domain=args.domain, hidden=args.hidden).to(device)
    model.train()
    prior = prior_model.predict(batch)
    target_index = torch.tensor(expected_blocks, device=device)
    target = batch["target_h"].index_select(1, target_index)
    prior_nmse = float(sample_linear_nmse(prior, target).mean())
    print(
        json.dumps(
            {
                "split": "train",
                "test_split_used": False,
                "samples": args.samples,
                "prior_nmse_linear": prior_nmse,
                "prior_nmse_db": 10.0 * math.log10(max(prior_nmse, 1e-12)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.spatial_anchors(batch, prior)
        loss, _ = prist_ris_loss(prediction, target)
        loss.backward()
        if step == 0 or step == args.steps or step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        **_diagnostics(prediction, prior, target),
                        "gradient_norms": parameter_group_gradient_norms(model),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step < args.steps:
            optimizer.step()


if __name__ == "__main__":
    main()
