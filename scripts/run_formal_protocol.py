from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from prist_ris.contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    SPATIAL_PROTOCOL_VERSION,
)


SEEDS = (123, 456, 789)


def run(arguments: list[object], *, dry_run: bool) -> None:
    command = [sys.executable, str(PROJECT / "main.py"), *(str(value) for value in arguments)]
    if dry_run:
        print("DRY-RUN:", subprocess.list2cmdline(command))
    else:
        subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only PriST-RIS formal protocol")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--quasi-prior", required=True)
    parser.add_argument("--mobility-prior", required=True)
    parser.add_argument("--quasi-best-result", required=True)
    parser.add_argument("--mobility-best-result", required=True)
    parser.add_argument("--output-root", default="runs/v3_2_formal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quasi-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    root = Path(args.output_root)
    protocol = {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "fp32": True,
        "test_split_used": False,
        "stage_f_enabled": False,
        "seeds": list(SEEDS),
        "quasi_batch": args.quasi_batch_size,
        "mobility_batch": 32,
        "eval_batch": 64,
        "workers": 8,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for domain, prior, best_path in (
        ("quasi", args.quasi_prior, args.quasi_best_result),
        ("mobility", args.mobility_prior, args.mobility_best_result),
    ):
        best = json.loads(Path(best_path).read_text(encoding="utf-8"))["best_hyperparameters"]
        for seed in SEEDS:
            name = f"v32_{domain}_prist_ris_full_seed{seed}"
            run(
                [
                    "train", "--domain", domain, "--model", "prist_ris_full",
                    "--mode", "full", "--seed", seed, "--hidden", best["hidden"],
                    "--learning-rate", best["learning_rate"],
                    "--blocks-per-stage", ",".join(str(v) for v in best.get("blocks_per_stage", [3, 3, 4])),
                    "--final-refine-blocks", best.get("final_refine_blocks", 4),
                    "--temporal-rank", best.get("temporal_rank", 2),
                    "--prior", prior, "--data-root", args.data_root,
                    "--device", args.device, "--batch-size", args.quasi_batch_size if domain == "quasi" else 32,
                    "--eval-batch-size", 64, "--workers", 8,
                    "--epochs", 100, "--min-epochs", 40, "--patience", 15,
                    "--run-name", name, "--output-root", root / "final",
                ],
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
