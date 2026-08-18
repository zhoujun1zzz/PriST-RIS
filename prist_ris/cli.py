from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import h5py
import torch

from .checkpoint import load_checkpoint
from .complexity import profile_model
from .contracts import MODEL_ALIASES, MODEL_KEYS, DataSemantics, canonical_model_key
from .data import (
    EXPECTED_MOBILITY_COUNTS,
    make_loader,
    nested_fraction_indices,
    resolve_dataset_path,
    write_index_manifest,
)
from .engine import TrainingConfig, evaluate, seed_everything, train
from .experiments import (
    CAPACITY_HIDDEN,
    LEARNING_RATES,
    MECHANISM_ABLATIONS,
    TEMPORAL_RANKS,
    TRANSFER_FRACTIONS,
    TRANSFER_PROTOCOLS,
    late_window_score,
    targeted_tuning_plan,
    write_plan,
)
from .manifests import (
    freeze_experiment,
    import_baseline_manifest,
    validate_test_unlock,
)
from .metrics import MetricAccumulator
from .models import build_model
from .prior import RidgePrior, RidgeStatistics, file_sha256


PROJECT = Path(__file__).resolve().parents[1]
ALL_MODEL_KEYS = MODEL_KEYS + tuple(MODEL_ALIASES)


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _allowed_test(args: argparse.Namespace) -> bool:
    if getattr(args, "split", None) != "test":
        return False
    if args.freeze_manifest is None or getattr(args, "checkpoint", None) is None:
        raise PermissionError("Test requires --freeze-manifest and the exact frozen checkpoint.")
    validate_test_unlock(args.freeze_manifest, args.checkpoint)
    return True


def audit_command(args: argparse.Namespace) -> dict[str, object]:
    splits = ["train", "validation"]
    if args.include_test:
        if args.freeze_manifest is None:
            raise PermissionError("Audit excludes test before a freeze manifest is supplied.")
        frozen = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
        if frozen.get("test_unlocked") is not True:
            raise PermissionError("The supplied freeze manifest does not unlock test.")
        splits.append("test")
    rows = []
    for domain in ("quasi", "mobility"):
        for split in splits:
            path = resolve_dataset_path(args.data_root, domain, split)
            with h5py.File(path, "r") as handle:
                input_key = "input_da" if domain == "quasi" and split == "train" else "Yd"
                target_key = "output_da" if domain == "quasi" and split == "train" else "Hd"
                count = int(handle[input_key].shape[-1])
                rows.append(
                    {
                        "domain": domain,
                        "split": split,
                        "path": str(path),
                        "input_key": input_key,
                        "target_key": target_key,
                        "input_shape": list(handle[input_key].shape),
                        "target_shape": list(handle[target_key].shape),
                        "samples": count,
                        "semantics_hash": DataSemantics.for_domain(domain).stable_hash(),
                    }
                )
                if domain == "mobility" and count != EXPECTED_MOBILITY_COUNTS[split]:
                    raise ValueError(f"Unexpected Mobility {split} sample count: {count}")
    result = {"method": "PriST-RIS", "test_included": "test" in splits, "files": rows}
    _json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def profile_command(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    model = build_model(
        args.model,
        domain=args.domain,
        hidden=args.hidden,
        heads=args.heads,
        temporal_rank=args.temporal_rank,
        antenna_branch=not args.ris_only,
        temporal_residual=not args.no_temporal_residual,
    ).to(device)
    result = profile_model(model, domain=args.domain, device=device)
    if args.output:
        _json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def _evaluate_prior(prior: RidgePrior, loader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float | int]:
    metrics = MetricAccumulator()
    blocks = torch.tensor(prior.target_blocks)
    for batch in loader:
        prediction = prior.predict(batch)
        target = batch["target_h"].index_select(1, blocks)
        metrics.update(prediction, target)
    return metrics.compute()


def fit_prior_command(args: argparse.Namespace) -> dict[str, object]:
    target_blocks = args.target_blocks
    if tuple(target_blocks) != (0,):
        raise ValueError("The initial PriST-RIS protocol freezes the spatial anchor target to block 0.")
    train_loader = make_loader(
        args.data_root,
        args.domain,
        "train",
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        max_samples=args.max_train,
        shuffle=False,
    )
    validation_loader = make_loader(
        args.data_root,
        args.domain,
        "validation",
        batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed,
        max_samples=args.max_validation,
        shuffle=False,
    )
    statistics = RidgeStatistics.accumulate(train_loader, tuple(target_blocks))
    candidates = []
    winner: tuple[float, RidgePrior, dict[str, float | int]] | None = None
    semantics = DataSemantics.for_domain(args.domain)
    for regularization in args.regularizations:
        prior = statistics.solve(regularization, semantics)
        validation = _evaluate_prior(prior, validation_loader)
        candidates.append({"regularization": regularization, "validation": validation})
        score = float(validation["nmse_linear"])
        if winner is None or score < winner[0]:
            winner = (score, prior, validation)
    assert winner is not None
    artifact = winner[1].save(args.output)
    result = {
        "status": "validation_prior_fit",
        "method": "PriST-RIS Ridge spatial anchor",
        "domain": args.domain,
        "fit_split": "train",
        "selection_split": "validation",
        "test_split_used": False,
        "target_blocks": list(target_blocks),
        "candidates": candidates,
        "selected_validation": winner[2],
        "artifact": artifact,
    }
    _json(Path(args.output).with_suffix(".json"), result)
    print(json.dumps(result, indent=2))
    return result


def _indices_from_file(path: Path | None, fraction: float | None) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fraction is None:
        values = payload.get("indices")
    else:
        values = payload.get("fractions", {}).get(f"{fraction:.2f}")
    if not isinstance(values, list):
        raise ValueError("Sample-index manifest does not contain the requested indices.")
    return [int(value) for value in values]


def train_command(args: argparse.Namespace) -> dict[str, object]:
    key = canonical_model_key(args.model)
    if args.mode == "smoke":
        max_train, max_validation, default_epochs = 64, 16, 1
    elif args.mode == "dev":
        max_train, max_validation, default_epochs = 4096, 1800, 30
    else:
        max_train, max_validation, default_epochs = None, None, 100
    indices = _indices_from_file(args.sample_index_manifest, args.fraction)
    train_loader = make_loader(
        args.data_root,
        args.domain,
        "train",
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        max_samples=None if indices is not None else max_train,
        indices=indices,
    )
    validation_loader = make_loader(
        args.data_root,
        args.domain,
        "validation",
        batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed,
        max_samples=max_validation,
        shuffle=False,
    )
    target_blocks = args.target_blocks
    if target_blocks is None and key != "prist_ris_full":
        target_blocks = (0,)
    config = TrainingConfig(
        domain=args.domain,
        model_key=key,
        mode=args.mode,
        seed=args.seed,
        hidden=args.hidden,
        blocks_per_stage=(2, 2, 3),
        heads=args.heads,
        temporal_rank=args.temporal_rank,
        antenna_branch=not args.ris_only,
        temporal_residual=not args.no_temporal_residual,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs or default_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        charbonnier_weight=args.charbonnier_weight,
        amp=args.amp,
        target_blocks=target_blocks,
        adaptation=args.adaptation,
    )
    run_name = args.run_name or f"{args.domain}_{key}_{args.mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    return train(
        config,
        train_loader,
        validation_loader,
        run_dir=Path(args.output_root) / run_name,
        device=torch.device(args.device),
        prior_path=args.prior,
        resume=args.resume,
        pretrained=args.pretrained,
        sample_indices=indices,
        stop_after_epoch=args.stop_after_epoch,
    )


def _invoke(arguments: list[object], *, dry_run: bool) -> None:
    command = [sys.executable, str(PROJECT / "main.py"), *(str(value) for value in arguments)]
    if dry_run:
        print("DRY-RUN:", subprocess.list2cmdline(command))
        return
    subprocess.run(command, cwd=PROJECT, check=True)


def tune_command(args: argparse.Namespace) -> dict[str, object]:
    plan = targeted_tuning_plan(args.domain, args.seed)
    root = Path(args.output_root) / (args.study_name or f"targeted_{args.domain}_seed{args.seed}")
    write_plan(root / "tuning_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.prior is None:
        raise ValueError("Targeted PriST-RIS tuning requires --prior.")

    def run_trial(name: str, *, hidden: int, lr: float, rank: int, epochs: int) -> Path:
        trial_root = root / "trials"
        _invoke(
            [
                "train", "--domain", args.domain, "--model", "prist_ris_full",
                "--mode", "full", "--seed", args.seed, "--hidden", hidden,
                "--learning-rate", lr, "--temporal-rank", rank, "--epochs", epochs,
                "--min-epochs", epochs + 1, "--prior", args.prior,
                "--data-root", args.data_root, "--device", args.device,
                "--batch-size", args.batch_size, "--eval-batch-size", args.eval_batch_size,
                "--workers", args.workers, "--run-name", name, "--output-root", trial_root,
            ],
            dry_run=False,
        )
        return trial_root / name

    capacity = []
    for hidden in CAPACITY_HIDDEN:
        run_dir = run_trial(f"capacity_h{hidden}", hidden=hidden, lr=5e-4, rank=2, epochs=30)
        capacity.append({"hidden": hidden, **late_window_score(run_dir / "results" / "training_history.csv", 25, 30)})
    capacity.sort(key=lambda row: (row["median_linear_nmse"], row["best_linear_nmse"]))
    selected_hidden = int(capacity[0]["hidden"])
    learning_rates = []
    for lr in LEARNING_RATES:
        token = f"{lr:.0e}".replace("-0", "-")
        run_dir = run_trial(f"lr_{token}", hidden=selected_hidden, lr=lr, rank=2, epochs=40)
        learning_rates.append({"learning_rate": lr, **late_window_score(run_dir / "results" / "training_history.csv", 31, 40)})
    learning_rates.sort(key=lambda row: (row["median_linear_nmse"], row["best_linear_nmse"]))
    selected_lr = float(learning_rates[0]["learning_rate"])
    ranks = []
    if args.domain == "mobility":
        for rank in TEMPORAL_RANKS:
            run_dir = run_trial(f"rank_{rank}", hidden=selected_hidden, lr=selected_lr, rank=rank, epochs=40)
            ranks.append({"temporal_rank": rank, **late_window_score(run_dir / "results" / "training_history.csv", 31, 40)})
        ranks.sort(key=lambda row: (row["median_linear_nmse"], row["best_linear_nmse"]))
    selected_rank = int(ranks[0]["temporal_rank"]) if ranks else 2
    result = {
        "status": "validation_search",
        "method": "PriST-RIS",
        "domain": args.domain,
        "test_split_used": False,
        "capacity_ranking": capacity,
        "learning_rate_ranking": learning_rates,
        "rank_ranking": ranks,
        "best_hyperparameters": {
            "hidden": selected_hidden,
            "learning_rate": selected_lr,
            "temporal_rank": selected_rank,
            "blocks_per_stage": [2, 2, 3],
            "heads": 4,
            "dropout": 0,
            "weight_decay": 1e-5,
        },
    }
    _json(root / "best_result.json", result)
    return result


def ablate_command(args: argparse.Namespace) -> dict[str, object]:
    plan = {
        "method": "PriST-RIS",
        "domain": "mobility",
        "seed": 123,
        "variants": list(MECHANISM_ABLATIONS),
        "reference_retrained": False,
        "test_split_used": False,
    }
    root = Path(args.output_root) / (args.study_name or "prist_ris_mechanism_seed123")
    write_plan(root / "ablation_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.prior is None or args.reference_checkpoint is None:
        raise ValueError("Executing ablation requires --prior and --reference-checkpoint.")
    reference = load_checkpoint(args.reference_checkpoint, torch.device("cpu"))
    if reference.get("method") != "PriST-RIS" or reference.get("model_config", {}).get("model_key") != "prist_ris_full":
        raise ValueError("Ablation reference must be the frozen PriST-RIS Full seed-123 checkpoint.")
    frozen = reference.get("training_config", {})
    if frozen.get("domain") != "mobility" or int(frozen.get("seed", -1)) != 123:
        raise ValueError("Ablation reference must use Mobility and seed 123.")
    mapping = {
        "ris_only_control": ("prist_ris_a", True, False),
        "structured_progressive": ("prist_ris_a", False, False),
        "prior_guided": ("prist_ris_b", False, False),
        "prior_cross_attention": ("prist_ris_c", False, False),
        "low_rank_temporal": ("prist_ris_full", False, True),
    }
    rows = []
    for variant, (model, ris_only, no_residual) in mapping.items():
        arguments: list[object] = [
            "train", "--domain", "mobility", "--model", model, "--mode", "full",
            "--seed", 123, "--data-root", args.data_root,
            "--device", args.device, "--run-name", variant, "--output-root", root / "trials",
            "--hidden", frozen["hidden"], "--heads", frozen["heads"],
            "--temporal-rank", frozen["temporal_rank"],
            "--learning-rate", frozen["learning_rate"],
            "--weight-decay", frozen["weight_decay"],
            "--epochs", frozen["epochs"], "--min-epochs", frozen["min_epochs"],
            "--patience", frozen["patience"],
        ]
        if model != "prist_ris_a":
            arguments.extend(["--prior", args.prior])
        if ris_only:
            arguments.append("--ris-only")
        if no_residual:
            arguments.append("--no-temporal-residual")
        _invoke(arguments, dry_run=False)
        final = json.loads((root / "trials" / variant / "results" / "final_result.json").read_text())
        rows.append({"variant": variant, "status": "trained", "result": final})
    rows.append(
        {
            "variant": "full_reference",
            "status": "reused_not_retrained",
            "best_validation_nmse_linear": reference["best_validation_nmse_linear"],
            "checkpoint": str(Path(args.reference_checkpoint).resolve()),
        }
    )
    result = {**plan, "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()), "results": rows}
    _json(root / "summary.json", result)
    return result


def transfer_command(args: argparse.Namespace) -> dict[str, object]:
    total = EXPECTED_MOBILITY_COUNTS["train"]
    nested = nested_fraction_indices(total, TRANSFER_FRACTIONS, seed=123)
    root = Path(args.output_root) / (args.study_name or "quasi_to_mobility_peft_seed123")
    manifest = root / "nested_sample_indices.json"
    write_index_manifest(manifest, nested, 123)
    plan = {
        "method": "PriST-RIS",
        "source_domain": "quasi",
        "target_domain": "mobility",
        "seed": 123,
        "fractions": list(TRANSFER_FRACTIONS),
        "protocols": list(TRANSFER_PROTOCOLS),
        "runs": 25,
        "nested_sample_manifest": str(manifest.resolve()),
        "test_split_used": False,
    }
    write_plan(root / "transfer_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.source_checkpoint is None or args.prior is None:
        raise ValueError("Executing transfer requires --source-checkpoint and Mobility --prior.")
    source = load_checkpoint(args.source_checkpoint, torch.device("cpu"))
    source_config = source.get("training_config", {})
    source_model = source.get("model_config", {})
    if source.get("method") != "PriST-RIS" or source_config.get("domain") != "quasi":
        raise ValueError("Transfer source must be a Quasi PriST-RIS checkpoint.")
    if source_model.get("model_key") != "prist_ris_full":
        raise ValueError("Transfer source must use prist_ris_full.")
    for fraction in TRANSFER_FRACTIONS:
        for protocol in TRANSFER_PROTOCOLS:
            name = f"fraction_{fraction:.2f}_{protocol}"
            arguments: list[object] = [
                "train", "--domain", "mobility", "--model", "prist_ris_full",
                "--mode", "full", "--seed", 123, "--fraction", fraction,
                "--sample-index-manifest", manifest, "--adaptation", protocol,
                "--prior", args.prior, "--data-root", args.data_root,
                "--device", args.device, "--hidden", source_config["hidden"],
                "--heads", source_config["heads"],
                "--temporal-rank", source_config["temporal_rank"],
                "--learning-rate", source_config["learning_rate"],
                "--weight-decay", source_config["weight_decay"],
                "--run-name", name, "--output-root", root / "runs",
            ]
            if protocol != "target_only_scratch":
                arguments.extend(["--pretrained", args.source_checkpoint])
            _invoke(arguments, dry_run=False)
    return plan


def evaluate_command(args: argparse.Namespace) -> dict[str, object]:
    allow_test = _allowed_test(args)
    device = torch.device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    if state.get("method") != "PriST-RIS":
        raise ValueError("Checkpoint is not a PriST-RIS checkpoint.")
    model_config = dict(state["model_config"])
    model = build_model(**model_config).to(device)
    model.load_state_dict(state["model_state"])
    prior_metadata = state.get("prior_metadata")
    prior = None
    if isinstance(prior_metadata, dict):
        prior_path = args.prior or Path(prior_metadata["path"])
        if file_sha256(prior_path) != prior_metadata.get("sha256"):
            raise ValueError("Evaluation prior is not the exact artifact recorded by the checkpoint.")
        prior = RidgePrior.load(prior_path)
    loader = make_loader(
        args.data_root,
        model_config["domain"],
        args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        allow_test=allow_test,
    )
    config = state["training_config"]
    result = evaluate(
        model,
        loader,
        device,
        prior=prior,
        target_blocks=tuple(config["target_blocks"]) if config.get("target_blocks") else None,
    )
    payload = {
        "method": "PriST-RIS",
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "freeze_manifest": str(Path(args.freeze_manifest).resolve()) if args.freeze_manifest else None,
        **result,
    }
    _json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return payload


def import_command(args: argparse.Namespace) -> dict[str, object]:
    result = import_baseline_manifest(args.source, args.output, require_checkpoints=args.require_checkpoints)
    print(json.dumps(result, indent=2))
    return result


def freeze_command(args: argparse.Namespace) -> dict[str, object]:
    if args.unlock_test_after_freeze and args.confirm != "FREEZE_PRIOR_AND_MODELS":
        raise ValueError("Unlocking test requires --confirm FREEZE_PRIOR_AND_MODELS.")
    result = freeze_experiment(
        args.output,
        project=PROJECT,
        checkpoints=list(args.checkpoints),
        prior_paths=list(args.priors),
        baseline_manifest=args.baseline_manifest,
        unlock_test_after_freeze=args.unlock_test_after_freeze,
    )
    print(json.dumps(result, indent=2))
    return result


def report_command(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for path in Path(args.runs_root).rglob("final_result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("method") == "PriST-RIS":
            rows.append({"path": str(path.resolve()), **value})
    baselines = None
    if args.baseline_manifest and args.baseline_manifest.is_file():
        baselines = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    result = {
        "method": "PriST-RIS",
        "validation_runs": rows,
        "external_baselines": baselines,
        "test_results_included": any(row.get("split") == "test" for row in rows),
    }
    _json(args.output, result)
    return result


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=ALL_MODEL_KEYS, default="prist_ris_full")
    parser.add_argument("--domain", choices=("quasi", "mobility"), required=True)
    parser.add_argument("--hidden", type=int, default=80)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-rank", type=int, choices=(2, 3), default=2)
    parser.add_argument("--ris-only", action="store_true")
    parser.add_argument("--no-temporal-residual", action="store_true")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=os.environ.get("PRIST_RIS_DATA_ROOT", str(PROJECT / "data")))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="PriST-RIS standalone research CLI")
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    add_runtime_arguments(audit)
    audit.add_argument("--include-test", action="store_true")
    audit.add_argument("--freeze-manifest", type=Path)
    audit.add_argument("--output", type=Path, default=Path("reports/generated/data_audit.json"))
    audit.set_defaults(func=audit_command)

    profile = commands.add_parser("profile")
    add_model_arguments(profile)
    profile.add_argument("--device", default="cpu")
    profile.add_argument("--output", type=Path)
    profile.set_defaults(func=profile_command)

    prior = commands.add_parser("fit-prior")
    add_runtime_arguments(prior)
    prior.add_argument("--domain", choices=("quasi", "mobility"), required=True)
    prior.add_argument("--target-blocks", type=csv_ints, default=(0,))
    prior.add_argument("--regularizations", type=csv_floats, default=(1e-5, 1e-4, 1e-3))
    prior.add_argument("--batch-size", type=int, default=64)
    prior.add_argument("--eval-batch-size", type=int, default=64)
    prior.add_argument("--seed", type=int, default=123)
    prior.add_argument("--max-train", type=int)
    prior.add_argument("--max-validation", type=int)
    prior.add_argument("--output", type=Path, required=True)
    prior.set_defaults(func=fit_prior_command)

    training = commands.add_parser("train")
    add_model_arguments(training)
    add_runtime_arguments(training)
    training.add_argument("--mode", choices=("smoke", "dev", "full"), default="dev")
    training.add_argument("--seed", type=int, default=123)
    training.add_argument("--batch-size", type=int, default=24)
    training.add_argument("--eval-batch-size", type=int, default=48)
    training.add_argument("--learning-rate", type=float, default=5e-4)
    training.add_argument("--weight-decay", type=float, default=1e-5)
    training.add_argument("--epochs", type=int)
    training.add_argument("--min-epochs", type=int, default=40)
    training.add_argument("--patience", type=int, default=15)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--charbonnier-weight", type=float, default=0.05)
    training.add_argument("--amp", action="store_true")
    training.add_argument("--target-blocks", type=csv_ints)
    training.add_argument("--prior", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--pretrained", type=Path)
    training.add_argument("--adaptation", choices=TRANSFER_PROTOCOLS + ("full",), default="full")
    training.add_argument("--fraction", type=float)
    training.add_argument("--sample-index-manifest", type=Path)
    training.add_argument("--stop-after-epoch", type=int, help="Graceful preemption point; does not alter the frozen training config.")
    training.add_argument("--run-name")
    training.add_argument("--output-root", type=Path, default=Path("runs"))
    training.set_defaults(func=train_command)

    tune = commands.add_parser("tune")
    add_runtime_arguments(tune)
    tune.add_argument("--domain", choices=("quasi", "mobility"), required=True)
    tune.add_argument("--seed", type=int, default=123)
    tune.add_argument("--prior", type=Path)
    tune.add_argument("--batch-size", type=int, default=32)
    tune.add_argument("--eval-batch-size", type=int, default=64)
    tune.add_argument("--execute", action="store_true")
    tune.add_argument("--study-name")
    tune.add_argument("--output-root", type=Path, default=Path("runs/tuning"))
    tune.set_defaults(func=tune_command)

    ablate = commands.add_parser("ablate")
    add_runtime_arguments(ablate)
    ablate.add_argument("--prior", type=Path)
    ablate.add_argument("--reference-checkpoint", type=Path)
    ablate.add_argument("--execute", action="store_true")
    ablate.add_argument("--study-name")
    ablate.add_argument("--output-root", type=Path, default=Path("runs/ablation"))
    ablate.set_defaults(func=ablate_command)

    transfer = commands.add_parser("transfer")
    add_runtime_arguments(transfer)
    transfer.add_argument("--source-checkpoint", type=Path)
    transfer.add_argument("--prior", type=Path)
    transfer.add_argument("--hidden", type=int, default=80)
    transfer.add_argument("--execute", action="store_true")
    transfer.add_argument("--study-name")
    transfer.add_argument("--output-root", type=Path, default=Path("runs/transfer"))
    transfer.set_defaults(func=transfer_command)

    evaluation = commands.add_parser("evaluate")
    add_runtime_arguments(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--prior", type=Path, help="Optional relocated copy of the checkpoint's exact Ridge artifact.")
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--freeze-manifest", type=Path)
    evaluation.add_argument("--batch-size", type=int, default=64)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.set_defaults(func=evaluate_command)

    imported = commands.add_parser("import-baselines")
    imported.add_argument("--source", type=Path, required=True)
    imported.add_argument("--output", type=Path, default=Path("external_results/baseline_manifest.json"))
    imported.add_argument("--require-checkpoints", action="store_true")
    imported.set_defaults(func=import_command)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    freeze.add_argument("--priors", type=Path, nargs="+", required=True)
    freeze.add_argument("--baseline-manifest", type=Path)
    freeze.add_argument("--unlock-test-after-freeze", action="store_true")
    freeze.add_argument("--confirm")
    freeze.add_argument("--output", type=Path, default=Path("runs/frozen_experiment_manifest.json"))
    freeze.set_defaults(func=freeze_command)

    report = commands.add_parser("report")
    report.add_argument("--runs-root", type=Path, default=Path("runs"))
    report.add_argument("--baseline-manifest", type=Path)
    report.add_argument("--output", type=Path, default=Path("reports/generated/results.json"))
    report.set_defaults(func=report_command)
    return root


def main() -> None:
    args = parser().parse_args()
    seed_everything(getattr(args, "seed", 123))
    args.func(args)
