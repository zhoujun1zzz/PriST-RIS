from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

from .anchor_cache import make_anchor_cache_loader, write_spatial_anchor_cache
from .checkpoint import load_checkpoint
from .complexity import profile_model
from .contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
    MODEL_ALIASES,
    MODEL_KEYS,
    DataSemantics,
    canonical_model_key,
)
from .data import (
    EXPECTED_MOBILITY_COUNTS,
    make_loader,
    nested_fraction_indices,
    resolve_dataset_source,
    validate_dataset_source,
    write_index_manifest,
)
from .engine import (
    TrainingConfig,
    evaluate,
    require_checkpoint_contract,
    seed_everything,
    train,
)
from .experiments import (
    CAPACITY_HIDDEN,
    LEARNING_RATES,
    MECHANISM_ABLATIONS,
    TEMPORAL_RANKS,
    TRANSFER_FRACTIONS,
    TRANSFER_PROTOCOLS,
    SPATIAL_ABLATION_TARGET_BLOCKS,
    TEMPORAL_ABLATION_TARGET_BLOCKS,
    late_window_score,
    targeted_tuning_plan,
    write_plan,
)
from .manifests import (
    freeze_experiment,
    import_baseline_manifest,
    validate_test_unlock,
)
from .metrics import MetricAccumulator, PerQueryMetricAccumulator
from .models import build_model
from .prior import RidgePrior, RidgeStatistics, file_sha256
from .screening import (
    SPATIAL_MODULE_CANDIDATES,
    SPATIAL_MODULE_REFERENCE_DB,
    TEMPORAL_MODULE_CANDIDATES,
    POSITION_SCREENING_CANDIDATES,
    SPATIAL_SCREENING_CANDIDATES,
    position_candidate_training_arguments,
    position_screening_plan,
    read_training_history,
    recommend_long_followup,
    should_extend_to_40,
    spatial_candidate_training_arguments,
    spatial_module_screening_plan,
    spatial_module_training_arguments,
    spatial_screening_plan,
    summarize_position_screening,
    summarize_spatial_modules,
    summarize_spatial_screening,
    summarize_temporal_modules,
    temporal_module_screening_plan,
    temporal_module_training_arguments,
)
from .temporal_audit import audit_temporal_loaders, write_temporal_audit


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
        if (
            frozen.get("architecture_version") != ARCHITECTURE_VERSION
            or frozen.get("test_unlocked") is not True
        ):
            raise PermissionError("The supplied freeze manifest does not unlock test.")
        splits.append("test")
    rows = []
    for domain in ("quasi", "mobility"):
        for split in splits:
            rows.append(validate_dataset_source(resolve_dataset_source(args.data_root, domain, split)))
    result = {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "test_included": "test" in splits,
        "files": rows,
    }
    _json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def profile_command(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    model = build_model(
        args.model,
        domain=args.domain,
        hidden=args.hidden,
        blocks_per_stage=args.blocks_per_stage,
        final_refine_blocks=args.final_refine_blocks,
        temporal_rank=args.temporal_rank,
        temporal_residual=not args.no_temporal_residual,
        spatial_multiscale_supervision=args.spatial_multiscale_supervision,
        spatial_channel_attention=args.spatial_channel_attention,
        coordinate_enabled=args.coordinate_enabled,
        backbone_ris_coordinate_enabled=args.backbone_ris_coordinate_enabled,
        backbone_antenna_index_enabled=args.backbone_antenna_index_enabled,
        backbone_ris_coordinate_mode=args.backbone_ris_coordinate_mode,
        attention_enabled=args.attention_enabled,
        attention_ris_coordinate_enabled=args.attention_ris_coordinate_enabled,
        attention_antenna_index_enabled=args.attention_antenna_index_enabled,
        observed_dense_attention_heads=args.observed_dense_attention_heads,
        spatial_residual_style=args.spatial_residual_style,
        temporal_mode=args.temporal_mode,
        temporal_base_mode=args.temporal_base_mode,
        temporal_learned_residual_enabled=args.temporal_learned_residual_enabled,
    ).to(device)
    result = profile_model(model, domain=args.domain, device=device)
    if args.output:
        _json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


def audit_temporal_command(args: argparse.Namespace) -> dict[str, object]:
    loaders = {
        split: make_loader(
            args.data_root,
            "mobility",
            split,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            max_samples=(args.max_train if split == "train" else args.max_validation),
            shuffle=False,
        )
        for split in ("train", "validation")
    }
    result = audit_temporal_loaders(loaders)
    result.update(
        {
            "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
            "train_samples_requested": args.max_train,
            "validation_samples_requested": args.max_validation,
        }
    )
    write_temporal_audit(result, args.output_json, args.output_csv)
    print(json.dumps(result, indent=2))
    return result


def cache_spatial_anchors_command(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    require_checkpoint_contract(
        state, "Spatial anchor cache", expected_domain="mobility"
    )
    model_config = dict(state["model_config"])
    if model_config.get("model_key") == "prist_ris_full":
        raise ValueError("Anchor cache requires a q0/q3 spatial checkpoint, not Full.")
    prior = RidgePrior.load(args.prior)
    semantics = DataSemantics.for_domain("mobility")
    if prior.semantics_hash != semantics.stable_hash() or prior.target_blocks != (0, 3):
        raise ValueError("Anchor cache Ridge prior must use Mobility q0/q3 semantics.")
    checkpoint_prior = state.get("prior_metadata")
    if isinstance(checkpoint_prior, dict) and checkpoint_prior.get("sha256") != file_sha256(
        args.prior
    ):
        raise ValueError("Checkpoint and anchor-cache Ridge SHA256 mismatch.")
    model = build_model(**model_config).to(device)
    model.load_state_dict(state["model_state"])
    outputs = []
    for split, maximum in (
        ("train", args.max_train),
        ("validation", args.max_validation),
    ):
        loader = make_loader(
            args.data_root,
            "mobility",
            split,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            max_samples=maximum,
            shuffle=False,
        )
        outputs.append(
            write_spatial_anchor_cache(
                Path(args.output_root) / f"{split}.h5",
                model=model,
                prior=prior,
                loader=loader,
                device=device,
                split=split,
                checkpoint_path=args.checkpoint,
                prior_path=args.prior,
            )
        )
    result = {
        "method": "PriST-RIS",
        "workflow": "cache_spatial_anchors",
        "caches": outputs,
        "test_split_used": False,
    }
    _json(Path(args.output_root) / "cache_manifest.json", result)
    print(json.dumps(result, indent=2))
    return result


def evaluate_temporal_cache_command(args: argparse.Namespace) -> dict[str, object]:
    loader = make_anchor_cache_loader(
        Path(args.anchor_cache_root) / "validation.h5",
        resolve_dataset_source(args.data_root, "mobility", "validation"),
        expected_checkpoint=args.spatial_checkpoint,
        expected_prior=args.prior,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        shuffle=False,
    )
    model = build_model(
        "prist_ris_full",
        domain="mobility",
        hidden=80,
        blocks_per_stage=(3, 3, 2),
        final_refine_blocks=1,
        backbone_ris_coordinate_enabled=True,
        backbone_antenna_index_enabled=False,
        backbone_ris_coordinate_mode="direct_add",
        attention_enabled=False,
        attention_ris_coordinate_enabled=False,
        attention_antenna_index_enabled=False,
        temporal_base_mode="linear_trend",
        temporal_learned_residual_enabled=False,
        temporal_residual=False,
    ).to(torch.device(args.device))
    result = evaluate(model, loader, torch.device(args.device), prior=None, target_blocks=None)
    payload = {
        "method": "PriST-RIS",
        "candidate": "T1_linear_trend",
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        **result,
        "test_split_used": False,
    }
    _json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return payload


def spatial_screen_command(args: argparse.Namespace) -> dict[str, object]:
    """Plan, execute, or summarize the fixed S1 Mobility-B Pareto screen."""

    root = Path(args.output_root) / (
        args.study_name or f"v32_s1_spatial_pareto_seed{args.seed}"
    )
    prior_value: str | Path = args.prior if args.prior is not None else "$PRIOR"
    plan = spatial_screening_plan(args.seed)
    commands = {
        candidate.name: [
            "prist-ris",
            *(
                str(value)
                for value in spatial_candidate_training_arguments(
                    candidate,
                    prior=prior_value,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                )
            ),
        ]
        for candidate in SPATIAL_SCREENING_CANDIDATES
    }
    plan["commands"] = commands
    plan["physical_gpu_preflight"] = f"nvidia-smi -i {args.physical_gpu_index}"
    write_plan(root / "screening_plan.json", plan)

    if args.summarize_only:
        summary = summarize_spatial_screening(
            root,
            reference_run=args.reference_run,
            reference_profile=args.reference_profile,
        )
        _json(root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return summary
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.prior is None:
        raise ValueError("Executing S1 screening requires the fixed q0/q3 --prior artifact.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
            raise PermissionError(
                f"Set CUDA_VISIBLE_DEVICES={args.physical_gpu_index} before executing S1."
            )
        if not args.confirm_gpu_free:
            raise PermissionError(
                "Inspect GPU 3 first, then pass --confirm-gpu-free to start the serial queue."
            )
    profiles = root / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for candidate in SPATIAL_SCREENING_CANDIDATES:
        run_dir = root / "runs" / candidate.name
        final_path = run_dir / "results" / "final_result.json"
        if final_path.is_file():
            print(f"reuse completed screening run: {run_dir}", flush=True)
        elif run_dir.exists():
            raise FileExistsError(
                f"Incomplete run exists and will not be overwritten: {run_dir}"
            )
        else:
            if device.type == "cuda":
                subprocess.run(
                    ["nvidia-smi", "-i", str(args.physical_gpu_index)], check=True
                )
            _invoke(
                spatial_candidate_training_arguments(
                    candidate,
                    prior=args.prior,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                ),
                dry_run=False,
            )
        profile_path = profiles / f"{candidate.name}.json"
        if not profile_path.is_file():
            model = build_model(
                "prist_ris_b",
                domain="mobility",
                hidden=candidate.hidden,
                blocks_per_stage=candidate.blocks_per_stage,
                final_refine_blocks=candidate.final_refine_blocks,
                backbone_ris_coordinate_enabled=False,
                backbone_antenna_index_enabled=False,
                backbone_ris_coordinate_mode="off",
                attention_enabled=False,
                attention_ris_coordinate_enabled=False,
                attention_antenna_index_enabled=False,
            ).to(device)
            _json(
                profile_path,
                profile_model(model, domain="mobility", device=device),
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = summarize_spatial_screening(
        root,
        reference_run=args.reference_run,
        reference_profile=args.reference_profile,
    )
    _json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def position_screen_command(args: argparse.Namespace) -> dict[str, object]:
    """Plan, execute, or summarize the factor-isolated P1-P3 screen."""

    root = Path(args.output_root) / (
        args.study_name or f"position_semantics_p1_p3_seed{args.seed}"
    )
    prior_value: str | Path = args.prior if args.prior is not None else "$PRIOR"
    plan = position_screening_plan(args.seed)
    plan["commands"] = {
        candidate.name: [
            "prist-ris",
            *(
                str(value)
                for value in position_candidate_training_arguments(
                    candidate,
                    prior=prior_value,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                )
            ),
        ]
        for candidate in POSITION_SCREENING_CANDIDATES
    }
    plan["physical_gpu_preflight"] = f"nvidia-smi -i {args.physical_gpu_index}"
    write_plan(root / "screening_plan.json", plan)
    if args.summarize_only:
        summary = summarize_position_screening(root)
        _json(root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return summary
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.prior is None:
        raise ValueError("Executing P1-P3 requires the fixed q0/q3 --prior artifact.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
            raise PermissionError(
                f"Set CUDA_VISIBLE_DEVICES={args.physical_gpu_index} before executing P1-P3."
            )
        if not args.confirm_gpu_free:
            raise PermissionError(
                "Inspect the physical GPU first, then pass --confirm-gpu-free."
            )
    profiles = root / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for candidate in POSITION_SCREENING_CANDIDATES:
        run_dir = root / "runs" / candidate.name
        final_path = run_dir / "results" / "final_result.json"
        if final_path.is_file():
            print(f"reuse completed position run: {run_dir}", flush=True)
        elif run_dir.exists():
            raise FileExistsError(
                f"Incomplete run exists and will not be overwritten: {run_dir}"
            )
        else:
            if device.type == "cuda":
                subprocess.run(
                    ["nvidia-smi", "-i", str(args.physical_gpu_index)], check=True
                )
            _invoke(
                position_candidate_training_arguments(
                    candidate,
                    prior=args.prior,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                ),
                dry_run=False,
            )
        profile_path = profiles / f"{candidate.name}.json"
        if not profile_path.is_file():
            model = build_model(
                "prist_ris_b",
                domain="mobility",
                backbone_ris_coordinate_enabled=candidate.backbone_ris_coordinate_enabled,
                backbone_antenna_index_enabled=False,
                backbone_ris_coordinate_mode=candidate.backbone_ris_coordinate_mode,
                attention_enabled=candidate.attention_enabled,
                attention_ris_coordinate_enabled=(
                    candidate.attention_ris_coordinate_enabled
                ),
                attention_antenna_index_enabled=False,
            ).to(device)
            _json(profile_path, profile_model(model, domain="mobility", device=device))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = summarize_position_screening(root)
    _json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def _module_gpu_preflight(args: argparse.Namespace) -> None:
    if torch.device(args.device).type != "cuda":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
        raise PermissionError(
            f"Set CUDA_VISIBLE_DEVICES={args.physical_gpu_index} before execution."
        )
    if not args.confirm_gpu_free:
        raise PermissionError("Inspect the physical GPU and pass --confirm-gpu-free.")
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(args.physical_gpu_index),
            "--query-compute-apps=pid", "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            f"GPU {args.physical_gpu_index} already has compute PID(s); queue stopped."
        )


def _completed_epoch(run_dir: Path) -> int:
    final = run_dir / "results" / "final_result.json"
    if not final.is_file():
        if run_dir.exists():
            raise FileExistsError(f"Incomplete run will not be overwritten: {run_dir}")
        return 0
    history = read_training_history(run_dir)
    return max(int(row["epoch"]) for row in history)


def spatial_modules_screen_command(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_root) / (
        args.study_name or f"s2_s3_modules_seed{args.seed}"
    )
    prior_value: str | Path = args.prior if args.prior is not None else "$PRIOR"
    plan = spatial_module_screening_plan(args.seed)
    plan["commands"] = {
        candidate.name: [
            "prist-ris",
            *map(
                str,
                spatial_module_training_arguments(
                    candidate,
                    prior=prior_value,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                ),
            ),
        ]
        for candidate in SPATIAL_MODULE_CANDIDATES
    }
    write_plan(root / "screening_plan.json", plan)
    if args.summarize_only:
        summary = summarize_spatial_modules(root)
        _json(root / "summary.json", summary)
        return summary
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.prior is None:
        raise ValueError("Spatial module screening requires --prior.")
    _module_gpu_preflight(args)
    profiles = root / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for candidate in SPATIAL_MODULE_CANDIDATES:
        run = root / "runs" / candidate.name
        completed = _completed_epoch(run)
        if completed == 0:
            _module_gpu_preflight(args)
            _invoke(
                spatial_module_training_arguments(
                    candidate,
                    prior=args.prior,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                ),
                dry_run=False,
            )
            completed = 30
        history = read_training_history(run)
        if completed == 30 and should_extend_to_40(
            history, reference_db=SPATIAL_MODULE_REFERENCE_DB
        ):
            _module_gpu_preflight(args)
            _invoke(
                spatial_module_training_arguments(
                    candidate,
                    prior=args.prior,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                    stop_epoch=40,
                    resume=run / "checkpoints" / "last_checkpoint.pth",
                ),
                dry_run=False,
            )
            history = read_training_history(run)
            completed = 40
        if (
            args.run_long_followups
            and completed == 40
            and recommend_long_followup(
                history, reference_db=SPATIAL_MODULE_REFERENCE_DB
            )
        ):
            _module_gpu_preflight(args)
            _invoke(
                spatial_module_training_arguments(
                    candidate,
                    prior=args.prior,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                    stop_epoch=100,
                    epochs=100,
                    resume=run / "checkpoints" / "last_checkpoint.pth",
                ),
                dry_run=False,
            )
        profile_path = profiles / f"{candidate.name}.json"
        if not profile_path.is_file():
            model = build_model(
                "prist_ris_b", domain="mobility", hidden=80,
                blocks_per_stage=(3, 3, 2), final_refine_blocks=1,
                backbone_ris_coordinate_enabled=True,
                backbone_antenna_index_enabled=False,
                backbone_ris_coordinate_mode="direct_add",
                attention_enabled=False,
                attention_ris_coordinate_enabled=False,
                attention_antenna_index_enabled=False,
                spatial_multiscale_supervision=candidate.multiscale,
                spatial_channel_attention=candidate.channel_attention,
            ).to(torch.device(args.device))
            _json(profile_path, profile_model(model, domain="mobility", device=torch.device(args.device)))
    summary = summarize_spatial_modules(root)
    summary["long_followups_executed"] = bool(args.run_long_followups)
    _json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def temporal_modules_screen_command(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_root) / (
        args.study_name or f"t1_t4_modules_seed{args.seed}"
    )
    plan = temporal_module_screening_plan(
        args.seed, include_curvature=args.include_curvature
    )
    placeholder_prior: str | Path = args.prior or "$PRIOR"
    placeholder_checkpoint: str | Path = args.spatial_checkpoint or "$SPATIAL_CHECKPOINT"
    placeholder_cache: str | Path = args.anchor_cache_root or "$ANCHOR_CACHE_ROOT"
    plan["commands"] = {
        candidate.name: [
            "prist-ris",
            *map(
                str,
                temporal_module_training_arguments(
                    candidate,
                    prior=placeholder_prior,
                    spatial_checkpoint=placeholder_checkpoint,
                    anchor_cache_root=placeholder_cache,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                ),
            ),
        ]
        for candidate in TEMPORAL_MODULE_CANDIDATES
        if args.include_curvature or candidate.name != "T4_curvature"
    }
    write_plan(root / "screening_plan.json", plan)
    if args.summarize_only:
        summary = summarize_temporal_modules(root)
        _json(root / "summary.json", summary)
        return summary
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    if args.prior is None or args.spatial_checkpoint is None or args.anchor_cache_root is None:
        raise ValueError(
            "Temporal screening requires --prior, --spatial-checkpoint, and --anchor-cache-root."
        )
    _module_gpu_preflight(args)
    t1_path = root / "T1_linear_trend.json"
    if not t1_path.is_file():
        _invoke(
            [
                "evaluate-temporal-cache", "--prior", args.prior,
                "--spatial-checkpoint", args.spatial_checkpoint,
                "--anchor-cache-root", args.anchor_cache_root,
                "--data-root", args.data_root, "--device", args.device,
                "--workers", args.workers, "--seed", args.seed,
                "--output", t1_path,
            ],
            dry_run=False,
        )
    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    reference_db = float(t1["nmse_db"])
    profiles = root / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    t1_profile = profiles / "T1_linear_trend.json"
    if not t1_profile.is_file():
        model = build_model(
            "prist_ris_full", domain="mobility", hidden=80,
            blocks_per_stage=(3, 3, 2), final_refine_blocks=1,
            backbone_ris_coordinate_enabled=True,
            backbone_antenna_index_enabled=False,
            backbone_ris_coordinate_mode="direct_add",
            attention_enabled=False,
            attention_ris_coordinate_enabled=False,
            attention_antenna_index_enabled=False,
            temporal_base_mode="linear_trend",
            temporal_learned_residual_enabled=False,
            temporal_residual=False,
        ).to(torch.device(args.device))
        _json(
            t1_profile,
            profile_model(
                model, domain="mobility", device=torch.device(args.device)
            ),
        )
    candidates = [
        candidate
        for candidate in TEMPORAL_MODULE_CANDIDATES
        if args.include_curvature or candidate.name != "T4_curvature"
    ]
    for candidate in candidates:
        run = root / "runs" / candidate.name
        completed = _completed_epoch(run)
        if completed == 0:
            _module_gpu_preflight(args)
            _invoke(
                temporal_module_training_arguments(
                    candidate,
                    prior=args.prior,
                    spatial_checkpoint=args.spatial_checkpoint,
                    anchor_cache_root=args.anchor_cache_root,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                ),
                dry_run=False,
            )
            completed = 30
        history = read_training_history(run)
        if completed == 30 and should_extend_to_40(history, reference_db=reference_db):
            _module_gpu_preflight(args)
            _invoke(
                temporal_module_training_arguments(
                    candidate,
                    prior=args.prior,
                    spatial_checkpoint=args.spatial_checkpoint,
                    anchor_cache_root=args.anchor_cache_root,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                    stop_epoch=40,
                    resume=run / "checkpoints" / "last_checkpoint.pth",
                ),
                dry_run=False,
            )
            history = read_training_history(run)
            completed = 40
        if (
            args.run_long_followups
            and completed == 40
            and recommend_long_followup(history, reference_db=reference_db)
        ):
            _module_gpu_preflight(args)
            _invoke(
                temporal_module_training_arguments(
                    candidate,
                    prior=args.prior,
                    spatial_checkpoint=args.spatial_checkpoint,
                    anchor_cache_root=args.anchor_cache_root,
                    data_root=args.data_root,
                    output_root=root / "runs",
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                    stop_epoch=100,
                    epochs=100,
                    resume=run / "checkpoints" / "last_checkpoint.pth",
                ),
                dry_run=False,
            )
        profile_path = profiles / f"{candidate.name}.json"
        if not profile_path.is_file():
            model = build_model(
                "prist_ris_full", domain="mobility", hidden=80,
                blocks_per_stage=(3, 3, 2), final_refine_blocks=1,
                backbone_ris_coordinate_enabled=True,
                backbone_antenna_index_enabled=False,
                backbone_ris_coordinate_mode="direct_add",
                attention_enabled=False,
                attention_ris_coordinate_enabled=False,
                attention_antenna_index_enabled=False,
                temporal_base_mode="linear_trend",
                temporal_learned_residual_enabled=True,
            ).to(torch.device(args.device))
            _json(profile_path, profile_model(model, domain="mobility", device=torch.device(args.device)))
    summary = summarize_temporal_modules(root)
    summary["long_followups_executed"] = bool(args.run_long_followups)
    _json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def _evaluate_prior(prior: RidgePrior, loader: Iterable[dict[str, torch.Tensor]]) -> dict[str, object]:
    metrics = MetricAccumulator()
    diagnostics = PerQueryMetricAccumulator(prior.target_blocks)
    blocks = torch.tensor(prior.target_blocks)
    for batch in loader:
        prediction = prior.predict(batch)
        target = batch["target_h"].index_select(1, blocks)
        metrics.update(prediction, target)
        diagnostics.update(prediction, target)
    return {**metrics.compute(), "diagnostics": diagnostics.compute()}


def fit_prior_command(args: argparse.Namespace) -> dict[str, object]:
    target_blocks = args.target_blocks or ((0,) if args.domain == "quasi" else (0, 3))
    allowed = tuple(range(1 if args.domain == "quasi" else 6))
    if not target_blocks or any(value not in allowed for value in target_blocks):
        raise ValueError(f"Invalid target blocks for {args.domain}: {target_blocks}")
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
    winner: tuple[float, RidgePrior, dict[str, object]] | None = None
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
        "architecture_version": ARCHITECTURE_VERSION,
        "mobility_contract_version": (
            MOBILITY_CONTRACT_VERSION if args.domain == "mobility" else None
        ),
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
    if args.anchor_cache_root is not None:
        if (
            args.domain != "mobility"
            or key != "prist_ris_full"
            or args.spatial_reference_checkpoint is None
            or args.prior is None
        ):
            raise ValueError(
                "Anchor-cache training requires Mobility Full, --prior, and "
                "--spatial-reference-checkpoint."
            )
        cache_root = Path(args.anchor_cache_root)
        train_loader = make_anchor_cache_loader(
            cache_root / "train.h5",
            resolve_dataset_source(args.data_root, "mobility", "train"),
            expected_checkpoint=args.spatial_reference_checkpoint,
            expected_prior=args.prior,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            shuffle=True,
        )
        validation_loader = make_anchor_cache_loader(
            cache_root / "validation.h5",
            resolve_dataset_source(args.data_root, "mobility", "validation"),
            expected_checkpoint=args.spatial_reference_checkpoint,
            expected_prior=args.prior,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            seed=args.seed,
            shuffle=False,
        )
        indices = list(train_loader.dataset.source_dataset.indices)  # type: ignore[attr-defined]
    else:
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
    if target_blocks is None and args.domain == "mobility" and key != "prist_ris_full":
        target_blocks = (0, 3)
    config = TrainingConfig(
        domain=args.domain,
        model_key=key,
        mode=args.mode,
        seed=args.seed,
        hidden=args.hidden,
        blocks_per_stage=args.blocks_per_stage,
        final_refine_blocks=args.final_refine_blocks,
        temporal_rank=args.temporal_rank,
        temporal_residual=not args.no_temporal_residual,
        spatial_multiscale_supervision=args.spatial_multiscale_supervision,
        spatial_channel_attention=args.spatial_channel_attention,
        coordinate_enabled=args.coordinate_enabled,
        backbone_ris_coordinate_enabled=args.backbone_ris_coordinate_enabled,
        backbone_antenna_index_enabled=args.backbone_antenna_index_enabled,
        backbone_ris_coordinate_mode=args.backbone_ris_coordinate_mode,
        attention_enabled=args.attention_enabled,
        attention_ris_coordinate_enabled=args.attention_ris_coordinate_enabled,
        attention_antenna_index_enabled=args.attention_antenna_index_enabled,
        observed_dense_attention_heads=args.observed_dense_attention_heads,
        spatial_residual_style=args.spatial_residual_style,
        temporal_mode=args.temporal_mode,
        temporal_base_mode=args.temporal_base_mode,
        temporal_learned_residual_enabled=args.temporal_learned_residual_enabled,
        temporal_delta_loss_weight=args.temporal_delta_loss_weight,
        temporal_curvature_loss_weight=args.temporal_curvature_loss_weight,
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
    run_name = args.run_name or f"v32_{args.domain}_{key}_{args.mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    return train(
        config,
        train_loader,
        validation_loader,
        run_dir=Path(args.output_root) / run_name,
        device=torch.device(args.device),
        prior_path=args.prior,
        resume=args.resume,
        pretrained=args.pretrained,
        spatial_reference=args.spatial_reference_checkpoint,
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
        "architecture_version": ARCHITECTURE_VERSION,
        "domain": args.domain,
        "test_split_used": False,
        "capacity_ranking": capacity,
        "learning_rate_ranking": learning_rates,
        "rank_ranking": ranks,
        "best_hyperparameters": {
            "hidden": selected_hidden,
            "learning_rate": selected_lr,
            "temporal_rank": selected_rank,
            "blocks_per_stage": [3, 3, 4],
            "final_refine_blocks": 4,
            "weight_decay": 1e-5,
        },
    }
    _json(root / "best_result.json", result)
    return result


def ablate_command(args: argparse.Namespace) -> dict[str, object]:
    plan = {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "domain": "mobility",
        "seed": 123,
        "mechanisms": list(MECHANISM_ABLATIONS),
        "spatial_table": {
            "variants": [
                "physical_grid_spatial",
                "prior_guided_dual_anchor",
                "coordinate_observed_dense_attention",
            ],
            "target_blocks": list(SPATIAL_ABLATION_TARGET_BLOCKS),
            "score_column": "observed_anchor_aggregate",
        },
        "temporal_table": {
            "variants": [
                "static_last_anchor",
                "no_delta_conditioning",
                "trend_conditioned_temporal",
                "full",
            ],
            "target_blocks": list(TEMPORAL_ABLATION_TARGET_BLOCKS),
            "score_column": "overall",
        },
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
    require_checkpoint_contract(
        reference, "Ablation reference", expected_domain="mobility"
    )
    if (
        reference.get("method") != "PriST-RIS"
        or reference.get("architecture_version") != ARCHITECTURE_VERSION
        or reference.get("model_config", {}).get("model_key") != "prist_ris_full"
    ):
        raise ValueError("Ablation reference must be the frozen PriST-RIS V3.2 Full seed-123 checkpoint.")
    frozen = reference.get("training_config", {})
    if frozen.get("domain") != "mobility" or int(frozen.get("seed", -1)) != 123:
        raise ValueError("Ablation reference must use Mobility and seed 123.")
    mapping = {
        "physical_grid_spatial": ("prist_ris_a", "0,3", "trend", True),
        "prior_guided_dual_anchor": ("prist_ris_b", "0,3", "trend", True),
        "coordinate_observed_dense_attention": (
            "prist_ris_c", "0,3", "trend", True
        ),
        "static_last_anchor": ("prist_ris_full", "0,1,2,3,4,5", "static", True),
        "no_delta_conditioning": ("prist_ris_full", "0,1,2,3,4,5", "no_delta", True),
        "trend_conditioned_temporal": ("prist_ris_full", "0,1,2,3,4,5", "trend", True),
    }
    rows = []
    for variant, (model, target_scope, temporal_mode, no_residual) in mapping.items():
        arguments: list[object] = [
            "train", "--domain", "mobility", "--model", model, "--mode", "full",
            "--seed", 123, "--data-root", args.data_root,
            "--device", args.device, "--run-name", variant, "--output-root", root / "trials",
            "--hidden", frozen["hidden"],
            "--blocks-per-stage", ",".join(str(v) for v in frozen["blocks_per_stage"]),
            "--final-refine-blocks", frozen["final_refine_blocks"],
            "--temporal-rank", frozen["temporal_rank"],
            "--observed-dense-attention-heads", frozen.get("observed_dense_attention_heads", 4),
            "--spatial-residual-style", frozen.get("spatial_residual_style", "scaled_true_residual"),
            "--temporal-mode", temporal_mode,
            "--target-blocks", target_scope,
            "--learning-rate", frozen["learning_rate"],
            "--weight-decay", frozen["weight_decay"],
            "--epochs", frozen["epochs"], "--min-epochs", frozen["min_epochs"],
            "--patience", frozen["patience"],
        ]
        if model != "prist_ris_a":
            arguments.extend(["--prior", args.prior])
        if no_residual:
            arguments.append("--no-temporal-residual")
        _invoke(arguments, dry_run=False)
        final = json.loads((root / "trials" / variant / "results" / "final_result.json").read_text())
        rows.append({"variant": variant, "status": "trained", "result": final})
    rows.append(
        {
            "variant": "full",
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
        "runs": 20,
        "transfer_scope": "spatial_only",
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
    require_checkpoint_contract(
        source, "Transfer source", expected_domain="quasi"
    )
    source_config = source.get("training_config", {})
    source_model = source.get("model_config", {})
    if (
        source.get("method") != "PriST-RIS"
        or source.get("architecture_version") != ARCHITECTURE_VERSION
        or source_config.get("domain") != "quasi"
    ):
        raise ValueError("Transfer source must be a Quasi PriST-RIS V3.2 checkpoint.")
    if source_model.get("model_key") not in {"prist_ris_c", "prist_ris_full"}:
        raise ValueError("Transfer source must use coordinate-enabled Quasi spatial weights.")
    for fraction in TRANSFER_FRACTIONS:
        for protocol in TRANSFER_PROTOCOLS:
            name = f"fraction_{fraction:.2f}_{protocol}"
            arguments: list[object] = [
                "train", "--domain", "mobility", "--model", "prist_ris_full",
                "--mode", "full", "--seed", 123, "--fraction", fraction,
                "--sample-index-manifest", manifest, "--adaptation", protocol,
                "--prior", args.prior, "--data-root", args.data_root,
                "--device", args.device, "--hidden", source_config["hidden"],
                "--blocks-per-stage", ",".join(str(v) for v in source_config["blocks_per_stage"]),
                "--final-refine-blocks", source_config["final_refine_blocks"],
                "--temporal-rank", source_config["temporal_rank"],
                "--observed-dense-attention-heads", source_config.get("observed_dense_attention_heads", 4),
                "--spatial-residual-style", source_config.get("spatial_residual_style", "scaled_true_residual"),
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
        raise ValueError("Evaluation requires a PriST-RIS V3.2 checkpoint; legacy versions are not loaded silently.")
    model_config = dict(state["model_config"])
    require_checkpoint_contract(
        state, "Evaluation", expected_domain=str(model_config["domain"])
    )
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
        "architecture_version": ARCHITECTURE_VERSION,
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
    legacy_rows = []
    for path in Path(args.runs_root).rglob("final_result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("method") == "PriST-RIS":
            row = {"path": str(path.resolve()), **value}
            if value.get("architecture_version") == ARCHITECTURE_VERSION:
                rows.append(row)
            else:
                legacy_rows.append(row)
    baselines = None
    if args.baseline_manifest and args.baseline_manifest.is_file():
        baselines = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    result = {
        "method": "PriST-RIS",
        "architecture_version": ARCHITECTURE_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "validation_runs": rows,
        "legacy_runs_preserved": legacy_rows,
        "external_baselines": baselines,
        "test_results_included": any(row.get("split") == "test" for row in rows),
    }
    _json(args.output, result)
    return result


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=ALL_MODEL_KEYS, default="prist_ris_full")
    parser.add_argument("--domain", choices=("quasi", "mobility"), required=True)
    parser.add_argument("--hidden", type=int, default=80)
    parser.add_argument("--blocks-per-stage", type=csv_ints, default=(3, 3, 4))
    parser.add_argument("--final-refine-blocks", type=int, default=4)
    parser.add_argument("--temporal-rank", type=int, choices=(2, 3), default=2)
    parser.add_argument("--no-temporal-residual", action="store_true")
    parser.add_argument(
        "--spatial-multiscale-supervision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--spatial-channel-attention", choices=("off", "se"), default="off"
    )
    parser.add_argument(
        "--coordinate-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility alias coupling all available position paths.",
    )
    parser.add_argument(
        "--backbone-ris-coordinate-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--backbone-antenna-index-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--backbone-ris-coordinate-mode",
        choices=("off", "direct_add", "zero_init_gated"),
        default=None,
    )
    parser.add_argument(
        "--attention-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--attention-ris-coordinate-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--attention-antenna-index-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--observed-dense-attention-heads", type=int, default=4)
    parser.add_argument(
        "--spatial-residual-style",
        choices=("post_activation", "scaled_true_residual"),
        default="scaled_true_residual",
        help="Canonical V3.2 true residual; post_activation remains a legacy ablation only.",
    )
    parser.add_argument("--temporal-mode", choices=("trend", "no_delta", "static"), default="trend")
    parser.add_argument(
        "--temporal-base-mode", choices=("static", "linear_trend"), default=None
    )
    parser.add_argument(
        "--temporal-learned-residual-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


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

    temporal_audit = commands.add_parser("audit-temporal")
    add_runtime_arguments(temporal_audit)
    temporal_audit.add_argument("--seed", type=int, default=123)
    temporal_audit.add_argument("--batch-size", type=int, default=16)
    temporal_audit.add_argument("--max-train", type=int)
    temporal_audit.add_argument("--max-validation", type=int)
    temporal_audit.add_argument(
        "--output-json", type=Path, default=Path("reports/generated/temporal_audit.json")
    )
    temporal_audit.add_argument(
        "--output-csv", type=Path, default=Path("reports/generated/temporal_audit.csv")
    )
    temporal_audit.set_defaults(func=audit_temporal_command)

    profile = commands.add_parser("profile")
    add_model_arguments(profile)
    profile.add_argument("--device", default="cpu")
    profile.add_argument("--output", type=Path)
    profile.set_defaults(func=profile_command)

    prior = commands.add_parser("fit-prior")
    add_runtime_arguments(prior)
    prior.add_argument("--domain", choices=("quasi", "mobility"), required=True)
    prior.add_argument("--target-blocks", type=csv_ints)
    prior.add_argument("--regularizations", type=csv_floats, default=(1e-5, 1e-4, 1e-3))
    prior.add_argument("--batch-size", type=int, default=64)
    prior.add_argument("--eval-batch-size", type=int, default=64)
    prior.add_argument("--seed", type=int, default=123)
    prior.add_argument("--max-train", type=int)
    prior.add_argument("--max-validation", type=int)
    prior.add_argument("--output", type=Path, required=True)
    prior.set_defaults(func=fit_prior_command)

    anchor_cache = commands.add_parser("cache-spatial-anchors")
    add_runtime_arguments(anchor_cache)
    anchor_cache.add_argument("--checkpoint", type=Path, required=True)
    anchor_cache.add_argument("--prior", type=Path, required=True)
    anchor_cache.add_argument("--seed", type=int, default=123)
    anchor_cache.add_argument("--batch-size", type=int, default=16)
    anchor_cache.add_argument("--max-train", type=int)
    anchor_cache.add_argument("--max-validation", type=int)
    anchor_cache.add_argument("--output-root", type=Path, required=True)
    anchor_cache.set_defaults(func=cache_spatial_anchors_command)

    temporal_cache_eval = commands.add_parser("evaluate-temporal-cache")
    add_runtime_arguments(temporal_cache_eval)
    temporal_cache_eval.add_argument("--prior", type=Path, required=True)
    temporal_cache_eval.add_argument("--spatial-checkpoint", type=Path, required=True)
    temporal_cache_eval.add_argument("--anchor-cache-root", type=Path, required=True)
    temporal_cache_eval.add_argument("--seed", type=int, default=123)
    temporal_cache_eval.add_argument("--batch-size", type=int, default=32)
    temporal_cache_eval.add_argument("--output", type=Path, required=True)
    temporal_cache_eval.set_defaults(func=evaluate_temporal_cache_command)

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
    training.add_argument("--temporal-delta-loss-weight", type=float, default=0.0)
    training.add_argument("--temporal-curvature-loss-weight", type=float, default=0.0)
    training.add_argument("--amp", action="store_true")
    training.add_argument("--target-blocks", type=csv_ints)
    training.add_argument("--prior", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--pretrained", type=Path)
    training.add_argument("--spatial-reference-checkpoint", type=Path)
    training.add_argument("--anchor-cache-root", type=Path)
    training.add_argument(
        "--adaptation", choices=TRANSFER_PROTOCOLS + ("temporal_only", "full"), default="full"
    )
    training.add_argument("--fraction", type=float)
    training.add_argument("--sample-index-manifest", type=Path)
    training.add_argument("--stop-after-epoch", type=int, help="Graceful preemption point; does not alter the frozen training config.")
    training.add_argument("--run-name")
    training.add_argument("--output-root", type=Path, default=Path("runs/v3_2_dev"))
    training.set_defaults(func=train_command)

    screening = commands.add_parser("screen-spatial")
    add_runtime_arguments(screening)
    screening.add_argument("--prior", type=Path)
    screening.add_argument("--seed", type=int, default=123)
    screening.add_argument("--physical-gpu-index", type=int, default=3)
    screening.add_argument("--confirm-gpu-free", action="store_true")
    action = screening.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--summarize-only", action="store_true")
    screening.add_argument("--reference-run", type=Path)
    screening.add_argument("--reference-profile", type=Path)
    screening.add_argument("--study-name")
    screening.add_argument(
        "--output-root", type=Path, default=Path("runs/spatial_screening")
    )
    screening.set_defaults(func=spatial_screen_command)

    position_screening = commands.add_parser("screen-position")
    add_runtime_arguments(position_screening)
    position_screening.add_argument("--prior", type=Path)
    position_screening.add_argument("--seed", type=int, default=123)
    position_screening.add_argument("--physical-gpu-index", type=int, default=3)
    position_screening.add_argument("--confirm-gpu-free", action="store_true")
    position_action = position_screening.add_mutually_exclusive_group()
    position_action.add_argument("--execute", action="store_true")
    position_action.add_argument("--summarize-only", action="store_true")
    position_screening.add_argument("--study-name")
    position_screening.add_argument(
        "--output-root", type=Path, default=Path("runs/position_screening")
    )
    position_screening.set_defaults(func=position_screen_command)

    spatial_modules = commands.add_parser("screen-spatial-modules")
    add_runtime_arguments(spatial_modules)
    spatial_modules.add_argument("--prior", type=Path)
    spatial_modules.add_argument("--seed", type=int, default=123)
    spatial_modules.add_argument("--physical-gpu-index", type=int, default=3)
    spatial_modules.add_argument("--confirm-gpu-free", action="store_true")
    spatial_modules.add_argument("--execute", action="store_true")
    spatial_modules.add_argument("--summarize-only", action="store_true")
    spatial_modules.add_argument("--run-long-followups", action="store_true")
    spatial_modules.add_argument("--study-name")
    spatial_modules.add_argument(
        "--output-root", type=Path, default=Path("runs/spatial_modules")
    )
    spatial_modules.set_defaults(func=spatial_modules_screen_command)

    temporal_modules = commands.add_parser("screen-temporal")
    add_runtime_arguments(temporal_modules)
    temporal_modules.add_argument("--prior", type=Path)
    temporal_modules.add_argument("--spatial-checkpoint", type=Path)
    temporal_modules.add_argument("--anchor-cache-root", type=Path)
    temporal_modules.add_argument("--seed", type=int, default=123)
    temporal_modules.add_argument("--physical-gpu-index", type=int, default=3)
    temporal_modules.add_argument("--confirm-gpu-free", action="store_true")
    temporal_modules.add_argument("--execute", action="store_true")
    temporal_modules.add_argument("--summarize-only", action="store_true")
    temporal_modules.add_argument("--include-curvature", action="store_true")
    temporal_modules.add_argument("--run-long-followups", action="store_true")
    temporal_modules.add_argument("--study-name")
    temporal_modules.add_argument(
        "--output-root", type=Path, default=Path("runs/temporal_modules")
    )
    temporal_modules.set_defaults(func=temporal_modules_screen_command)

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
