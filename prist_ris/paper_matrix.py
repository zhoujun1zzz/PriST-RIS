from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from .checkpoint import load_checkpoint
from .contracts import (
    ARCHITECTURE_VERSION,
    MOBILITY_CONTRACT_VERSION,
    POSITION_SEMANTICS_VERSION,
    SPATIAL_PROTOCOL_VERSION,
    SPATIAL_SUPERVISION_PROTOCOL_VERSION,
    TEMPORAL_PROTOCOL_VERSION,
    DataSemantics,
)
from .data import EXPECTED_MOBILITY_COUNTS, nested_fraction_indices
from .engine import require_checkpoint_contract
from .prior import RidgePrior, file_sha256


PAPER_MATRIX_SCHEMA = "prist_ris.paper_matrix.v1"
PAPER_FROZEN_PROTOCOL_SCHEMA = "prist_ris.paper_frozen_protocol.v1"
PAPER_SEEDS = (123, 456, 789)
PAPER_DATA_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
PAPER_TRANSFER_FRACTIONS = (0.05, 0.10, 0.25)
PAPER_TRANSFER_PROTOCOLS = ("scratch", "full_finetune", "selective")
CONVERGENCE_THRESHOLDS_DB = (-18.0, -19.0, -20.0)
PAPER_PHASES = ("data-efficiency", "transfer")

PAPER_SPATIAL_CONFIG: dict[str, object] = {
    "hidden": 80,
    "blocks_per_stage": [3, 3, 2],
    "final_refine_blocks": 1,
    "backbone_ris_coordinate_enabled": True,
    "backbone_ris_coordinate_mode": "direct_add",
    "backbone_antenna_index_enabled": False,
    "attention_enabled": False,
    "attention_ris_coordinate_enabled": False,
    "attention_antenna_index_enabled": False,
    "spatial_multiscale_supervision": False,
    "spatial_channel_attention": "se",
    "spatial_residual_style": "scaled_true_residual",
}
PAPER_OPTIMIZER_CONFIG: dict[str, object] = {
    "name": "AdamW",
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "min_learning_rate": 5e-6,
    "epochs": 100,
    "batch_size": 32,
    "eval_batch_size": 64,
    "amp": False,
}


@dataclass(frozen=True)
class ExperimentSpec:
    phase: str
    name: str
    domain: str
    seed: int
    fraction: float
    method_variant: str
    model_key: str
    prior_path: str | None
    pretrained_checkpoint: str | None
    sample_manifest: str
    run_dir: str
    training_config: dict[str, object]
    dependencies: tuple[str, ...]
    target_scope: str = "mobility_q0_q3"
    test_split_used: bool = False

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["dependencies"] = list(self.dependencies)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExperimentSpec":
        return cls(
            phase=str(value["phase"]),
            name=str(value["name"]),
            domain=str(value["domain"]),
            seed=int(value["seed"]),
            fraction=float(value["fraction"]),
            method_variant=str(value["method_variant"]),
            model_key=str(value["model_key"]),
            prior_path=(str(value["prior_path"]) if value.get("prior_path") else None),
            pretrained_checkpoint=(
                str(value["pretrained_checkpoint"])
                if value.get("pretrained_checkpoint")
                else None
            ),
            sample_manifest=str(value["sample_manifest"]),
            run_dir=str(value["run_dir"]),
            training_config=dict(value["training_config"]),  # type: ignore[arg-type]
            dependencies=tuple(str(item) for item in value.get("dependencies", [])),
            target_scope=str(value.get("target_scope", "mobility_q0_q3")),
            test_split_used=bool(value.get("test_split_used", False)),
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def indices_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_json(list(indices)).encode("utf-8")).hexdigest()


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise FileExistsError(f"Existing reproducibility artifact differs: {path}")
        return
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def frozen_protocol_manifest(git_head: str) -> dict[str, object]:
    return {
        "schema": PAPER_FROZEN_PROTOCOL_SCHEMA,
        "method": "PriST-RIS",
        "architecture_frozen": True,
        "architecture_version": ARCHITECTURE_VERSION,
        "git_head": git_head,
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "mobility_contract_version": MOBILITY_CONTRACT_VERSION,
        "spatial_protocol_version": SPATIAL_PROTOCOL_VERSION,
        "position_semantics_version": POSITION_SEMANTICS_VERSION,
        "spatial_supervision_protocol_version": SPATIAL_SUPERVISION_PROTOCOL_VERSION,
        "temporal_protocol_version": TEMPORAL_PROTOCOL_VERSION,
        "spatial": PAPER_SPATIAL_CONFIG,
        "optimizer": PAPER_OPTIMIZER_CONFIG,
        "temporal": {
            "preferred": "trend_residual",
            "final_checkpoint": None,
            "anchor_cache": None,
            "status": "pending_final_S3_cache_T2_validation",
        },
        "transfer": {"final_quasi_checkpoint": None},
        "selection_split": "validation",
        "test_split_used": False,
    }


def require_temporal_binding(manifest: dict[str, object]) -> tuple[str, str]:
    temporal = manifest.get("temporal")
    if not isinstance(temporal, dict):
        raise ValueError("Frozen protocol lacks temporal configuration.")
    checkpoint = temporal.get("final_checkpoint")
    cache = temporal.get("anchor_cache")
    if not checkpoint or not cache:
        raise RuntimeError("Final temporal T2 checkpoint / anchor cache is not bound.")
    return str(checkpoint), str(cache)


def _subset_manifest(
    *, seed: int, fractions: Sequence[float], total: int
) -> dict[str, object]:
    return {
        "schema": PAPER_MATRIX_SCHEMA,
        "seed": seed,
        "nested": True,
        "total_train_samples": total,
        "fractions": nested_fraction_indices(total, fractions, seed),
        "test_split_used": False,
    }


def _training_config(seed: int, adaptation: str) -> dict[str, object]:
    return {
        "seed": seed,
        **PAPER_SPATIAL_CONFIG,
        **PAPER_OPTIMIZER_CONFIG,
        "domain": "mobility",
        "mode": "full",
        "target_blocks": [0, 3],
        "adaptation": adaptation,
        "min_epochs": 101,
        "patience": 15,
        "formal_fp32": True,
        "test_split_used": False,
    }


def _prior_job(
    *, root: Path, phase: str, seed: int, fraction: float, manifest: Path
) -> dict[str, object]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    indices = payload["fractions"][f"{fraction:.2f}"]
    output = (
        root
        / "priors"
        / "mobility"
        / phase.replace("-", "_")
        / f"seed{seed}"
        / f"fraction_{fraction:.2f}.npz"
    )
    return {
        "phase": phase,
        "seed": seed,
        "fraction": fraction,
        "sample_count": len(indices),
        "sample_manifest": str(manifest.resolve()),
        "sample_manifest_sha256": file_sha256(manifest),
        "indices_hash": indices_hash(indices),
        "output": str(output.resolve()),
        "fit_split": "train",
        "selection_split": "validation",
        "test_split_used": False,
    }


def _spec(
    *,
    root: Path,
    phase: str,
    seed: int,
    fraction: float,
    method: str,
    model_key: str,
    prior_path: str | None,
    manifest: Path,
    adaptation: str,
    dependency: str | None = None,
) -> ExperimentSpec:
    name = f"{phase.replace('-', '_')}_seed{seed}_fraction_{fraction:.2f}_{method}"
    dependencies = tuple(
        value for value in (prior_path, dependency) if value is not None
    )
    return ExperimentSpec(
        phase=phase,
        name=name,
        domain="mobility",
        seed=seed,
        fraction=fraction,
        method_variant=method,
        model_key=model_key,
        prior_path=prior_path,
        pretrained_checkpoint=None,
        sample_manifest=str(manifest.resolve()),
        run_dir=str((root / "runs" / phase.replace("-", "_") / name).resolve()),
        training_config=_training_config(seed, adaptation),
        dependencies=dependencies,
    )


def build_paper_matrix_plan(
    output_root: str | Path,
    *,
    phase: str,
    seeds: Sequence[int] = PAPER_SEEDS,
    project_root: str | Path | None = None,
    git_head: str | None = None,
) -> dict[str, object]:
    if phase not in {*PAPER_PHASES, "all"}:
        raise ValueError(f"Unsupported paper matrix phase {phase!r}.")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or any(seed <= 0 for seed in normalized_seeds):
        raise ValueError("Paper matrix seeds must be positive integers.")
    root = Path(output_root).resolve()
    project = Path(project_root).resolve() if project_root else Path.cwd()
    head = git_head or _git_head(project)
    frozen = frozen_protocol_manifest(head)
    _write_json_exact(root / "manifests" / "frozen_protocol.json", frozen)
    specs: list[ExperimentSpec] = []
    prior_jobs: list[dict[str, object]] = []
    total = EXPECTED_MOBILITY_COUNTS["train"]

    if phase in {"data-efficiency", "all"}:
        for seed in normalized_seeds:
            manifest = root / "manifests" / f"data_efficiency_seed{seed}.json"
            _write_json_exact(
                manifest,
                _subset_manifest(seed=seed, fractions=PAPER_DATA_FRACTIONS, total=total),
            )
            for fraction in PAPER_DATA_FRACTIONS:
                prior = _prior_job(
                    root=root,
                    phase="data-efficiency",
                    seed=seed,
                    fraction=fraction,
                    manifest=manifest,
                )
                prior_jobs.append(prior)
                specs.append(
                    _spec(
                        root=root,
                        phase="data-efficiency",
                        seed=seed,
                        fraction=fraction,
                        method="direct_s3",
                        model_key="prist_ris_a",
                        prior_path=None,
                        manifest=manifest,
                        adaptation="full",
                    )
                )
                specs.append(
                    _spec(
                        root=root,
                        phase="data-efficiency",
                        seed=seed,
                        fraction=fraction,
                        method="prior_s3",
                        model_key="prist_ris_b",
                        prior_path=str(prior["output"]),
                        manifest=manifest,
                        adaptation="full",
                    )
                )

    if phase in {"transfer", "all"}:
        for seed in normalized_seeds:
            manifest = root / "manifests" / f"transfer_seed{seed}.json"
            _write_json_exact(
                manifest,
                _subset_manifest(seed=seed, fractions=PAPER_TRANSFER_FRACTIONS, total=total),
            )
            for fraction in PAPER_TRANSFER_FRACTIONS:
                prior = _prior_job(
                    root=root,
                    phase="transfer",
                    seed=seed,
                    fraction=fraction,
                    manifest=manifest,
                )
                prior_jobs.append(prior)
                for protocol, adaptation in (
                    ("scratch", "target_only_scratch"),
                    ("full_finetune", "full_finetune"),
                    ("selective", "selective"),
                ):
                    specs.append(
                        _spec(
                            root=root,
                            phase="transfer",
                            seed=seed,
                            fraction=fraction,
                            method=protocol,
                            model_key="prist_ris_b",
                            prior_path=str(prior["output"]),
                            manifest=manifest,
                            adaptation=adaptation,
                            dependency=(
                                "final_quasi_spatial_checkpoint"
                                if protocol != "scratch"
                                else None
                            ),
                        )
                    )

    profile_jobs = [
        {
            "variant": "direct_s3",
            "model_key": "prist_ris_a",
            "output": str((root / "profiles" / "direct_s3.json").resolve()),
            "test_split_used": False,
        },
        {
            "variant": "prior_s3_final_spatial",
            "model_key": "prist_ris_b",
            "output": str(
                (root / "profiles" / "prior_s3_final_spatial.json").resolve()
            ),
            "test_split_used": False,
        },
    ]
    plan = {
        "schema": PAPER_MATRIX_SCHEMA,
        "method": "PriST-RIS",
        "phase": phase,
        "seeds": list(normalized_seeds),
        "gpu_runs": len(specs),
        "prior_jobs": prior_jobs,
        "experiments": [spec.to_dict() for spec in specs],
        "profile_jobs": profile_jobs,
        "estimated_sample_counts": {
            spec.name: round(total * spec.fraction) for spec in specs
        },
        "dependencies": {
            "final_quasi_spatial_checkpoint": None,
            "final_temporal_T2_checkpoint": None,
            "final_temporal_anchor_cache": None,
        },
        "serial_execution": True,
        "planner_uses_gpu": False,
        "selection_split": "validation",
        "test_split_used": False,
    }
    commands: dict[str, list[str]] = {}
    for job in prior_jobs:
        commands[f"prior:{job['phase']}:seed{job['seed']}:fraction{float(job['fraction']):.2f}"] = [
            str(value)
            for value in prior_arguments(job, data_root="$DATA_ROOT", workers=8)
        ]
    for spec in specs:
        spec_path = root / "manifests" / "specs" / f"{spec.name}.json"
        command = training_arguments(
            spec,
            data_root="$DATA_ROOT",
            device="cuda",
            workers=8,
            spec_path=spec_path,
        )
        if spec.phase == "transfer" and spec.method_variant != "scratch":
            command.extend(("--pretrained", "$FINAL_QUASI_CHECKPOINT"))
        commands[spec.name] = [str(value) for value in command]
    for job in profile_jobs:
        commands[f"profile:{job['variant']}"] = [
            str(value) for value in profile_arguments(job)
        ]
    plan["commands"] = commands
    if any(spec.test_split_used for spec in specs):
        raise PermissionError("Paper matrix planner must never include TEST.")
    phase_slug = phase.replace("-", "_")
    _write_json_exact(root / f"paper_matrix_plan_{phase_slug}.json", plan)
    _write_json_atomic(root / "paper_matrix_plan.json", plan)
    for spec in specs:
        _write_json_exact(
            root / "manifests" / "specs" / f"{spec.name}.json", spec.to_dict()
        )
    return plan


def first_threshold_crossings(
    history: Iterable[dict[str, object]],
    thresholds: Sequence[float] = CONVERGENCE_THRESHOLDS_DB,
) -> dict[str, dict[str, float | int] | None]:
    rows = list(history)
    result: dict[str, dict[str, float | int] | None] = {}
    for threshold in thresholds:
        crossing = next(
            (
                row
                for row in rows
                if float(row["validation_nmse_db"]) <= float(threshold)
            ),
            None,
        )
        result[f"{threshold:.1f}"] = (
            {
                "epoch": int(crossing["epoch"]),
                "wall_clock_seconds": float(crossing["wall_clock_seconds"]),
            }
            if crossing is not None
            else None
        )
    return result


def decide_run(spec: ExperimentSpec, *, resume_incomplete: bool) -> dict[str, object]:
    run = Path(spec.run_dir)
    final = run / "results" / "final_result.json"
    recorded_spec = run / "paper_experiment_spec.json"
    expected = spec.to_dict()
    if final.is_file():
        if not recorded_spec.is_file() or json.loads(
            recorded_spec.read_text(encoding="utf-8")
        ) != expected:
            raise ValueError(f"Completed run spec mismatch: {run}")
        return {"action": "reuse", "run_dir": str(run)}
    if not run.exists():
        return {"action": "run", "run_dir": str(run)}
    if not resume_incomplete:
        raise FileExistsError(f"Incomplete paper run will not be overwritten: {run}")
    if not recorded_spec.is_file() or json.loads(
        recorded_spec.read_text(encoding="utf-8")
    ) != expected:
        raise ValueError(f"Incomplete run spec mismatch: {run}")
    checkpoint = run / "checkpoints" / "last_checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Incomplete run has no resumable checkpoint: {run}")
    return {"action": "resume", "run_dir": str(run), "checkpoint": str(checkpoint)}


def _expected_prior_metadata(job: dict[str, object]) -> dict[str, object]:
    return {
        "seed": int(job["seed"]),
        "fraction": float(job["fraction"]),
        "sample_count": int(job["sample_count"]),
        "sample_manifest_sha256": str(job["sample_manifest_sha256"]),
        "indices_hash": str(job["indices_hash"]),
        "semantics_hash": DataSemantics.for_domain("mobility").stable_hash(),
        "fit_split": "train",
        "selection_split": "validation",
        "test_split_used": False,
    }


def validate_prior_artifact(path: str | Path, job: dict[str, object]) -> None:
    prior = RidgePrior.load(path)
    metadata = prior.metadata()
    expected = _expected_prior_metadata(job)
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Existing fraction Ridge metadata mismatch: {mismatched}")


def prior_arguments(job: dict[str, object], *, data_root: str | Path, workers: int) -> list[object]:
    return [
        "fit-prior", "--domain", "mobility", "--target-blocks", "0,3",
        "--seed", job["seed"], "--fraction", job["fraction"],
        "--sample-index-manifest", job["sample_manifest"],
        "--data-root", data_root, "--workers", workers,
        "--batch-size", 64, "--eval-batch-size", 64,
        "--output", job["output"],
    ]


def profile_arguments(job: dict[str, object]) -> list[object]:
    return [
        "profile", "--domain", "mobility", "--model", job["model_key"],
        "--hidden", 80, "--blocks-per-stage", "3,3,2",
        "--final-refine-blocks", 1,
        "--backbone-ris-coordinate-enabled", "--backbone-ris-coordinate-mode", "direct_add",
        "--no-backbone-antenna-index-enabled", "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled", "--no-attention-antenna-index-enabled",
        "--no-spatial-multiscale-supervision", "--spatial-channel-attention", "se",
        "--spatial-residual-style", "scaled_true_residual",
        "--device", "cpu", "--output", job["output"],
    ]


def training_arguments(
    spec: ExperimentSpec,
    *,
    data_root: str | Path,
    device: str,
    workers: int,
    spec_path: str | Path,
    resume: str | Path | None = None,
) -> list[object]:
    config = spec.training_config
    arguments: list[object] = [
        "train", "--domain", "mobility", "--model", spec.model_key,
        "--mode", "full", "--seed", spec.seed,
        "--fraction", spec.fraction, "--sample-index-manifest", spec.sample_manifest,
        "--target-blocks", "0,3", "--data-root", data_root,
        "--device", device, "--workers", workers,
        "--batch-size", config["batch_size"],
        "--eval-batch-size", config["eval_batch_size"],
        "--hidden", config["hidden"],
        "--blocks-per-stage", ",".join(str(v) for v in config["blocks_per_stage"]),
        "--final-refine-blocks", config["final_refine_blocks"],
        "--backbone-ris-coordinate-enabled", "--backbone-ris-coordinate-mode", "direct_add",
        "--no-backbone-antenna-index-enabled", "--no-attention-enabled",
        "--no-attention-ris-coordinate-enabled", "--no-attention-antenna-index-enabled",
        "--no-spatial-multiscale-supervision", "--spatial-channel-attention", "se",
        "--spatial-residual-style", "scaled_true_residual",
        "--learning-rate", config["learning_rate"],
        "--weight-decay", config["weight_decay"],
        "--scheduler", config["scheduler"],
        "--min-learning-rate", config["min_learning_rate"],
        "--epochs", config["epochs"], "--min-epochs", config["min_epochs"],
        "--patience", config["patience"], "--adaptation", config["adaptation"],
        "--run-name", spec.name, "--output-root", Path(spec.run_dir).parent,
        "--experiment-spec", spec_path,
    ]
    if spec.prior_path is not None:
        arguments.extend(("--prior", spec.prior_path))
    if spec.pretrained_checkpoint is not None:
        arguments.extend(("--pretrained", spec.pretrained_checkpoint))
    if resume is not None:
        arguments.extend(("--resume", resume))
    return arguments


def validate_quasi_source_checkpoint(path: str | Path) -> str:
    source = Path(path).resolve()
    state = load_checkpoint(source, torch.device("cpu"))
    require_checkpoint_contract(state, "Paper transfer source", expected_domain="quasi")
    model = state.get("model_config")
    training = state.get("training_config")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("Paper transfer source lacks model/training configuration.")
    expected = PAPER_SPATIAL_CONFIG
    checks = {
        "hidden": model.get("hidden"),
        "blocks_per_stage": list(model.get("blocks_per_stage", [])),
        "final_refine_blocks": model.get("final_refine_blocks"),
        "backbone_ris_coordinate_enabled": model.get("backbone_ris_coordinate_enabled"),
        "backbone_ris_coordinate_mode": model.get("backbone_ris_coordinate_mode"),
        "backbone_antenna_index_enabled": model.get("backbone_antenna_index_enabled"),
        "attention_enabled": model.get("attention_enabled"),
        "spatial_multiscale_supervision": model.get("spatial_multiscale_supervision"),
        "spatial_channel_attention": model.get("spatial_channel_attention"),
    }
    for key, value in checks.items():
        expected_value = expected[key]
        if value != expected_value:
            raise ValueError(f"Quasi source is not frozen D1+RISCoord+SE: {key}={value!r}.")
    for key in ("scheduler", "min_learning_rate", "epochs"):
        if training.get(key) != PAPER_OPTIMIZER_CONFIG[key]:
            raise ValueError(f"Quasi source training protocol mismatch: {key}.")
    return file_sha256(source)


def gpu_preflight(*, device: str, physical_gpu_index: int, confirm_gpu_free: bool) -> None:
    if torch.device(device).type != "cuda":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu_index):
        raise PermissionError(
            f"Set CUDA_VISIBLE_DEVICES={physical_gpu_index}; use cuda:0 inside the process."
        )
    if not confirm_gpu_free:
        raise PermissionError("Inspect the physical GPU and pass --confirm-gpu-free.")
    subprocess.run(["nvidia-smi", "-i", str(physical_gpu_index)], check=True)
    query = subprocess.run(
        [
            "nvidia-smi", "-i", str(physical_gpu_index),
            "--query-compute-apps=pid", "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    foreign = {
        int(line.strip())
        for line in query.stdout.splitlines()
        if line.strip().isdigit() and int(line.strip()) != os.getpid()
    }
    if foreign:
        raise RuntimeError(
            f"GPU {physical_gpu_index} has foreign compute PID(s) {sorted(foreign)}; stopped."
        )


Invoke = Callable[[list[object]], None]


def execute_paper_matrix(
    plan: dict[str, object],
    *,
    output_root: str | Path,
    data_root: str | Path,
    device: str,
    workers: int,
    physical_gpu_index: int,
    confirm_gpu_free: bool,
    resume_incomplete: bool,
    quasi_checkpoint: str | Path | None,
    invoke: Invoke,
) -> dict[str, object]:
    if plan.get("test_split_used") is not False:
        raise PermissionError("Paper runner rejects any plan that uses TEST.")
    root = Path(output_root).resolve()
    phase = str(plan["phase"])
    quasi_hash = None
    if phase in {"transfer", "all"}:
        if quasi_checkpoint is None:
            raise RuntimeError("Transfer execution requires the final Quasi spatial checkpoint.")
        quasi_hash = validate_quasi_source_checkpoint(quasi_checkpoint)

    for job in plan.get("profile_jobs", []):
        output = Path(str(job["output"]))
        if output.is_file():
            profile = json.loads(output.read_text(encoding="utf-8"))
            if (
                profile.get("model_key") != job["model_key"]
                or profile.get("spatial_channel_attention") != "se"
                or profile.get("test_split_used") is not False
            ):
                raise ValueError(f"Existing paper complexity profile mismatch: {output}")
        else:
            invoke(profile_arguments(job))

    for job in plan["prior_jobs"]:  # type: ignore[index]
        output = Path(str(job["output"]))
        if output.is_file():
            validate_prior_artifact(output, job)
        else:
            if output.exists():
                raise FileExistsError(f"Prior path exists but is not an artifact: {output}")
            invoke(prior_arguments(job, data_root=data_root, workers=workers))
            validate_prior_artifact(output, job)

    results = []
    for raw_spec in plan["experiments"]:  # type: ignore[index]
        spec = ExperimentSpec.from_dict(raw_spec)
        if spec.test_split_used:
            raise PermissionError("Paper experiment spec uses TEST.")
        if spec.phase == "transfer" and spec.method_variant != "scratch":
            assert quasi_checkpoint is not None and quasi_hash is not None
            spec = replace(
                spec,
                pretrained_checkpoint=str(Path(quasi_checkpoint).resolve()),
                dependencies=tuple(
                    value
                    for value in spec.dependencies
                    if value != "final_quasi_spatial_checkpoint"
                )
                + (f"quasi_checkpoint_sha256:{quasi_hash}",),
            )
        executed_spec_path = root / "manifests" / "executed_specs" / f"{spec.name}.json"
        _write_json_exact(executed_spec_path, spec.to_dict())
        decision = decide_run(spec, resume_incomplete=resume_incomplete)
        if decision["action"] == "reuse":
            results.append({"name": spec.name, "status": "reused"})
            continue
        gpu_preflight(
            device=device,
            physical_gpu_index=physical_gpu_index,
            confirm_gpu_free=confirm_gpu_free,
        )
        invoke(
            training_arguments(
                spec,
                data_root=data_root,
                device=device,
                workers=workers,
                spec_path=executed_spec_path,
                resume=decision.get("checkpoint"),
            )
        )
        results.append({"name": spec.name, "status": decision["action"]})
    return {
        "method": "PriST-RIS",
        "phase": phase,
        "results": results,
        "test_split_used": False,
    }


def _read_history(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]], group_keys: Sequence[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        value = row.get("best_validation_nmse_db")
        if value is None:
            continue
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(float(value))
    output = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        output.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "mean_best_validation_nmse_db": statistics.fmean(values),
                "std_best_validation_nmse_db": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "n": len(values),
            }
        )
    return output


def summarize_paper_matrix(
    output_root: str | Path,
    plan: dict[str, object],
    *,
    final_spatial_run: str | Path | None = None,
) -> dict[str, object]:
    root = Path(output_root).resolve()
    data_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    missing = []
    for raw_spec in plan["experiments"]:  # type: ignore[index]
        planned = ExperimentSpec.from_dict(raw_spec)
        executed_path = root / "manifests" / "executed_specs" / f"{planned.name}.json"
        spec = (
            ExperimentSpec.from_dict(json.loads(executed_path.read_text(encoding="utf-8")))
            if executed_path.is_file()
            else planned
        )
        run = Path(spec.run_dir)
        final_path = run / "results" / "final_result.json"
        history_path = run / "results" / "training_history.csv"
        if not final_path.is_file() or not history_path.is_file():
            missing.append(spec.name)
            continue
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("test_split_used") is not False:
            raise PermissionError(f"Paper summary rejects TEST evidence: {run}")
        history = _read_history(history_path)
        last = final.get("last_validation", {})
        base: dict[str, object] = {
            "name": spec.name,
            "seed": spec.seed,
            "fraction": spec.fraction,
            "sample_count": round(EXPECTED_MOBILITY_COUNTS["train"] * spec.fraction),
            "method_variant": spec.method_variant,
            "target_scope": spec.target_scope,
            "best_validation_nmse_db": final.get("best_validation_nmse_db"),
            "last_validation_nmse_db": (
                last.get("nmse_db") if isinstance(last, dict) else None
            ),
            "best_epoch": min(
                history, key=lambda row: float(row["validation_nmse_linear"])
            )["epoch"],
            "wall_clock_seconds": final.get("wall_clock_seconds"),
            "parameters": final.get("metadata", {}).get("total_parameters"),
            "trainable_parameters": final.get("metadata", {}).get("trainable_parameters"),
            "prior_path": spec.prior_path,
            "sample_manifest": spec.sample_manifest,
            "test_split_used": False,
        }
        if spec.phase == "data-efficiency":
            data_rows.append(base)
            crossings = first_threshold_crossings(history)
            for threshold, crossing in crossings.items():
                convergence_rows.append(
                    {
                        "name": spec.name,
                        "seed": spec.seed,
                        "fraction": spec.fraction,
                        "method_variant": spec.method_variant,
                        "threshold_db": threshold,
                        "epoch": crossing["epoch"] if crossing else None,
                        "wall_clock_seconds": (
                            crossing["wall_clock_seconds"] if crossing else None
                        ),
                        "target_scope": spec.target_scope,
                        "test_split_used": False,
                    }
                )
        else:
            transfer_rows.append(base)

    profiles = []
    profile_root = root / "profiles"
    if profile_root.is_dir():
        for path in sorted(profile_root.glob("*.json")):
            profile = json.loads(path.read_text(encoding="utf-8"))
            profiles.append(
                {
                    "variant": path.stem,
                    "target_scope": "mobility_q0_q3",
                    "parameters": profile.get("parameters"),
                    "trainable_parameters": profile.get("trainable_parameters"),
                    "gmacs": profile.get("gmacs"),
                    "gflops": profile.get("gflops"),
                    "latency_ms_batch1": profile.get("latency_ms_batch1"),
                    "test_split_used": False,
                }
            )
    if final_spatial_run is not None:
        final_path = Path(final_spatial_run) / "results" / "final_result.json"
        if not final_path.is_file():
            raise FileNotFoundError(f"Missing final spatial result: {final_path}")
        final_spatial = json.loads(final_path.read_text(encoding="utf-8"))
        if final_spatial.get("test_split_used") is not False:
            raise PermissionError("Final spatial evidence must be VALIDATION-only.")
        spatial_profile = next(
            (
                row
                for row in profiles
                if row["variant"] == "prior_s3_final_spatial"
            ),
            {},
        )
        profiles.append(
            {
                "variant": "final_spatial_validation_evidence",
                "target_scope": "mobility_q0_q3",
                "best_validation_nmse_db": final_spatial.get(
                    "best_validation_nmse_db"
                ),
                "last_validation_nmse_db": final_spatial.get(
                    "last_validation", {}
                ).get("nmse_db"),
                "parameters": spatial_profile.get("parameters"),
                "trainable_parameters": spatial_profile.get(
                    "trainable_parameters"
                ),
                "gmacs": spatial_profile.get("gmacs"),
                "gflops": spatial_profile.get("gflops"),
                "latency_ms_batch1": spatial_profile.get(
                    "latency_ms_batch1"
                ),
                "wall_clock_seconds": final_spatial.get("wall_clock_seconds"),
                "test_split_used": False,
            }
        )
    summaries = root / "summaries"
    _write_csv(summaries / "data_efficiency.csv", data_rows)
    _write_csv(summaries / "convergence_efficiency.csv", convergence_rows)
    _write_csv(summaries / "transfer.csv", transfer_rows)
    _write_csv(summaries / "complexity.csv", profiles)
    result = {
        "schema": PAPER_MATRIX_SCHEMA,
        "method": "PriST-RIS",
        "target_scope_separation": {
            "spatial": "mobility_q0_q3",
            "temporal": "mobility_q0_q5_not_ranked_here",
        },
        "data_efficiency": data_rows,
        "data_efficiency_aggregate": _aggregate(
            data_rows, ("fraction", "method_variant")
        ),
        "convergence_efficiency": convergence_rows,
        "transfer": transfer_rows,
        "transfer_aggregate": _aggregate(
            transfer_rows, ("fraction", "method_variant")
        ),
        "complexity": profiles,
        "missing_experiments": missing,
        "test_split_used": False,
    }
    _write_json_atomic(summaries / "summary.json", result)
    return result
