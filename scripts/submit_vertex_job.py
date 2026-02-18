#!/usr/bin/env python3

#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#
#  Licensed under the MIT License.
#  For details: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   RIAWELC — Welding Defect Classification & Segmentation Pipeline

"""Vertex AI Training Job Submission CLI — EfficientNet / ResNet.

Submits Vertex AI CustomJobs for welding defect classification with:
- GPU (T4) or CPU-only machine configurations
- Cost estimation before submission
- Spot VM support (~60-70% cost savings)
- Dry-run mode to preview without spending money
- Custom config files and checkpoint resumption for fine-tuning

Usage:
    # Dry run — see what would be submitted + estimated cost
    python scripts/submit_vertex_job.py --model efficientnetb0 --dry-run

    # Submit a real GPU job (Spot VM)
    python scripts/submit_vertex_job.py --model resnet50v2 --yes

    # CPU-only smoke test (cheapest option)
    python scripts/submit_vertex_job.py --model efficientnetb0 --machine-type cpu-only --yes

    # Full training with on-demand VM (no preemption risk)
    python scripts/submit_vertex_job.py --model efficientnetb0 --no-spot --yes

    # Use a custom config file
    python scripts/submit_vertex_job.py --model efficientnetb0 \\
        --config configs/efficientnetb0_ht_1.yaml --dry-run

    # Fine-tune from a Phase 1 checkpoint (FUSE path)
    python scripts/submit_vertex_job.py --model efficientnetb0 \\
        --config configs/efficientnetb0_ht_1.yaml \\
        --resume-from /gcs/YOUR_ARTIFACTS_BUCKET/outputs/\\
    checkpoints/efficientnetb0/v1/feature_extraction/best.keras --yes

    # Fine-tune from a gs:// URL (auto-converted to /gcs/ FUSE path)
    python scripts/submit_vertex_job.py --model efficientnetb0 \\
        --config configs/efficientnetb0_ht_1.yaml \\
        --resume-from gs://YOUR_ARTIFACTS_BUCKET/outputs/\\
    checkpoints/efficientnetb0/v1/feature_extraction/best.keras --yes
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Constants
# =============================================================================

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
REGION = os.environ.get("GCP_REGION", "europe-west3")
DEFAULT_SERVICE_ACCOUNT = os.environ.get(
    "GCP_SERVICE_ACCOUNT", "YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com"
)

DATA_BUCKET = os.environ.get("RIAWELC_DATA_BUCKET", "your-data-bucket")
ARTIFACTS_BUCKET = os.environ.get("RIAWELC_ARTIFACTS_BUCKET", "your-artifacts-bucket")
STAGING_BUCKET = f"gs://{ARTIFACTS_BUCKET}/staging"

# Container image in Artifact Registry (built by cloudbuild.yaml)
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/training/riawelc-training:latest"

# GCS paths via /gcs/ FUSE mount inside Vertex AI containers
GCS_DATA_DIR = f"/gcs/{DATA_BUCKET}"
GCS_CHECKPOINT_DIR = f"/gcs/{ARTIFACTS_BUCKET}/outputs/checkpoints"
GCS_MLFLOW_URI = f"/gcs/{ARTIFACTS_BUCKET}/mlruns"

# Supported models (must match MODEL_REGISTRY in riawelc.models.registry)
SUPPORTED_MODELS = [
    "efficientnetb0",
    "resnet50v2",
]

# ── Pricing (approximate, europe-west3 as of 2025) ──────────────────
HOURLY_COSTS_ON_DEMAND: dict[str, float] = {
    "gpu-t4": 0.57,  # n1-standard-4 + NVIDIA T4
    "gpu-t4-highmem": 0.67,  # n1-standard-8 + NVIDIA T4 (30 GB RAM)
    "cpu-only": 0.21,  # n1-standard-4
}
HOURLY_COSTS_SPOT: dict[str, float] = {
    "gpu-t4": 0.19,  # ~67% discount
    "gpu-t4-highmem": 0.23,  # ~66% discount
    "cpu-only": 0.07,  # ~67% discount
}

# Estimated training time per job (hours)
TRAINING_TIME_ESTIMATES: dict[str, dict[str, float]] = {
    "efficientnetb0": {"gpu-t4": 1.0, "gpu-t4-highmem": 1.0, "cpu-only": 5.0},
    "resnet50v2": {"gpu-t4": 1.5, "gpu-t4-highmem": 1.5, "cpu-only": 6.0},
}
DEFAULT_TRAINING_TIME: dict[str, float] = {
    "gpu-t4": 2.0,
    "gpu-t4-highmem": 2.0,
    "cpu-only": 8.0,
}


# =============================================================================
# Machine Configurations
# =============================================================================


@dataclass(frozen=True)
class MachineConfig:
    """Configuration for a Vertex AI worker pool."""

    name: str
    machine_type: str
    accelerator_type: str | None
    accelerator_count: int
    hourly_cost: float

    @property
    def has_gpu(self) -> bool:
        return self.accelerator_type is not None


def _build_machine_configs(spot: bool = True) -> dict[str, MachineConfig]:
    """Build machine configs with on-demand or spot pricing."""
    costs = HOURLY_COSTS_SPOT if spot else HOURLY_COSTS_ON_DEMAND
    return {
        "gpu-t4": MachineConfig(
            name="gpu-t4",
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            hourly_cost=costs["gpu-t4"],
        ),
        "gpu-t4-highmem": MachineConfig(
            name="gpu-t4-highmem",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            hourly_cost=costs["gpu-t4-highmem"],
        ),
        "cpu-only": MachineConfig(
            name="cpu-only",
            machine_type="n1-standard-4",
            accelerator_type=None,
            accelerator_count=0,
            hourly_cost=costs["cpu-only"],
        ),
    }


# =============================================================================
# Cost Estimation
# =============================================================================


def estimate_cost(
    model_name: str,
    machine_type: str,
    spot: bool = True,
) -> dict[str, float]:
    """Estimate training cost for a single job."""
    time_estimates = TRAINING_TIME_ESTIMATES.get(model_name, DEFAULT_TRAINING_TIME)
    hours = time_estimates.get(machine_type, DEFAULT_TRAINING_TIME[machine_type])
    costs = HOURLY_COSTS_SPOT if spot else HOURLY_COSTS_ON_DEMAND
    hourly_rate = costs.get(machine_type, 0.50)

    return {
        "hours": hours,
        "hourly_rate": hourly_rate,
        "total_cost": hours * hourly_rate,
    }


def print_cost_estimate(
    model_name: str,
    machine_type: str,
    machine_config: MachineConfig,
    spot: bool = True,
) -> None:
    """Print formatted cost estimate."""
    est = estimate_cost(model_name, machine_type, spot=spot)
    vm_label = "Spot" if spot else "On-Demand"

    print(f"\n{'=' * 60}")
    print("Cost Estimation")
    print("=" * 60)
    print(f"  Model:           {model_name}")
    print(f"  Machine type:    {machine_type} ({vm_label})")
    print(f"    Instance:      {machine_config.machine_type}")
    if machine_config.has_gpu:
        print(f"    GPU:           {machine_config.accelerator_type}")
    print(f"  Est. duration:   ~{est['hours']:.1f}h")
    print(f"  Hourly rate:     \u20ac{est['hourly_rate']:.2f}/hr")
    print(f"  Estimated cost:  \u20ac{est['total_cost']:.2f}")
    print("=" * 60)
    print("  Note: Actual costs depend on training duration + data size.")


# =============================================================================
# Job Building
# =============================================================================


def build_job_config(
    model_name: str,
    machine_config: MachineConfig,
    data_dir: str,
    checkpoint_dir: str,
    mlflow_uri: str,
    config_path: str | None = None,
    resume_from: str | None = None,
    tracking: str = "both",
) -> dict:
    """Build a Vertex AI CustomJob configuration.

    Args:
        model_name: Model architecture name (e.g. efficientnetb0).
        machine_config: Machine specification.
        data_dir: GCS path to dataset via /gcs/ FUSE mount.
        checkpoint_dir: GCS path for checkpoints via /gcs/ mount.
        mlflow_uri: GCS path for MLflow tracking via /gcs/ mount.
        config_path: YAML config file path (relative to container /app/).
            Defaults to ``configs/{model_name}_baseline.yaml``.
        resume_from: Path to a checkpoint for fine-tuning (/gcs/ FUSE mount).
        tracking: Experiment tracking backend.

    Returns:
        Dictionary with job_name and worker_pool_specs for Vertex AI.
    """
    is_finetune = resume_from is not None
    job_name = f"riawelc-{model_name}-finetune" if is_finetune else f"riawelc-{model_name}-train"
    resolved_config = config_path or f"configs/{model_name}_baseline.yaml"

    args = [
        "--model",
        model_name,
        "--config",
        resolved_config,
        "--tracking",
        tracking,
        "--data-dir",
        data_dir,
        "--checkpoint-dir",
        checkpoint_dir,
        "--mlflow-uri",
        mlflow_uri,
    ]

    if resume_from is not None:
        args.extend(["--resume-from", resume_from])

    worker_pool_spec: dict = {
        "machine_spec": {
            "machine_type": machine_config.machine_type,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": CONTAINER_URI,
            "command": ["python", "scripts/01_train_classifier.py"],
            "args": args,
        },
    }

    if machine_config.has_gpu:
        worker_pool_spec["machine_spec"]["accelerator_type"] = machine_config.accelerator_type
        worker_pool_spec["machine_spec"]["accelerator_count"] = machine_config.accelerator_count

    return {
        "job_name": job_name,
        "worker_pool_specs": [worker_pool_spec],
    }


# =============================================================================
# Job Submission
# =============================================================================


def submit_job(
    job_config: dict,
    spot: bool = True,
    wait: bool = False,
    service_account: str | None = None,
) -> object:
    """Submit a single Vertex AI CustomJob.

    Args:
        job_config: Configuration from build_job_config().
        spot: Use Spot VMs for cost savings.
        wait: Block until job completes.
        service_account: SA email for the training VM. Explicitly setting this
            causes Vertex AI to grant the ``cloud-platform`` OAuth scope,
            which is required for Vertex AI Experiments / Metadata API calls
            from inside the container.

    Returns:
        Submitted CustomJob object.
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types.custom_job import Scheduling

    scheduling_strategy = Scheduling.Strategy.SPOT if spot else Scheduling.Strategy.ON_DEMAND
    vm_type = "Spot" if spot else "On-Demand"

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    print(f"  Submitting ({vm_type}): {job_config['job_name']}")

    job = aiplatform.CustomJob(
        display_name=job_config["job_name"],
        worker_pool_specs=job_config["worker_pool_specs"],
        staging_bucket=STAGING_BUCKET,
    )

    submit_kwargs: dict = {"scheduling_strategy": scheduling_strategy}
    if service_account:
        submit_kwargs["service_account"] = service_account

    if wait:
        job.run(sync=True, **submit_kwargs)
    else:
        job.submit(**submit_kwargs)

    # Print console URL
    job_id = job.resource_name.split("/")[-1]
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
        f"/{REGION}/{job_id}?project={PROJECT_ID}"
    )
    print(f"    Console: {console_url}")

    return job


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Vertex AI training jobs for RIAWELC welding defect classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with cost estimate
  python scripts/submit_vertex_job.py --model efficientnetb0 --dry-run

  # Submit GPU job (Spot VM, ~\u20ac0.19/hr)
  python scripts/submit_vertex_job.py --model resnet50v2 --yes

  # CPU-only smoke test (~\u20ac0.07/hr)
  python scripts/submit_vertex_job.py --model efficientnetb0 --machine-type cpu-only --yes

  # Wait for job to complete
  python scripts/submit_vertex_job.py --model resnet50v2 --wait --yes

  # Custom config file
  python scripts/submit_vertex_job.py --model efficientnetb0 \\
      --config configs/efficientnetb0_ht_1.yaml --dry-run

  # Fine-tune from a Phase 1 checkpoint
  python scripts/submit_vertex_job.py --model efficientnetb0 \\
      --config configs/efficientnetb0_ht_1.yaml \\
      --resume-from gs://YOUR_ARTIFACTS_BUCKET/outputs/\\
  checkpoints/efficientnetb0/v1/feature_extraction/best.keras --yes
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file path relative to container /app/ "
        "(default: configs/{model}_baseline.yaml).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path for fine-tuning. Accepts /gcs/ FUSE paths or gs:// URLs "
        "(gs:// URLs are auto-converted to /gcs/ FUSE paths).",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default="gpu-t4",
        choices=["gpu-t4", "gpu-t4-highmem", "cpu-only"],
        help="Machine config (default: gpu-t4).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=GCS_DATA_DIR,
        help=f"GCS data path via /gcs/ mount (default: {GCS_DATA_DIR}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=GCS_CHECKPOINT_DIR,
        help=f"GCS checkpoint path (default: {GCS_CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=GCS_MLFLOW_URI,
        help=f"GCS MLflow tracking path (default: {GCS_MLFLOW_URI}).",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["mlflow", "vertex", "both", "none"],
        default="both",
        help="Experiment tracking backend (default: both).",
    )
    parser.add_argument(
        "--service-account",
        type=str,
        default=DEFAULT_SERVICE_ACCOUNT,
        help="Service account email for the training VM. Explicitly setting this "
        "causes Vertex AI to grant the cloud-platform OAuth scope, required for "
        f"Vertex AI Experiments (default: {DEFAULT_SERVICE_ACCOUNT}).",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Use on-demand VMs (default: Spot).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job to complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without submitting.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    return parser.parse_args()


def _convert_gs_to_fuse(gs_url: str) -> str:
    """Convert a gs:// URL to a /gcs/ FUSE mount path."""
    if gs_url.startswith("gs://"):
        return "/gcs/" + gs_url[len("gs://") :]
    return gs_url


def main() -> None:
    args = parse_args()
    use_spot = not args.no_spot
    vm_label = "Spot" if use_spot else "On-Demand"
    machine_configs = _build_machine_configs(spot=use_spot)
    machine_config = machine_configs[args.machine_type]

    # Resolve config path and resume-from
    config_path = args.config
    resume_from = args.resume_from
    if resume_from is not None:
        resume_from = _convert_gs_to_fuse(resume_from)

    resolved_config = config_path or f"configs/{args.model}_baseline.yaml"

    # ── Print job summary ──
    print("=" * 60)
    print("Vertex AI Training Job Submission")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Config:          {resolved_config}")
    print(f"  Machine:         {args.machine_type} ({vm_label})")
    print(f"    Instance:      {machine_config.machine_type}")
    if machine_config.has_gpu:
        print(f"    GPU:           {machine_config.accelerator_type}")
    print(f"  Container:       {CONTAINER_URI}")
    print(f"  Service account: {args.service_account}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    print(f"  MLflow URI:      {args.mlflow_uri}")
    if resume_from is not None:
        print(f"  Resume from:     {resume_from}")

    # ── Cost estimate ──
    print_cost_estimate(args.model, args.machine_type, machine_config, spot=use_spot)

    # ── Build job config ──
    job_config = build_job_config(
        model_name=args.model,
        machine_config=machine_config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        mlflow_uri=args.mlflow_uri,
        config_path=config_path,
        resume_from=resume_from,
        tracking=args.tracking,
    )

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN — Job Configuration")
        print("=" * 60)
        print(json.dumps(job_config, indent=2))
        print(f"\n{'=' * 60}")
        print("Dry run complete. No jobs submitted.")
        print("=" * 60)
        return

    # ── Confirmation ──
    if not args.yes:
        est = estimate_cost(args.model, args.machine_type, spot=use_spot)
        response = (
            input(f"\n  Submit job (est. \u20ac{est['total_cost']:.2f})? [y/N]: ").strip().lower()
        )
        if response != "y":
            print("  Aborted.")
            return

    # ── Submit ──
    print(f"\n{'=' * 60}")
    print("Submitting Job")
    print("=" * 60)

    job = submit_job(
        job_config,
        spot=use_spot,
        wait=args.wait,
        service_account=args.service_account,
    )

    if args.wait:
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        print(f"\n  Final status: {state}")

    print(f"\n{'=' * 60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
