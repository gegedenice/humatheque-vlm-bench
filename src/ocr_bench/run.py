"""VLM model orchestration — launch HF Jobs for metadata extraction models."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import structlog
from huggingface_hub import HfApi, get_token

from ocr_bench.task_config import (
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_SOURCE_DATASET,
    build_default_task_prompt,
)

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Configuration for a single OCR model."""

    script: str
    model_id: str
    size: str
    default_flavor: str = "l4x1"
    default_args: list[str] = field(default_factory=list)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3-vl-4b-instruct": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/firered-ocr.py",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        size="4B",
        default_flavor="l4x1",
    ),
    "nanonets-ocr2-3b": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2.py",
        model_id="nanonets/Nanonets-OCR2-3B",
        size="3B",
        default_flavor="l4x1",
    ),
    "gemma-4-e4b-it": ModelConfig(
        script="https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py",
        model_id="google/gemma-4-E4B-it",
        size="4B",
        default_flavor="l4x1",
    ),
}

DEFAULT_MODELS = ["qwen3-vl-4b-instruct", "nanonets-ocr2-3b", "gemma-4-e4b-it"]
DEFAULT_TASK_PROMPT = build_default_task_prompt()


@dataclass
class JobRun:
    """Tracks a launched HF Job."""

    model_slug: str
    job_id: str
    job_url: str
    status: str = "running"


def list_models() -> list[str]:
    """Return sorted list of available model slugs."""
    return sorted(MODEL_REGISTRY.keys())


def build_script_args(
    input_dataset: str,
    output_repo: str,
    config_name: str,
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    extra_args: list[str] | None = None,
    prompt: str | None = None,
) -> list[str]:
    """Build script arguments for uv-scripts/ocr inference jobs.

    These scripts expect positional ``input_dataset output_dataset`` plus flags
    such as ``--output-column`` (not ``--config`` / ``--create-pr``).
    """
    args = [
        input_dataset,
        output_repo,
        "--image-column",
        DEFAULT_IMAGE_COLUMN,
        "--output-column",
        config_name,
    ]
    if prompt is not None:
        args += ["--prompt", prompt]
    if max_samples is not None:
        args += ["--max-samples", str(max_samples)]
    if shuffle:
        args.append("--shuffle")
    if seed != 42:
        args += ["--seed", str(seed)]
    if extra_args:
        args += extra_args
    return args


def launch_ocr_jobs(
    input_dataset: str,
    output_repo: str,
    *,
    models: list[str] | None = None,
    max_samples: int | None = None,
    split: str = "train",
    shuffle: bool = False,
    seed: int = 42,
    prompt: str | None = None,
    flavor_override: str | None = None,
    timeout: str = "4h",
    api: HfApi | None = None,
) -> list[JobRun]:
    """Launch HF Jobs for each model. Returns list of JobRun tracking objects."""
    if api is None:
        api = HfApi()

    token = get_token()
    if not token:
        raise RuntimeError("No HF token found. Log in with `hf login` or set HF_TOKEN.")

    selected = models or DEFAULT_MODELS
    for slug in selected:
        if slug not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {slug}. Available: {', '.join(MODEL_REGISTRY.keys())}"
            )

    jobs: list[JobRun] = []
    validated_scripts: set[str] = set()
    for slug in selected:
        config = MODEL_REGISTRY[slug]
        if config.script not in validated_scripts:
            _validate_remote_script(config.script)
            validated_scripts.add(config.script)
        flavor = flavor_override or config.default_flavor
        script_args = build_script_args(
            input_dataset,
            output_repo,
            slug,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            extra_args=config.default_args or None,
            prompt=prompt,
        )

        logger.info("launching_job", model=slug, flavor=flavor, script=config.script)
        job = api.run_uv_job(
            script=config.script,
            script_args=script_args,
            flavor=flavor,
            secrets={"HF_TOKEN": token},
            timeout=timeout,
        )
        jobs.append(JobRun(model_slug=slug, job_id=job.id, job_url=job.url))
        logger.info("job_launched", model=slug, job_id=job.id, url=job.url)

    return jobs


def _validate_remote_script(script_url: str) -> None:
    """Validate that a remote HF script URL resolves to a Python file, not an error page."""
    try:
        with urllib.request.urlopen(script_url, timeout=15) as response:
            header = response.read(256).decode("utf-8", errors="ignore").strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not access script URL: {script_url}") from exc

    if "Entry not found" in header:
        raise RuntimeError(
            f"Script URL points to a missing file: {script_url}. "
            "Please update MODEL_REGISTRY to a valid uv-scripts/ocr raw path."
        )


_TERMINAL_STAGES = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


def poll_jobs(
    jobs: list[JobRun],
    *,
    interval: int = 30,
    api: HfApi | None = None,
) -> list[JobRun]:
    """Poll until all jobs complete or fail. Updates status in-place and returns the list."""
    if api is None:
        api = HfApi()

    pending = {j.job_id: j for j in jobs if j.status == "running"}

    while pending:
        time.sleep(interval)
        still_running: dict[str, JobRun] = {}
        for job_id, job_run in pending.items():
            info = api.inspect_job(job_id=job_id)
            stage = info.status.stage
            if stage in _TERMINAL_STAGES:
                job_run.status = stage.lower()
                logger.info("job_finished", model=job_run.model_slug, status=job_run.status)
            else:
                still_running[job_id] = job_run
        pending = still_running
        if pending:
            slugs = [j.model_slug for j in pending.values()]
            logger.info("jobs_pending", models=slugs)

    return jobs
