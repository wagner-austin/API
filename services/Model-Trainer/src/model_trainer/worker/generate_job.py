"""Generate job processing."""

from __future__ import annotations

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import generate_key
from typing_extensions import TypedDict

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.model import GenerateConfig
from model_trainer.core.contracts.queue import GenerateJobPayload
from model_trainer.worker.job_utils import (
    materialize_run_artifacts,
    redis_client,
    setup_job_logging,
)
from model_trainer.worker.manifest import as_model_family, load_manifest_from_text


class _GenerateCacheModel(TypedDict, total=False):
    status: str
    outputs: list[str] | None
    steps: int | None
    eos_terminated: list[bool] | None


def process_generate_job(payload: GenerateJobPayload) -> None:
    """Process a generate inference job."""
    settings = _test_hooks.load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    request_id = payload["request_id"]

    running: _GenerateCacheModel = {"status": "running"}
    r.set(generate_key(run_id, request_id), dump_json_str(running))

    try:
        normalized = materialize_run_artifacts(settings, r, run_id, purpose="generate")

        manifest_path = normalized / "manifest.json"
        if not manifest_path.exists():
            raise AppError(
                ModelTrainerErrorCode.MODEL_NOT_FOUND,
                f"manifest missing for run_id={run_id}",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_NOT_FOUND),
            )

        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = load_manifest_from_text(manifest_text)

        tokenizer_id = manifest["tokenizer_id"]
        tok_handle = (
            _test_hooks.load_tokenizer_for_training(settings, tokenizer_id)
            if tokenizer_id is not None
            else None
        )
        container = _test_hooks.service_container_from_settings(settings)
        backend = container.model_registry.get(as_model_family(manifest["model_family"]))

        cfg = GenerateConfig(
            prompt_text=payload["prompt_text"],
            prompt_path=payload["prompt_path"],
            max_new_tokens=payload["max_new_tokens"],
            temperature=payload["temperature"],
            top_k=payload["top_k"],
            top_p=payload["top_p"],
            stop_on_eos=payload["stop_on_eos"],
            stop_sequences=list(payload["stop_sequences"]),
            seed=payload["seed"],
            num_return_sequences=payload["num_return_sequences"],
        )

        prepared = backend.load(str(normalized), settings, tokenizer=tok_handle)
        result = backend.generate(prepared=prepared, cfg=cfg, settings=settings)

        out: _GenerateCacheModel = {
            "status": "completed",
            "outputs": list(result["outputs"]),
            "steps": result["steps"],
            "eos_terminated": list(result["eos_terminated"]),
        }
    except Exception as e:
        out_failed: _GenerateCacheModel = {"status": "failed"}
        log.exception("Generate failed run_id=%s request_id=%s error=%s", run_id, request_id, e)
        r.set(generate_key(run_id, request_id), dump_json_str(out_failed))
        raise
    else:
        r.set(generate_key(run_id, request_id), dump_json_str(out))
        log.info("Generate job completed run_id=%s request_id=%s", run_id, request_id)
