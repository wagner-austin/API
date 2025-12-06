"""Generate job processing."""

from __future__ import annotations

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import artifact_file_id_key, generate_key
from typing_extensions import TypedDict

from model_trainer.core.config.settings import load_settings
from model_trainer.core.contracts.model import GenerateConfig
from model_trainer.core.contracts.queue import GenerateJobPayload
from model_trainer.core.infra.paths import models_dir
from model_trainer.core.services.container import ServiceContainer
from model_trainer.worker.job_utils import (
    load_tokenizer_for_training,
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
    settings = load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    request_id = payload["request_id"]

    running: _GenerateCacheModel = {"status": "running"}
    r.set(generate_key(run_id, request_id), dump_json_str(running))

    try:
        file_id = r.get(artifact_file_id_key(run_id))
        if not isinstance(file_id, str) or file_id.strip() == "":
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "artifact pointer not found for generate",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        from platform_ml import ArtifactStore

        api_url = settings["app"]["data_bank_api_url"]
        api_key = settings["app"]["data_bank_api_key"]
        store = ArtifactStore(api_url, api_key)
        models_root = models_dir(settings)
        expected_root = f"model-{run_id}"
        normalized = models_root / run_id

        if not normalized.exists():
            out_root = store.download_artifact(
                file_id.strip(),
                dest_dir=models_root,
                request_id=run_id,
                expected_root=expected_root,
            )
            out_root.rename(normalized)

        manifest_path = normalized / "manifest.json"
        if not manifest_path.exists():
            raise AppError(
                ModelTrainerErrorCode.MODEL_NOT_FOUND,
                f"manifest missing for run_id={run_id}",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_NOT_FOUND),
            )

        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = load_manifest_from_text(manifest_text)

        tok_handle = load_tokenizer_for_training(settings, manifest["tokenizer_id"])
        container = ServiceContainer.from_settings(settings)
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
