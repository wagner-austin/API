"""Score job processing."""

from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import artifact_file_id_key, score_key
from typing_extensions import TypedDict

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.model import ScoreConfig
from model_trainer.core.contracts.queue import ScoreJobPayload
from model_trainer.core.infra.paths import models_dir
from model_trainer.worker.job_utils import redis_client, setup_job_logging
from model_trainer.worker.manifest import as_model_family, load_manifest_from_text


class _ScoreCacheModel(TypedDict, total=False):
    status: str
    loss: float | None
    perplexity: float | None
    surprisal: list[float] | None
    topk: list[list[list[str | float]]] | None
    tokens: list[str] | None


def process_score_job(payload: ScoreJobPayload) -> None:
    """Process a score inference job."""
    settings = _test_hooks.load_settings()
    setup_job_logging(settings)

    log = get_logger(__name__)
    r = redis_client(settings)
    run_id = payload["run_id"]
    request_id = payload["request_id"]

    running: _ScoreCacheModel = {"status": "running"}
    r.set(score_key(run_id, request_id), dump_json_str(running))

    try:
        file_id = r.get(artifact_file_id_key(run_id))
        if not isinstance(file_id, str) or file_id.strip() == "":
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "artifact pointer not found for score",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        api_url = settings["app"]["data_bank_api_url"]
        api_key = settings["app"]["data_bank_api_key"]
        store = _test_hooks.artifact_store_factory(api_url, api_key)
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

        tok_handle = _test_hooks.load_tokenizer_for_training(settings, manifest["tokenizer_id"])
        container = _test_hooks.service_container_from_settings(settings)
        backend = container.model_registry.get(as_model_family(manifest["model_family"]))

        detail_level: Literal["summary", "per_char"] = (
            "per_char" if payload["detail_level"] == "per_char" else "summary"
        )

        cfg = ScoreConfig(
            text=payload["text"],
            path=payload["path"],
            detail_level=detail_level,
            top_k=payload["top_k"],
            seed=payload["seed"],
        )

        prepared = backend.load(str(normalized), settings, tokenizer=tok_handle)
        result = backend.score(prepared=prepared, cfg=cfg, settings=settings)

        topk_json: list[list[list[str | float]]] | None = None
        if result["topk"] is not None:
            topk_json = []
            for pos in result["topk"]:
                pos_list: list[list[str | float]] = []
                for tok, prob in pos:
                    pos_list.append([tok, prob])
                topk_json.append(pos_list)

        out: _ScoreCacheModel = {
            "status": "completed",
            "loss": result["loss"],
            "perplexity": result["perplexity"],
            "surprisal": list(result["surprisal"]) if result["surprisal"] is not None else None,
            "topk": topk_json,
            "tokens": list(result["tokens"]) if result["tokens"] is not None else None,
        }
    except Exception as e:
        out_failed: _ScoreCacheModel = {"status": "failed"}
        log.exception("Score failed run_id=%s request_id=%s error=%s", run_id, request_id, e)
        r.set(score_key(run_id, request_id), dump_json_str(out_failed))
        raise
    else:
        r.set(score_key(run_id, request_id), dump_json_str(out))
        log.info("Score job completed run_id=%s request_id=%s", run_id, request_id)
