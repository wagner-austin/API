"""Score job processing."""

from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import score_key
from typing_extensions import TypedDict

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.model import ScoreConfig
from model_trainer.core.contracts.queue import ScoreJobPayload
from model_trainer.worker.job_utils import (
    materialize_run_artifacts,
    redis_client,
    setup_job_logging,
)
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
        normalized = materialize_run_artifacts(settings, r, run_id, purpose="score")

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
