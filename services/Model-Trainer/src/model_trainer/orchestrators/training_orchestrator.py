from __future__ import annotations

from datetime import datetime

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger
from platform_core.trainer_keys import (
    artifact_file_id_key,
    eval_key,
    heartbeat_key,
)
from platform_workers.redis import RedisStrProto
from typing_extensions import TypedDict

from ..api.schemas.pointers import ArtifactPointer
from ..api.schemas.runs import (
    EvaluateRequest,
    EvaluateResponse,
    RunStatusResponse,
    TrainRequest,
    TrainResponse,
)
from ..core.config.settings import Settings
from ..core.contracts.queue import EvalJobPayload, TrainJobPayload, TrainRequestPayload
from ..core.infra.redis_utils import get_with_retry, set_with_retry
from ..core.services.queue.rq_adapter import RQEnqueuer
from ..core.services.registries import ModelRegistry
from ..infra.persistence.models import EvalCache
from ..infra.storage.run_store import RunStore
from ..worker.trainer_job_store import TrainerJobStore

_logger = get_logger(__name__)


class EnqueueOut(TypedDict):
    run_id: str
    job_id: str


class TrainingOrchestrator:
    def __init__(
        self: TrainingOrchestrator,
        *,
        settings: Settings,
        redis_client: RedisStrProto,
        enqueuer: RQEnqueuer,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer
        self._store = RunStore(settings["app"]["artifacts_root"])
        self._models = model_registry
        self._job_store = TrainerJobStore(redis_client)

    def enqueue_training(self: TrainingOrchestrator, req: TrainRequest) -> TrainResponse:
        # Early validation via registry if available
        if self._models is not None:
            try:
                _ = self._models.get(req["model_family"])
            except AppError:
                _logger.info(
                    "unsupported model family",
                    extra={
                        "category": "orchestrator",
                        "service": "training",
                        "event": "model_backend_unavailable",
                        "model_family": req["model_family"],
                    },
                )
                raise
        run_id = self._store.create_run(req["model_family"], req["model_size"])
        # Pass corpus_file_id through to worker; worker resolves locally
        fid = req["corpus_file_id"].strip()
        if fid == "":  # should not occur due to schema min_length
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_FOUND,
                "corpus_file_id must be non-empty",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
            )

        request_payload: TrainRequestPayload = {
            "model_family": req["model_family"],
            "model_size": req["model_size"],
            "max_seq_len": req["max_seq_len"],
            "num_epochs": req["num_epochs"],
            "batch_size": req["batch_size"],
            "learning_rate": req["learning_rate"],
            "corpus_file_id": fid,
            "tokenizer_id": req["tokenizer_id"],
            "holdout_fraction": req["holdout_fraction"],
            "seed": req["seed"],
            "pretrained_run_id": req["pretrained_run_id"],
            "freeze_embed": req["freeze_embed"],
            "gradient_clipping": req["gradient_clipping"],
            "optimizer": req["optimizer"],
            "device": req["device"],
            "precision": req["precision"],
            "data_num_workers": req.get("data_num_workers"),
            "data_pin_memory": req.get("data_pin_memory"),
            "early_stopping_patience": req["early_stopping_patience"],
            "test_split_ratio": req["test_split_ratio"],
            "finetune_lr_cap": req["finetune_lr_cap"],
        }
        payload: TrainJobPayload = {
            "run_id": run_id,
            "request": request_payload,
            "user_id": int(req["user_id"]),
        }

        job_id = self._enq.enqueue_train(payload)
        now = datetime.utcnow()
        self._job_store.save(
            {
                "job_id": run_id,
                "user_id": int(req["user_id"]),
                "status": "queued",
                "progress": 0,
                "message": "queued",
                "created_at": now,
                "updated_at": now,
                "error": None,
                "artifact_file_id": None,
            },
        )
        _logger.info(
            "training enqueued",
            extra={
                "category": "training",
                "service": "orchestrator",
                "run_id": run_id,
                "event": "enqueued",
            },
        )
        return TrainResponse(run_id=run_id, job_id=job_id)

    def get_status(self: TrainingOrchestrator, run_id: str) -> RunStatusResponse:
        from typing import Literal

        status_obj = self._job_store.load(run_id)
        if status_obj is None:
            _logger.info(
                "run not found",
                extra={
                    "category": "orchestrator",
                    "service": "training",
                    "run_id": run_id,
                    "event": "run_not_found",
                },
            )
            raise AppError(
                ModelTrainerErrorCode.RUN_NOT_FOUND,
                "run not found",
                model_trainer_status_for(ModelTrainerErrorCode.RUN_NOT_FOUND),
            )
        status_v = status_obj["status"]
        status_literal: Literal["queued", "running", "completed", "failed"]
        if status_v == "queued":
            status_literal = "queued"
        elif status_v == "processing":
            status_literal = "running"
        elif status_v == "completed":
            status_literal = "completed"
        else:
            # status_v == "failed" is the only remaining case per JobStatusLiteral
            status_literal = "failed"
        hb_raw = get_with_retry(self._redis, heartbeat_key(run_id))
        hb = float(hb_raw) if hb_raw is not None else None
        return RunStatusResponse(
            run_id=run_id,
            status=status_literal,
            last_heartbeat_ts=hb,
            message=status_obj["message"],
        )

    def enqueue_evaluation(
        self: TrainingOrchestrator, run_id: str, req: EvaluateRequest
    ) -> EvaluateResponse:
        status = self._job_store.load(run_id)
        if status is None:
            return EvaluateResponse(
                run_id=run_id,
                split=req["split"],
                status="failed",
                loss=None,
                perplexity=None,
                artifact_path=None,
            )
        payload: EvalJobPayload = {
            "run_id": run_id,
            "split": req["split"],
            "path_override": req.get("path_override"),
        }

        _ = self._enq.enqueue_eval(payload)
        cache: EvalCache = {
            "status": "queued",
            "split": req["split"],
            "loss": None,
            "ppl": None,
            "artifact": None,
        }
        from platform_core.json_utils import dump_json_str

        set_with_retry(
            self._redis,
            eval_key(run_id),
            dump_json_str(cache),
        )
        _logger.info(
            "eval enqueued",
            extra={
                "category": "training",
                "service": "orchestrator",
                "run_id": run_id,
                "event": "eval_enqueued",
                "split": req["split"],
            },
        )
        return EvaluateResponse(
            run_id=run_id,
            split=req["split"],
            status="queued",
            loss=None,
            perplexity=None,
            artifact_path=None,
        )

    def get_artifact_pointer(self: TrainingOrchestrator, run_id: str) -> ArtifactPointer:
        key = artifact_file_id_key(run_id)
        fid = get_with_retry(self._redis, key)
        if fid is None or str(fid).strip() == "":
            _logger.info(
                "artifact pointer not found",
                extra={
                    "category": "orchestrator",
                    "service": "training",
                    "run_id": run_id,
                    "event": "artifact_not_found",
                },
            )
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "artifact pointer not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )
        return ArtifactPointer(storage="data-bank", file_id=str(fid))

    def get_evaluation(self: TrainingOrchestrator, run_id: str) -> EvaluateResponse:
        raw = get_with_retry(self._redis, eval_key(run_id))
        if raw is None:
            _logger.info(
                "eval not found",
                extra={
                    "category": "orchestrator",
                    "service": "training",
                    "run_id": run_id,
                    "event": "eval_not_found",
                },
            )
            raise AppError(
                ModelTrainerErrorCode.EVAL_NOT_FOUND,
                "evaluation not found",
                model_trainer_status_for(ModelTrainerErrorCode.EVAL_NOT_FOUND),
            )
        from platform_core.json_utils import load_json_str

        obj = load_json_str(str(raw))
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.EVAL_NOT_FOUND,
                "evaluation cache corrupt",
                model_trainer_status_for(ModelTrainerErrorCode.EVAL_NOT_FOUND),
            )
        from typing import Literal

        status_v = obj.get("status")
        split_v = obj.get("split")
        loss_v = obj.get("loss")
        ppl_v = obj.get("ppl")
        art_v = obj.get("artifact")
        # Narrow status to expected values
        if status_v == "queued":
            status_literal: Literal["queued", "running", "completed", "failed"] = "queued"
        elif status_v == "running":
            status_literal = "running"
        elif status_v == "completed":
            status_literal = "completed"
        elif status_v == "failed":
            status_literal = "failed"
        else:
            status_literal = "failed"
        return EvaluateResponse(
            run_id=run_id,
            split=str(split_v) if isinstance(split_v, str) else "",
            status=status_literal,
            loss=float(loss_v) if isinstance(loss_v, int | float) else None,
            perplexity=float(ppl_v) if isinstance(ppl_v, int | float) else None,
            artifact_path=str(art_v) if isinstance(art_v, str) else None,
        )
