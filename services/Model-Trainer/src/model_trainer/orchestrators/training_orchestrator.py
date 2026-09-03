from __future__ import annotations

from datetime import datetime

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger
from platform_core.trainer_keys import (
    artifact_file_id_key,
    cancel_key,
    eval_key,
    heartbeat_key,
    job_id_key,
)
from platform_workers.redis import RedisStrProto
from typing_extensions import TypedDict

from ..api.schemas.pointers import ArtifactPointer
from ..api.schemas.runs import (
    CancelResponse,
    EvaluateRequest,
    EvaluateResponse,
    ProgressResponse,
    RunStatusResponse,
    TrainRequest,
    TrainResponse,
)
from ..core import _test_hooks
from ..core.config.settings import Settings
from ..core.contracts.queue import EvalJobPayload, TrainJobPayload, TrainRequestPayload
from ..core.infra.redis_utils import get_with_retry, set_with_retry
from ..core.services.queue.rq_adapter import RQEnqueuer
from ..core.services.registries import ModelRegistry
from ..core.services.training.checkpoint import checkpoint_exists
from ..core.services.training.liveness import (
    WORKER_HEARTBEAT_TIMEOUT_SECONDS,
    seconds_since_last_sign_of_life,
    worker_death_message,
    worker_has_died,
)
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

    def _build_request_payload(
        self: TrainingOrchestrator, req: TrainRequest
    ) -> TrainRequestPayload:
        """Build the queue request payload from an API training request.

        Shared by fresh enqueues and resumes so both executions of a run
        travel through one encoding.

        Args:
            req: The decoded API training request.

        Returns:
            The queue-ready request payload.

        Raises:
            AppError: With ``CORPUS_NOT_FOUND`` when the corpus file id
                is blank.
        """
        # Pass corpus_file_id through to worker; worker resolves locally
        fid = req["corpus_file_id"].strip()
        if fid == "":  # should not occur due to schema min_length
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_FOUND,
                "corpus_file_id must be non-empty",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
            )
        return {
            "model_family": req["model_family"],
            "model_size": req["model_size"],
            "max_seq_len": req["max_seq_len"],
            "num_epochs": req["num_epochs"],
            "batch_size": req["batch_size"],
            "learning_rate": req["learning_rate"],
            "corpus_file_id": fid,
            "corpus_format": req["corpus_format"],
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
            "loss_mask_prefix_separator": req["loss_mask_prefix_separator"],
            "hub_model_id": req["hub_model_id"],
            "finetuning_strategy": req["finetuning_strategy"],
            "lora": req["lora"],
            "cartridge": req["cartridge"],
            "quantization": req["quantization"],
            "gguf_export": req["gguf_export"],
        }

    def _enqueue_execution(
        self: TrainingOrchestrator, run_id: str, req: TrainRequest, *, resume: bool
    ) -> TrainResponse:
        """Enqueue one execution of a run and record it as queued.

        Args:
            run_id: The run this execution belongs to.
            req: The decoded API training request.
            resume: Whether the worker continues from the run's checkpoint.

        Returns:
            TrainResponse naming the run and the queue job.
        """
        payload: TrainJobPayload = {
            "run_id": run_id,
            "request": self._build_request_payload(req),
            "user_id": int(req["user_id"]),
            "resume": resume,
        }
        # Enqueueing an execution supersedes any earlier cancellation of this
        # run id: the cancel flag has no expiry, so without this delete a run
        # cancelled once could never be resumed. The worker would read the
        # stale flag at its first cancellation check and stop immediately.
        # Deleting BEFORE enqueue keeps the flag's normal path intact: a
        # cancel issued after this point targets the new execution.
        _ = self._redis.delete(cancel_key(run_id))
        job_id = self._enq.enqueue_train(payload)
        # The queue knows jobs, not runs, and a run can be enqueued more than
        # once through resume. Cancelling a run that has not started yet has
        # to remove its job, so the mapping has to outlive this call.
        set_with_retry(self._redis, job_id_key(run_id), job_id)
        now = datetime.utcnow()
        self._job_store.save(
            {
                "job_id": run_id,
                "user_id": int(req["user_id"]),
                "status": "queued",
                "progress": 0,
                "message": "resume queued" if resume else "queued",
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
                "event": "resume_enqueued" if resume else "enqueued",
            },
        )
        return TrainResponse(run_id=run_id, job_id=job_id)

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
        return self._enqueue_execution(run_id, req, resume=False)

    def enqueue_resume(self: TrainingOrchestrator, run_id: str, req: TrainRequest) -> TrainResponse:
        """Re-enqueue a failed run to continue from its checkpoint.

        The run keeps its id: a resume is another execution of the same
        run, not a new run. The submitted request must carry the original
        config; the worker refuses a mismatch against the checkpoint's
        recorded fingerprint before touching the model.

        Args:
            run_id: The interrupted run to continue.
            req: The training request, identical to the original.

        Returns:
            TrainResponse naming the run and the new queue job.

        Raises:
            AppError: With ``RUN_NOT_FOUND`` when the run is unknown;
                ``RUN_NOT_RESUMABLE`` when the run is queued, genuinely still
                running, or already completed; ``CHECKPOINT_NOT_FOUND`` when no
                checkpoint file exists for the run. A run still reading
                ``processing`` whose worker has gone silent past the heartbeat
                timeout is resumable, because that status reflects a killed
                worker rather than work in progress.
        """
        status_obj = self._job_store.load(run_id)
        if status_obj is None:
            raise AppError(
                ModelTrainerErrorCode.RUN_NOT_FOUND,
                "run not found",
                model_trainer_status_for(ModelTrainerErrorCode.RUN_NOT_FOUND),
            )
        status_v = status_obj["status"]
        if status_v != "failed":
            # A run whose worker was killed still reads `processing`, because
            # nothing ran to record otherwise. Those are precisely the runs
            # worth resuming -- interrupted rather than broken, and usually
            # holding a checkpoint -- so the same predicate the status endpoint
            # uses admits them here instead of making an operator edit Redis.
            hb_raw = get_with_retry(self._redis, heartbeat_key(run_id))
            if not worker_has_died(
                status=status_v,
                last_heartbeat_ts=float(hb_raw) if hb_raw is not None else None,
                status_updated_at=status_obj["updated_at"],
                now_ts=_test_hooks.time_wall_clock(),
                timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
            ):
                raise AppError(
                    ModelTrainerErrorCode.RUN_NOT_RESUMABLE,
                    f"run '{run_id}' has status '{status_v}'; only a failed run, or one "
                    f"whose worker died, can resume",
                    model_trainer_status_for(ModelTrainerErrorCode.RUN_NOT_RESUMABLE),
                )
        if not checkpoint_exists(self._settings, run_id):
            raise AppError(
                ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND,
                f"no checkpoint exists for run '{run_id}'; resubmit it as a fresh run",
                model_trainer_status_for(ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND),
            )
        return self._enqueue_execution(run_id, req, resume=True)

    def cancel(self: TrainingOrchestrator, run_id: str) -> CancelResponse:
        """Cancel a run, removing its job from the queue if it has not started.

        The cancellation flag alone is not enough for queued work. A worker
        that dequeues a flagged job still loads the model before reaching its
        first cancellation check, so cancelling a queued run used to cost a
        full model load and left the run advertising `queued` while a job for
        it was still pending. Removing the job is what makes the cancellation
        immediate.

        The flag is set regardless, because the job may be taken between the
        status read and the removal attempt; the flag is what stops it then.

        Args:
            run_id: The run to cancel.

        Returns:
            CancelResponse reporting `dequeued` when the job was pending and
            was removed, or `cancellation-requested` when a worker already
            holds it and must stop itself.
        """
        set_with_retry(self._redis, cancel_key(run_id), "1")

        job_id = get_with_retry(self._redis, job_id_key(run_id))
        if job_id is None or not self._enq.remove_queued_job(job_id):
            return CancelResponse(status="cancellation-requested")

        # Nothing will ever run this job now, so this call owns the run's
        # terminal state. Leaving it `queued` would strand it exactly the way
        # a dead worker strands a `processing` run.
        now = datetime.utcnow()
        status_obj = self._job_store.load(run_id)
        self._job_store.save(
            {
                "job_id": run_id,
                "user_id": status_obj["user_id"] if status_obj is not None else 0,
                "status": "failed",
                "progress": 0,
                "message": "cancelled before training started",
                "created_at": status_obj["created_at"] if status_obj is not None else now,
                "updated_at": now,
                "error": ModelTrainerErrorCode.TRAINING_CANCELLED.value,
                "artifact_file_id": None,
            },
        )
        _logger.info(
            "queued run cancelled",
            extra={
                "category": "training",
                "service": "orchestrator",
                "run_id": run_id,
                "event": "cancel_dequeued",
            },
        )
        return CancelResponse(status="dequeued")

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
        hb_raw = get_with_retry(self._redis, heartbeat_key(run_id))
        hb = float(hb_raw) if hb_raw is not None else None

        # A killed container writes nothing on its way out, so `processing` on
        # its own is not evidence the run is alive. The heartbeat is, and it is
        # read here rather than by a reaper so that a caller polling for a
        # terminal state gets the truth on the very next poll.
        now_ts = _test_hooks.time_wall_clock()
        if worker_has_died(
            status=status_v,
            last_heartbeat_ts=hb,
            status_updated_at=status_obj["updated_at"],
            now_ts=now_ts,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        ):
            silence = seconds_since_last_sign_of_life(
                last_heartbeat_ts=hb,
                status_updated_at=status_obj["updated_at"],
                now_ts=now_ts,
            )
            return RunStatusResponse(
                run_id=run_id,
                status="failed",
                last_heartbeat_ts=hb,
                message=worker_death_message(run_id=run_id, silent_for_seconds=silence),
                error=ModelTrainerErrorCode.RUN_WORKER_DIED.value,
            )

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
        return RunStatusResponse(
            run_id=run_id,
            status=status_literal,
            last_heartbeat_ts=hb,
            message=status_obj["message"],
            error=status_obj["error"],
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

    def get_progress(self: TrainingOrchestrator, run_id: str) -> ProgressResponse:
        """Get detailed training progress for a run.

        Args:
            run_id: Training run identifier.

        Returns:
            ProgressResponse with current training metrics and phase.

        Raises:
            AppError: If run or progress not found.
        """
        from ..worker.progress_store import ProgressStore

        progress_store = ProgressStore(self._redis)
        progress = progress_store.load(run_id)
        if progress is None:
            # Check if job exists at all
            status_obj = self._job_store.load(run_id)
            if status_obj is None:
                _logger.info(
                    "progress not found",
                    extra={
                        "category": "orchestrator",
                        "service": "training",
                        "run_id": run_id,
                        "event": "progress_not_found",
                    },
                )
                raise AppError(
                    ModelTrainerErrorCode.RUN_NOT_FOUND,
                    "run not found",
                    model_trainer_status_for(ModelTrainerErrorCode.RUN_NOT_FOUND),
                )
            # Job exists but no progress yet - return initial state
            from datetime import datetime

            return ProgressResponse(
                run_id=run_id,
                phase="queued",
                epoch=0,
                total_epochs=0,
                step=0,
                total_steps=0,
                train_loss=0.0,
                train_ppl=0.0,
                grad_norm=0.0,
                samples_per_sec=0.0,
                val_loss=None,
                val_ppl=None,
                updated_at=datetime.utcnow().isoformat(),
            )
        return ProgressResponse(
            run_id=progress["run_id"],
            phase=progress["phase"],
            epoch=progress["epoch"],
            total_epochs=progress["total_epochs"],
            step=progress["step"],
            total_steps=progress["total_steps"],
            train_loss=progress["train_loss"],
            train_ppl=progress["train_ppl"],
            grad_norm=progress["grad_norm"],
            samples_per_sec=progress["samples_per_sec"],
            val_loss=progress["val_loss"],
            val_ppl=progress["val_ppl"],
            updated_at=progress["updated_at"],
        )
