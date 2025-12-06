from __future__ import annotations

import uuid
from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import generate_key, score_key
from platform_workers.redis import RedisStrProto

from ..api.schemas.runs import GenerateRequest, GenerateResponse, ScoreRequest, ScoreResponse
from ..core.config.settings import Settings
from ..core.contracts.queue import GenerateJobPayload, ScoreJobPayload
from ..core.infra.redis_utils import get_with_retry, set_with_retry
from ..core.services.queue.rq_adapter import RQEnqueuer

_logger = get_logger(__name__)


def _parse_score_topk(topk_v: JSONValue) -> list[list[tuple[str, float]]] | None:
    """Parse topk from JSON value."""
    if not isinstance(topk_v, list):
        return None
    topk: list[list[tuple[str, float]]] = []
    for pos in topk_v:
        if isinstance(pos, list):
            pos_list: list[tuple[str, float]] = []
            for item in pos:
                if isinstance(item, list) and len(item) == 2:
                    tok, prob = item[0], item[1]
                    if isinstance(tok, str) and isinstance(prob, int | float):
                        pos_list.append((tok, float(prob)))
            topk.append(pos_list)
    return topk


def _narrow_status(status_v: JSONValue) -> Literal["queued", "running", "completed", "failed"]:
    """Narrow status value to expected literal type."""
    if status_v == "queued":
        return "queued"
    if status_v == "running":
        return "running"
    if status_v == "completed":
        return "completed"
    return "failed"


class InferenceOrchestrator:
    """Orchestrates score and generate inference jobs."""

    def __init__(
        self: InferenceOrchestrator,
        *,
        settings: Settings,
        redis_client: RedisStrProto,
        enqueuer: RQEnqueuer,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer

    def enqueue_score(self: InferenceOrchestrator, run_id: str, req: ScoreRequest) -> ScoreResponse:
        """Enqueue a score job and return initial response with request_id."""
        request_id = str(uuid.uuid4())

        payload: ScoreJobPayload = {
            "run_id": run_id,
            "request_id": request_id,
            "text": req["text"],
            "path": req["path"],
            "detail_level": req["detail_level"],
            "top_k": req["top_k"],
            "seed": req["seed"],
        }

        _ = self._enq.enqueue_score(payload)

        # Store initial cache state
        cache: dict[str, JSONValue] = {
            "status": "queued",
            "loss": None,
            "perplexity": None,
            "surprisal": None,
            "topk": None,
            "tokens": None,
        }
        set_with_retry(self._redis, score_key(run_id, request_id), dump_json_str(cache))

        _logger.info(
            "score enqueued",
            extra={
                "category": "inference",
                "service": "orchestrator",
                "run_id": run_id,
                "request_id": request_id,
                "event": "score_enqueued",
            },
        )

        return {
            "request_id": request_id,
            "status": "queued",
            "loss": None,
            "perplexity": None,
            "surprisal": None,
            "topk": None,
            "tokens": None,
        }

    def get_score(self: InferenceOrchestrator, run_id: str, request_id: str) -> ScoreResponse:
        """Get the result of a score job."""
        raw = get_with_retry(self._redis, score_key(run_id, request_id))
        if raw is None:
            _logger.info(
                "score not found",
                extra={
                    "category": "inference",
                    "service": "orchestrator",
                    "run_id": run_id,
                    "request_id": request_id,
                    "event": "score_not_found",
                },
            )
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "score request not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        obj = load_json_str(str(raw))
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "score cache corrupt",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        status_literal = _narrow_status(obj.get("status"))

        # Extract values with type narrowing
        loss_v = obj.get("loss")
        ppl_v = obj.get("perplexity")
        surprisal_v = obj.get("surprisal")
        tokens_v = obj.get("tokens")

        loss: float | None = float(loss_v) if isinstance(loss_v, int | float) else None
        perplexity: float | None = float(ppl_v) if isinstance(ppl_v, int | float) else None

        surprisal: list[float] | None = None
        if isinstance(surprisal_v, list):
            surprisal = [float(x) for x in surprisal_v if isinstance(x, int | float)]

        topk = _parse_score_topk(obj.get("topk"))

        tokens: list[str] | None = None
        if isinstance(tokens_v, list):
            tokens = [str(t) for t in tokens_v if isinstance(t, str)]

        return {
            "request_id": request_id,
            "status": status_literal,
            "loss": loss,
            "perplexity": perplexity,
            "surprisal": surprisal,
            "topk": topk,
            "tokens": tokens,
        }

    def enqueue_generate(
        self: InferenceOrchestrator, run_id: str, req: GenerateRequest
    ) -> GenerateResponse:
        """Enqueue a generate job and return initial response with request_id."""
        request_id = str(uuid.uuid4())

        payload: GenerateJobPayload = {
            "run_id": run_id,
            "request_id": request_id,
            "prompt_text": req["prompt_text"],
            "prompt_path": req["prompt_path"],
            "max_new_tokens": req["max_new_tokens"],
            "temperature": req["temperature"],
            "top_k": req["top_k"],
            "top_p": req["top_p"],
            "stop_on_eos": req["stop_on_eos"],
            "stop_sequences": list(req["stop_sequences"]),
            "seed": req["seed"],
            "num_return_sequences": req["num_return_sequences"],
        }

        _ = self._enq.enqueue_generate(payload)

        # Store initial cache state
        cache: dict[str, JSONValue] = {
            "status": "queued",
            "outputs": None,
            "steps": None,
            "eos_terminated": None,
        }
        set_with_retry(self._redis, generate_key(run_id, request_id), dump_json_str(cache))

        _logger.info(
            "generate enqueued",
            extra={
                "category": "inference",
                "service": "orchestrator",
                "run_id": run_id,
                "request_id": request_id,
                "event": "generate_enqueued",
            },
        )

        return {
            "request_id": request_id,
            "status": "queued",
            "outputs": None,
            "steps": None,
            "eos_terminated": None,
        }

    def get_generate(self: InferenceOrchestrator, run_id: str, request_id: str) -> GenerateResponse:
        """Get the result of a generate job."""
        raw = get_with_retry(self._redis, generate_key(run_id, request_id))
        if raw is None:
            _logger.info(
                "generate not found",
                extra={
                    "category": "inference",
                    "service": "orchestrator",
                    "run_id": run_id,
                    "request_id": request_id,
                    "event": "generate_not_found",
                },
            )
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "generate request not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        obj = load_json_str(str(raw))
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "generate cache corrupt",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        status_literal = _narrow_status(obj.get("status"))

        # Extract values with type narrowing
        outputs_v = obj.get("outputs")
        steps_v = obj.get("steps")
        eos_v = obj.get("eos_terminated")

        outputs: list[str] | None = None
        if isinstance(outputs_v, list):
            outputs = [str(o) for o in outputs_v if isinstance(o, str)]

        steps: int | None = int(steps_v) if isinstance(steps_v, int) else None

        eos_terminated: list[bool] | None = None
        if isinstance(eos_v, list):
            eos_terminated = [bool(e) for e in eos_v if isinstance(e, bool)]

        return {
            "request_id": request_id,
            "status": status_literal,
            "outputs": outputs,
            "steps": steps,
            "eos_terminated": eos_terminated,
        }
