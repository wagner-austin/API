from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Generator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.params import Depends as DependsParamType
from fastapi.responses import PlainTextResponse, StreamingResponse
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger

from ...core.infra.paths import model_logs_path
from ...core.logging.types import LoggingExtra
from ...core.services.container import ServiceContainer
from ..middleware import api_key_dependency
from ..schemas.runs import (
    ArtifactPointerResponse,
    CancelResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    RunStatusResponse,
    ScoreRequest,
    ScoreResponse,
    TrainRequest,
    TrainResponse,
)
from ..validators.runs import (
    _decode_chat_request,
    _decode_generate_request,
    _decode_score_request,
    _decode_train_request,
)

_logger = get_logger(__name__)


class _RunsRoutes:
    c: ServiceContainer
    # Test seam: injectable sleep to make streaming deterministic
    _sleep_fn: Callable[[float], None]
    _follow_max_loops: int | None

    def __init__(self: _RunsRoutes, container: ServiceContainer) -> None:
        self.c = container
        # Defaults (production): real time.sleep, unlimited follow
        # These can be overridden in tests to avoid non-deterministic sleeps
        import time as _time  # local import to avoid top-level import side effects

        self._sleep_fn = _time.sleep

        self._follow_max_loops = None

    async def start_training(self: _RunsRoutes, request: Request) -> TrainResponse:
        raw_body = await request.body()
        from platform_core.json_utils import load_json_str

        body = load_json_str(raw_body.decode("utf-8"))
        req: TrainRequest = _decode_train_request(body)
        orchestrator = self.c.training_orchestrator
        extra: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "event": "runs_enqueue",
            "model_family": req["model_family"],
            "model_size": req["model_size"],
        }
        _logger.info("runs enqueue", extra=extra)
        out = orchestrator.enqueue_training(req)
        return {"run_id": out["run_id"], "job_id": out["job_id"]}

    def run_status(self: _RunsRoutes, run_id: str) -> RunStatusResponse:
        orchestrator = self.c.training_orchestrator
        res = orchestrator.get_status(run_id)
        return {
            "run_id": res["run_id"],
            "status": res["status"],
            "last_heartbeat_ts": res["last_heartbeat_ts"],
            "message": res["message"],
        }

    def run_evaluate(self: _RunsRoutes, run_id: str, req: EvaluateRequest) -> EvaluateResponse:
        orchestrator = self.c.training_orchestrator
        extra2: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "run_id": run_id,
            "event": "runs_enqueue_eval",
            "split": req["split"],
        }
        _logger.info("runs enqueue eval", extra=extra2)
        res = orchestrator.enqueue_evaluation(run_id, req)
        return {
            "run_id": res["run_id"],
            "split": res["split"],
            "status": res["status"],
            "loss": res["loss"],
            "perplexity": res["perplexity"],
            "artifact_path": res["artifact_path"],
        }

    def run_eval_result(self: _RunsRoutes, run_id: str) -> EvaluateResponse:
        orchestrator = self.c.training_orchestrator
        res = orchestrator.get_evaluation(run_id)
        return {
            "run_id": res["run_id"],
            "split": res["split"],
            "status": res["status"],
            "loss": res["loss"],
            "perplexity": res["perplexity"],
            "artifact_path": res["artifact_path"],
        }

    def run_artifact_pointer(self: _RunsRoutes, run_id: str) -> ArtifactPointerResponse:
        orchestrator = self.c.training_orchestrator
        ptr = orchestrator.get_artifact_pointer(run_id)
        return {"storage": ptr["storage"], "file_id": ptr["file_id"]}

    def run_logs(self: _RunsRoutes, run_id: str, tail: int = 200) -> PlainTextResponse:
        path = str(model_logs_path(self.c.settings, run_id))
        if not os.path.exists(path):
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "logs not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail_n = max(1, int(tail))
            content = "".join(lines[-tail_n:])
            extra3: LoggingExtra = {
                "category": "api",
                "service": "runs",
                "run_id": run_id,
                "event": "runs_logs",
                "tail": tail_n,
            }
            _logger.info("runs logs", extra=extra3)
            return PlainTextResponse(content)
        except OSError:
            raise AppError(
                ModelTrainerErrorCode.LOGS_READ_FAILED,
                "failed to read logs",
                model_trainer_status_for(ModelTrainerErrorCode.LOGS_READ_FAILED),
            ) from None

    def _sse_iter(
        self: _RunsRoutes, path: str, tail: int, follow: bool
    ) -> Generator[bytes, None, None]:
        try:
            # Emit last `tail` lines immediately
            with open(path, "rb") as f:
                last: deque[bytes] = deque(maxlen=max(1, int(tail)))
                for line in f:
                    last.append(line)
            for line in last:
                yield b"data: " + line.rstrip(b"\n") + b"\n\n"
            if not follow:
                return
            # Follow the file
            with open(path, "rb") as f2:
                f2.seek(0, os.SEEK_END)
                loops = 0
                while True:
                    chunk = f2.readline()
                    if chunk:
                        yield b"data: " + chunk.rstrip(b"\n") + b"\n\n"
                    else:
                        self._sleep_fn(0.5)
                        if self._follow_max_loops is not None:
                            loops += 1
                            if loops >= self._follow_max_loops:
                                return
        except OSError as e:
            _logger.error(
                "SSE file read error",
                extra={
                    "category": "api",
                    "service": "runs",
                    "event": "runs_logs_stream_error",
                    "reason": str(e),
                },
            )
            return

    def run_logs_stream(
        self: _RunsRoutes,
        run_id: str,
        tail: int = 200,
        follow: bool = Query(True, description="Follow the log file for new lines"),
    ) -> StreamingResponse:
        path = str(model_logs_path(self.c.settings, run_id))
        if not os.path.exists(path):
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "logs not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )
        headers = {"Cache-Control": "no-cache"}
        extra4: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "run_id": run_id,
            "event": "runs_logs_stream",
            "tail": max(1, int(tail)),
        }
        _logger.info("runs logs stream", extra=extra4)
        return StreamingResponse(
            self._sse_iter(path, tail, follow), media_type="text/event-stream", headers=headers
        )

    def cancel_run(self: _RunsRoutes, run_id: str) -> CancelResponse:
        r = self.c.redis
        from platform_core.trainer_keys import cancel_key

        r.set(cancel_key(run_id), "1")
        extra5: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "run_id": run_id,
            "event": "runs_cancel",
        }
        _logger.info("runs cancel", extra=extra5)
        return CancelResponse(status="cancellation-requested")

    async def enqueue_score(self: _RunsRoutes, run_id: str, request: Request) -> ScoreResponse:
        raw_body = await request.body()
        from platform_core.json_utils import load_json_str

        body = load_json_str(raw_body.decode("utf-8"))
        req: ScoreRequest = _decode_score_request(body)
        orchestrator = self.c.inference_orchestrator
        extra: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "run_id": run_id,
            "event": "runs_enqueue_score",
        }
        _logger.info("runs enqueue score", extra=extra)
        out = orchestrator.enqueue_score(run_id, req)
        return {
            "request_id": out["request_id"],
            "status": out["status"],
            "loss": out["loss"],
            "perplexity": out["perplexity"],
            "surprisal": out["surprisal"],
            "topk": out["topk"],
            "tokens": out["tokens"],
        }

    def get_score(self: _RunsRoutes, run_id: str, request_id: str) -> ScoreResponse:
        orchestrator = self.c.inference_orchestrator
        res = orchestrator.get_score(run_id, request_id)
        return {
            "request_id": res["request_id"],
            "status": res["status"],
            "loss": res["loss"],
            "perplexity": res["perplexity"],
            "surprisal": res["surprisal"],
            "topk": res["topk"],
            "tokens": res["tokens"],
        }

    async def enqueue_generate(
        self: _RunsRoutes, run_id: str, request: Request
    ) -> GenerateResponse:
        raw_body = await request.body()
        from platform_core.json_utils import load_json_str

        body = load_json_str(raw_body.decode("utf-8"))
        req: GenerateRequest = _decode_generate_request(body)
        orchestrator = self.c.inference_orchestrator
        extra: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "run_id": run_id,
            "event": "runs_enqueue_generate",
        }
        _logger.info("runs enqueue generate", extra=extra)
        out = orchestrator.enqueue_generate(run_id, req)
        return {
            "request_id": out["request_id"],
            "status": out["status"],
            "outputs": out["outputs"],
            "steps": out["steps"],
            "eos_terminated": out["eos_terminated"],
        }

    def get_generate(self: _RunsRoutes, run_id: str, request_id: str) -> GenerateResponse:
        orchestrator = self.c.inference_orchestrator
        res = orchestrator.get_generate(run_id, request_id)
        return {
            "request_id": res["request_id"],
            "status": res["status"],
            "outputs": res["outputs"],
            "steps": res["steps"],
            "eos_terminated": res["eos_terminated"],
        }

    async def enqueue_chat(self: _RunsRoutes, run_id: str, request: Request) -> ChatResponse:
        raw_body = await request.body()
        from platform_core.json_utils import load_json_str

        body = load_json_str(raw_body.decode("utf-8"))
        req: ChatRequest = _decode_chat_request(body)
        orchestrator = self.c.conversation_orchestrator
        extra: LoggingExtra = {
            "category": "api",
            "service": "runs",
            "run_id": run_id,
            "event": "runs_enqueue_chat",
        }
        _logger.info("runs enqueue chat", extra=extra)
        out = orchestrator.enqueue_chat(run_id, req)
        return {
            "session_id": out["session_id"],
            "status": out["status"],
            "request_id": out["request_id"],
            "response": out["response"],
        }

    def get_chat_result(
        self: _RunsRoutes, run_id: str, session_id: str, request_id: str
    ) -> ChatResponse:
        orchestrator = self.c.conversation_orchestrator
        res = orchestrator.get_chat_result(run_id, session_id, request_id)
        return {
            "session_id": res["session_id"],
            "status": res["status"],
            "request_id": res["request_id"],
            "response": res["response"],
        }

    def get_chat_history(self: _RunsRoutes, run_id: str, session_id: str) -> ChatHistoryResponse:
        orchestrator = self.c.conversation_orchestrator
        res = orchestrator.get_history(run_id, session_id)
        return {
            "session_id": res["session_id"],
            "run_id": res["run_id"],
            "messages": list(res["messages"]),
            "created_at": res["created_at"],
        }

    def delete_chat_session(self: _RunsRoutes, run_id: str, session_id: str) -> CancelResponse:
        orchestrator = self.c.conversation_orchestrator
        orchestrator.delete_session(run_id, session_id)
        return CancelResponse(status="cancellation-requested")


def build_router(container: ServiceContainer) -> APIRouter:
    # Require API key for all routes under /runs
    api_dep: DependsParamType = Depends(api_key_dependency(container.settings))
    router = APIRouter(dependencies=[api_dep])
    h = _RunsRoutes(container)
    router.add_api_route("/train", h.start_training, methods=["POST"])
    router.add_api_route("/{run_id}", h.run_status, methods=["GET"])
    router.add_api_route("/{run_id}/evaluate", h.run_evaluate, methods=["POST"])
    router.add_api_route("/{run_id}/eval", h.run_eval_result, methods=["GET"])
    router.add_api_route("/{run_id}/artifact", h.run_artifact_pointer, methods=["GET"])
    router.add_api_route("/{run_id}/logs", h.run_logs, methods=["GET"])
    router.add_api_route("/{run_id}/logs/stream", h.run_logs_stream, methods=["GET"])
    router.add_api_route("/{run_id}/cancel", h.cancel_run, methods=["POST"])
    # Inference routes
    router.add_api_route("/{run_id}/score", h.enqueue_score, methods=["POST"])
    router.add_api_route("/{run_id}/score/{request_id}", h.get_score, methods=["GET"])
    router.add_api_route("/{run_id}/generate", h.enqueue_generate, methods=["POST"])
    router.add_api_route("/{run_id}/generate/{request_id}", h.get_generate, methods=["GET"])
    # Chat routes (conversation with memory)
    router.add_api_route("/{run_id}/chat", h.enqueue_chat, methods=["POST"])
    router.add_api_route(
        "/{run_id}/chat/{session_id}/{request_id}", h.get_chat_result, methods=["GET"]
    )
    router.add_api_route("/{run_id}/chat/{session_id}", h.get_chat_history, methods=["GET"])
    router.add_api_route("/{run_id}/chat/{session_id}", h.delete_chat_session, methods=["DELETE"])
    return router
