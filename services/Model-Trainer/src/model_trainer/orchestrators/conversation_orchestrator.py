"""Orchestrator for conversation/chat with persistent memory."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_core.logging import get_logger
from platform_core.trainer_keys import conversation_key, conversation_meta_key
from platform_workers.redis import RedisStrProto

from ..api.schemas.runs import ChatHistoryResponse, ChatMessage, ChatRequest, ChatResponse
from ..core.config.settings import Settings
from ..core.contracts.conversation import ChatJobPayload
from ..core.infra.redis_utils import get_with_retry, set_with_retry
from ..core.services.queue.rq_adapter import RQEnqueuer

_logger = get_logger(__name__)

_CHAT_RESULT_KEY_PREFIX = "runs:chat:"
_DEFAULT_SESSION_TTL_SEC = 3600  # 1 hour


def _chat_result_key(run_id: str, session_id: str, request_id: str) -> str:
    return f"{_CHAT_RESULT_KEY_PREFIX}{run_id}:{session_id}:{request_id}"


def _narrow_status(status_v: JSONValue) -> Literal["queued", "running", "completed", "failed"]:
    """Narrow status value to expected literal type."""
    if status_v == "queued":
        return "queued"
    if status_v == "running":
        return "running"
    if status_v == "completed":
        return "completed"
    return "failed"


def _decode_messages_from_json(obj: JSONValue) -> list[ChatMessage]:
    """Decode chat messages from JSON value."""
    if not isinstance(obj, list):
        return []
    messages: list[ChatMessage] = []
    for item in obj:
        if isinstance(item, dict):
            role_v = item.get("role")
            content_v = item.get("content")
            if role_v in ("user", "assistant") and isinstance(content_v, str):
                role: Literal["user", "assistant"] = "user" if role_v == "user" else "assistant"
                messages.append({"role": role, "content": content_v})
    return messages


class ConversationOrchestrator:
    """Orchestrates chat sessions with conversation memory in Redis."""

    def __init__(
        self: ConversationOrchestrator,
        *,
        settings: Settings,
        redis_client: RedisStrProto,
        enqueuer: RQEnqueuer,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer

    def enqueue_chat(self: ConversationOrchestrator, run_id: str, req: ChatRequest) -> ChatResponse:
        """Enqueue a chat job and return initial response with session_id and request_id."""
        # Get or create session
        session_id = req["session_id"]
        if session_id is None:
            session_id = str(uuid.uuid4())
            # Initialize new session
            self._init_session(run_id, session_id)

        request_id = str(uuid.uuid4())

        payload: ChatJobPayload = {
            "run_id": run_id,
            "session_id": session_id,
            "request_id": request_id,
            "prompt": req["message"],
            "max_new_tokens": req["max_new_tokens"],
            "temperature": req["temperature"],
            "top_k": req["top_k"],
            "top_p": req["top_p"],
        }

        _ = self._enq.enqueue_chat(payload)

        # Store initial cache state
        cache: dict[str, JSONValue] = {
            "status": "queued",
            "response": None,
        }
        result_key = _chat_result_key(run_id, session_id, request_id)
        set_with_retry(self._redis, result_key, dump_json_str(cache))

        _logger.info(
            "chat enqueued",
            extra={
                "category": "conversation",
                "service": "orchestrator",
                "run_id": run_id,
                "session_id": session_id,
                "request_id": request_id,
                "event": "chat_enqueued",
            },
        )

        return {
            "session_id": session_id,
            "status": "queued",
            "request_id": request_id,
            "response": None,
        }

    def get_chat_result(
        self: ConversationOrchestrator, run_id: str, session_id: str, request_id: str
    ) -> ChatResponse:
        """Get the result of a chat job."""
        result_key = _chat_result_key(run_id, session_id, request_id)
        raw = get_with_retry(self._redis, result_key)
        if raw is None:
            _logger.info(
                "chat result not found",
                extra={
                    "category": "conversation",
                    "service": "orchestrator",
                    "run_id": run_id,
                    "session_id": session_id,
                    "request_id": request_id,
                    "event": "chat_not_found",
                },
            )
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "chat request not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        obj = load_json_str(str(raw))
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "chat cache corrupt",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        status_literal = _narrow_status(obj.get("status"))
        response_v = obj.get("response")
        response: str | None = str(response_v) if isinstance(response_v, str) else None

        return {
            "session_id": session_id,
            "status": status_literal,
            "request_id": request_id,
            "response": response,
        }

    def get_history(
        self: ConversationOrchestrator, run_id: str, session_id: str
    ) -> ChatHistoryResponse:
        """Get conversation history for a session."""
        conv_key = conversation_key(run_id, session_id)
        meta_key = conversation_meta_key(run_id, session_id)

        raw_messages = get_with_retry(self._redis, conv_key)
        raw_meta = get_with_retry(self._redis, meta_key)

        if raw_meta is None:
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "session not found",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        meta_obj = load_json_str(str(raw_meta))
        if not isinstance(meta_obj, dict):
            raise AppError(
                ModelTrainerErrorCode.DATA_NOT_FOUND,
                "session meta corrupt",
                model_trainer_status_for(ModelTrainerErrorCode.DATA_NOT_FOUND),
            )

        created_at_v = meta_obj.get("created_at")
        created_at = str(created_at_v) if isinstance(created_at_v, str) else ""

        messages: list[ChatMessage] = []
        if raw_messages is not None and isinstance(raw_messages, str):
            msg_obj = load_json_str(raw_messages)
            messages = _decode_messages_from_json(msg_obj)

        return {
            "session_id": session_id,
            "run_id": run_id,
            "messages": messages,
            "created_at": created_at,
        }

    def delete_session(self: ConversationOrchestrator, run_id: str, session_id: str) -> None:
        """Delete a conversation session."""
        conv_key = conversation_key(run_id, session_id)
        meta_key = conversation_meta_key(run_id, session_id)

        self._redis.delete(conv_key)
        self._redis.delete(meta_key)

        _logger.info(
            "session deleted",
            extra={
                "category": "conversation",
                "service": "orchestrator",
                "run_id": run_id,
                "session_id": session_id,
                "event": "session_deleted",
            },
        )

    def _init_session(self: ConversationOrchestrator, run_id: str, session_id: str) -> None:
        """Initialize a new conversation session in Redis."""
        conv_key = conversation_key(run_id, session_id)
        meta_key = conversation_meta_key(run_id, session_id)

        # Initialize empty message list
        set_with_retry(self._redis, conv_key, dump_json_str([]))

        # Store metadata
        now = datetime.now(UTC).isoformat()
        meta: dict[str, JSONValue] = {
            "run_id": run_id,
            "created_at": now,
            "session_ttl_sec": _DEFAULT_SESSION_TTL_SEC,
        }
        set_with_retry(self._redis, meta_key, dump_json_str(meta))

        # Set TTL on keys
        self._redis.expire(conv_key, _DEFAULT_SESSION_TTL_SEC)
        self._redis.expire(meta_key, _DEFAULT_SESSION_TTL_SEC)

        _logger.info(
            "session initialized",
            extra={
                "category": "conversation",
                "service": "orchestrator",
                "run_id": run_id,
                "session_id": session_id,
                "event": "session_init",
            },
        )


__all__ = ["ConversationOrchestrator"]
