from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class ConversationMessage(TypedDict, total=True):
    """A single message in a conversation history."""

    role: Literal["user", "assistant"]
    content: str


class ConversationMeta(TypedDict, total=True):
    """Metadata for a conversation session."""

    run_id: str
    created_at: str
    session_ttl_sec: int


class ConversationState(TypedDict, total=True):
    """Full conversation state stored in Redis."""

    messages: list[ConversationMessage]
    meta: ConversationMeta


class ChatJobPayload(TypedDict, total=True):
    """Payload for a chat job enqueued to RQ."""

    run_id: str
    session_id: str
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float


__all__ = [
    "ChatJobPayload",
    "ConversationMessage",
    "ConversationMeta",
    "ConversationState",
]
