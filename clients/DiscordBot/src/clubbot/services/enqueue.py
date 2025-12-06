from __future__ import annotations

from typing import Protocol, TypeVar

P = TypeVar("P", bound=object, contravariant=True)
R = TypeVar("R", bound=object, covariant=True)


class EnqueueClient(Protocol[P, R]):
    def enqueue(self, payload: P, *, user_id: int, request_id: str) -> R: ...


def enqueue_job(client: EnqueueClient[P, R], payload: P, *, user_id: int, request_id: str) -> R:
    return client.enqueue(payload, user_id=user_id, request_id=request_id)


__all__ = ["EnqueueClient", "enqueue_job"]
