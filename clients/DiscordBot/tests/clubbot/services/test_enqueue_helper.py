from __future__ import annotations

from typing import TypedDict

from clubbot.services.enqueue import EnqueueClient, enqueue_job


class _Payload(TypedDict):
    x: int


class _Result(TypedDict):
    ok: bool
    ident: str


class _FakeClient(EnqueueClient[_Payload, _Result]):
    def enqueue(self, payload: _Payload, *, user_id: int, request_id: str) -> _Result:
        return {"ok": True, "ident": f"{user_id}:{request_id}:{payload['x']}"}


def test_enqueue_job_generic_helper() -> None:
    c = _FakeClient()
    payload: _Payload = {"x": 3}
    out = enqueue_job(c, payload, user_id=7, request_id="r")
    assert out == {"ok": True, "ident": "7:r:3"}
