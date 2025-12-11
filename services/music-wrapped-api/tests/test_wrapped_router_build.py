from __future__ import annotations

from music_wrapped_api.api.routes.wrapped import build_router


def test_build_router_executes() -> None:
    router = build_router()
    # Simple assertion to ensure execution path and object type
    typename = str(type(router))
    assert "APIRouter" in typename
