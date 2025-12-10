from __future__ import annotations

from fastapi import APIRouter, Request
from platform_core.json_utils import load_json_bytes
from starlette.responses import Response

from ...generator import generate_png
from ...types import QROptions
from ...validators import Defaults, _decode_qr_payload


def build_router(defaults: Defaults) -> APIRouter:
    router = APIRouter()

    async def _qr_handler(request: Request) -> Response:
        body = await request.body()
        payload = load_json_bytes(body)
        opts: QROptions = _decode_qr_payload(payload, defaults)
        png = generate_png(opts)
        return Response(content=png, media_type="image/png")

    router.add_api_route("/v1/qr", _qr_handler, methods=["POST"])
    return router


__all__ = ["build_router"]
