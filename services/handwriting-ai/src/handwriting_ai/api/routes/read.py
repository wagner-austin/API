"""Read/predict routes for handwriting-ai."""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from typing import Annotated, Protocol

from fastapi import APIRouter, File, Header, Request, UploadFile
from fastapi.params import Depends as DependsParamType
from PIL import Image, ImageFile, UnidentifiedImageError
from platform_core.errors import AppError, ErrorCode, HandwritingErrorCode, handwriting_status_for
from platform_core.logging import get_logger
from platform_core.security import ApiKeyCheckFn
from starlette.datastructures import FormData

from ...config import Limits, Settings
from ...inference.engine import InferenceEngine
from ...preprocess import PreprocessOptions, run_preprocess
from ..schemas import PredictResponse

# Non-recursive JSON value type for flat API responses
_FlatJsonValue = str | int | float | bool | list[float] | None

# Ensure truncated images fail cleanly
ImageFile.LOAD_TRUNCATED_IMAGES = False


class _DependsCtor(Protocol):
    """Protocol for FastAPI Depends constructor with typed return."""

    def __call__(self, dependency: ApiKeyCheckFn) -> DependsParamType: ...


def _typed_depends(dep: ApiKeyCheckFn) -> DependsParamType:
    """Create Depends instance with proper typing via dynamic import."""
    fastapi_mod = __import__("fastapi")
    depends_ctor: _DependsCtor = fastapi_mod.Depends
    result: DependsParamType = depends_ctor(dep)
    return result


def _raise_if_too_large(raw: bytes, limits: Limits) -> None:
    if len(raw) > limits["max_bytes"]:
        raise AppError(
            ErrorCode.PAYLOAD_TOO_LARGE,
            "File exceeds size limit",
        )


def _strict_validate_multipart(form: FormData) -> None:
    # Reject any unexpected form fields
    for key in form:
        if key != "file":
            raise AppError(
                HandwritingErrorCode.malformed_multipart,
                "Unexpected form field",
                handwriting_status_for(HandwritingErrorCode.malformed_multipart),
            )
    # Require exactly one file part
    n_files = len(form.getlist("file"))
    if n_files != 1:
        raise AppError(
            HandwritingErrorCode.malformed_multipart,
            "Multiple file parts not allowed" if n_files > 1 else "Missing file part",
            handwriting_status_for(HandwritingErrorCode.malformed_multipart),
        )


def _open_image_bytes(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise AppError(
            HandwritingErrorCode.invalid_image,
            "Failed to decode image",
            handwriting_status_for(HandwritingErrorCode.invalid_image),
        ) from None
    except Image.DecompressionBombError:
        raise AppError(
            ErrorCode.PAYLOAD_TOO_LARGE,
            "Decompression bomb triggered",
        ) from None


def _validate_image_dimensions(img: Image.Image, limits: Limits) -> None:
    w, h = img.size
    if max(w, h) > limits["max_side_px"]:
        raise AppError(
            HandwritingErrorCode.bad_dimensions,
            "Image dimensions too large",
            handwriting_status_for(HandwritingErrorCode.bad_dimensions),
        )


def _ensure_supported_content_type(ctype: str) -> None:
    if ctype not in ("image/png", "image/jpeg", "image/jpg"):
        raise AppError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "Only PNG and JPEG are supported",
        )


def build_router(
    provide_engine: Callable[[], InferenceEngine],
    provide_settings: Callable[[], Settings],
    provide_limits: Callable[[], Limits],
    api_key_dep: ApiKeyCheckFn,
) -> APIRouter:
    """Build read router with /v1/read and /v1/predict endpoints."""
    router = APIRouter()
    api_dep: DependsParamType = _typed_depends(api_key_dep)

    async def _read_digit(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        invert: bool | None = None,
        center: bool = True,
        visualize: bool = False,
        content_length: int | None = Header(default=None, alias="Content-Length"),
    ) -> dict[str, _FlatJsonValue]:
        engine = provide_engine()
        settings = provide_settings()
        limits = provide_limits()

        # Enforce strict multipart structure: exactly one 'file' part and no extras.
        form = await request.form()
        _strict_validate_multipart(form)
        ctype = (file.content_type or "").lower()
        _ensure_supported_content_type(ctype)

        if content_length is not None and content_length > limits["max_bytes"]:
            raise AppError(
                ErrorCode.PAYLOAD_TOO_LARGE,
                "Request body too large",
            )

        raw = await file.read()
        _raise_if_too_large(raw, limits)
        img = _open_image_bytes(raw)

        _validate_image_dimensions(img, limits)

        opts: PreprocessOptions = {
            "invert": invert,
            "center": center,
            "visualize": visualize,
            "visualize_max_kb": int(settings["digits"]["visualize_max_kb"]),
        }

        t0 = time.perf_counter()
        pre = run_preprocess(img, opts)

        from concurrent.futures import TimeoutError as _FutTimeout

        fut = engine.submit_predict(pre["tensor"])
        try:
            out = fut.result(timeout=float(settings["digits"]["predict_timeout_seconds"]))
        except _FutTimeout:
            fut.cancel()
            raise AppError(
                ErrorCode.TIMEOUT,
                "Prediction timed out",
            ) from None
        except RuntimeError as _err:
            # Surface a clear error when engine has no model loaded
            msg = str(_err)
            if "Model not loaded" in msg:
                raise AppError(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    "Model not loaded. Upload or train a model.",
                ) from None
            raise

        dt_ms = int((time.perf_counter() - t0) * 1000.0)

        uncertain = out["confidence"] < float(settings["digits"]["uncertain_threshold"])
        visual_b64: str | None = (
            base64.b64encode(pre["visual_png"]).decode("ascii") if pre["visual_png"] else None
        )
        # Structured log for successful read
        get_logger("handwriting_ai").info(
            "read_finished",
            extra={
                "latency_ms": dt_ms,
                "digit": int(out["digit"]),
                "confidence": float(out["confidence"]),
                "model_id": out["model_id"],
                "uncertain": bool(uncertain),
            },
        )

        return {
            "digit": int(out["digit"]),
            "confidence": float(out["confidence"]),
            "probs": [float(p) for p in out["probs"]],
            "model_id": out["model_id"],
            "visual_png_b64": visual_b64,
            "uncertain": bool(uncertain),
            "latency_ms": dt_ms,
        }

    router.add_api_route(
        "/v1/read",
        _read_digit,
        methods=["POST"],
        response_model=PredictResponse,
        dependencies=[api_dep],
    )
    router.add_api_route(
        "/v1/predict",
        _read_digit,
        methods=["POST"],
        response_model=PredictResponse,
        dependencies=[api_dep],
    )
    return router


__all__ = ["build_router"]
