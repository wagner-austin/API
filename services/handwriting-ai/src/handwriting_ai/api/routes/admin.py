"""Admin routes for handwriting-ai."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Protocol

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.params import Depends as DependsParamType
from platform_core.errors import AppError, HandwritingErrorCode, handwriting_status_for
from platform_core.logging import get_logger
from platform_core.security import ApiKeyCheckFn

from ...config import Settings
from ...inference.engine import InferenceEngine
from ...inference.engine import _load_state_dict_file as _engine_load_state_dict_file
from ...inference.engine import _validate_state_dict as _engine_validate_state_dict
from ...inference.manifest import from_json_manifest

# Non-recursive JSON value type for flat API responses
_FlatJsonValue = str | int | float | bool | list[float] | None

# Defaults for admin upload endpoint parameters (satisfy Ruff B008).
_form_model_id: str = Form(...)
_form_activate: bool = Form(False)
_file_required_manifest: UploadFile = File(...)
_file_required_model: UploadFile = File(...)


class _DependsCtor(Protocol):
    """Protocol for FastAPI Depends constructor with typed return."""

    def __call__(self, dependency: ApiKeyCheckFn) -> DependsParamType: ...


def _typed_depends(dep: ApiKeyCheckFn) -> DependsParamType:
    """Create Depends instance with proper typing via dynamic import."""
    fastapi_mod = __import__("fastapi")
    depends_ctor: _DependsCtor = fastapi_mod.Depends
    result: DependsParamType = depends_ctor(dep)
    return result


def build_router(
    engine: InferenceEngine,
    provide_settings: Callable[[], Settings],
    api_key_dep: ApiKeyCheckFn,
) -> APIRouter:
    """Build admin router with /v1/admin/models/upload endpoint."""
    router = APIRouter()
    api_dep: DependsParamType = _typed_depends(api_key_dep)

    async def _upload_model(
        model_id: str = _form_model_id,
        activate: bool = _form_activate,
        manifest: UploadFile = _file_required_manifest,
        model: UploadFile = _file_required_model,
    ) -> dict[str, _FlatJsonValue]:
        get_logger("handwriting_ai").info(
            f"upload_files_received manifest.filename={manifest.filename} "
            f"model.filename={model.filename} same_object={manifest is model}"
        )
        man_bytes = await manifest.read()
        try:
            man = from_json_manifest(man_bytes.decode("utf-8"))
        except ValueError:
            code = HandwritingErrorCode.preprocessing_failed
            raise AppError(code, "Invalid manifest", handwriting_status_for(code)) from None
        from ...preprocess import preprocess_signature as _sig

        if man["preprocess_hash"] != _sig():
            code = HandwritingErrorCode.preprocessing_failed
            raise AppError(code, "Preprocess signature mismatch", handwriting_status_for(code))
        if man["model_id"] != model_id:
            code = HandwritingErrorCode.preprocessing_failed
            raise AppError(code, "Model id mismatch", handwriting_status_for(code))

        s = provide_settings()
        dest = s["digits"]["model_dir"] / model_id
        dest.mkdir(parents=True, exist_ok=True)
        model_bytes = await model.read()
        get_logger("handwriting_ai").info(
            f"admin_upload_received model_bytes={len(model_bytes)} manifest_bytes={len(man_bytes)}"
        )
        (dest / "manifest.json").write_text(man_bytes.decode("utf-8"), encoding="utf-8")
        (dest / "model.pt").write_bytes(model_bytes)
        # Verify write completed successfully
        written_size = (dest / "model.pt").stat().st_size
        get_logger("handwriting_ai").info(f"admin_upload_written size_bytes={written_size}")

        # Two-phase validation per design:
        # - When activate=True: strictly validate by loading the state dict and checking shape,
        #   then optionally reload the active engine.
        # - When activate=False: ensure the model file is non-empty (transport sanity) but do not
        #   attempt to parse weights; the model may be validated at activation time.
        if activate:
            load_errors = (RuntimeError, ValueError, TypeError, OSError, pickle.UnpicklingError)
            try:
                sd = _engine_load_state_dict_file(dest / "model.pt")
                _engine_validate_state_dict(sd, man["arch"], int(man["n_classes"]))
            except load_errors:
                raise AppError(
                    HandwritingErrorCode.invalid_model,
                    "Invalid model file",
                    handwriting_status_for(HandwritingErrorCode.invalid_model),
                ) from None

            if model_id == s["digits"]["active_model"]:
                try:
                    engine.try_load_active()
                except (RuntimeError, ValueError, OSError, TypeError) as exc:
                    get_logger("handwriting_ai").error(
                        "admin_reload_failed error_type=%s error=%s",
                        type(exc).__name__,
                        str(exc),
                    )
                    raise
        else:
            # Basic transport-level check: avoid persisting empty artifacts
            if written_size <= 0:
                raise AppError(
                    HandwritingErrorCode.invalid_model,
                    "Invalid model file",
                    handwriting_status_for(HandwritingErrorCode.invalid_model),
                )

        out: dict[str, _FlatJsonValue] = {
            "ok": True,
            "model_id": model_id,
            "run_id": man["created_at"].isoformat(),
        }
        return out

    router.add_api_route(
        "/v1/admin/models/upload",
        _upload_model,
        methods=["POST"],
        dependencies=[api_dep],
    )
    return router


__all__ = ["build_router"]
