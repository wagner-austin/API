from __future__ import annotations

from collections.abc import Callable
from weakref import WeakKeyDictionary

from fastapi import FastAPI
from platform_core.errors import ErrorCode
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import get_logger, setup_logging
from platform_core.request_context import install_request_id_middleware
from platform_core.security import ApiKeyCheckFn, create_api_key_dependency

from .. import _test_hooks
from ..config import Limits, Settings, ensure_settings, limits_from_settings, load_settings
from ..inference.engine import InferenceEngine
from .routes import admin as routes_admin
from .routes import health as routes_health
from .routes import models as routes_models
from .routes import read as routes_read
from .routes import training as routes_training


def _setup_optional_reloader(
    app: FastAPI, engine: InferenceEngine, reload_interval_seconds: float | None
) -> None:
    """Optionally attach a background reloader for model artifacts.

    If `reload_interval_seconds` is falsy or non-positive, no handlers are added.
    """
    if reload_interval_seconds is None or float(reload_interval_seconds) <= 0.0:
        return

    stop_evt: _test_hooks.EventProtocol | None = None
    thread: _test_hooks.ThreadProtocol | None = None

    def _start_bg_reloader() -> None:
        nonlocal stop_evt, thread
        stop_evt = _test_hooks.event_factory()

        def _loop() -> None:
            interval = float(reload_interval_seconds)
            # Use wait on the event to allow prompt shutdown
            while not stop_evt.is_set():
                engine.reload_if_changed()
                stop_evt.wait(interval)

        thread = _test_hooks.thread_factory(target=_loop, daemon=True, name="model-reloader")
        thread.start()

    def _stop_bg_reloader() -> None:
        nonlocal stop_evt, thread
        if stop_evt is not None:
            stop_evt.set()
        if thread is not None:
            thread.join(timeout=1.0)

    app.add_event_handler("startup", _start_bg_reloader)
    app.add_event_handler("shutdown", _stop_bg_reloader)
    # Expose handlers for white-box tests via a weak map keyed by app instance
    _RELOADER_HANDLES[app] = (_start_bg_reloader, _stop_bg_reloader)


_RELOADER_HANDLES: WeakKeyDictionary[FastAPI, tuple[Callable[[], None], Callable[[], None]]] = (
    WeakKeyDictionary()
)


def _debug_invoke_reloader_start(app: FastAPI) -> None:
    pair = _RELOADER_HANDLES.get(app)
    if pair is not None:
        start, _ = pair
        start()


def _debug_invoke_reloader_stop(app: FastAPI) -> None:
    pair = _RELOADER_HANDLES.get(app)
    if pair is not None:
        _, stop = pair
        stop()


def _create_engine(settings: Settings) -> InferenceEngine:
    engine = InferenceEngine(settings)
    engine.try_load_active()
    return engine


def _seed_initial_model_if_needed(settings: Settings) -> bool:
    """Copy seed artifacts into the configured model directory when missing.

    Returns True when a seed was copied; False when no action was needed.
    """
    seed_root = settings["digits"]["seed_root"]
    model_id = settings["digits"]["active_model"]
    model_dir = settings["digits"]["model_dir"]
    dest = model_dir / model_id
    seed = seed_root / model_id

    seed_model = seed / "model.pt"
    seed_manifest = seed / "manifest.json"
    dest_model = dest / "model.pt"
    dest_manifest = dest / "manifest.json"

    should_seed = seed_model.exists() and seed_manifest.exists()
    dest_missing = not (dest_model.exists() and dest_manifest.exists())
    if not (should_seed and dest_missing):
        return False

    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_model, dest_model)
    shutil.copy2(seed_manifest, dest_manifest)
    get_logger("handwriting_ai").info("seeded_initial_model", extra={"model_id": model_id})
    return True


def create_app(
    settings: Settings | None = None,
    engine_provider: Callable[[], InferenceEngine] | None = None,
    *,
    reload_interval_seconds: float | None = None,
    enforce_api_key: bool | None = None,
) -> FastAPI:
    """Application factory.

    Parameters:
    - `settings`: Optional pre-loaded settings; when omitted, loads defaults.
    - `engine_provider`: Optional provider for a custom `InferenceEngine` (primarily for tests).
    - `reload_interval_seconds`: When provided and > 0, starts a background thread on startup
      that periodically calls `engine.reload_if_changed()` to pick up model artifact changes.
    """
    if settings is None and enforce_api_key is None:
        enforce_api_key = False
    base_settings = settings or load_settings()
    s = ensure_settings(base_settings, create_dirs=True)
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="handwriting-ai",
        instance_id=None,
        extra_fields=["request_id", "latency_ms", "digit", "confidence", "model_id", "uncertain"],
    )
    from ..version import version_string

    app = FastAPI(title="handwriting-ai", version=version_string())
    # Normalize exception groups first so downstream handlers see standard Exceptions
    from ..middleware import ExceptionNormalizeMiddleware

    install_request_id_middleware(app)
    # Add last so it becomes the outermost catcher for grouped exceptions.
    app.add_middleware(ExceptionNormalizeMiddleware)

    # Create API key dependency for protected routes
    api_enabled = s["security"]["api_key_enabled"]
    api_required_key = s["security"]["api_key"]
    if enforce_api_key is False or (enforce_api_key is None and not api_enabled):
        api_required_key = ""
    api_key_dep: ApiKeyCheckFn = create_api_key_dependency(
        required_key=api_required_key,
        error_code=ErrorCode.UNAUTHORIZED,
        http_status=401,
    )
    # Store on app state for route registration
    app.state.api_key_dep = api_key_dep

    # Perform an eager seed of initial artifacts (idempotent) before engine init so
    # that a default engine can load the active model during startup deterministically.
    _ = _seed_initial_model_if_needed(s)

    # Build shared engine instance (or use provided one) used across routes and reloader
    engine: InferenceEngine = (
        engine_provider() if engine_provider is not None else _create_engine(s)
    )
    limits = limits_from_settings(s)

    # Exception handlers
    install_exception_handlers_fastapi(app, logger_name="handwriting-ai")

    def _provide_engine() -> InferenceEngine:
        return engine

    def _provide_settings() -> Settings:
        return s

    def _provide_limits() -> Limits:
        return limits

    # Expose providers for dependency overrides in tests
    app.state.provide_engine = _provide_engine
    app.state.provide_settings = _provide_settings
    app.state.provide_limits = _provide_limits

    # Include standardized route modules
    app.include_router(routes_health.build_router(engine))
    app.include_router(routes_models.build_router(engine))
    app.include_router(routes_admin.build_router(engine, _provide_settings, api_key_dep))
    app.include_router(
        routes_read.build_router(_provide_engine, _provide_settings, _provide_limits, api_key_dep)
    )
    app.include_router(routes_training.build_router(api_key_dep))

    # Optional background reloader for model artifacts
    _setup_optional_reloader(app, engine, reload_interval_seconds)

    # Seed initial model on startup if needed (idempotent). Kept to ensure
    # environments that add seed files late still converge without a reloader.
    def _seed_startup() -> None:
        _ = _seed_initial_model_if_needed(s)

    app.add_event_handler("startup", _seed_startup)

    return app
