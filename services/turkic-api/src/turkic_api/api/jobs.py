from __future__ import annotations

import io
from collections.abc import Generator, Mapping
from datetime import datetime
from typing import Final

from platform_core.data_bank_client import DataBankClientError
from platform_core.job_events import default_events_channel
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import get_logger
from platform_core.queues import TURKIC_QUEUE
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto
from typing_extensions import TypedDict

from turkic_api import _test_hooks
from turkic_api.api.config import Settings, settings_from_env
from turkic_api.api.job_store import TurkicJobStore
from turkic_api.api.types import LoggerProtocol
from turkic_api.core.models import ProcessSpec, is_language, is_source


# TypedDicts for job processing
class JobParams(TypedDict):
    """Parameters for job processing from queue."""

    user_id: int
    source: str
    language: str
    script: str | None
    max_sentences: int
    transliterate: bool
    confidence_threshold: float


class JobResult(TypedDict):
    """Result of job processing."""

    job_id: str
    status: str
    result: str


def _parse_script_param(script_val: str | None) -> str | None:
    """Parse and validate the optional script filter parameter.

    Raises:
        JSONTypeError: If script value is not one of 'Latn', 'Cyrl', or 'Arab'.
    """
    if script_val is None:
        return None
    s = script_val.strip()
    if not s:
        return None
    norm = s[0:1].upper() + s[1:].lower()
    if norm not in ("Latn", "Cyrl", "Arab"):
        raise JSONTypeError("Invalid script; expected 'Latn', 'Cyrl', or 'Arab'")
    return norm


def _decode_job_params(raw: dict[str, JSONValue]) -> JobParams:
    """Parse and validate job parameters from queue payload.

    Raises:
        JSONTypeError: If required fields have wrong types or invalid values.
    """
    user_id_val = raw.get("user_id")
    if not isinstance(user_id_val, int):
        raise JSONTypeError("user_id must be an int")

    src_val = raw.get("source", "")
    lang_val = raw.get("language", "")
    max_val = raw.get("max_sentences", 1000)
    translit_val = raw.get("transliterate", True)
    thr_val = raw.get("confidence_threshold", 0.95)

    if not isinstance(src_val, str) or not isinstance(lang_val, str):
        raise JSONTypeError("source and language must be strings")
    if not isinstance(max_val, int):
        raise JSONTypeError("max_sentences must be int")
    if not isinstance(translit_val, bool):
        raise JSONTypeError("transliterate must be bool")
    if not isinstance(thr_val, (int, float)):
        raise JSONTypeError("confidence_threshold must be a number")

    script_raw = raw.get("script")
    if script_raw is not None and not isinstance(script_raw, str):
        raise JSONTypeError("script must be a string or null")
    script = _parse_script_param(script_raw)

    if not is_source(src_val.strip()) or not is_language(lang_val.strip()):
        raise JSONTypeError("Invalid source or language in job parameters")

    return {
        "user_id": user_id_val,
        "source": src_val.strip(),
        "language": lang_val.strip(),
        "script": script,
        "max_sentences": max_val,
        "transliterate": translit_val,
        "confidence_threshold": float(thr_val),
    }


def process_corpus_impl(
    job_id: str,
    params: JobParams,
    *,
    redis: RedisStrProto,
    settings: Settings,
    logger: LoggerProtocol,
) -> JobResult:
    """Implementation for corpus processing with explicit injected deps."""
    user_id = params["user_id"]
    store = TurkicJobStore(redis)
    ctx: JobContext = make_job_context(
        redis=redis,
        domain="turkic",
        events_channel=default_events_channel("turkic"),
        job_id=job_id,
        user_id=user_id,
        queue_name=TURKIC_QUEUE,
    )
    created_at = datetime.utcnow()
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "processing",
            "progress": 0,
            "message": "started",
            "result_url": None,
            "file_id": None,
            "upload_status": None,
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
        },
    )
    ctx.publish_started()

    spec = _build_spec(params)
    norm_script = _normalize_script(params["script"])
    _ensure_corpus(spec, settings, norm_script)
    body_stream = _result_stream(job_id, user_id, spec, settings, store, ctx, created_at)
    file_id, result_bytes = _upload_and_record(
        settings, logger, job_id, user_id, body_stream, store, ctx, created_at
    )
    _mark_completed(job_id, user_id, logger, store, ctx, file_id, result_bytes, created_at)
    return {"job_id": job_id, "status": "completed", "result": f"/api/v1/jobs/{job_id}/result"}


def _ensure_corpus(
    spec: ProcessSpec,
    settings: Settings,
    script: str | None,
) -> None:
    lang_model = None
    filtering_needed = spec["confidence_threshold"] > 0.0 or script is not None
    if filtering_needed:
        lang_model = _test_hooks.load_langid_model(settings["data_dir"])
    _test_hooks.ensure_corpus_file(
        spec,
        settings["data_dir"],
        script,
        langid_model=lang_model,
    )


def _result_stream(
    job_id: str,
    user_id: int,
    spec: ProcessSpec,
    settings: Settings,
    store: TurkicJobStore,
    ctx: JobContext,
    created_at: datetime,
) -> Generator[bytes, None, None]:
    svc = _test_hooks.local_corpus_service_factory(settings["data_dir"])
    for idx, line in enumerate(svc.stream(spec), start=1):
        text_line = _test_hooks.to_ipa(line, spec["language"]) if spec["transliterate"] else line
        if idx % 50 == 0:
            now = datetime.utcnow()
            current_progress = min(99, idx)
            store.save(
                {
                    "job_id": job_id,
                    "user_id": user_id,
                    "status": "processing",
                    "progress": current_progress,
                    "message": "processing",
                    "result_url": None,
                    "file_id": None,
                    "upload_status": None,
                    "created_at": created_at,
                    "updated_at": now,
                    "error": None,
                },
            )
            ctx.publish_progress(current_progress, "processing")
        yield (text_line + "\n").encode("utf-8")


def _build_spec(params: JobParams) -> ProcessSpec:
    src = params["source"]
    lang = params["language"]
    assert is_source(src), "Invalid source"
    assert is_language(lang), "Invalid language"
    return ProcessSpec(
        source=src,
        language=lang,
        max_sentences=params["max_sentences"],
        transliterate=params["transliterate"],
        confidence_threshold=params["confidence_threshold"],
    )


def _normalize_script(val: str | None) -> str | None:
    if val is None:
        return None
    trimmed = val.strip()
    return None if not trimmed else trimmed[0:1].upper() + trimmed[1:].lower()


def _upload_and_record(
    settings: Settings,
    logger: LoggerProtocol,
    job_id: str,
    user_id: int,
    body_stream: Generator[bytes, None, None],
    store: TurkicJobStore,
    ctx: JobContext,
    created_at: datetime,
) -> tuple[str, int]:
    url_cfg: Final[str] = settings["data_bank_api_url"]
    key_cfg: Final[str] = settings["data_bank_api_key"]
    if url_cfg.strip() == "" or key_cfg.strip() == "":
        logger.error(
            "data-bank configuration missing",
            extra={
                "job_id": job_id,
                "has_url": bool(url_cfg.strip()),
                "has_key": bool(key_cfg.strip()),
            },
        )
        _fail_upload(job_id, user_id, "config_missing", store, ctx, created_at)
        raise DataBankClientError("data-bank configuration missing")

    logger.info("Starting upload to data-bank-api", extra={"job_id": job_id, "url": url_cfg})

    # Collect generator output into BytesIO for upload
    buffer = io.BytesIO()
    for chunk in body_stream:
        buffer.write(chunk)
    result_bytes = buffer.tell()
    buffer.seek(0)

    # Use DataBankClient via test hook
    client = _test_hooks.data_bank_client_factory(url_cfg, key_cfg, timeout_seconds=600.0)
    try:
        response = client.upload(
            file_id=f"{job_id}.txt",
            stream=buffer,
            content_type="text/plain; charset=utf-8",
            request_id=job_id,
        )
    except DataBankClientError as exc:
        _fail_upload(job_id, user_id, str(exc), store, ctx, created_at)
        logger.error("data-bank upload failed", extra={"job_id": job_id, "error": str(exc)})
        raise

    store.save_upload_metadata(job_id, response)
    fid = response["file_id"]
    now = datetime.utcnow()
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "processing",
            "progress": 100,
            "message": "uploading",
            "result_url": None,
            "file_id": fid,
            "upload_status": "uploaded",
            "created_at": created_at,
            "updated_at": now,
            "error": None,
        },
    )
    logger.info("data-bank upload succeeded", extra={"job_id": job_id, "file_id": fid})
    return fid, result_bytes


def _fail_upload(
    job_id: str,
    user_id: int,
    error: str,
    store: TurkicJobStore,
    ctx: JobContext,
    created_at: datetime,
) -> None:
    now = datetime.utcnow()
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "failed",
            "progress": 100,
            "message": "upload_failed",
            "result_url": None,
            "file_id": None,
            "upload_status": None,
            "created_at": created_at,
            "updated_at": now,
            "error": error,
        },
    )
    ctx.publish_failed("system", error)


def _mark_completed(
    job_id: str,
    user_id: int,
    logger: LoggerProtocol,
    store: TurkicJobStore,
    ctx: JobContext,
    file_id: str,
    result_bytes: int,
    created_at: datetime,
) -> None:
    now = datetime.utcnow()
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "completed",
            "progress": 100,
            "message": "done",
            "result_url": None,
            "file_id": file_id,
            "upload_status": "uploaded",
            "created_at": created_at,
            "updated_at": now,
            "error": None,
        },
    )
    ctx.publish_completed(file_id, result_bytes)
    logger.info("Job completed", extra={"job_id": job_id, "file_id": file_id})


def _decode_process_corpus(job_id: str, params: dict[str, JSONValue]) -> JobResult:
    """RQ job entry point (via _decode prefix). Loads deps from env and delegates to the impl.

    Note: params is dict[str, JSONValue] from RQ queue, decoded internally.
    Logging is initialized by worker_entry.py before the worker starts processing jobs.
    """
    settings = settings_from_env()
    logger = get_logger(__name__)
    client = _get_redis_client(settings["redis_url"])

    # Decode params from JSONValue to typed JobParams
    decoded_params = _decode_job_params(params)

    try:
        return process_corpus_impl(
            job_id, decoded_params, redis=client, settings=settings, logger=logger
        )
    finally:
        client.close()


def _get_redis_client(url: str) -> RedisStrProto:
    return _test_hooks.redis_factory(url)


def process_corpus(job_id: str, params: Mapping[str, JSONValue]) -> JobResult:
    """Public RQ job entry point for corpus processing jobs.

    This is the function that RQ workers call. It delegates to the internal
    implementation after loading settings and decoding the payload.
    """
    # Convert Mapping to concrete dict - JSONValue is compatible with JSONValue
    return _decode_process_corpus(job_id, dict(params))
