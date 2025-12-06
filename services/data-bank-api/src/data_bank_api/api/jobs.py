from __future__ import annotations

import io
from collections.abc import Generator
from pathlib import Path
from typing import Final, Protocol, TypedDict

from platform_core.data_bank_client import DataBankClient, DataBankClientError
from platform_core.job_events import JobDomain, default_events_channel
from platform_core.job_types import job_key
from platform_core.queues import DATA_BANK_QUEUE
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto

from ..core.corpus_download import ensure_corpus_file
from .config import Settings


class _LogExtraDict(TypedDict, total=False):
    job_id: str


class LoggerLike(Protocol):
    def info(self, msg: str, *, extra: _LogExtraDict | None = None) -> None: ...

    def error(self, msg: str, *, extra: _LogExtraDict | None = None) -> None: ...


class JobParams(TypedDict):
    source: str
    language: str
    max_sentences: int
    transliterate: bool
    confidence_threshold: float


class StatusDict(TypedDict):
    status: str


class LocalCorpusService:
    """Reads corpus lines from data_dir/corpus/{source}_{language}.txt."""

    def __init__(self, data_dir: str) -> None:
        self._root: Final[Path] = Path(data_dir) / "corpus"

    def stream(self, spec: JobParams) -> Generator[str, None, None]:
        """Stream lines from the corpus file for the given spec.

        Yields non-empty lines up to max_sentences.
        """
        path = self._root / f"{spec['source']}_{spec['language']}.txt"
        if not path.exists():
            return
        max_lines = int(spec["max_sentences"])
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped:
                    continue
                yield stripped
                count += 1
                if max_lines > 0 and count >= max_lines:
                    break


_DATABANK_DOMAIN: JobDomain = "databank"
_EVENTS_CHANNEL = default_events_channel(_DATABANK_DOMAIN)
_DEFAULT_USER_ID = 0


def process_corpus_impl(
    job_id: str,
    *,
    params: JobParams,
    redis: RedisStrProto,
    settings: Settings,
    logger: LoggerLike,
    user_id: int = _DEFAULT_USER_ID,
    queue_name: str = DATA_BANK_QUEUE,
) -> StatusDict:
    """Process a corpus and upload the results to the Data Bank API.

    The implementation is intentionally minimal and strongly typed to satisfy the
    integration behavior expected by tests.
    """
    logger.info("start job", extra={"job_id": job_id})

    ctx: JobContext = make_job_context(
        redis=redis,
        domain=_DATABANK_DOMAIN,
        events_channel=_EVENTS_CHANNEL,
        job_id=job_id,
        user_id=user_id,
        queue_name=queue_name,
    )
    ctx.publish_started()

    try:
        # Ensure corpus availability (tests monkeypatch this to a no-op)
        ensure_corpus_file(
            source=params["source"],
            language=params["language"],
            data_dir=settings["data_dir"],
            max_sentences=params["max_sentences"],
            transliterate=params["transliterate"],
            confidence_threshold=params["confidence_threshold"],
        )

        # Stream corpus into buffer
        service = LocalCorpusService(settings["data_dir"])
        buffer = io.BytesIO()
        for line in service.stream(params):
            buffer.write(f"{line}\n".encode())
        buffer.seek(0)

        # Upload to Data Bank API using DataBankClient
        api_url = settings["data_bank_api_url"]
        api_key = settings["data_bank_api_key"]
        if api_url.strip() == "" or api_key.strip() == "":
            raise DataBankClientError("data-bank configuration missing")

        client = DataBankClient(api_url, api_key, timeout_seconds=600.0)
        ctx.publish_progress(5, "uploading corpus")
        response = client.upload(
            file_id=f"{job_id}.txt",
            stream=buffer,
            content_type="text/plain; charset=utf-8",
            request_id=job_id,
        )

        redis.hset(job_key("databank", job_id), mapping={"file_id": response["file_id"]})
        result_bytes = int(response["size"])
        ctx.publish_completed(response["file_id"], result_bytes)
        logger.info("job completed", extra={"job_id": job_id})
        return {"status": "completed"}
    except Exception as exc:
        logger.error("data-bank upload failed", extra={"job_id": job_id})
        ctx.publish_failed("system", str(exc))
        raise


__all__ = [
    "JobParams",
    "LocalCorpusService",
    "LoggerLike",
    "process_corpus_impl",
]
