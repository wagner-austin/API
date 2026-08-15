"""The production implementation behind every hook.

These are what the service actually runs: real Redis, real HTTP, the real
filesystem, the real transliterator. They are separated from the hook variables
so that reading "what does this hook do in production" does not mean reading
past two hundred lines of Protocol declarations.

Several import from :mod:`turkic_api.core` inside the function body rather than
at module scope. That is deliberate: those modules import this package's hooks,
and a module-scope import would close the cycle.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Generator
from pathlib import Path

import requests
from platform_core.config import _optional_env_str
from platform_core.data_bank_client import DataBankClient
from platform_core.json_utils import JSONValue
from platform_workers.redis import RedisStrProto, redis_for_kv

from turkic_api._hook_protocols import (
    DataBankDownloaderProtocol,
    DataBankUploaderProtocol,
    LangIdModelFactoryProtocol,
    LangIdModelProtocol,
    LocalCorpusServiceProtocol,
    RawStreamProtocol,
    WikipediaRequestsResponseProtocol,
)
from turkic_api.core.models import ProcessSpec


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return _optional_env_str(key)


def _default_redis_for_kv(url: str) -> RedisStrProto:
    """Production implementation - creates real Redis client."""
    return redis_for_kv(url)


def _default_local_corpus_factory(data_dir: str) -> LocalCorpusServiceProtocol:
    """Production implementation - creates real LocalCorpusService."""
    # Import lazily to avoid circular import
    from turkic_api.core.corpus import LocalCorpusService

    return LocalCorpusService(data_dir)


def _default_data_bank_uploader_factory(
    api_url: str, api_key: str, *, timeout_seconds: float
) -> DataBankUploaderProtocol:
    """Production implementation - creates real DataBankClient."""
    return DataBankClient(api_url, api_key, timeout_seconds=timeout_seconds)


def _default_data_bank_downloader_factory(
    api_url: str, api_key: str, *, timeout_seconds: float
) -> DataBankDownloaderProtocol:
    """Production implementation - creates real DataBankClient for downloading."""
    return DataBankClient(api_url, api_key, timeout_seconds=timeout_seconds)


def _default_ensure_corpus_file(
    spec: ProcessSpec,
    data_dir: str,
    script: str | None = None,
    *,
    langid_model: LangIdModelProtocol | None = None,
) -> Path:
    """Production implementation - calls real ensure_corpus_file."""
    from turkic_api.core.corpus_download import ensure_corpus_file as _ensure

    return _ensure(spec, data_dir, script, langid_model=langid_model)


def _default_load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModelProtocol:
    """Production implementation - loads real langid model."""
    from turkic_api.core.langid import load_langid_model as _load

    return _load(data_dir, prefer_218e=prefer_218e)


def _default_langid_download(url: str, dest: Path) -> None:
    """Production implementation - downloads file via requests."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def _default_langid_ensure_model_path(data_dir: str, prefer_218e: bool = True) -> Path:
    """Production implementation - ensures model path exists.

    Downloads through the ``langid_download`` *hook* rather than through
    :func:`_default_langid_download` directly, so that a test overriding the
    download hook also governs what this does. While every hook lived in one
    module that indirection was an unqualified name and easy to miss; it is
    named explicitly here.

    Args:
        data_dir (str): Directory the model lives under.
        prefer_218e (bool): Prefer lid218e over lid176.

    Returns:
        Path: Path to the model file, downloaded if it was absent.
    """
    from turkic_api import _test_hooks
    from turkic_api.core.langid import (
        _MODEL_DIRNAME,
        _URL_176,
        _URL_218E,
    )

    base = Path(data_dir) / _MODEL_DIRNAME
    path_218e = base / "lid218e.bin"
    path_176 = base / "lid.176.bin"
    if prefer_218e:
        if not path_218e.exists():
            _test_hooks.langid_download(url=_URL_218E, dest=path_218e)
        return path_218e
    if not path_176.exists():
        _test_hooks.langid_download(url=_URL_176, dest=path_176)
    return path_176


def _default_langid_get_fasttext_factory() -> LangIdModelFactoryProtocol:
    """Production implementation - gets FastText model factory."""
    ft_module = __import__("fasttext.FastText", fromlist=["_FastText"])
    factory: LangIdModelFactoryProtocol = ft_module._FastText
    return factory


def _default_wikipedia_requests_get(
    url: str, *, stream: bool, timeout: int
) -> WikipediaRequestsResponseProtocol:
    """Production implementation - uses real requests.get.

    Note: requests.Response structurally implements WikipediaRequestsResponseProtocol
    even though mypy can't prove it due to __enter__ returning Self. At runtime
    this works correctly because we only access .raw and .raise_for_status().
    """
    # The response implements the protocol structurally at runtime.
    # We use a factory function to satisfy the hook type while letting
    # the real requests.Response be used.
    resp = requests.get(url, stream=stream, timeout=timeout)

    class _Adapter:
        """Adapter to satisfy WikipediaRequestsResponseProtocol."""

        @property
        def raw(self) -> RawStreamProtocol:
            """Return the raw response stream."""
            return resp.raw

        def raise_for_status(self) -> None:
            resp.raise_for_status()

        def __enter__(self) -> WikipediaRequestsResponseProtocol:
            resp.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            resp.__exit__(exc_type, exc_val, exc_tb)

    return _Adapter()


def _default_to_ipa(text: str, lang: str) -> str:
    """Production implementation - calls real to_ipa.

    Args:
        text: Source text to transliterate.
        lang: ISO 639 language code controlling the transliteration table.

    Returns:
        IPA representation of ``text`` for the given language.
    """
    from turkic_api.core.translit import to_ipa as _to_ipa

    return _to_ipa(text, lang)


def _default_build_lang_script_filter(
    *,
    target_lang: str,
    script: str | None,
    threshold: float,
    model: LangIdModelProtocol,
) -> Callable[[str], bool]:
    """Production implementation - calls real build_lang_script_filter.

    Args:
        target_lang: ISO 639 language code to keep.
        script: Optional ISO 15924 script code that lines must also match.
        threshold: Minimum FastText probability for the language label.
        model: Language identification model.

    Returns:
        Callable taking a line and returning True if it passes the filter.
    """
    from turkic_api.core.langid import build_lang_script_filter as _build

    return _build(target_lang=target_lang, script=script, threshold=threshold, model=model)


def _default_decode_required_literal(
    val: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str:
    """Production implementation - uses real validator."""
    from turkic_api.api.validators import _decode_required_literal as _decode

    return _decode(val, field, allowed)


def _default_decode_optional_literal(
    val: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    """Production implementation - uses real validator."""
    from turkic_api.api.validators import _decode_optional_literal as _decode

    return _decode(val, field, allowed)


def _default_path_exists(path: Path) -> bool:
    """Production implementation - checks if path exists on filesystem."""
    return path.exists()


def _default_path_unlink(path: Path, *, missing_ok: bool = False) -> None:
    """Production implementation - unlinks a file from the filesystem."""
    path.unlink(missing_ok=missing_ok)


def _default_stream_oscar(lang: str) -> Generator[str, None, None]:
    """Production implementation - streams from OSCAR dataset."""
    from turkic_api.core.corpus_download import stream_oscar as _stream

    yield from _stream(lang)


def _default_stream_wikipedia_xml(lang: str) -> Generator[str, None, None]:
    """Production implementation - streams from Wikipedia XML dump."""
    from turkic_api.core.corpus_download import stream_wikipedia_xml as _stream

    yield from _stream(lang)


def _default_stream_culturax(lang: str) -> Generator[str, None, None]:
    """Production implementation - streams from CulturaX dataset."""
    from turkic_api.core.corpus_download import stream_culturax as _stream

    yield from _stream(lang)


def _path_unlink_with_missing_ok(path: Path) -> None:
    """Default path_unlink hook - unlinks with missing_ok=True."""
    _default_path_unlink(path, missing_ok=True)


__all__ = [
    "_default_build_lang_script_filter",
    "_default_data_bank_downloader_factory",
    "_default_data_bank_uploader_factory",
    "_default_decode_optional_literal",
    "_default_decode_required_literal",
    "_default_ensure_corpus_file",
    "_default_get_env",
    "_default_langid_download",
    "_default_langid_ensure_model_path",
    "_default_langid_get_fasttext_factory",
    "_default_load_langid_model",
    "_default_local_corpus_factory",
    "_default_path_exists",
    "_default_path_unlink",
    "_default_redis_for_kv",
    "_default_stream_culturax",
    "_default_stream_oscar",
    "_default_stream_wikipedia_xml",
    "_default_to_ipa",
    "_default_wikipedia_requests_get",
    "_path_unlink_with_missing_ok",
]
