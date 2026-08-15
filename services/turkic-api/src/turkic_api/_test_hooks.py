"""Dependency injection for turkic-api.

This module is the registry: one module-level name per injectable dependency,
each bound to its production implementation at import time. Production code calls
the hook directly and never asks whether it has been replaced —

    from turkic_api import _test_hooks
    client = _test_hooks.redis_factory(url)

— and a test rebinds the name before exercising the code under test, restoring
it afterwards (``tests/conftest.py`` does this automatically for every hook).
There is no conditional anywhere, because there is nothing to be conditional
about: the hook is always present and always callable.

The contracts live in :mod:`turkic_api._hook_protocols` and the production
implementations in :mod:`turkic_api._hook_defaults`; this module holds only the
bindings. Both are re-exported through ``__all__`` below, so a caller needing a
Protocol or a default can keep importing it from here.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

from platform_workers.redis import RedisStrProto

from turkic_api._hook_defaults import (
    _default_build_lang_script_filter,
    _default_data_bank_downloader_factory,
    _default_data_bank_uploader_factory,
    _default_decode_optional_literal,
    _default_decode_required_literal,
    _default_ensure_corpus_file,
    _default_get_env,
    _default_langid_download,
    _default_langid_ensure_model_path,
    _default_langid_get_fasttext_factory,
    _default_load_langid_model,
    _default_local_corpus_factory,
    _default_path_exists,
    _default_path_unlink,
    _default_redis_for_kv,
    _default_stream_culturax,
    _default_stream_oscar,
    _default_stream_wikipedia_xml,
    _default_to_ipa,
    _default_wikipedia_requests_get,
    _path_unlink_with_missing_ok,
)
from turkic_api._hook_protocols import (
    BuildLangScriptFilterProtocol,
    DataBankDownloaderFactoryProtocol,
    DataBankDownloaderProtocol,
    DataBankUploaderFactoryProtocol,
    DataBankUploaderProtocol,
    DecodeOptionalLiteralProtocol,
    DecodeRequiredLiteralProtocol,
    EnsureCorpusProtocol,
    LangIdDownloadProtocol,
    LangIdEnsureModelPathProtocol,
    LangIdFastTextFactoryProtocol,
    LangIdModelFactoryProtocol,
    LangIdModelLoaderProtocol,
    LangIdModelProtocol,
    LocalCorpusServiceFactoryProtocol,
    LocalCorpusServiceProtocol,
    RawStreamProtocol,
    StreamDownloadResponseProtocol,
    ToIpaProtocol,
    WikipediaRequestsGetProtocol,
    WikipediaRequestsResponseProtocol,
    WorkerRunnerProtocol,
)

test_runner: WorkerRunnerProtocol | None = None
get_env: Callable[[str], str | None] = _default_get_env
redis_factory: Callable[[str], RedisStrProto] = _default_redis_for_kv
local_corpus_service_factory: LocalCorpusServiceFactoryProtocol = _default_local_corpus_factory
data_bank_client_factory: DataBankUploaderFactoryProtocol = _default_data_bank_uploader_factory
data_bank_downloader_factory: DataBankDownloaderFactoryProtocol = (
    _default_data_bank_downloader_factory
)
ensure_corpus_file: EnsureCorpusProtocol = _default_ensure_corpus_file
load_langid_model: LangIdModelLoaderProtocol = _default_load_langid_model
to_ipa: ToIpaProtocol = _default_to_ipa
build_lang_script_filter: BuildLangScriptFilterProtocol = _default_build_lang_script_filter
path_exists: Callable[[Path], bool] = _default_path_exists
path_unlink: Callable[[Path], None] = _path_unlink_with_missing_ok
stream_oscar_hook: Callable[[str], Generator[str, None, None]] = _default_stream_oscar
stream_wikipedia_xml_hook: Callable[[str], Generator[str, None, None]] = (
    _default_stream_wikipedia_xml
)
stream_culturax_hook: Callable[[str], Generator[str, None, None]] = _default_stream_culturax
wikipedia_requests_get: WikipediaRequestsGetProtocol = _default_wikipedia_requests_get
langid_download: LangIdDownloadProtocol = _default_langid_download
langid_ensure_model_path: LangIdEnsureModelPathProtocol = _default_langid_ensure_model_path
langid_get_fasttext_factory: LangIdFastTextFactoryProtocol = _default_langid_get_fasttext_factory
decode_required_literal: DecodeRequiredLiteralProtocol = _default_decode_required_literal
decode_optional_literal: DecodeOptionalLiteralProtocol = _default_decode_optional_literal
source_map: dict[str, str] = {
    "oscar": "oscar",
    "wikipedia": "wikipedia",
    "culturax": "culturax",
}
language_map: dict[str, str] = {
    "kk": "kk",
    "ky": "ky",
    "uz": "uz",
    "tr": "tr",
    "ug": "ug",
    "fi": "fi",
    "az": "az",
    "en": "en",
    "ru": "ru",
}


__all__ = [
    "BuildLangScriptFilterProtocol",
    "DataBankDownloaderFactoryProtocol",
    "DataBankDownloaderProtocol",
    "DataBankUploaderFactoryProtocol",
    "DataBankUploaderProtocol",
    "DecodeOptionalLiteralProtocol",
    "DecodeRequiredLiteralProtocol",
    "EnsureCorpusProtocol",
    "LangIdDownloadProtocol",
    "LangIdEnsureModelPathProtocol",
    "LangIdFastTextFactoryProtocol",
    "LangIdModelFactoryProtocol",
    "LangIdModelLoaderProtocol",
    "LangIdModelProtocol",
    "LocalCorpusServiceFactoryProtocol",
    "LocalCorpusServiceProtocol",
    "RawStreamProtocol",
    "StreamDownloadResponseProtocol",
    "ToIpaProtocol",
    "WikipediaRequestsGetProtocol",
    "WikipediaRequestsResponseProtocol",
    "WorkerRunnerProtocol",
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
    "build_lang_script_filter",
    "data_bank_client_factory",
    "data_bank_downloader_factory",
    "decode_optional_literal",
    "decode_required_literal",
    "ensure_corpus_file",
    "get_env",
    "langid_download",
    "langid_ensure_model_path",
    "langid_get_fasttext_factory",
    "language_map",
    "load_langid_model",
    "local_corpus_service_factory",
    "path_exists",
    "path_unlink",
    "redis_factory",
    "source_map",
    "stream_culturax_hook",
    "stream_oscar_hook",
    "stream_wikipedia_xml_hook",
    "test_runner",
    "to_ipa",
    "wikipedia_requests_get",
]
