from __future__ import annotations

import bz2
import html
import re
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Final, Protocol
from xml.etree import ElementTree as ET

from platform_core.json_utils import JSONValue
from typing_extensions import TypedDict

from turkic_api.core.langid import LangIdModel, build_lang_script_filter
from turkic_api.core.models import ProcessSpec

# NOTE: We deliberately avoid Any/casts/ignores. External library usage is
# narrowed to typed access patterns.


_OSCAR_DATASET: Final[str] = "oscar-corpus/OSCAR-2301"
_CULTURAX_DATASET: Final[str] = "uonlp/CulturaX"


def _stream_hf_dataset(dataset: str, lang: str) -> Generator[str, None, None]:
    """Stream sentences from a HuggingFace dataset (streaming mode).

    Generic helper for OSCAR-style datasets. Uses HF_TOKEN from environment
    if present for gated datasets.
    """
    ds = _get_ds().load_dataset(
        dataset,
        lang,
        split="train",
        streaming=True,
        trust_remote_code=True,
        token=None,
    )

    def _decode_text_from(row: Mapping[str, str | int]) -> str | None:
        v = row.get("text")
        if isinstance(v, str):
            s2 = v.strip()
            return s2 if s2 else None
        return None

    for row in ds:
        if not isinstance(row, dict):
            continue
        s = _decode_text_from(row)
        if s is not None:
            yield s


class _DSRow(TypedDict, total=False):
    """Row from dataset - flexible dict with text field."""

    text: str


class _DSIterable(Protocol):
    def __iter__(self) -> Generator[dict[str, str | int] | int, None, None]: ...


class _DSMod(Protocol):
    def load_dataset(self, *args: JSONValue, **kwargs: JSONValue) -> _DSIterable: ...


def _get_ds() -> _DSMod:
    import importlib

    return importlib.import_module("datasets")


def stream_oscar(lang: str) -> Generator[str, None, None]:
    """Stream sentences from OSCAR via Hugging Face datasets (streaming mode).

    Requires the "datasets" package at runtime. Uses HF_TOKEN from environment
    if present for gated datasets.
    """
    yield from _stream_hf_dataset(_OSCAR_DATASET, lang)


def stream_culturax(lang: str) -> Generator[str, None, None]:
    """Stream sentences from CulturaX via Hugging Face datasets (streaming mode).

    CulturaX combines mC4 and OSCAR corpora with deduplication.
    Requires the "datasets" package at runtime. Uses HF_TOKEN from environment
    if present for gated datasets.
    """
    yield from _stream_hf_dataset(_CULTURAX_DATASET, lang)


def stream_wikipedia_xml(lang: str) -> Generator[str, None, None]:
    """Stream sentences from Wikipedia XML dump for language "lang".

    Uses the latest dump and streams/decompresses on the fly.
    Uses _test_hooks.wikipedia_requests_get for the HTTP request.
    """
    from turkic_api import _test_hooks

    dump_version = "latest"
    dump_name = f"{lang}wiki-{dump_version}-pages-articles.xml.bz2"
    url = f"https://dumps.wikimedia.org/{lang}wiki/{dump_version}/{dump_name}"
    with _test_hooks.wikipedia_requests_get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        bz_stream = bz2.BZ2File(resp.raw)
        for _, elem in ET.iterparse(bz_stream, events=("end",)):
            if (elem.tag.endswith("}text") or elem.tag == "text") and elem.text:
                etxt = "" if elem.text is None else str(elem.text)
                txt = html.unescape(re.sub(r"(?s)<.*?>", " ", etxt))
                for s in re.split(r"[.!?]", txt):
                    s_str: str = s.strip()
                    if s_str:
                        yield s_str
                elem.clear()
            else:
                elem.clear()


def _write_lines(dest: Path, lines: Generator[str, None, None], limit: int) -> int:
    count = 0
    with dest.open("w", encoding="utf-8") as fh:
        for s in lines:
            fh.write(s.replace("\n", " ").replace("\r", " ").strip() + "\n")
            count += 1
            if count >= limit:
                break
    return count


def _stream_for_source(source: str, language: str) -> Generator[str, None, None]:
    """Return a corpus stream generator for the given source.

    Uses hooks from _test_hooks to allow test injection.
    """
    from turkic_api import _test_hooks

    if source == "oscar":
        return _test_hooks.stream_oscar_hook(language)
    if source == "culturax":
        return _test_hooks.stream_culturax_hook(language)
    if source == "wikipedia":
        return _test_hooks.stream_wikipedia_xml_hook(language)
    raise ValueError(f"Unsupported corpus source: {source}")


def ensure_corpus_file(
    spec: ProcessSpec,
    data_dir: str,
    script: str | None = None,
    *,
    langid_model: LangIdModel | None = None,
) -> Path:
    """Ensure a local corpus file exists for the given spec.

    Creates data_dir/corpus/{source}_{language}.txt if missing by streaming
    from the configured remote source. Returns the path.
    """
    corpus_dir = Path(data_dir) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / f"{spec['source']}_{spec['language']}.txt"
    if path.exists():
        return path

    stream = _stream_for_source(spec["source"], spec["language"])

    # Optionally filter by language/script using FastText when a positive
    # confidence threshold is provided OR when a script filter is requested.
    if spec["confidence_threshold"] > 0.0 or script is not None:
        if langid_model is None:
            raise ValueError("langid_model is required when filtering is enabled")
        keep = build_lang_script_filter(
            target_lang=spec["language"],
            script=script,
            threshold=spec["confidence_threshold"],
            model=langid_model,
        )

        def _filtered(src: Generator[str, None, None]) -> Generator[str, None, None]:
            for s in src:
                if keep(s):
                    yield s

        source_iter: Generator[str, None, None] = _filtered(stream)
    else:
        source_iter = stream

    written = _write_lines(path, source_iter, spec["max_sentences"])
    if written == 0:
        from platform_core.logging import get_logger

        from turkic_api import _test_hooks

        logger = get_logger("turkic_api")
        try:
            _test_hooks.path_unlink(path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("corpus_zero_unlink_failed path=%s error=%s", path, exc)
        raise RuntimeError("No sentences were written for the requested corpus")
    # Touch mtime for traceability
    path.touch(exist_ok=True)
    return path
