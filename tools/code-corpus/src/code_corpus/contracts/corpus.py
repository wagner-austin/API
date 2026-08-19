"""The corpus contract: emitted source-file records and the emission manifest.

A corpus file is a list of documents with no memory of how it was produced,
and a style-training run whose inputs cannot be reconstructed proves nothing.
Every emission therefore writes a manifest beside the JSONL outputs recording
the repositories' commits, the counts of what was kept and excluded, and the
digests of what was written.

The holdout split is recorded because the guard-pass evaluation depends on it:
prompts are built from held-out files, and a manifest that claimed a split
while holding none would pin the evaluation to nothing. The dirty flag on each
repository pin exists because a corpus emitted from an uncommitted working
tree cannot be reproduced from the recorded commit alone; the flag makes that
condition visible instead of silently untrue.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

SHA256_HEX_LENGTH = 64
COMMIT_HEX_LENGTH = 40

_HEX_DIGITS = frozenset("0123456789abcdef")


class LanguageStats(TypedDict):
    """Per-language totals over every file the corpus kept.

    Attributes:
        files: Files of this language across train and holdout.
        tokens_approx: Estimated tokens across those files, chars/4, used for
            sizing runs and not as a measurement.
    """

    files: int
    tokens_approx: int


class RepoPin(TypedDict):
    """The state one repository was in when the corpus was emitted.

    Attributes:
        name: Short repository name, used as the path prefix in document
            headers; never contains a slash.
        commit: Full commit hash of the repository's HEAD.
        dirty: Whether the working tree differed from that commit. A dirty
            emission cannot be reproduced from the commit alone.
    """

    name: str
    commit: str
    dirty: bool


class SourceFileRecord(TypedDict):
    """One corpus document: a source file rendered for training.

    Attributes:
        repo: Name of the repository the file came from.
        path: Repository-relative path, forward-slashed.
        language: Language the file was selected as.
        sha256: Digest of the file's normalized content (UTF-8, LF endings),
            NOT of ``text``. It identifies the source file for deduplication
            and provenance; ``text`` additionally carries the path header.
        tokens_approx: Estimated tokens of ``text``, chars/4, for sizing only.
        text: The training document: a path-header comment line followed by
            the file's content.
    """

    repo: str
    path: str
    language: str
    sha256: str
    tokens_approx: int
    text: str


class CodeCorpusManifest(TypedDict):
    """Provenance for one corpus emission.

    Attributes:
        train_output: Filename of the emitted training JSONL.
        train_sha256: Digest of the training file's emitted bytes.
        holdout_output: Filename of the emitted holdout JSONL.
        holdout_sha256: Digest of the holdout file's emitted bytes.
        seed: Seed behind the holdout sample and the train-order shuffle.
        holdout_fraction: Fraction of selected files held out, by file.
        repos: State of every repository that contributed files.
        files_train: Documents written to the training file.
        files_holdout: Documents written to the holdout file.
        excluded_generated: Files skipped because they are generator output.
        excluded_duplicate: Files skipped because an identical file was
            already kept.
        excluded_empty: Files skipped because they held only whitespace.
        languages: Per-language totals over the kept files.
        tokens_approx_train: Estimated tokens across training documents.
        tokens_approx_holdout: Estimated tokens across holdout documents.
    """

    train_output: str
    train_sha256: str
    holdout_output: str
    holdout_sha256: str
    seed: int
    holdout_fraction: float
    repos: list[RepoPin]
    files_train: int
    files_holdout: int
    excluded_generated: int
    excluded_duplicate: int
    excluded_empty: int
    languages: dict[str, LanguageStats]
    tokens_approx_train: int
    tokens_approx_holdout: int


def _require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field that must not be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def _require_hex(obj: dict[str, JSONValue], key: str, length: int) -> str:
    """Read a required lowercase-hex digest field of a fixed length.

    Args:
        obj: Object being decoded.
        key: Field name.
        length: Exact number of hex characters required.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, of the wrong
            length, or holds a non-hex or uppercase character. A truncated or
            re-cased digest no longer names the bytes it was taken from.
    """
    value = require_str(obj, key)
    if len(value) != length or any(ch not in _HEX_DIGITS for ch in value):
        raise JSONTypeError(
            f"Field '{key}' must be {length} lowercase hex characters, got {value!r}"
        )
    return value


def _require_count(obj: dict[str, JSONValue], key: str) -> int:
    """Read a required count field that cannot be negative.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not an integer, or negative.
            A negative count means the emitter miscounted, which is not a
            state any reader should try to interpret.
    """
    value = require_int(obj, key)
    if value < 0:
        raise JSONTypeError(f"Field '{key}' must not be negative, got {value}")
    return value


def _require_repo_name(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required repository-name field.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, empty, or
            contains a slash. Repository names prefix document paths, so a
            slash would make ``repo/path`` ambiguous.
    """
    value = _require_nonempty_str(obj, key)
    if "/" in value or "\\" in value:
        raise JSONTypeError(f"Field '{key}' must not contain a slash, got {value!r}")
    return value


def _require_relative_posix_path(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required repository-relative forward-slashed path field.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, empty, absolute,
            backslashed, or escaping its repository via a ``..`` segment.
    """
    value = _require_nonempty_str(obj, key)
    if "\\" in value:
        raise JSONTypeError(f"Field '{key}' must be forward-slashed, got {value!r}")
    if value.startswith("/"):
        raise JSONTypeError(f"Field '{key}' must be repository-relative, got {value!r}")
    if ".." in value.split("/"):
        raise JSONTypeError(f"Field '{key}' must not escape its repository, got {value!r}")
    return value


def _require_object(value: JSONValue, what: str) -> dict[str, JSONValue]:
    """Narrow a decoded JSON value to an object.

    Args:
        value: Value produced by the JSON loader.
        what: Name of the thing being decoded, used in the error message.

    Returns:
        The value as a JSON object.

    Raises:
        JSONTypeError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def encode_language_stats(stats: LanguageStats) -> dict[str, JSONValue]:
    """Encode per-language totals to a JSON object.

    Args:
        stats: Totals to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "files": stats["files"],
        "tokens_approx": stats["tokens_approx"],
    }


def decode_language_stats(value: JSONValue, language: str) -> LanguageStats:
    """Decode and validate a JSON value into per-language totals.

    Args:
        value: Value produced by the JSON loader.
        language: Language name, used in error messages.

    Returns:
        Validated totals.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or a total is below one. A language entry with no files
            or no tokens describes nothing and must not have been written.
    """
    obj = _require_object(value, f"language stats for '{language}'")
    files = require_int(obj, "files")
    if files < 1:
        raise JSONTypeError(
            f"Field 'files' for language '{language}' must be at least 1, got {files}"
        )
    tokens = require_int(obj, "tokens_approx")
    if tokens < 1:
        raise JSONTypeError(
            f"Field 'tokens_approx' for language '{language}' must be at least 1, got {tokens}"
        )
    return LanguageStats(files=files, tokens_approx=tokens)


def encode_repo_pin(pin: RepoPin) -> dict[str, JSONValue]:
    """Encode a repository pin to a JSON object.

    Args:
        pin: Pin to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "name": pin["name"],
        "commit": pin["commit"],
        "dirty": pin["dirty"],
    }


def decode_repo_pin(value: JSONValue) -> RepoPin:
    """Decode and validate a JSON value into a repository pin.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated pin.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the name is empty or slashed, or the commit is not a
            full 40-character lowercase hash.
    """
    obj = _require_object(value, "repo pin")
    return RepoPin(
        name=_require_repo_name(obj, "name"),
        commit=_require_hex(obj, "commit", COMMIT_HEX_LENGTH),
        dirty=require_bool(obj, "dirty"),
    )


def encode_source_file_record(record: SourceFileRecord) -> dict[str, JSONValue]:
    """Encode a source-file record to a JSON object.

    Args:
        record: Record to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "repo": record["repo"],
        "path": record["path"],
        "language": record["language"],
        "sha256": record["sha256"],
        "tokens_approx": record["tokens_approx"],
        "text": record["text"],
    }


def decode_source_file_record(value: JSONValue) -> SourceFileRecord:
    """Decode and validate a JSON value into a source-file record.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated record.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the path is not repository-relative, the digest is not
            64 lowercase hex characters, the token estimate is below one, or
            the text is empty. An empty document trains nothing and must have
            been excluded at selection.
    """
    obj = _require_object(value, "source file record")
    tokens = require_int(obj, "tokens_approx")
    if tokens < 1:
        raise JSONTypeError(f"Field 'tokens_approx' must be at least 1, got {tokens}")
    return SourceFileRecord(
        repo=_require_repo_name(obj, "repo"),
        path=_require_relative_posix_path(obj, "path"),
        language=_require_nonempty_str(obj, "language"),
        sha256=_require_hex(obj, "sha256", SHA256_HEX_LENGTH),
        tokens_approx=tokens,
        text=_require_nonempty_str(obj, "text"),
    )


def encode_code_corpus_manifest(manifest: CodeCorpusManifest) -> dict[str, JSONValue]:
    """Encode a corpus manifest to a JSON object.

    Args:
        manifest: Manifest to encode.

    Returns:
        JSON-serialisable mapping carrying every field of the manifest.
    """
    repos: list[JSONValue] = [encode_repo_pin(pin) for pin in manifest["repos"]]
    languages: dict[str, JSONValue] = {
        name: encode_language_stats(stats) for name, stats in manifest["languages"].items()
    }
    return {
        "train_output": manifest["train_output"],
        "train_sha256": manifest["train_sha256"],
        "holdout_output": manifest["holdout_output"],
        "holdout_sha256": manifest["holdout_sha256"],
        "seed": manifest["seed"],
        "holdout_fraction": manifest["holdout_fraction"],
        "repos": repos,
        "files_train": manifest["files_train"],
        "files_holdout": manifest["files_holdout"],
        "excluded_generated": manifest["excluded_generated"],
        "excluded_duplicate": manifest["excluded_duplicate"],
        "excluded_empty": manifest["excluded_empty"],
        "languages": languages,
        "tokens_approx_train": manifest["tokens_approx_train"],
        "tokens_approx_holdout": manifest["tokens_approx_holdout"],
    }


def _decode_repos(obj: dict[str, JSONValue]) -> list[RepoPin]:
    """Decode the manifest's repository pins.

    Args:
        obj: Manifest object being decoded.

    Returns:
        Validated pins.

    Raises:
        JSONTypeError: If the list is missing, empty, holds an invalid pin,
            or names the same repository twice. A corpus with no repository
            has no provenance; two pins under one name are unresolvable.
    """
    raw = require_list(obj, "repos")
    if raw == []:
        raise JSONTypeError("Field 'repos' must not be empty")
    pins = [decode_repo_pin(item) for item in raw]
    names = [pin["name"] for pin in pins]
    if len(set(names)) != len(names):
        raise JSONTypeError(f"Field 'repos' must not repeat a name, got {names}")
    return pins


def _decode_languages(obj: dict[str, JSONValue]) -> dict[str, LanguageStats]:
    """Decode the manifest's per-language totals.

    Args:
        obj: Manifest object being decoded.

    Returns:
        Validated totals by language.

    Raises:
        JSONTypeError: If the mapping is missing, empty, or holds an invalid
            entry. A corpus that kept files in no language kept no files.
    """
    raw = require_dict(obj, "languages")
    if raw == {}:
        raise JSONTypeError("Field 'languages' must not be empty")
    return {name: decode_language_stats(value, name) for name, value in raw.items()}


def _check_holdout_consistency(fraction: float, files_holdout: int) -> None:
    """Reject a manifest whose split claim and split contents disagree.

    Args:
        fraction: Decoded ``holdout_fraction``.
        files_holdout: Decoded ``files_holdout``.

    Raises:
        JSONTypeError: If a positive fraction comes with zero holdout files,
            which pins the evaluation to nothing, or holdout files come with
            a zero fraction, which claims files that no split produced.
    """
    if fraction > 0.0 and files_holdout == 0:
        raise JSONTypeError(
            f"Field 'files_holdout' must be positive when 'holdout_fraction' is {fraction}"
        )
    if fraction == 0.0 and files_holdout != 0:
        raise JSONTypeError(
            f"Field 'files_holdout' must be 0 when 'holdout_fraction' is 0, got {files_holdout}"
        )


def decode_code_corpus_manifest(value: JSONValue) -> CodeCorpusManifest:
    """Decode and validate a JSON value into a corpus manifest.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated manifest.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the fraction is outside ``[0, 1)``, the split claim and
            split contents disagree, the repository or language sections are
            empty or inconsistent, or the per-language totals do not sum to
            the file and token totals -- a manifest that disagrees with
            itself describes no corpus.
    """
    obj = _require_object(value, "code corpus manifest")

    fraction = require_float(obj, "holdout_fraction")
    if fraction < 0.0 or fraction >= 1.0:
        raise JSONTypeError(f"Field 'holdout_fraction' must be in [0, 1), got {fraction}")

    files_train = _require_count(obj, "files_train")
    files_holdout = _require_count(obj, "files_holdout")
    _check_holdout_consistency(fraction, files_holdout)

    languages = _decode_languages(obj)
    language_files = sum(stats["files"] for stats in languages.values())
    if language_files != files_train + files_holdout:
        raise JSONTypeError(
            f"Field 'languages' counts {language_files} files, "
            f"but the split holds {files_train + files_holdout}"
        )

    tokens_train = _require_count(obj, "tokens_approx_train")
    tokens_holdout = _require_count(obj, "tokens_approx_holdout")
    language_tokens = sum(stats["tokens_approx"] for stats in languages.values())
    if language_tokens != tokens_train + tokens_holdout:
        raise JSONTypeError(
            f"Field 'languages' counts {language_tokens} tokens, "
            f"but the split holds {tokens_train + tokens_holdout}"
        )

    return CodeCorpusManifest(
        train_output=_require_nonempty_str(obj, "train_output"),
        train_sha256=_require_hex(obj, "train_sha256", SHA256_HEX_LENGTH),
        holdout_output=_require_nonempty_str(obj, "holdout_output"),
        holdout_sha256=_require_hex(obj, "holdout_sha256", SHA256_HEX_LENGTH),
        seed=require_int(obj, "seed"),
        holdout_fraction=fraction,
        repos=_decode_repos(obj),
        files_train=files_train,
        files_holdout=files_holdout,
        excluded_generated=_require_count(obj, "excluded_generated"),
        excluded_duplicate=_require_count(obj, "excluded_duplicate"),
        excluded_empty=_require_count(obj, "excluded_empty"),
        languages=languages,
        tokens_approx_train=tokens_train,
        tokens_approx_holdout=tokens_holdout,
    )


__all__ = [
    "COMMIT_HEX_LENGTH",
    "SHA256_HEX_LENGTH",
    "CodeCorpusManifest",
    "LanguageStats",
    "RepoPin",
    "SourceFileRecord",
    "decode_code_corpus_manifest",
    "decode_language_stats",
    "decode_repo_pin",
    "decode_source_file_record",
    "encode_code_corpus_manifest",
    "encode_language_stats",
    "encode_repo_pin",
    "encode_source_file_record",
]
