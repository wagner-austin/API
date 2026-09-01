from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

#: Every corpus format, and the ONE place the accepted set is written.
#:
#: ``lines`` reads every non-blank line of a ``.txt`` corpus, stripped. That is
#: right for prose, where a line is a paragraph and its surrounding whitespace
#: carries nothing, and it is what every wiki corpus and every HPC3 arm to date
#: has trained under.
#:
#: ``documents`` reads one record per line of a ``.jsonl`` corpus and takes its
#: ``text`` field verbatim. Source code needs this: stripping a Python file's
#: indentation destroys its syntax, so a code corpus read as lines trains the
#: model on text that would not parse.
#:
#: The ``Literal`` is written out at each field rather than aliased, matching
#: ``model_family`` and ``finetuning_strategy``. ``scripts/guard.py`` fails the
#: lint when this tuple and those annotations stop agreeing, which is the check
#: an alias would have bought and mypy cannot make on its own.
CORPUS_FORMATS: tuple[Literal["lines", "documents"], ...] = ("lines", "documents")


def as_corpus_format(raw: str, field: str) -> Literal["lines", "documents"]:
    """Narrow a string to a corpus format, or refuse it.

    The single narrowing in the service. Three paths need it -- the queue
    payload decoder, the checkpoint decoder, and the manifest reader -- and
    a second copy would be a second place for the accepted set to drift from
    the ``Literal`` on each field.

    Args:
        raw: The string to narrow.
        field: Name of the field it came from, for the error message.

    Returns:
        The narrowed corpus format.

    Raises:
        JSONTypeError: If the string names no known format.
    """
    for known in CORPUS_FORMATS:
        if raw == known:
            return known
    raise JSONTypeError(f"Field '{field}' must be one of {CORPUS_FORMATS}, got '{raw}'")


def require_corpus_format(obj: JSONObject, field: str) -> Literal["lines", "documents"]:
    """Read a required corpus-format field, narrowing it to the Literal.

    Args:
        obj: JSON object to read from.
        field: Name of the field holding the format.

    Returns:
        The narrowed corpus format.

    Raises:
        JSONTypeError: If the field is absent, is not a string, or names no
            known format.
    """
    return as_corpus_format(require_str(obj, field), field)


class CorpusSplit(TypedDict):
    """Three disjoint partitions of a corpus, carried as units rather than paths.

    A unit is what :data:`CORPUS_FORMATS` divides the corpus into -- a stripped
    line under ``lines``, a whole source file under ``documents``. Both reduce
    to a tuple of strings here, so everything downstream of the split is
    shared between the two formats.

    The partitions are units because ``holdout_fraction`` and
    ``test_split_ratio`` are fractions of the corpus, not of its file list.
    Splitting by file made a single-file corpus -- the prevailing layout in this
    service and in every experiment run against it -- collapse to the same file
    in all three partitions, so validation loss, early stopping and
    best-checkpoint selection all ran against training data while reporting
    numbers indistinguishable from real ones.
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


class DatasetConfig:
    """Configuration for dataset splitting."""

    corpus_path: str
    holdout_fraction: float
    test_split_ratio: float
    corpus_format: Literal["lines", "documents"]

    def __init__(
        self: DatasetConfig,
        corpus_path: str,
        corpus_format: Literal["lines", "documents"],
        holdout_fraction: float = 0.01,
        test_split_ratio: float = 0.15,
    ) -> None:
        """Initialize dataset configuration.

        Both ratios are fractions of the corpus's content units, not of its
        file list. A corpus is one logical body of text however many files it
        happens to occupy, so a single-file corpus splits exactly like a
        thousand-file one.

        Args:
            corpus_path: Path to corpus directory or file.
            corpus_format: Whether the corpus carries one training unit per
                stripped line or one per JSONL record. Required and without a
                default, for the same reason ``deterministic`` is required on
                an HPC3 project: it partitions results rather than improving
                them. The same path read under the two formats yields
                different units, so a run's format is part of what the run IS.
                A default would let a code corpus be read as lines by
                omission, and a stripped Python file does not parse.
            holdout_fraction: Fraction of corpus units held out for validation
                (default 0.01). Zero trains without a validation split.
            test_split_ratio: Fraction of corpus units held out for testing
                (default 0.15). Zero trains without a test split.
        """
        self.corpus_path = corpus_path
        self.holdout_fraction = holdout_fraction
        self.test_split_ratio = test_split_ratio
        self.corpus_format = corpus_format


class DatasetBuilder(Protocol):
    """Protocol for dataset builders that partition a corpus."""

    def split(self: DatasetBuilder, cfg: DatasetConfig) -> CorpusSplit:
        """Partition a corpus into disjoint train/validation/test lines.

        Args:
            cfg: Dataset configuration with corpus path and split ratios.

        Returns:
            The three disjoint partitions, as corpus lines.
        """
        ...
