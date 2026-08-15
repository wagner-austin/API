"""The published description each rule file implements.

Every IPA rule file states, in its own header, the phonological description
its mappings come from. That statement is a structured block rather than
prose, so it can be read as data:

.. code-block:: text

    # Source-Authors: McCollum, A. G.
    # Source-Year: 2020
    # Source-Title: Vowel harmony and positional variation in Kyrgyz
    # Source-Container: Laboratory Phonology 11(1): article 25
    # Source-Id: https://doi.org/10.5334/labphon.247

Two things follow. This service can answer "what backs the Kazakh rules" as
data rather than as a comment. And a gold-standard test can declare the
source its expected values came from and have that checked against what the
rule file itself declares — so the provenance the rules carry is inherited
by the tests that verify them, rather than restated on trust. A test that
cites a source the rules never cite is how a wrong Kyrgyz <ж> survived
upstream for months.

This mirrors ``turkic_translit.rule_provenance`` field for field, because
the rule files are vendored from that project and their headers are written
to its convention. Keeping the reader compatible is what lets the headers
survive re-vendoring unchanged.

Parsing is strict and total. A header missing a field, or carrying one
twice, or carrying one with no value, fails at the boundary naming the file
and the field, because a rule set whose provenance cannot be read is a rule
set whose claims cannot be checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

from platform_core.json_utils import JSONObject, require_int, require_str

FIELD_PREFIX: Final = "# Source-"
AUTHORS_FIELD: Final = "Authors"
YEAR_FIELD: Final = "Year"
TITLE_FIELD: Final = "Title"
CONTAINER_FIELD: Final = "Container"
IDENTIFIER_FIELD: Final = "Id"

REQUIRED_FIELDS: Final = (
    AUTHORS_FIELD,
    YEAR_FIELD,
    TITLE_FIELD,
    CONTAINER_FIELD,
    IDENTIFIER_FIELD,
)

# Keys of the decoded record that must carry text. ``year`` is absent
# because it is an integer and validated as one.
TEXT_KEYS: Final = ("authors", "title", "container", "identifier")

# Header field name to decoded record key.
_FIELD_KEYS: Final = {
    AUTHORS_FIELD: "authors",
    YEAR_FIELD: "year",
    TITLE_FIELD: "title",
    CONTAINER_FIELD: "container",
    IDENTIFIER_FIELD: "identifier",
}

ERR_MALFORMED_LINE: Final = "TURKIC_RULE_SOURCE_001_MALFORMED_LINE"
ERR_FIELD_MISSING: Final = "TURKIC_RULE_SOURCE_002_FIELD_MISSING"
ERR_FIELD_EMPTY: Final = "TURKIC_RULE_SOURCE_003_FIELD_EMPTY"


class RuleSourceError(Exception):
    """Base class for unreadable rule-file provenance.

    Args:
        code (str): Stable, greppable code from this module.
        message (str): Description naming the file and the offending field.
    """

    def __init__(self, code: str, message: str) -> None:
        """Render ``code: message`` as the string form."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class RuleSourceMalformedLineError(RuleSourceError):
    """Raised when a ``# Source-`` line cannot be read as a field.

    Args:
        origin (str): Name of the rule file.
        line (str): The offending line, quoted back to the caller.
    """

    def __init__(self, origin: str, line: str) -> None:
        """Name the file and the line that could not be read."""
        super().__init__(ERR_MALFORMED_LINE, f"{origin} has an unreadable source line: {line!r}")
        self.origin = origin
        self.line = line


class RuleSourceFieldMissingError(RuleSourceError):
    """Raised when a required provenance field is absent.

    Args:
        origin (str): Name of the rule file.
        field (str): The field that should have been present.
    """

    def __init__(self, origin: str, field: str) -> None:
        """Name the file and the field it does not declare."""
        super().__init__(ERR_FIELD_MISSING, f"{origin} declares no {FIELD_PREFIX}{field} field")
        self.origin = origin
        self.field = field


class RuleSourceFieldEmptyError(RuleSourceError):
    """Raised when a required provenance field is present but empty.

    An empty field is worse than an absent one: it looks answered.

    Args:
        origin (str): Name of the rule file.
        field (str): The field whose value is empty.
    """

    def __init__(self, origin: str, field: str) -> None:
        """Name the file and the field it left empty."""
        super().__init__(ERR_FIELD_EMPTY, f"{origin} declares {field} with no value")
        self.origin = origin
        self.field = field


class RuleSource(TypedDict):
    """The published description one rule file implements.

    Attributes:
        authors (str): Author list as the source itself gives it.
        year (int): Year of publication.
        title (str): Title of the work.
        container (str): Journal, series or publisher, with volume and
            pages where the source has them.
        identifier (str): Canonical resolvable identifier — a DOI URL where
            the work has a DOI, a permanent URL otherwise.
    """

    authors: str
    year: int
    title: str
    container: str
    identifier: str


def encode_rule_source(record: RuleSource) -> JSONObject:
    """Render a source record to a plain mapping, for writing as JSON.

    Args:
        record (RuleSource): The record to encode.

    Returns:
        JSONObject: A mapping carrying exactly the five provenance fields.
    """
    return {
        "authors": record["authors"],
        "year": record["year"],
        "title": record["title"],
        "container": record["container"],
        "identifier": record["identifier"],
    }


def decode_rule_source(source: JSONObject, origin: str) -> RuleSource:
    """Validate a loosely-typed mapping into a :class:`RuleSource`.

    Args:
        source (JSONObject): Mapping holding the five provenance fields.
        origin (str): Name of the file the mapping came from, for error
            messages.

    Returns:
        RuleSource: A fully validated source record.

    Raises:
        JSONTypeError: If a field is absent, or a text field is not a
            string, or the year is not an integer.
        RuleSourceFieldEmptyError: If a text field is empty.
    """
    for key in TEXT_KEYS:
        if not require_str(source, key):
            raise RuleSourceFieldEmptyError(origin, key)
    return RuleSource(
        authors=require_str(source, "authors"),
        year=require_int(source, "year"),
        title=require_str(source, "title"),
        container=require_str(source, "container"),
        identifier=require_str(source, "identifier"),
    )


def _collect_fields(text: str, origin: str) -> dict[str, str]:
    """Gather every ``# Source-`` field from rule-file text.

    Args:
        text (str): Full contents of a rule file.
        origin (str): Name of the file, for error messages.

    Returns:
        dict[str, str]: Field name to its stripped value.

    Raises:
        RuleSourceMalformedLineError: If a line carries no ``:`` separator,
            or names a field that was already given.
    """
    collected: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(FIELD_PREFIX):
            continue
        body = line[len(FIELD_PREFIX) :]
        if ":" not in body:
            raise RuleSourceMalformedLineError(origin, line)
        name, _, value = body.partition(":")
        field = name.strip()
        if field in collected:
            raise RuleSourceMalformedLineError(origin, line)
        collected[field] = value.strip()
    return collected


def parse_rule_source(text: str, origin: str) -> RuleSource:
    """Read the structured provenance block out of rule-file text.

    Only lines beginning ``# Source-`` are considered, so ordinary comments
    and the rules themselves are ignored. Every field in
    :data:`REQUIRED_FIELDS` must appear exactly once with a value.

    Args:
        text (str): Full contents of a rule file.
        origin (str): Name of the file, used in error messages.

    Returns:
        RuleSource: The validated source record the header declares.

    Raises:
        RuleSourceMalformedLineError: If a source line is unreadable, or
            the year is not a plain sequence of digits.
        RuleSourceFieldMissingError: If a required field is absent.
        RuleSourceFieldEmptyError: If a required field is empty.
    """
    collected = _collect_fields(text, origin)
    for field in REQUIRED_FIELDS:
        if field not in collected:
            raise RuleSourceFieldMissingError(origin, field)

    year_text = collected[YEAR_FIELD]
    if not year_text.isdigit():
        raise RuleSourceMalformedLineError(origin, f"{FIELD_PREFIX}{YEAR_FIELD}: {year_text}")

    record: JSONObject = {_FIELD_KEYS[field]: collected[field] for field in REQUIRED_FIELDS}
    record["year"] = int(year_text)
    return decode_rule_source(record, origin)


def read_rule_source(path: Path) -> RuleSource:
    """Read one rule file's declared source from disk.

    Args:
        path (Path): Path to a ``.rules`` file.

    Returns:
        RuleSource: The validated source record its header declares.

    Raises:
        FileNotFoundError: When the rule file does not exist.
        RuleSourceError: When its provenance header cannot be read.
    """
    return parse_rule_source(path.read_text(encoding="utf-8"), path.name)


__all__ = [
    "AUTHORS_FIELD",
    "CONTAINER_FIELD",
    "ERR_FIELD_EMPTY",
    "ERR_FIELD_MISSING",
    "ERR_MALFORMED_LINE",
    "FIELD_PREFIX",
    "IDENTIFIER_FIELD",
    "REQUIRED_FIELDS",
    "TEXT_KEYS",
    "TITLE_FIELD",
    "YEAR_FIELD",
    "RuleSource",
    "RuleSourceError",
    "RuleSourceFieldEmptyError",
    "RuleSourceFieldMissingError",
    "RuleSourceMalformedLineError",
    "decode_rule_source",
    "encode_rule_source",
    "parse_rule_source",
    "read_rule_source",
]
