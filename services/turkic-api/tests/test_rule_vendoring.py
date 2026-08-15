"""``PROVENANCE.md`` must describe the rule files that are actually here.

The rules are a vendored copy of an upstream project, and this service has
no dependency on that project — so the only record of *which* upstream
version is in the tree is the table in ``PROVENANCE.md``. A record that can
drift from what it describes is worse than no record, because it will be
believed. This test parses the table and checks it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

_RULES_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent / "src" / "turkic_api" / "core" / "rules"
)
_PROVENANCE: Final[Path] = _RULES_DIR / "PROVENANCE.md"

# | `name.rules` | `<64 hex>` | origin |
_ROW: Final = re.compile(
    r"^\|\s*`(?P<name>[a-z_]+\.rules)`\s*\|\s*`(?P<digest>[0-9a-f]{64})`\s*\|"
    r"\s*(?P<origin>[^|]+?)\s*\|$",
    re.MULTILINE,
)
_COMMIT_LABEL: Final = "Commit"


def _content_digest(path: Path) -> str:
    """Return the SHA-256 of a rule file's content, ignoring line endings.

    ``.gitattributes`` marks ``*.rules`` as ``text``, so git stores LF and
    checks out whatever the platform uses — CRLF on Windows. Hashing raw bytes
    would therefore assert the checkout's newline convention rather than the
    rules, and would pass on Linux while failing on a fresh Windows clone.
    Normalising first makes the recorded hash mean what it claims to: the
    content.

    Args:
        path (Path): The rule file to hash.

    Returns:
        str: Hex SHA-256 of the file with CRLF normalised to LF.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _documented() -> dict[str, str]:
    """Return the file-to-digest table recorded in ``PROVENANCE.md``."""
    text = _PROVENANCE.read_text(encoding="utf-8")
    return {match.group("name"): match.group("digest") for match in _ROW.finditer(text)}


def _recorded_commits() -> list[str]:
    """Return every commit SHA the ``Commit`` table row records.

    Read by splitting the markdown row rather than with a capturing group,
    because a regular expression group is typed as possibly-``Any`` and this
    project admits no ``Any`` into its tests.

    Returns:
        list[str]: The recorded values, with surrounding backticks stripped,
        in file order.
    """
    found: list[str] = []
    for line in _PROVENANCE.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == 4 and cells[1] == _COMMIT_LABEL:
            found.append(cells[2].strip("`"))
    return found


def test_provenance_lists_every_rule_file() -> None:
    """The table must name every rule file and no file that is absent."""
    on_disk = {path.name for path in _RULES_DIR.glob("*.rules")}
    documented = set(_documented())
    assert documented == on_disk, (
        f"PROVENANCE.md and the rules directory disagree.\n"
        f"  undocumented files: {sorted(on_disk - documented)}\n"
        f"  documented but absent: {sorted(documented - on_disk)}"
    )


def test_every_documented_hash_matches_the_file() -> None:
    """Each recorded SHA-256 must be the hash of the file it names.

    Line endings are normalised first; see :func:`_content_digest`.
    """
    for name, expected in sorted(_documented().items()):
        actual = _content_digest(_RULES_DIR / name)
        assert actual == expected, (
            f"{name} does not match its recorded hash.\n"
            f"  recorded: {expected}\n"
            f"  actual:   {actual}\n"
            f"Either the file was edited in place — the rules are vendored and "
            f"belong to the upstream project — or PROVENANCE.md is stale."
        )


def test_provenance_records_exactly_one_full_upstream_commit() -> None:
    """The upstream commit must be recorded once, as a full 40-character SHA.

    An abbreviated commit is ambiguous and a missing one cannot be resolved
    at all, either of which defeats the point of recording it. Two would
    leave it unclear which version is actually vendored.
    """
    recorded = _recorded_commits()

    assert len(recorded) == 1, (
        f"PROVENANCE.md must record exactly one upstream commit row, found {len(recorded)}"
    )
    sha = recorded[0]
    assert len(sha) == 40, f"upstream commit {sha!r} is {len(sha)} characters, not a full SHA"
    assert set(sha) <= set("0123456789abcdef"), (
        f"upstream commit {sha!r} is not lowercase hexadecimal"
    )
