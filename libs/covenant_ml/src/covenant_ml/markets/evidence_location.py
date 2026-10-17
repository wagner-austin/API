"""Where a recorded evidence path is now, once its evidence has moved hosts.

Reports and campaign manifests record each source by the absolute path it had
when it was written, and every reader re-hashes the file at that path. Evidence
written on Windows names ``C:\\Users\\...`` paths, which no Linux host can open,
so a study whose host is reimaged could no longer be verified anywhere
(MCPs board task e2ffed0d). A host declares where moved evidence now lives in
``~/.config/covenant_ml/evidence-relocations.json``; a recorded path under a
declared prefix is read at the same relative place under that prefix's current
location. Nothing else changes: the SHA-256 each reader checks is still the one
recorded, so a relocation can only point at a byte-identical copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from .validation import read_document

#: The declaration's path under the reading account's home.
RELOCATIONS_FILE = Path(".config") / "covenant_ml" / "evidence-relocations.json"


class EvidenceRelocation(TypedDict):
    """One moved evidence tree: its recorded prefix and where it is now."""

    recorded_prefix: str
    current_prefix: str


def _separators(path: str) -> str:
    """Spell a path with forward slashes and no trailing one.

    Args:
        path: A recorded or declared path in either platform's spelling.

    Returns:
        The path with ``\\`` as ``/`` and any trailing ``/`` removed.
    """
    return path.replace("\\", "/").rstrip("/")


def decode_relocations(document: JSONObject) -> tuple[EvidenceRelocation, ...]:
    """Validate a host's relocation declaration.

    Args:
        document: The declaration's JSON object, schema version 1.

    Returns:
        Each declared relocation, in the declaration's order.

    Raises:
        ValueError: EVIDENCE-RELOCATION-001 for an unknown schema version, an
            empty prefix, a current prefix that is not absolute on this host,
            or a recorded prefix declared twice.
        TypeError: Required fields are missing or of the wrong type.
    """
    if require_int(document, "schema_version") != 1:
        raise ValueError("[EVIDENCE-RELOCATION-001] unknown relocation schema version")
    relocations: list[EvidenceRelocation] = []
    seen: set[str] = set()
    for value in require_list(document, "relocations"):
        entry = narrow_json_to_dict(value)
        recorded = _separators(require_str(entry, "recorded_prefix"))
        current = require_str(entry, "current_prefix")
        if recorded == "" or current == "":
            raise ValueError("[EVIDENCE-RELOCATION-001] a relocation prefix is empty")
        if not Path(current).is_absolute():
            raise ValueError("[EVIDENCE-RELOCATION-001] a current prefix is not absolute")
        if recorded in seen:
            raise ValueError("[EVIDENCE-RELOCATION-001] a recorded prefix is declared twice")
        seen.add(recorded)
        relocations.append(EvidenceRelocation(recorded_prefix=recorded, current_prefix=current))
    return tuple(relocations)


def read_relocations(path: Path) -> tuple[EvidenceRelocation, ...]:
    """Read a relocation declaration, or none where the host declares none.

    Args:
        path: The declaration file.

    Returns:
        Its relocations; empty when the file does not exist.

    Raises:
        ValueError: The declaration is malformed or invalid.
        OSError: The file exists but cannot be read.
    """
    if not path.exists():
        return ()
    return decode_relocations(read_document(path))


def read_host_relocations() -> tuple[EvidenceRelocation, ...]:
    """Read the reading account's declaration under its home.

    Returns:
        The relocations :data:`RELOCATIONS_FILE` declares for this host.
    """
    return read_relocations(Path.home() / RELOCATIONS_FILE)


def locate_evidence(recorded: str, relocations: Sequence[EvidenceRelocation]) -> Path:
    """Resolve a recorded evidence path to where it is read on this host.

    Args:
        recorded: The absolute path a report or manifest recorded.
        relocations: The host's declared relocations.

    Returns:
        The recorded path under the current prefix of the first relocation
        whose recorded prefix contains it, else the recorded path itself.
    """
    spelled = _separators(recorded)
    for relocation in relocations:
        prefix = _separators(relocation["recorded_prefix"])
        if spelled == prefix or spelled.startswith(prefix + "/"):
            rest = spelled[len(prefix) :].lstrip("/")
            current = Path(relocation["current_prefix"])
            return current / rest if rest != "" else current
    return Path(recorded)


__all__ = [
    "RELOCATIONS_FILE",
    "EvidenceRelocation",
    "decode_relocations",
    "locate_evidence",
    "read_host_relocations",
    "read_relocations",
]
