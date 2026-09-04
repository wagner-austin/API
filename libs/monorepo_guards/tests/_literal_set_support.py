"""Materialising the declared sets a synthetic monorepo has to carry.

``LiteralSetRule`` resolves its declaring module from the monorepo root, so a
test that builds a fake monorepo and expects a clean run has to give it the
declarations a real one has. Written from :data:`REGISTERED_SETS` rather than
listed here, so registering a fourth set does not silently make these trees
incomplete -- which would surface as an unrelated integration test failing on
a rule it was never about.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.literal_set_rules import REGISTERED_SETS, LiteralSet

DECLARING_PACKAGE = Path("libs") / "declaring_pkg" / "src"
"""Where these fixtures put a declaring module: two levels down, src-layout."""


def config_for(monorepo_root: Path) -> GuardConfig:
    """Build a config whose monorepo root is the test's own tree.

    Args:
        monorepo_root: The test's temporary directory, standing in for the
            repository the rule resolves declarations from.

    Returns:
        A config the rule can resolve a declaring module through.
    """
    return GuardConfig(
        root=monorepo_root,
        monorepo_root=monorepo_root,
        directories=("src",),
        exclude_parts=(),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )


def write_source(path: Path, text: str) -> Path:
    """Write a source file for the rule to scan.

    Args:
        path: Where to write it.
        text: The source.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def declaring_source(declared: LiteralSet, members: tuple[str, ...]) -> str:
    """Render a module binding one set's tuple.

    Args:
        declared: The set to declare.
        members: The members to declare it with.

    Returns:
        Module source binding the tuple under its annotated name.
    """
    literal = ", ".join(f'"{member}"' for member in members)
    return (
        "from typing import Literal\n\n"
        f"{declared.tuple_name}: tuple[Literal[{literal}], ...] = ({literal},)\n"
    )


def write_declared_sets(monorepo_root: Path) -> list[Path]:
    """Give a synthetic monorepo the declarations every registered set needs.

    Args:
        monorepo_root: Root of the tree being built.

    Returns:
        The modules written, one per registered set.

    Raises:
        ValueError: If a registered set declares no members to write, which
            would produce a module the rule reads as having lost its tuple.
    """
    written: list[Path] = []
    for declared in REGISTERED_SETS:
        members = DECLARED_MEMBERS.get(declared.subject)
        if not members:
            raise ValueError(
                f"{declared.subject} has no members in DECLARED_MEMBERS; a set "
                "registered without one leaves every synthetic monorepo missing "
                "a declaration and fails tests that are about something else"
            )
        path = monorepo_root / DECLARING_PACKAGE / declared.defining_module
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(declaring_source(declared, members), encoding="utf-8")
        written.append(path)
    return written


DECLARED_MEMBERS: dict[str, tuple[str, ...]] = {
    "corpus-format": ("lines", "documents"),
    "risk-tier": ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
    "strategy-name": ("full", "lora", "qlora"),
}
"""The members each registered set is materialised with.

Only the tuple's existence and shape matter to the tests that use this; the
values are the real ones so a reader is not misled about what the sets hold.
"""


__all__ = [
    "DECLARED_MEMBERS",
    "DECLARING_PACKAGE",
    "config_for",
    "declaring_source",
    "write_declared_sets",
    "write_source",
]
