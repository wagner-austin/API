"""Whether the research index carries a SECTION for a registered project.

The step of registration nothing can generate is the prose: what a project
measures and what its provenance does not cover. ``test_committed_runs.py``
held it to that by asserting each registered name appeared in
``docs/RESEARCH.md`` -- as `` `name` `` anywhere in the file.

THAT CHECK HAD BEEN VACUOUS SINCE THE TABLE WAS GENERATED. The rendered
project table writes every registered name as `` `name` `` in its first
column, so once ``hpc3-research-index --write`` had run, a project with no
section at all passed the presence check on the strength of its own table
row. The gate that was meant to make a person write a section was satisfied
by a generator.

So the property is now stated as what it always meant: a level-three heading
naming the project, inside the ``## Registered with the hpc3 CLI`` part of
the file. Inside, because the file's ``## Not registered anywhere`` part names
projects too -- ``sirius`` is described there as a deliberate
non-registration, and a heading there is the opposite of the claim this
checks.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Final

#: The heading that opens the part of the index describing registered work.
REGISTERED_HEADING: Final[str] = "## Registered with the hpc3 CLI"

#: How every second-level heading begins, which is where the part ends.
_PART_PREFIX: Final[str] = "## "


def registered_part(text: str) -> str:
    """Cut the registered part out of the research index.

    Args:
        text: The whole index, LF line endings.

    Returns:
        The text after :data:`REGISTERED_HEADING` up to the next
        second-level heading, or to the end of the file.

    Raises:
        ValueError: If the heading is absent. Reporting every project as
            missing a section would send a reader to write eight sections
            when one heading was renamed.
    """
    lines = text.split("\n")
    if REGISTERED_HEADING not in lines:
        raise ValueError(f"the research index has no {REGISTERED_HEADING!r} heading")
    part: list[str] = []
    for line in lines[lines.index(REGISTERED_HEADING) + 1 :]:
        if line.startswith(_PART_PREFIX):
            break
        part.append(line)
    return "\n".join(part)


def section_heading(project: str) -> str:
    """Spell the heading that opens a project's section.

    Args:
        project: The project's name.

    Returns:
        ``### `<project>```, which a section's heading line equals or
        continues with a space -- ``### `tankpit` — TankpitBot, ...``.
    """
    return f"### `{project}`"


def projects_without_section(text: str, projects: Collection[str]) -> list[str]:
    """Name the projects the registered part carries no section for.

    Args:
        text: The whole index, LF line endings.
        projects: The project names to look for.

    Returns:
        Those without a heading, sorted. Matched against whole heading
        tokens, so ``mi``'s section does not stand in for ``mi-cu128``'s.

    Raises:
        ValueError: If the index has no registered part at all.
    """
    lines = registered_part(text).split("\n")
    return sorted(
        project
        for project in projects
        if not any(
            line == section_heading(project) or line.startswith(section_heading(project) + " ")
            for line in lines
        )
    )


__all__ = [
    "REGISTERED_HEADING",
    "projects_without_section",
    "registered_part",
    "section_heading",
]
