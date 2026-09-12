"""The part of the research index a human must not maintain by hand.

``docs/RESEARCH.md`` is prose about every body of work that produces numbers
someone compares, and prose is the right shape for most of it: what a project
measures, what its provenance does not cover, why a gap is still open. None of
that is derivable.

But some of it is. A project's cores, memory, wall clock, partition and image
are declared in the hpc3 workspace documents, and when the index restated them
it restated them WRONG:

* The `rusted` entry said one CPU, 45 minutes and no image, against a
  workspace document committed seven minutes earlier declaring four CPUs, 100
  minutes and an sha256-pinned image.
* The `cleargbm` entry said its record shape was `BenchmarkManifest` and not
  `RunRecord`, months after `benchmark_run_record` landed. A session acting on
  that sentence rewrote a module that already existed.
* The `code-style` entry said its training run recorded no fingerprint, while
  the artifact carried a full one.

All three passed ``test_committed_runs.py`` throughout, because it asserts that
every registered project APPEARS in the index and never that what the index
says about one is true. Presence was enforced; agreement was not.

Restating a declared fact in prose is the whole failure. So the declared facts
are rendered from the documents into a marked block, and a test fails when the
block on disk is not what the registry would produce now. What cannot drift is
what nobody retypes.

WHAT THIS DOES NOT COVER, and the boundary is worth stating plainly. Only
facts the workspace documents declare are generated. A claim about what a
package's code does -- the `cleargbm` failure -- is not derivable from the
registry and this block cannot check it. That gap stays open.

THE SECOND CLASS, and it needs the opposite remedy. Some facts in the index
are restatements of a source that CANNOT be rendered from, because the source
is neither committed nor shared: ``tools/hpc3/runs/ledger.jsonl`` is
machine-local and deliberately untracked, being state rather than
configuration. A row count read off it is therefore uncheckable by anyone who
is not sitting at the machine that produced it, and on 2026-09-12 both entries
carrying one were wrong -- ``mi`` low by a factor of four while also claiming a
superlative that belonged to ``rusted``, ``cleargbm`` stale by two.

Generating those is impossible, so the remedy is to refuse them.
:func:`ledger_state_claims` is the refusal, and it is not a style rule: the
index's own ``tankpit`` entry reached the conclusion ("run state is the
ledger's answer and this file should not assert it") while three entries went
on asserting it, which is what a conclusion does when nothing executes it.
"""

from __future__ import annotations

from typing import Final

from hpc3.contracts.project import ProjectConfig

#: Opening marker of the generated block, and the ONE place it is written.
BLOCK_START = "<!-- generated: hpc3-projects. Do not edit by hand. -->"

#: Closing marker.
BLOCK_END = "<!-- /generated: hpc3-projects -->"

#: What a reader is told to run when the block is stale.
REGENERATE_HINT = "hpc3-research-index --write"


def _image_cell(project: ProjectConfig) -> str:
    """Render a project's image as a table cell.

    Args:
        project: The project's declared configuration.

    Returns:
        The pinned digest's first twelve characters. Twelve is enough to tell
        two images apart by eye and short enough to keep the row readable;
        the whole digest lives in the workspace document.

        There is no ``none`` case. Every project declares an image, so the
        cell that used to say so is unreachable -- see
        :func:`~hpc3.contracts.project._require_project_image`.
    """
    return f"`{project['image']['sha256'][:12]}`"


def _gpu_cell(project: ProjectConfig) -> str:
    """Render a project's GPU request as a table cell.

    Args:
        project: The project's declared configuration.

    Returns:
        The pinned model and count, or ``cpu`` when the project pins none.
        Rendered as a word rather than left blank, because a blank in a table
        reads as a formatting fault instead of as the absence it records, and
        rendered field by field rather than by formatting the mapping, which
        would put Python dict syntax into a markdown document.
    """
    gpu = project["gpu"]
    if gpu is None:
        return "cpu"
    return f"`{gpu['model']}` x{gpu['count']}"


def render_project_row(name: str, project: ProjectConfig) -> str:
    """Render one registered project as a table row.

    Args:
        name: The project's name.
        project: Its declared configuration.

    Returns:
        A markdown table row.
    """
    return (
        f"| `{name}` "
        f"| {project['partition']} "
        f"| {_gpu_cell(project)} "
        f"| {project['cpus']} "
        f"| {project['mem_gb']} "
        f"| {project['minutes']} "
        f"| {_image_cell(project)} "
        f"| {'yes' if project['deterministic'] else 'no'} "
        f"| {'yes' if project['resumes_from_checkpoint'] else 'no'} |"
    )


def render_projects_block(projects: dict[str, ProjectConfig]) -> str:
    """Render every registered project as a generated markdown block.

    Args:
        projects: Declared projects, keyed by name.

    Returns:
        The block, markers included, sorted by project name so two renderings
        of the same registry are byte-identical.
    """
    columns = (
        "project",
        "partition",
        "gpu",
        "cpus",
        "mem GiB",
        "minutes",
        "image",
        "deterministic",
        "resumes",
    )
    header = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    rows = [render_project_row(name, projects[name]) for name in sorted(projects)]
    return "\n".join(
        [
            BLOCK_START,
            "",
            f"Rendered from `tools/hpc3/runs/hpc3*.json`. Regenerate with `{REGENERATE_HINT}`.",
            "",
            *header,
            *rows,
            "",
            BLOCK_END,
        ]
    )


def replace_projects_block(text: str, block: str) -> str:
    """Substitute the generated block into a document.

    Args:
        text: The document.
        block: The rendered block.

    Returns:
        The document with its block replaced.

    Raises:
        ValueError: If the markers are missing or out of order. Appending the
            block instead would put a second table in the file and leave the
            stale one above it, which is worse than refusing.
    """
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start == -1 or end == -1:
        raise ValueError(f"document carries no generated block; expected {BLOCK_START!r}")
    if end < start:
        raise ValueError("generated block markers are out of order")
    return text[:start] + block + text[end + len(BLOCK_END) :]


#: The bullet field every asserted row count was written into. Banned as a
#: FIELD rather than policed as a value: all three instances lived here, and a
#: heading that invites a number nobody can check is the affordance, not the
#: typo. Matched after stripping indentation so a nested bullet cannot smuggle
#: one back in.
SCALE_FIELD: Final[str] = "- **Scale:**"

#: The noun a ledger count is spelled with. Checked in addition to the field,
#: so deleting the heading and writing the same claim as prose is caught too.
LEDGER_ROW_UNIT: Final[str] = "ledger row"

#: Characters a written count is built from, so that "13,008" reads as one
#: number. Read character by character rather than through a regular
#: expression: this package's mypy settings ban an expression of type ``Any``,
#: and ``re.Match.group`` is typed to return one.
_COUNT_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789,")


def _count_before(text: str, index: int) -> str:
    """Read the number a phrase is quantified by, if it is quantified at all.

    Args:
        text: The document.
        index: Offset of the phrase's first character.

    Returns:
        The count immediately preceding the phrase, or the empty string when
        no digit precedes it. "the ledger row for `55715577`" yields nothing,
        because the job id follows the phrase rather than quantifying it,
        while "131 ledger rows" yields ``131``.
    """
    end = index
    while end > 0 and text[end - 1] == " ":
        end -= 1
    start = end
    while start > 0 and text[start - 1] in _COUNT_CHARACTERS:
        start -= 1
    count = text[start:end]
    return count if any(character.isdigit() for character in count) else ""


def ledger_state_claims(text: str) -> tuple[str, ...]:
    """Find every place the index asserts run state it cannot support.

    Args:
        text: The research index's full text.

    Returns:
        One human-readable claim per violation, in the order they appear.
        Empty when the document asserts none.

    WHAT THIS DELIBERATELY DOES NOT CATCH. A count phrased around the
    predicate -- "the ledger held five hundred rows", or a number written far
    from the noun -- passes. The two forms checked are the field that invited
    the claim and the spelling all of them used, which is what makes the check
    exact and false-positive-free on a document that legitimately cites job
    ids beside the word "ledger". A reader determined to assert run state can
    still do it; this stops the shape it arrived in twice, and the entry it
    fires on names the reason rather than the rule.
    """
    claims: list[str] = []
    for line in text.split("\n"):
        if line.strip().startswith(SCALE_FIELD):
            claims.append(f"the `Scale` field is back: {line.strip()[:80]}")
    index = text.find(LEDGER_ROW_UNIT)
    while index != -1:
        count = _count_before(text, index)
        if count:
            claims.append(f"an asserted ledger row count: {count} {LEDGER_ROW_UNIT}s")
        index = text.find(LEDGER_ROW_UNIT, index + len(LEDGER_ROW_UNIT))
    return tuple(claims)


#: Marks the sentence that restates a project's declared image. A ``.sif``
#: path is the anchor rather than the word "image", because the index cites
#: image DIGESTS constantly and legitimately -- historical generations, the
#: image an old job ran under -- and only a path claims to be the thing the
#: registry currently declares.
IMAGE_PATH_SUFFIX: Final[str] = ".sif"

#: How far past a ``.sif`` path a digest may sit and still be that image's.
#: Generous, because the claim wraps across markdown lines and normalising
#: whitespace does not close the gap; wide enough to reach the digest in both
#: spellings the file uses, short enough not to reach the next bullet.
_DIGEST_WINDOW: Final[int] = 200

#: Introduces a digest. Both live spellings put the token in backticks right
#: after this word.
_DIGEST_MARKER: Final[str] = "sha256 `"


def _section_of(text: str, index: int) -> str:
    """Name the project whose entry an offset falls in.

    Args:
        text: The index's full text.
        index: An offset into it.

    Returns:
        The project named by the nearest ``### `name`` heading above the
        offset, or the empty string when the offset precedes every heading.
    """
    heading = text.rfind("### `", 0, index)
    if heading == -1:
        return ""
    start = heading + len("### `")
    end = text.find("`", start)
    return text[start:end] if end != -1 else ""


def _digest_after(text: str, index: int) -> str:
    """Read the digest a path claims, if it states one nearby.

    Args:
        text: The index's full text.
        index: Offset just past the image path.

    Returns:
        The digest token, or the empty string when none sits within the
        window. The token stops at the first character that is not hex, which
        is how an elided ``0cfdd5592a1a…`` yields its twelve real characters
        rather than the ellipsis with them.
    """
    marker = text.find(_DIGEST_MARKER, index, index + _DIGEST_WINDOW)
    if marker == -1:
        return ""
    start = marker + len(_DIGEST_MARKER)
    end = start
    while end < len(text) and text[end] in "0123456789abcdef":
        end += 1
    return text[start:end]


def image_digest_claims(text: str, projects: dict[str, ProjectConfig]) -> tuple[str, ...]:
    """Find every restated image digest the registry contradicts.

    The generated block already renders each project's declared digest, and
    ``rusted``'s entry retyped it anyway and went stale beside it -- naming
    ``images/v4`` and ``b1eaaa2e`` while the registry declared v5 and
    ``97a80bdeb16d`` in a table two screens above. That is the third time that
    one entry disagreed with its own registry, so the restatement is checked
    rather than trusted.

    Args:
        text: The research index's full text.
        projects: Declared projects, keyed by name.

    Returns:
        One human-readable claim per contradiction, in order of appearance.

    WHAT THIS DOES NOT CATCH, deliberately. A digest with no ``.sif`` path
    beside it is left alone, because the index cites superseded images
    constantly and a rule convicting those would fire on correct history. So
    this covers the sentence that claims to state what a project DECLARES, and
    nothing else.
    """
    claims: list[str] = []
    index = text.find(IMAGE_PATH_SUFFIX)
    while index != -1:
        end = index + len(IMAGE_PATH_SUFFIX)
        project = _section_of(text, index)
        declared = projects.get(project)
        digest = _digest_after(text, end)
        if declared is not None and digest and not declared["image"]["sha256"].startswith(digest):
            claims.append(
                f"`{project}` restates an image digest the registry contradicts: "
                f"{digest} against {declared['image']['sha256'][:12]}"
            )
        index = text.find(IMAGE_PATH_SUFFIX, end)
    return tuple(claims)


def extract_projects_block(text: str) -> str:
    """Read the generated block out of a document.

    Args:
        text: The document.

    Returns:
        The block, markers included.

    Raises:
        ValueError: If the markers are missing or out of order.
    """
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start == -1 or end == -1:
        raise ValueError(f"document carries no generated block; expected {BLOCK_START!r}")
    if end < start:
        raise ValueError("generated block markers are out of order")
    return text[start : end + len(BLOCK_END)]


__all__ = [
    "BLOCK_END",
    "BLOCK_START",
    "IMAGE_PATH_SUFFIX",
    "LEDGER_ROW_UNIT",
    "REGENERATE_HINT",
    "SCALE_FIELD",
    "extract_projects_block",
    "image_digest_claims",
    "ledger_state_claims",
    "render_project_row",
    "render_projects_block",
    "replace_projects_block",
]
