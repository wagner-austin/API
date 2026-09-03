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
"""

from __future__ import annotations

from hpc3.contracts.workspace import ProjectConfig

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
        The pinned digest's first twelve characters, or ``none`` when the
        project declares no image. Twelve is enough to tell two images apart
        by eye and short enough to keep the row readable; the whole digest
        lives in the workspace document.
    """
    image = project["image"]
    if image is None:
        return "none"
    return f"`{image['sha256'][:12]}`"


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
        f"| {project['checkpoint_steps']} |"
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
        "ckpt steps",
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
    "REGENERATE_HINT",
    "extract_projects_block",
    "render_project_row",
    "render_projects_block",
    "replace_projects_block",
]
