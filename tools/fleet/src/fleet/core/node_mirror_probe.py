"""What a node's mirror of a remote already holds, asked before anything is sent to it.

Split out of :mod:`fleet.core.commit_stage` when staging a commit beside its
project's base branch (MCPs board task a48b5236) took that module past the
600-line ceiling: the probe is one concern, the script a node runs and the
reading of its answer, and both the commit's stage and a companion's ask it
(:mod:`fleet.core.companion_stage`).
"""

from __future__ import annotations

import re
import shlex
from typing import Final

from typing_extensions import TypedDict

from fleet.core import dialect

#: The node mirror's ref that names the last commit fetched into it, the
#: base the hub's next bundle for that node leaves out.
NODE_TIP_REF: Final = "refs/fleet/staged"

#: The node mirror's ref that names the last base branch tip fetched into
#: it, which keeps that tip's history reachable (MCPs board task a48b5236).
NODE_BASE_REF: Final = "refs/fleet/base"

#: What the node's prepare script prints when its mirror already holds the commit.
HELD_ANSWER: Final = "held"

#: What it prints when its mirror already holds the base branch tip.
BASE_HELD_ANSWER: Final = "base-held"

#: A commit id as the prepare script prints it.
_SHA: Final = re.compile(r"[0-9a-f]{40}")


class NodeMirror(TypedDict):
    """What the node's mirror holds, as its prepare script said.

    Attributes:
        tip: :data:`NODE_TIP_REF`'s commit, or None for a mirror that has
            never been fetched into or does not exist.
        held: Whether the mirror's :data:`NODE_TIP_REF` already reaches the
            commit being staged (:func:`mirror_probe_lines`).
        base_held: Whether it already reaches the base branch tip staged
            beside the commit; False when none is.
    """

    tip: str | None
    held: bool
    base_held: bool


def prepare_script(*, target: str, staged: str, mirror: str, sha: str, base_tip: str | None) -> str:
    """Make the export's directories and say what the node's mirror holds.

    Args:
        target: The export's directory.
        staged: Its staging directory.
        mirror: The node's mirror of the commit's remote.
        sha: The commit.
        base_tip: The project's base branch tip, which the mirror must hold
            too, or None.

    Returns:
        The script's text. It prints what :func:`mirror_probe_lines` prints
        once both directories exist.
    """
    return (
        f"{dialect.PROLOGUE}mkdir -p {shlex.quote(target)} {shlex.quote(staged)}\n"
        f"{mirror_probe_lines(mirror=mirror, sha=sha, base_tip=base_tip)}"
    )


def mirror_probe_lines(*, mirror: str, sha: str, base_tip: str | None) -> str:
    """The lines of a node script that say what the node's mirror holds.

    HELD MEANS :data:`NODE_TIP_REF` REACHES THE COMMIT (MCPs board task
    5280a25d), not that the object is there: two runs staging one remote on
    a node at once fetch into one mirror, and a small fetch writes its
    objects loose, the commit before its tree, while git moves the ref only
    once every object is written. A commit off the ref's history is sent
    again, which costs a bundle and never a tree missing its blobs. A base
    branch tip is held the same way, reached by :data:`NODE_TIP_REF` or by
    :data:`NODE_BASE_REF`, and answered on its own (MCPs board task
    a48b5236): ``git bundle create`` leaves out a ref its exclusions reach,
    so a bundle carries only the refs the node lacks and the node fetches
    exactly those.

    Args:
        mirror: The node's mirror of the remote.
        sha: The commit.
        base_tip: The base branch tip staged beside it, or None.

    Returns:
        The lines. They print :data:`NODE_TIP_REF`'s commit when the mirror
        has one, :data:`HELD_ANSWER` when the mirror reaches ``sha`` and
        :data:`BASE_HELD_ANSWER` when it reaches ``base_tip``, and nothing
        about a mirror that does not exist yet.
    """
    quoted = shlex.quote(mirror)
    tip = shlex.quote(f"{NODE_TIP_REF}^{{commit}}")
    lines = (
        f"if [ -d {quoted} ]; then\n"
        f"  git -C {quoted} rev-parse --verify --quiet {tip} || true\n"
        f"  if git -C {quoted} merge-base --is-ancestor {shlex.quote(sha)} {NODE_TIP_REF}"
        f" 2>/dev/null; then echo {HELD_ANSWER}; fi\n"
    )
    if base_tip is not None:
        base = shlex.quote(base_tip)
        lines += (
            f"  if git -C {quoted} merge-base --is-ancestor {base} {NODE_TIP_REF} 2>/dev/null"
            f" || git -C {quoted} merge-base --is-ancestor {base} {NODE_BASE_REF} 2>/dev/null;"
            f" then echo {BASE_HELD_ANSWER}; fi\n"
        )
    return f"{lines}fi\n"


def read_node_mirror(said: str) -> NodeMirror:
    """Read the prepare script's answer.

    Args:
        said: What :func:`prepare_script` printed.

    Returns:
        The mirror's tip and whether it holds the commit and the base tip.
    """
    words = said.split()
    tips = [word for word in words if _SHA.fullmatch(word) is not None]
    return NodeMirror(
        tip=tips[0] if tips else None,
        held=HELD_ANSWER in words,
        base_held=BASE_HELD_ANSWER in words,
    )


__all__ = [
    "BASE_HELD_ANSWER",
    "HELD_ANSWER",
    "NODE_BASE_REF",
    "NODE_TIP_REF",
    "NodeMirror",
    "mirror_probe_lines",
    "prepare_script",
    "read_node_mirror",
]
