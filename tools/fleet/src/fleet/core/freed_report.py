"""What retires freed on each node since an instant (MCPs board task 1526ece1).

The freed record (:mod:`fleet.contracts.freed_record`) holds one line per
retire done, written by the hub's runners as each retire's node answers with
the bytes it removed. This totals those lines per node over a window, every
node of the workspace named so a node whose retires freed nothing reads 0
rather than missing, and renders the totals as the one line the hub posts on
a board task (``python -m fleet.cli.freed_report``).
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence

from typing_extensions import TypedDict

from fleet.contracts.freed_record import FreedRecord


class NodeFreed(TypedDict):
    """What one node's retires freed in a window.

    Attributes:
        node: The node's workspace name.
        runs: How many retires were done there.
        freed_bytes: The bytes those retires removed.
    """

    node: str
    runs: int
    freed_bytes: int


def node_totals(
    records: Sequence[FreedRecord], *, since_unix: int, nodes: Sequence[str]
) -> tuple[NodeFreed, ...]:
    """Total each node's retires recorded at or after an instant.

    Args:
        records: The freed record's lines.
        since_unix: The window's start, whole seconds since the epoch.
        nodes: Every node of the workspace, each reported even with no line;
            a node a line names outside them is reported too, after them.

    Returns:
        One total per node: ``nodes`` in their order, then any other node a
        line in the window names, in the order first recorded.
    """
    runs: dict[str, int] = dict.fromkeys(nodes, 0)
    freed: dict[str, int] = dict.fromkeys(nodes, 0)
    for record in records:
        if record["at_unix"] < since_unix:
            continue
        node = record["node"]
        runs[node] = runs.get(node, 0) + 1
        freed[node] = freed.get(node, 0) + record["freed_bytes"]
    return tuple(NodeFreed(node=node, runs=runs[node], freed_bytes=freed[node]) for node in runs)


def report_line(totals: Sequence[NodeFreed], *, since_unix: int, source: str) -> str:
    """The note the hub posts: every node's runs retired and bytes freed.

    Args:
        totals: :func:`node_totals`' answer.
        since_unix: The window's start.
        source: The freed record the totals were read from.

    Returns:
        ``STAGE FREED since <instant>: <node> <runs> run(s) <bytes> bytes; ...;
        total <runs> run(s) <bytes> bytes, from <source>``.
    """
    since = datetime.datetime.fromtimestamp(since_unix, tz=datetime.UTC)
    nodes = "; ".join(
        f"{total['node']} {total['runs']} run(s) {total['freed_bytes']} bytes" for total in totals
    )
    runs = sum(total["runs"] for total in totals)
    freed = sum(total["freed_bytes"] for total in totals)
    return (
        f"STAGE FREED since {since.strftime('%Y-%m-%dT%H:%M:%SZ')}: {nodes}; "
        f"total {runs} run(s) {freed} bytes, from {source}"
    )


__all__ = ["NodeFreed", "node_totals", "report_line"]
