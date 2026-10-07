"""A fake of what the collect pass hands runs to (MCPs board task c1d48330).

The collect pass hands the serve's watch the runs it holds, each run it
adopts, and each run whose node missed its read
(:class:`fleet.cli.node_collect.RunHolder`). A case that runs the pass
without a watch records those hand-overs here and asserts on them.
"""

from __future__ import annotations


class RecordingHolder:
    """Records every hand-over, in order.

    Satisfies :class:`fleet.cli.node_collect.RunHolder`.

    Attributes:
        held: Each set of runs given to :meth:`hold`.
        owed: Each set of runs given to :meth:`owe`.
    """

    held: list[frozenset[str]]
    owed: list[frozenset[str]]

    def __init__(self) -> None:
        """Start with nothing handed over."""
        self.held = []
        self.owed = []

    def hold(self, run_ids: frozenset[str]) -> None:
        """Record runs to watch.

        Args:
            run_ids: The runs.
        """
        self.held.append(run_ids)

    def owe(self, run_ids: frozenset[str]) -> None:
        """Record runs owed a renewal.

        Args:
            run_ids: The runs.
        """
        self.owed.append(run_ids)


__all__ = ["RecordingHolder"]
