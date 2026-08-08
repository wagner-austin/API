"""Play queued matches until the queue stays empty.

The engine half of [[harness-match-service]], phase zero: a host-side worker
that claims jobs, leases clone indices from the allocator, and plays each
match through the same harness seams sweeps use -- so artifacts file exactly
where a sweep files them and every existing reader keeps working.

Usage::

    poetry run python -m scripts.match_worker <dsn> <worker-name> \
        <clones> [max-jobs]

``clones`` is a comma-separated list of the harness's clone indices this
worker may lease -- the runner's zero-based ordinals, so ``0,1,2,3`` names
``.game-w1`` through ``.game-w4``. Two workers given disjoint pools cannot
collide; two workers given the same pool cannot either, because the lease
table decides.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot.harness import _test_hooks
from rw_bot.service.worker import run_worker

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Poll the queue and play what it holds.

    Args:
        argv: ``<dsn> <worker-name> <clones> [max-jobs]``. ``None`` reads
            the process arguments.

    Returns:
        ``EXIT_OK`` when the worker drained the queue or spent its budget,
        ``EXIT_BAD_USAGE`` on a bad argument count.

    Raises:
        MatchServiceError: When a claimed row is unreadable.
        ValueError: When the clone list or budget is not numeric.
        Exception: Whatever the database driver raises when the queue is
            unreachable.
    """
    args = list(argv) if argv is not None else _test_hooks.read_argv()
    if len(args) not in (3, 4):
        _test_hooks.write_line("usage: match_worker <dsn> <worker-name> <clones> [max-jobs]")
        return EXIT_BAD_USAGE
    clone_pool = tuple(int(part) for part in args[2].split(","))
    max_jobs = int(args[3]) if len(args) == 4 else 0
    played = run_worker(args[0], args[1], clone_pool, max_jobs)
    _test_hooks.write_line(f"[worker] {args[1]} played {played} match(es)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
