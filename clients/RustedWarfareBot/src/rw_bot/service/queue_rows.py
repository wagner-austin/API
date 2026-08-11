"""Validating what the queue's statements return, one row shape at a time.

Split from :mod:`rw_bot.service.queue` at the size cap: the statements and
their transactions stay together, and the row-shape validators -- every
``RW-SERVICE-001`` refusal in the service -- live here. A poisoned queue is
a loud stop, not a skipped job, and these are the functions that stop it.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot import RwBotError


class MatchServiceError(RwBotError):
    """The queue answered with a shape the service cannot read.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming what was malformed.
    """


def _batch_name(row: Sequence[str | int]) -> str:
    """Validate one batch-name row.

    Args:
        row: One row of the batch listing query.

    Returns:
        The batch name.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 1 and isinstance(row[0], str):
        return row[0]
    raise MatchServiceError("RW-SERVICE-001", f"batch row has an unreadable shape: {row!r}")


def _running_columns(row: Sequence[str | int]) -> tuple[str, str, int, str, int]:
    """Validate one running-match row.

    Args:
        row: One row of the running-matches query.

    Returns:
        The batch, label, seed, worker and clone index.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if (
        len(row) == 5
        and isinstance(row[0], str)
        and isinstance(row[1], str)
        and isinstance(row[2], int)
        and isinstance(row[3], str)
        and isinstance(row[4], int)
    ):
        return row[0], row[1], row[2], row[3], row[4]
    raise MatchServiceError("RW-SERVICE-001", f"running row has an unreadable shape: {row!r}")


def _claim_columns(row: Sequence[str | int]) -> tuple[int, str, str, str, str]:
    """Validate the claim query's row shape.

    Args:
        row: What ``fetchone`` returned.

    Returns:
        The id, batch, encoded config, encoded match and encoded job.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if (
        len(row) == 5
        and isinstance(row[0], int)
        and isinstance(row[1], str)
        and isinstance(row[2], str)
        and isinstance(row[3], str)
        and isinstance(row[4], str)
    ):
        return row[0], row[1], row[2], row[3], row[4]
    raise MatchServiceError("RW-SERVICE-001", f"claim row has an unreadable shape: {row!r}")


def _match_payload(parsed: dict[str, str | int | float | bool]) -> dict[str, str | int]:
    """Narrow a parsed match payload to the types its codec reads.

    Args:
        parsed: The stored match object, parsed.

    Returns:
        The same mapping, with every value proven ``str`` or ``int``.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` when a value is neither --
            a stored match carries only paths and counts, so anything else
            is corruption.
    """
    narrowed: dict[str, str | int] = {}
    for key, value in parsed.items():
        if not isinstance(value, (str, int)):
            raise MatchServiceError(
                "RW-SERVICE-001", f"match field {key} has an unreadable type: {value!r}"
            )
        narrowed[key] = value
    return narrowed


def _result_columns(row: Sequence[str | int]) -> tuple[str, int, str, str]:
    """Validate one result row's shape.

    Args:
        row: One row of the results query.

    Returns:
        The label, seed, state and card text.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if (
        len(row) == 4
        and isinstance(row[0], str)
        and isinstance(row[1], int)
        and isinstance(row[2], str)
        and isinstance(row[3], str)
    ):
        return row[0], row[1], row[2], row[3]
    raise MatchServiceError("RW-SERVICE-001", f"result row has an unreadable shape: {row!r}")


def _status_columns(row: Sequence[str | int]) -> tuple[str, int]:
    """Validate one state-count row.

    Args:
        row: One row of the status query.

    Returns:
        The state and its count.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 2 and isinstance(row[0], str) and isinstance(row[1], int):
        return row[0], row[1]
    raise MatchServiceError("RW-SERVICE-001", f"status row has an unreadable shape: {row!r}")


def _lease_index(row: Sequence[str | int]) -> int:
    """Validate one lease row's clone index.

    Args:
        row: One row of the lease query.

    Returns:
        The leased clone index.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 1 and isinstance(row[0], int):
        return row[0]
    raise MatchServiceError("RW-SERVICE-001", f"lease row has an unreadable shape: {row!r}")


def _reaped_id(row: Sequence[str | int]) -> int:
    """Validate one requeued job id.

    Args:
        row: One row of the reap statement's RETURNING clause.

    Returns:
        The requeued job's id.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 1 and isinstance(row[0], int):
        return row[0]
    raise MatchServiceError("RW-SERVICE-001", f"reaped row has an unreadable shape: {row!r}")
