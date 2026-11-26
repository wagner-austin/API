"""THROWAWAY RED FORM for MCPs board task 330b0465: a declaration read with no floor.

A registry before the task held no checker figure and admission charged none,
so this reads whatever is written and holds nothing against the least grant.
Never published.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue
from typing_extensions import ReadOnly, TypedDict

#: The fields a declaration carries.
MUTATION_CHECKER_FIELDS: frozenset[str] = frozenset({"heap_mb", "ram_gb", "measurement"})


class MutationChecker(TypedDict):
    """What one checker is given.

    Attributes:
        heap_mb: Its heap cap.
        ram_gb: Its charge.
        measurement: Its measurement.
    """

    heap_mb: ReadOnly[int]
    ram_gb: ReadOnly[float]
    measurement: ReadOnly[str]


def mutation_floor_gb(checker: MutationChecker, *, worker_ram_gb: float) -> float:
    """Charge the checker alone, as a step that counted no process beside it.

    Args:
        checker: The checker.
        worker_ram_gb: Unread.

    Returns:
        The checker's charge.
    """
    return checker["ram_gb"] + 0.0 * worker_ram_gb


def decode_mutation_checker(
    value: JSONValue,
    *,
    worker_ram_gb: float,
    minimum_workers: int,
) -> MutationChecker | None:
    """Read the declaration with no check.

    Args:
        value: The key's value.
        worker_ram_gb: Unread.
        minimum_workers: Unread.

    Returns:
        The checker, or None for anything but an object.
    """
    if not isinstance(value, dict) or worker_ram_gb < 0 or minimum_workers < 0:
        return None
    heap = value.get("heap_mb")
    ram = value.get("ram_gb")
    measurement = value.get("measurement")
    return MutationChecker(
        heap_mb=heap if isinstance(heap, int) else 0,
        ram_gb=float(ram) if isinstance(ram, (int, float)) else 0.0,
        measurement=measurement if isinstance(measurement, str) else "",
    )


def encode_mutation_checker(checker: MutationChecker | None) -> JSONObject | None:
    """Encode it.

    Args:
        checker: The checker, or None.

    Returns:
        Its object, or None.
    """
    if checker is None:
        return None
    return {
        "heap_mb": checker["heap_mb"],
        "ram_gb": checker["ram_gb"],
        "measurement": checker["measurement"],
    }
