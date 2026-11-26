"""THROWAWAY RED FORM for MCPs board task 330b0465: the declaration read with no floor and no checks.

A registry before the task held no checker figure and admission charged none,
so this reads whatever is written and holds nothing against the least grant.
Never published.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue
from typing_extensions import ReadOnly, TypedDict

MUTATION_CHECKER_FIELDS: frozenset[str] = frozenset({"heap_mb", "ram_gb", "measurement"})
MB_PER_GB = 1024
WORKERS_BESIDE_ONE_CHECKER = 2


class MutationChecker(TypedDict):
    """What one checker is given, unread."""

    heap_mb: ReadOnly[int]
    ram_gb: ReadOnly[float]
    measurement: ReadOnly[str]


def mutation_floor_gb(checker: MutationChecker, *, worker_ram_gb: float) -> float:
    """Charge the checker alone, as a step that counted no process beside it."""
    return checker["ram_gb"] + 0.0 * worker_ram_gb


def decode_mutation_checker(
    value: JSONValue,
    *,
    worker_ram_gb: float,
    minimum_workers: int,
) -> MutationChecker | None:
    """Read the declaration with no check."""
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
    """Encode it."""
    if checker is None:
        return None
    return {"heap_mb": checker["heap_mb"], "ram_gb": checker["ram_gb"], "measurement": checker["measurement"]}
