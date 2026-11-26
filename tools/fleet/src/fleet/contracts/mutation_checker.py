"""The memory a project's mutation step gives each TypeScript checker, as the registry declares.

MCPs board task 330b0465. Every MCPs TypeScript package's ``make check`` ends
in a mutation step (MCPs ``mcp-shared/src/mutation``) that runs StrykerJS
with its TypeScript checker. StrykerJS 10.1.1 starts ``ceil(c/2)`` checker
processes and ``floor(c/2)`` test runners from its ``concurrency`` option
``c``, then ``c`` runners once checking ends (``ConcurrencyTokenProvider``).
Fleet job 752810f4 (MCPs/packages/db at 600a93bb3 on pendragon, 7 workers)
asked for 14 and was killed for memory on an 11.4 GB node whose admission
had booked the check 3.5 GB (7 x ``worker_ram_gb`` 0.5).

A CHECKER'S SIZE IS ITS HEAP CAP. Uncapped, a checker grows to V8's default
heap limit, which follows the machine's memory: 2240 MB on pendragon and
4288 MB on ziggurat (``v8.getHeapStatistics().heap_size_limit``), and
packages/db's grew past 2.7 GB in nine minutes of checking on ziggurat. Its
live set is far smaller: wiki-check's checker checked 47 mutants under a
700 MB cap and peaked at 0.86 GB RSS, against 1.66 GB uncapped (2026-11-26).
So a project declares ``heap_mb``, the cap the step passes as
``--max-old-space-size``, at which its checker was measured to finish, and
``ram_gb``, what the step charges a checker against the check's grant: the
cap and the memory V8 holds beside its heap.

The step's smallest run is StrykerJS's own process, one test runner and one
checker, the first two charged one worker each, then its own process and two
runners once the checker is freed: :func:`mutation_floor_gb`.
Admission books a run of a project that declares a checker at no less than
that floor, whatever its workers
(:func:`~fleet.contracts.project.run_ram_gb`), and refuses a node whose
memory left is under it (:func:`~fleet.core.capacity.assess`). The floor is
memory, so it is booked as memory: ``minimum_workers`` also books cores
(``minimum_workers x worker_cores``), and carrying a memory floor there would
refuse a node for cores the step does not use. The step reads the
declaration from the registry the runner staged and asks StrykerJS for no
more checkers and runners than the same booked memory holds.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

#: The fields a declaration carries, exactly.
MUTATION_CHECKER_FIELDS: frozenset[str] = frozenset({"heap_mb", "ram_gb", "measurement"})

#: Megabytes in the gigabyte ``ram_gb`` counts, as V8 counts ``--max-old-space-size``.
MB_PER_GB = 1024

#: The processes of the step's smallest run charged one worker each beside its
#: checker, or beside the second runner that follows it: StrykerJS's own
#: process and one test runner (a vitest host with one fork).
WORKERS_BESIDE_ONE_CHECKER = 2


class MutationChecker(TypedDict):
    """What one TypeScript checker of a project's mutation step is given.

    Attributes:
        heap_mb: Its V8 heap cap in MB, passed as ``--max-old-space-size``,
            at which it was measured to check its mutants to the end.
        ram_gb: What the step charges it against the check's grant: the cap
            and the memory V8 holds beside its heap. At least the cap.
        measurement: The run that measured both, so the numbers a reader
            meets in the registry carry where they came from.
    """

    heap_mb: ReadOnly[int]
    ram_gb: ReadOnly[float]
    measurement: ReadOnly[str]


def mutation_floor_gb(checker: MutationChecker, *, worker_ram_gb: float) -> float:
    """The least memory the mutation step can run in.

    StrykerJS's smallest run, at concurrency 2, holds its own process, one
    test runner and one checker while it checks, then its own process and two
    runners once the checker is freed, so it peaks at the larger phase.

    Args:
        checker: The project's declared checker.
        worker_ram_gb: The project's memory per test worker, charged for
            StrykerJS's own process and for each test runner.

    Returns:
        ``max(checker.ram_gb, worker_ram_gb) + 2 x worker_ram_gb``.
    """
    return max(checker["ram_gb"], worker_ram_gb) + WORKERS_BESIDE_ONE_CHECKER * worker_ram_gb


def decode_mutation_checker(value: JSONValue) -> MutationChecker | None:
    """Decode a project's optional ``mutation_checker``.

    Args:
        value: The key's value, or None when the key is absent: a project
            whose check runs no TypeScript checker.

    Returns:
        The checker, or None for a project that declares none.

    Raises:
        JSONTypeError: When the value is not an object of exactly
            ``heap_mb``, ``ram_gb`` and ``measurement``, ``heap_mb`` is not a
            positive whole number, ``ram_gb`` is under the cap, or the
            measurement is empty.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != MUTATION_CHECKER_FIELDS:
        raise JSONTypeError(
            "project.mutation_checker must be an object of exactly heap_mb, ram_gb and "
            f"measurement, got {value!r}"
        )
    heap_mb = require_int(value, "heap_mb")
    if heap_mb < 1:
        raise JSONTypeError(
            f"project.mutation_checker.heap_mb must be a positive whole number of MB, got "
            f"{heap_mb}; it is the heap cap the step gives each checker"
        )
    ram_gb = require_float(value, "ram_gb")
    if ram_gb * MB_PER_GB < heap_mb:
        raise JSONTypeError(
            f"project.mutation_checker.ram_gb {ram_gb} is under its heap_mb {heap_mb} MB; a "
            "checker is charged its heap cap and what V8 holds beside it"
        )
    measurement = require_str(value, "measurement")
    if not measurement.strip():
        raise JSONTypeError(
            "project.mutation_checker.measurement must name the run that measured heap_mb and "
            "ram_gb, got empty text"
        )
    return MutationChecker(heap_mb=heap_mb, ram_gb=ram_gb, measurement=measurement)


def encode_mutation_checker(checker: MutationChecker | None) -> JSONObject | None:
    """Encode a project's ``mutation_checker``.

    Args:
        checker: The checker, or None.

    Returns:
        Its JSON object, or None for a project that declares none.
    """
    if checker is None:
        return None
    return {
        "heap_mb": checker["heap_mb"],
        "ram_gb": checker["ram_gb"],
        "measurement": checker["measurement"],
    }
