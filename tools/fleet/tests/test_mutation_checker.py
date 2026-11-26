"""A project's mutation checker memory, declared in the registry (MCPs board task 330b0465).

Each booking below is worked by hand from the rule: a run is booked its
workers times ``worker_ram_gb``, or, for a project that declares a checker,
the step's smallest run when that is more: the checker's ``ram_gb`` beside
StrykerJS's own process and one runner, charged ``worker_ram_gb`` each.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from fleet.contracts.mutation_checker import (
    MutationChecker,
    decode_mutation_checker,
    encode_mutation_checker,
    mutation_floor_gb,
)
from fleet.contracts.project import (
    decode_project_config,
    encode_project_config,
    run_ram_gb,
)

#: How the declaration below was measured.
MEASURED = "checker_peak.py, 0.86 GB at 700 MB"

#: A declaration as the registry writes it: a 700 MB cap charged 1.0 GB.
DECLARED: JSONObject = {"heap_mb": 700, "ram_gb": 1.0, "measurement": MEASURED}

#: Its decoded form.
CHECKER = MutationChecker(heap_mb=700, ram_gb=1.0, measurement=MEASURED)


def _checker(heap_mb: int, ram_gb: float) -> JSONObject:
    """A declaration of a cap and a charge.

    Args:
        heap_mb: The cap.
        ram_gb: The charge.

    Returns:
        The declaration as the registry writes it.
    """
    return {"heap_mb": heap_mb, "ram_gb": ram_gb, "measurement": "m"}


def _decoded(heap_mb: int, ram_gb: float) -> MutationChecker:
    """The decoded form of :func:`_checker`.

    Args:
        heap_mb: The cap.
        ram_gb: The charge.

    Returns:
        The checker.
    """
    return MutationChecker(heap_mb=heap_mb, ram_gb=ram_gb, measurement="m")


def _project(**fields: JSONValue) -> JSONObject:
    """One project row, packages/db's worker and expected time, with the given fields over it.

    Args:
        **fields: The fields to set or replace.

    Returns:
        The row as the registry writes it.
    """
    row: JSONObject = {
        "worker_ram_gb": 0.5,
        "minimum_workers": 3,
        "expected_minutes": 25,
        "required_tags": [],
        "source": None,
    }
    row.update(fields)
    return row


class TestTheDeclaration:
    def test_it_decodes_encodes_and_is_absent_as_none(self) -> None:
        assert decode_mutation_checker(DECLARED) == CHECKER
        assert encode_mutation_checker(CHECKER) == DECLARED
        assert decode_mutation_checker(None) is None
        assert encode_mutation_checker(None) is None

    def test_the_floor_is_one_checker_beside_stryker_and_one_runner(self) -> None:
        # 1.0 + 2 x 0.5 and 1.7 + 2 x 0.4.
        assert mutation_floor_gb(CHECKER, worker_ram_gb=0.5) == 2.0
        assert mutation_floor_gb(_decoded(1500, 1.7), worker_ram_gb=0.4) == 2.5

    def test_a_checker_lighter_than_a_worker_floors_at_the_two_runners_after_it(self) -> None:
        # Checking holds 0.5 + 0.3 + 0.5 = 1.3; the two runners after it, beside
        # StrykerJS, hold 3 x 0.5 = 1.5, the larger.
        assert mutation_floor_gb(_decoded(300, 0.3), worker_ram_gb=0.5) == 1.5

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            (1.0, "exactly heap_mb, ram_gb and measurement"),
            ({"heap_mb": 700, "ram_gb": 1.0}, "exactly heap_mb, ram_gb and measurement"),
            ({**DECLARED, "runners": 2}, "exactly heap_mb, ram_gb and measurement"),
            ({**DECLARED, "heap_mb": 700.5}, "heap_mb"),
            ({**DECLARED, "heap_mb": 0}, "positive whole number of MB, got 0"),
            ({**DECLARED, "heap_mb": -1}, "positive whole number of MB, got -1"),
            ({**DECLARED, "ram_gb": "1.0"}, "ram_gb"),
            ({**DECLARED, "ram_gb": 0.68}, "ram_gb 0.68 is under its heap_mb 700 MB"),
            ({**DECLARED, "measurement": 7}, "measurement"),
            ({**DECLARED, "measurement": "  "}, "must name the run that measured heap_mb"),
        ],
    )
    def test_a_misspelt_declaration_is_refused(self, value: JSONValue, named: str) -> None:
        with pytest.raises(JSONTypeError, match=named):
            decode_mutation_checker(value)

    def test_a_charge_equal_to_its_cap_and_a_cap_of_one_mb_are_accepted(self) -> None:
        # 1024 MB is 1.0 GB; 1 MB is the least whole cap.
        assert decode_mutation_checker(_checker(1024, 1.0)) == _decoded(1024, 1.0)
        assert decode_mutation_checker(_checker(1, 0.001)) == _decoded(1, 0.001)

    def test_the_project_row_round_trips_with_and_without_a_checker(self) -> None:
        declared = decode_project_config(_project(mutation_checker=DECLARED))
        assert declared["mutation_checker"] == CHECKER
        assert encode_project_config(declared)["mutation_checker"] == DECLARED
        assert decode_project_config(encode_project_config(declared)) == declared
        bare = decode_project_config(_project())
        assert bare["mutation_checker"] is None
        assert encode_project_config(bare)["mutation_checker"] is None

    def test_minimum_workers_is_not_held_against_the_floor(self) -> None:
        # 3 x 0.5 = 1.5 under the 2.0 floor: admission books the floor as
        # memory instead, so the row decodes as written.
        row = decode_project_config(_project(mutation_checker=DECLARED))
        assert row["minimum_workers"] == 3


class TestTheBooking:
    @pytest.mark.parametrize(
        ("workers", "booked"),
        [
            # 1.5 GB of workers under the 2.0 floor: the floor.
            (3, 2.0),
            # Exactly the floor either way.
            (4, 2.0),
            # 3.5 GB of workers over the floor: the workers.
            (7, 3.5),
        ],
    )
    def test_a_project_with_a_checker_is_booked_at_least_its_floor(
        self, workers: int, booked: float
    ) -> None:
        project = decode_project_config(_project(mutation_checker=DECLARED))
        assert run_ram_gb(project, workers) == booked

    @pytest.mark.parametrize(("workers", "booked"), [(1, 0.5), (3, 1.5), (7, 3.5)])
    def test_a_project_without_one_is_booked_its_workers(self, workers: int, booked: float) -> None:
        project = decode_project_config(_project())
        assert run_ram_gb(project, workers) == booked

    def test_the_booking_is_the_larger_of_the_workers_and_the_floor_over_a_grid(self) -> None:
        # The reference: each phase of the smallest run summed by hand, the
        # larger taken, then compared with the workers by hand. A checker of
        # 0.65 under a worker of 0.75 floors at the two runners after it.
        for worker_ram_gb in (0.25, 0.3, 0.5, 0.75):
            for ram_gb in (0.65, 1.05, 2.0):
                row = _project(
                    worker_ram_gb=worker_ram_gb,
                    mutation_checker=_checker(int(ram_gb * 1024) - 200, ram_gb),
                )
                project = decode_project_config(row)
                checking = ram_gb + 2 * worker_ram_gb
                testing = worker_ram_gb + 2 * worker_ram_gb
                floor = checking if checking > testing else testing
                for workers in range(1, 17):
                    workers_gb = workers * worker_ram_gb
                    expected = floor if floor > workers_gb else workers_gb
                    assert run_ram_gb(project, workers) == expected
