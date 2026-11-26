"""A project's mutation checker memory, declared in the registry (MCPs board task 330b0465).

Each floor below is worked by hand from the decoder's rule: one checker's
``ram_gb`` beside StrykerJS's own process and one runner, charged
``worker_ram_gb`` each, against ``minimum_workers x worker_ram_gb``. The
grants are packages/db's registry row's: ``worker_ram_gb`` 0.5.
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
from fleet.contracts.project import decode_project_config, encode_project_config

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
        "minimum_workers": 4,
        "expected_minutes": 25,
        "required_tags": [],
        "source": None,
    }
    row.update(fields)
    return row


class TestTheDeclaration:
    def test_it_decodes_encodes_and_is_absent_as_none(self) -> None:
        assert decode_mutation_checker(DECLARED, worker_ram_gb=0.5, minimum_workers=4) == CHECKER
        assert encode_mutation_checker(CHECKER) == DECLARED
        assert decode_mutation_checker(None, worker_ram_gb=0.5, minimum_workers=4) is None
        assert encode_mutation_checker(None) is None

    def test_the_floor_is_one_checker_beside_stryker_and_one_runner(self) -> None:
        # 1.0 + 2 x 0.5 and 1.7 + 2 x 0.4.
        assert mutation_floor_gb(CHECKER, worker_ram_gb=0.5) == 2.0
        assert mutation_floor_gb(_decoded(1500, 1.7), worker_ram_gb=0.4) == 2.5

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
            decode_mutation_checker(value, worker_ram_gb=0.5, minimum_workers=4)

    def test_a_charge_equal_to_its_cap_is_accepted(self) -> None:
        # 1024 MB is 1.0 GB.
        decoded = decode_mutation_checker(_checker(1024, 1.0), worker_ram_gb=0.5, minimum_workers=4)
        assert decoded == _decoded(1024, 1.0)


class TestTheFloorAgainstTheLeastGrant:
    def test_a_least_grant_that_holds_the_floor_exactly_is_accepted(self) -> None:
        # 4 x 0.5 = 2.0 = 1.0 + 2 x 0.5.
        assert decode_mutation_checker(DECLARED, worker_ram_gb=0.5, minimum_workers=4) == CHECKER

    def test_a_least_grant_equal_but_for_float_rounding_is_accepted(self) -> None:
        # 6 x 0.3 is 1.7999999999999998 in floats and 1.2 + 2 x 0.3 is 1.8: the
        # same 1.8 GB, which the decoder reads as equal.
        assert 6 * 0.3 < 1.2 + 2 * 0.3
        decoded = decode_mutation_checker(_checker(1000, 1.2), worker_ram_gb=0.3, minimum_workers=6)
        assert decoded == _decoded(1000, 1.2)

    def test_a_least_grant_under_the_floor_is_refused_naming_the_minimum_that_holds_it(
        self,
    ) -> None:
        # 4 x 0.5 = 2.0 under 1.1 + 2 x 0.5 = 2.1; 5 x 0.5 = 2.5 holds it.
        with pytest.raises(JSONTypeError) as refused:
            decode_mutation_checker(_checker(900, 1.1), worker_ram_gb=0.5, minimum_workers=4)
        assert str(refused.value) == (
            "project.minimum_workers 4 x worker_ram_gb 0.5 grants 2.00 GB at least, under the "
            "2.10 GB its mutation step's smallest run holds (one checker of 1.1 GB beside "
            "StrykerJS and one runner of 0.5 GB each); raise minimum_workers to 5"
        )

    def test_the_project_decoder_holds_its_own_grant_against_the_floor(self) -> None:
        with pytest.raises(JSONTypeError, match="raise minimum_workers to 5"):
            decode_project_config(_project(mutation_checker=_checker(900, 1.1)))
        raised = decode_project_config(
            _project(minimum_workers=5, mutation_checker=_checker(900, 1.1))
        )
        assert raised["mutation_checker"] == _decoded(900, 1.1)
        # A worker of 0.6 GB: 4 x 0.6 = 2.4 holds 1.1 + 2 x 0.6 = 2.3.
        wider = decode_project_config(
            _project(worker_ram_gb=0.6, mutation_checker=_checker(900, 1.1))
        )
        assert wider["mutation_checker"] == _decoded(900, 1.1)

    def test_the_project_row_round_trips_with_and_without_a_checker(self) -> None:
        declared = decode_project_config(_project(mutation_checker=DECLARED))
        assert declared["mutation_checker"] == CHECKER
        assert encode_project_config(declared)["mutation_checker"] == DECLARED
        assert decode_project_config(encode_project_config(declared)) == declared
        bare = decode_project_config(_project())
        assert bare["mutation_checker"] is None
        assert encode_project_config(bare)["mutation_checker"] is None
