"""Tests for the dependency contract.

The id validation carries more weight than it looks like it should. Under
``--kill-on-invalid-dep`` a dependency on an id that never existed does not
wait -- it cancels the dependent job immediately, which reads exactly like the
pipeline failed. So a typo has to be refused here, where the message can say
what is wrong, rather than on the cluster where it cannot.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.dependency import (
    AFTER_ANY,
    AFTER_NOT_OK,
    AFTER_OK,
    DEPENDENCY_KINDS,
    Dependency,
    decode_dependency,
    dependency_argument,
    describe_dependency,
    encode_dependency,
)


def _payload(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid dependency payload.

    Args:
        **overrides: Fields to replace.

    Returns:
        A JSON object ready for decoding.
    """
    base: dict[str, JSONValue] = {"kind": AFTER_OK, "job_ids": ["55519937"]}
    base.update(overrides)
    return base


class TestDecode:
    def test_a_valid_dependency_decodes(self) -> None:
        assert decode_dependency(_payload(), "depends_on") == {
            "kind": "afterok",
            "job_ids": ["55519937"],
        }

    def test_null_means_the_job_waits_for_nothing(self) -> None:
        assert decode_dependency(None, "depends_on") is None

    def test_several_ids_are_kept_in_order(self) -> None:
        decoded = decode_dependency(_payload(job_ids=["1", "2", "3"]), "depends_on")
        if decoded is None:
            raise AssertionError("a stated dependency must decode")
        assert decoded["job_ids"] == ["1", "2", "3"]

    def test_every_kind_this_package_emits_is_accepted(self) -> None:
        for kind in DEPENDENCY_KINDS:
            decoded = decode_dependency(_payload(kind=kind), "depends_on")
            if decoded is None:
                raise AssertionError("a stated dependency must decode")
            assert decoded["kind"] == kind

    def test_the_three_kinds_are_exactly_these(self) -> None:
        """Slurm's `after` and `singleton` are deliberately absent: neither
        expresses "this stage consumes the previous stage's output", and
        `after` reads as if it did."""
        assert DEPENDENCY_KINDS == (AFTER_OK, AFTER_ANY, AFTER_NOT_OK)


class TestDecodeRefusals:
    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_dependency(["55519937"], "depends_on")

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="kind must be one of"):
            decode_dependency(_payload(kind="after"), "depends_on")

    def test_an_empty_id_list_is_refused(self) -> None:
        """`--dependency=afterok:` is malformed to sbatch, not absent."""
        with pytest.raises(JSONTypeError, match="at least one job id"):
            decode_dependency(_payload(job_ids=[]), "depends_on")

    def test_a_non_string_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be strings"):
            decode_dependency(_payload(job_ids=[55519937]), "depends_on")

    def test_a_non_numeric_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="numeric Slurm ids"):
            decode_dependency(_payload(job_ids=["job-1"]), "depends_on")

    def test_an_empty_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="numeric Slurm ids"):
            decode_dependency(_payload(job_ids=[""]), "depends_on")

    def test_a_repeated_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not repeat"):
            decode_dependency(_payload(job_ids=["1", "1"]), "depends_on")

    def test_a_missing_kind_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_dependency({"job_ids": ["1"]}, "depends_on")

    def test_a_missing_id_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_dependency({"kind": AFTER_OK}, "depends_on")


class TestEncode:
    def test_a_dependency_round_trips(self) -> None:
        payload = _payload(kind=AFTER_ANY, job_ids=["1", "2"])
        assert encode_dependency(decode_dependency(payload, "depends_on")) == payload

    def test_absence_encodes_as_null(self) -> None:
        assert encode_dependency(None) is None


class TestDependencyArgument:
    def test_one_id_renders_as_kind_colon_id(self) -> None:
        assert dependency_argument(Dependency(kind=AFTER_OK, job_ids=["1"])) == "afterok:1"

    def test_several_ids_are_colon_joined_which_is_slurms_and_form(self) -> None:
        """Comma is OR. A pipeline that started when ANY of its inputs
        finished would read the others mid-write."""
        argument = dependency_argument(Dependency(kind=AFTER_OK, job_ids=["1", "2", "3"]))
        assert argument == "afterok:1:2:3"
        assert "," not in argument

    def test_the_kind_leads(self) -> None:
        assert dependency_argument(Dependency(kind=AFTER_NOT_OK, job_ids=["9"])) == "afternotok:9"


class TestDescribe:
    def test_it_names_the_kind_and_the_ids(self) -> None:
        assert describe_dependency(Dependency(kind=AFTER_OK, job_ids=["1", "2"])) == "afterok 1,2"

    def test_absence_reads_as_nothing_rather_than_blank(self) -> None:
        """An empty string in a status line reads as a missing value."""
        assert describe_dependency(None) == "nothing"
