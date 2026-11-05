"""A worker is charged the cores its project declares (MCPs board task 51e3eac5).

Built with ``test_capacity.py``'s builders. MCPs/fleet-nixos's one worker is
a nix daemon that runs as many builds as the node's dispatch share has
cores. Charged one core, its cold run on lavender (fleet job 46eb328f)
shared that 8-core share with two checks the fleet placed beside it, 386.5
CPU-s beside its own 1463.8, and ran 306 s of its 300.
"""

from __future__ import annotations

import pytest
from platform_core.errors import FleetErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.node import LiveLoad
from fleet.contracts.project import (
    decode_project_config,
    decode_worker_cores,
    decode_worker_cores_measurement,
    encode_project_config,
)
from fleet.core.capacity import assess, job_ceiling
from tests.test_capacity import LINUX_TAGS, _node, _project, _state

#: What a declaration that is no whole number of at least one is refused with.
REFUSED = r"project\.worker_cores must be a whole number of at least 1 or absent"

#: What a count above one with no measurement is refused with.
MEASUREMENT_MISSING = r"project\.worker_cores 8 must carry worker_cores_measurement"

#: What a measurement beside one core a worker is refused with.
MEASUREMENT_STRAY = r"project\.worker_cores_measurement is '.+' beside worker_cores 1"

#: A live run of the flake check: one worker on 8 cores holding 16 GB.
FLAKE_RUN = LiveLoad(runs=1, workers=1, cores=8, ram_gb=16.0)


class TestTheDeclaration:
    def test_absent_reads_as_one_and_a_whole_number_as_itself(self) -> None:
        assert decode_worker_cores(None) == 1
        assert decode_worker_cores(8) == 8

    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
    def test_any_other_value_is_refused_by_name(self, value: JSONValue) -> None:
        with pytest.raises(JSONTypeError, match=REFUSED):
            decode_worker_cores(value)

    def test_a_declaration_survives_encoding(self) -> None:
        project = _project(worker_cores=8, worker_cores_measurement="fleet job 46eb328f")

        assert decode_project_config(encode_project_config(project)) == project
        assert decode_project_config(encode_project_config(_project())) == _project()


class TestTheMeasurement:
    """A count above one carries the run that set it, in the registry beside it."""

    def test_a_count_above_one_carries_its_measurement(self) -> None:
        assert decode_worker_cores_measurement("fleet job 46eb328f", worker_cores=8) == (
            "fleet job 46eb328f"
        )

    @pytest.mark.parametrize("value", [None, "", "  ", 8])
    def test_a_count_above_one_without_text_is_refused(self, value: JSONValue) -> None:
        with pytest.raises(JSONTypeError, match=MEASUREMENT_MISSING):
            decode_worker_cores_measurement(value, worker_cores=8)

    def test_one_core_carries_none_and_refuses_a_measurement(self) -> None:
        assert decode_worker_cores_measurement(None, worker_cores=1) is None
        with pytest.raises(JSONTypeError, match=MEASUREMENT_STRAY):
            decode_worker_cores_measurement("fleet job 46eb328f", worker_cores=1)

    def test_the_project_decoder_applies_it(self) -> None:
        row = encode_project_config(_project(worker_cores=8, worker_cores_measurement="x"))
        row["worker_cores_measurement"] = None

        with pytest.raises(JSONTypeError, match=MEASUREMENT_MISSING):
            decode_project_config(row)


class TestTheCharge:
    def test_a_worker_of_eight_cores_takes_one_of_fourteen_spare(self) -> None:
        """16 cores, 2 reserved: a one-core suite is granted 7, this one 1."""
        project = _project(minimum_workers=1, worker_cores=8)

        assert job_ceiling(_node(), project) == 1
        assert assess(_node(), _state(), project, LINUX_TAGS) == {
            "workers": 1,
            "code": None,
            "reason": "",
        }

    def test_the_ceiling_is_the_workers_half_the_spare_cores_hold(self) -> None:
        """Half of 14 is 7 cores, which hold 2 workers of 3."""
        project = _project(minimum_workers=1, worker_cores=3)

        assert job_ceiling(_node(), project) == 2
        assert assess(_node(), _state(), project, LINUX_TAGS)["workers"] == 2

    def test_a_live_run_s_cores_come_off_the_pool(self) -> None:
        """14 spare less the flake's 8 leaves 6, though it holds one worker."""
        state = _state(free_ram_gb=60.0, live=FLAKE_RUN)
        verdict = assess(_node(), state, _project(minimum_workers=1), LINUX_TAGS)

        assert verdict == {"workers": 6, "code": None, "reason": ""}

    def test_a_share_the_flake_fills_takes_nothing_beside_it(self) -> None:
        """lavender's shape: 16 cores, 8 reserved for its owner."""
        node = _node(cores=16, reserved_cores=8)
        verdict = assess(node, _state(live=FLAKE_RUN), _project(minimum_workers=1), LINUX_TAGS)

        assert (verdict["workers"], verdict["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
        assert verdict["reason"] == (
            "lavender has 27.0 GB free against a reservation of 4.0 GB for whoever is using it, "
            "and 16 cores against 8 reserved, and its 1 live fleet run(s) hold 1 worker(s) on 8 "
            "core(s) and 16.0 GB. Nothing is left for a dispatch; its own fleet runs hold the "
            "rest, so it takes the next job when one of them ends."
        )

    def test_a_share_smaller_than_one_worker_never_takes_the_project(self) -> None:
        """serendipity's shape: 8 cores, 2 reserved, idle."""
        node = _node(host="serendipity", cores=8, reserved_cores=2)
        project = _project(minimum_workers=1, worker_cores=8)
        verdict = assess(node, _state(host="serendipity"), project, LINUX_TAGS)

        assert (verdict["workers"], verdict["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
        assert verdict["reason"] == (
            "serendipity has 27.0 GB free against a reservation of 4.0 GB for whoever is using "
            "it, and 8 cores against 2 reserved. Nothing is left for a dispatch; one worker of "
            "this project occupies 8 cores and its owner's reservation leaves 6, so it never "
            "takes this project."
        )


class TestTheQuota:
    def test_a_quota_under_one_worker_s_cores_refuses_it(self) -> None:
        verdict = assess(
            _node(),
            _state(check_cpu_quota=6.0),
            _project(minimum_workers=1, worker_cores=8),
            LINUX_TAGS,
        )

        assert (verdict["workers"], verdict["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
        assert (
            "so 6 core(s) are left where the suite's minimum of 1 worker(s) occupies 8"
            in (verdict["reason"])
        )

    def test_the_quota_s_cores_bound_the_workers_they_hold(self) -> None:
        """32 cores leave 30 and a ceiling of 5 three-core workers; 7 CPUs hold 2."""
        verdict = assess(
            _node(cores=32),
            _state(check_cpu_quota=7.0),
            _project(minimum_workers=1, worker_cores=3),
            LINUX_TAGS,
        )

        assert verdict == {"workers": 2, "code": None, "reason": ""}
