"""A project's host target in the registry, and what its decoders refuse (MCPs board task 57791f39).

Criterion A2: osrs-lab/manual-desktop declares host target ``update`` on
node loki, and a malformed host target is refused by name. The values each
case expects are written by hand from the task: the shipped declaration is
the one the task names, and each refusal's field is the key its case
spoiled. The round-trip property runs over generated host targets, and each
spoiled form is one encode never produces.
"""

from __future__ import annotations

import random

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from fleet.cli import rolled
from fleet.contracts.host_target import (
    MAX_DEADLINE_SECONDS,
    HostTarget,
    decode_host_target,
    encode_host_target,
)
from fleet.core import host_target, remote
from fleet.contracts.project import decode_project_config, encode_project_config
from fleet.contracts.workspace import decode_fleet_workspace, encode_fleet_workspace
from fleet.core import run_lease
from tests._registry import shipped_workspace
from tests.test_contracts import _project, _workspace

#: The seed of the generated host targets, fixed so a failure reproduces.
SEED = 57791

#: How many generated host targets the property reads.
CASES = 300

#: A make target's first and later characters (TARGET_NAME), written out.
FIRST = "abcdefghijklmnopqrstuvwxyz"
LATER = FIRST + "0123456789-"


def _generated(rng: random.Random) -> HostTarget:
    """One host target the grammar admits, drawn from the generator.

    Args:
        rng: The seeded generator.

    Returns:
        The host target.
    """
    target = rng.choice(FIRST) + "".join(rng.choice(LATER) for _ in range(rng.randrange(12)))
    node = "".join(rng.choice(LATER) for _ in range(1 + rng.randrange(10)))
    return HostTarget(
        target=target,
        node=node,
        workers=1 + rng.randrange(64),
        # A minute to half an hour, the bounds the task gives, both ends included.
        deadline_seconds=rng.randrange(60, 1801),
    )


def _declared() -> JSONObject:
    """The task's host target, as a registry line spells it.

    Returns:
        ``update`` on loki with six workers and a 1800-second deadline.
    """
    return {"target": "update", "node": "loki", "workers": 6, "deadline_seconds": 1800}


class TestTheShippedRegistry:
    def test_manual_desktop_runs_update_on_loki(self) -> None:
        project = shipped_workspace()["projects"]["osrs-lab/manual-desktop"]

        assert project["host_target"] == HostTarget(
            target="update", node="loki", workers=6, deadline_seconds=1800
        )


class TestTheRoundTrip:
    def test_every_generated_host_target_survives_encode_then_decode(self) -> None:
        rng = random.Random(SEED)
        for _ in range(CASES):
            generated = _generated(rng)
            assert decode_host_target(encode_host_target(generated)) == generated

    def test_absent_reads_as_none_and_none_encodes_as_none(self) -> None:
        assert decode_host_target(None) is None
        assert encode_host_target(None) is None

    def test_a_project_carries_its_host_target_through_its_own_codec(self) -> None:
        encoded = {**encode_project_config(_project()), "host_target": _declared()}

        decoded = decode_project_config(encoded)

        assert decoded["host_target"] == HostTarget(
            target="update", node="loki", workers=6, deadline_seconds=1800
        )
        assert encode_project_config(decoded)["host_target"] == _declared()

    def test_a_project_without_the_key_declares_none(self) -> None:
        encoded = encode_project_config(_project())
        del encoded["host_target"]

        assert decode_project_config(encoded)["host_target"] is None

    def test_the_smallest_values_the_grammar_admits_decode(self) -> None:
        smallest: JSONObject = {"target": "a", "node": "n", "workers": 1, "deadline_seconds": 60}

        assert decode_host_target(smallest) == HostTarget(
            target="a", node="n", workers=1, deadline_seconds=60
        )

    def test_the_longest_deadline_decodes(self) -> None:
        assert decode_host_target({**_declared(), "deadline_seconds": 1800}) == HostTarget(
            target="update", node="loki", workers=6, deadline_seconds=1800
        )


SPOILED: list[tuple[str, JSONValue, str]] = [
    ("not an object", ["update", "loki"], r"project\.host_target must be an object of exactly"),
    ("a key missing", {"target": "update", "node": "loki", "workers": 6}, r"of exactly"),
    ("a key too many", {**_declared(), "user": "corvis"}, r"of exactly"),
    ("an upper-case target", {**_declared(), "target": "Update"}, r"host_target\.target"),
    ("a target opening on a digit", {**_declared(), "target": "1update"}, r"host_target\.target"),
    ("a target with a shell word", {**_declared(), "target": "update;id"}, r"host_target\.target"),
    ("an empty target", {**_declared(), "target": ""}, r"host_target\.target"),
    ("an empty node", {**_declared(), "node": ""}, r"project\.host_target\.node must be a node's"),
    ("a numeric node", {**_declared(), "node": 7}, r"project\.host_target\.node must be a node's"),
    ("no workers", {**_declared(), "workers": 0}, r"host_target\.workers must be at least 1"),
    (
        "a deadline written in minutes",
        {**_declared(), "deadline_seconds": 59},
        r"host_target\.deadline_seconds must be at least 60",
    ),
    (
        "a deadline past the launcher's wall",
        {**_declared(), "deadline_seconds": 1801},
        r"host_target\.deadline_seconds must be at most 1800",
    ),
]


class TestARefusalNamesItsField:
    @pytest.mark.parametrize(
        ("spoiled", "match"),
        [(value, match) for _, value, match in SPOILED],
        ids=[name for name, _, _ in SPOILED],
    )
    def test_the_decoder_refuses_it(self, spoiled: JSONValue, match: str) -> None:
        with pytest.raises(JSONTypeError, match=match):
            decode_host_target(spoiled)

    @pytest.mark.parametrize(
        ("spoiled", "match"),
        [(value, match) for _, value, match in SPOILED],
        ids=[name for name, _, _ in SPOILED],
    )
    def test_the_registry_decoder_refuses_the_project(self, spoiled: JSONValue, match: str) -> None:
        with pytest.raises(JSONTypeError, match=match):
            decode_project_config({**encode_project_config(_project()), "host_target": spoiled})

    def test_a_host_target_on_an_undeclared_node_refuses_the_workspace(self) -> None:
        encoded = encode_fleet_workspace(_workspace())
        project = {
            **encode_project_config(_project()),
            "host_target": {**_declared(), "node": "colossus"},
        }

        with pytest.raises(JSONTypeError) as raised:
            decode_fleet_workspace({**encoded, "projects": {"osrs-lab/manual-desktop": project}})

        assert str(raised.value) == (
            "project osrs-lab/manual-desktop runs its host target on node 'colossus', which the "
            "workspace does not declare (it declares lavender, loki); no host could run it"
        )

    def test_a_host_target_on_a_declared_node_decodes(self) -> None:
        encoded = encode_fleet_workspace(_workspace())
        project = {**encode_project_config(_project()), "host_target": _declared()}

        decoded = decode_fleet_workspace(
            {**encoded, "projects": {"osrs-lab/manual-desktop": project}}
        )

        assert decoded["projects"]["osrs-lab/manual-desktop"]["host_target"] == HostTarget(
            target="update", node="loki", workers=6, deadline_seconds=1800
        )


class TestTheLeaseWindow:
    def test_a_host_target_holds_its_lease_twice_its_deadline(self) -> None:
        declared = HostTarget(target="update", node="loki", workers=6, deadline_seconds=1800)

        # 1800 s at a slack of 2, worked by hand.
        assert run_lease.host_target_window(declared) == 3600

    def test_the_longest_deadline_ends_inside_the_rolled_launcher_s_wall(self) -> None:
        """The node's deadline, its kill and the transport's bound all end before the launcher's."""
        longest = MAX_DEADLINE_SECONDS + host_target.KILL_AFTER_SECONDS
        assert longest + remote.SSH_TIMEOUT_SECONDS < rolled.AGENT_WALL_SECONDS
        # 1800 + 30 + 120 = 1950 of 39 x 60 = 2340 s, worked by hand: 390 s
        # remain for the fetch, the stage and the retire.
        assert rolled.AGENT_WALL_SECONDS - (longest + remote.SSH_TIMEOUT_SECONDS) == 390

    def test_a_check_holds_its_lease_twice_its_estimate(self) -> None:
        # Ten minutes is 600 s, at a slack of 2.
        assert run_lease.check_window(_project(expected_minutes=10)) == 1200
