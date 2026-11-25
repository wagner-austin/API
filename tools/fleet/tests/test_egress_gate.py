"""A node that cannot reach the internet claims no check (MCPs board task e70cdf05).

The probe the fleet sends is RUN, its machine roots pointed at the fakes
``test_linux_capacity_probe`` uses, and in the first case inside a network
namespace of its own (``unshare --user --map-root-user --net``). That
namespace has no interface but a loopback that is down, so every address
outside it fails, as everything did on lavender at 2026-11-25T04:00Z once it
had lost its IPv4 lease. Its answer is read by the probe's own reader and
weighed by the node runner's own gate, :func:`fleet.core.capacity.room_for_any`,
whose refusal is the verdict the runner's tick carries to the queue.

The node is declared with no disk reserved and the I/O pressure file calm, so
neither the disk the test host has free nor the fake's I/O stall can refuse
the node for another reason first.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.core import probe
from fleet.core.capacity import room_for_any
from tests.conftest import IDLE
from tests.test_capacity import LINUX_TAGS, _node, _project
from tests.test_linux_capacity_probe import PRESSURE, _probe

#: An I/O pressure file nothing stalls on, so the gate's I/O bound refuses nothing.
CALM_IO = (
    "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
)

#: A user namespace mapping this account to root, which lets an unprivileged
#: account make the network namespace beside it, with no network.
NO_NETWORK = ("unshare", "--user", "--map-root-user", "--net")


def _gate(tmp_path: pathlib.Path, wrapper: tuple[str, ...]) -> str | None:
    """Run the probe, read its answer and ask the node runner's gate.

    Args:
        tmp_path: Where the fakes and the script go.
        wrapper: What the probe's shell runs under.

    Returns:
        The gate's refusal, or None when the node may claim.
    """
    completed = _probe(tmp_path, None, PRESSURE, CALM_IO, wrapper=wrapper)
    assert completed.returncode == 0, completed.stderr
    outcome = probe.read_state("lavender", completed.stdout, live=IDLE)
    state = outcome["state"]
    assert state is not None, outcome["reason"]
    return room_for_any(
        _node(max_disk_gb=0.0), state, (_project(minimum_workers=1),), LINUX_TAGS
    )


@pytest.mark.host_linux
class TestEgressGate:
    def test_a_node_with_no_network_claims_nothing_and_names_its_lost_egress(
        self, tmp_path: pathlib.Path
    ) -> None:
        verdict = _gate(tmp_path, NO_NETWORK)

        assert verdict is not None
        # The code the task names, then the node, then the first index a
        # Python check's setup downloads from (poetry's).
        assert verdict.startswith(
            "NODE_EGRESS_LOST: lavender cannot reach the internet: https://pypi.org/simple/pip/ "
        )
        # npm's index, which a TypeScript check's setup downloads from.
        assert "; https://registry.npmjs.org/ " in verdict

    def test_the_same_node_on_this_host_s_network_may_claim(self, tmp_path: pathlib.Path) -> None:
        """A fleet node running this check downloaded its dependencies in setup."""
        assert _gate(tmp_path, ()) is None
