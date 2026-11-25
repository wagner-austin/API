"""Egress readings like the fleet's, for tests that build a probe or a state (board task e70cdf05).

Every fake probe answer and every built :class:`fleet.contracts.node.NodeState`
carries one, since every probe prints the lines and the state requires them.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.egress_reading import EgressAttempt
from fleet.core.linux_egress_probe import TARGETS

#: What a node that reaches both package indexes prints.
REACHED_LINES: Final = f"egress_0=0 {TARGETS[0]}\negress_1=0 {TARGETS[1]}\negress_count=2\n"

#: Both indexes answered.
REACHED: Final = (
    EgressAttempt(target=TARGETS[0], curl_exit=0),
    EgressAttempt(target=TARGETS[1], curl_exit=0),
)

#: lavender at 2026-11-25T04:00Z: neither name resolved (curl exit 6), as
#: ``curl https://pypi.org/simple/pip/`` there answered 'Could not resolve host'.
LOST: Final = (
    EgressAttempt(target=TARGETS[0], curl_exit=6),
    EgressAttempt(target=TARGETS[1], curl_exit=6),
)
