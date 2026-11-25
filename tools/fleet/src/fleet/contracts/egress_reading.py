"""Whether a node reaches the internet, as its capacity probe read it (MCPs board task e70cdf05).

WHY A NODE ASKS. Every check's setup downloads what its lockfile pins:
poetry from pypi.org, npm from registry.npmjs.org. On 2026-11-25 lavender
lost its IPv4 lease to the LAN's DHCP. It still answered ssh over the
tailnet, so the hub's probe read it as healthy and the queue went on
handing it checks, and fleet job 376c6670 failed ``make setup`` in
packages/maketools with 'All attempts to connect to pypi.org failed', read
as maketools' own red. So the probe asks the two indexes itself, from the
node, the way a check's setup will.

WHAT REACHING IS. Each target is asked with ``curl --head``. Any HTTP
answer, whatever its status, proves the name resolved, a route existed and
TLS completed, so it is reach. A node reaches the internet when at least
one target answers. Requiring both would refuse every check on the fleet
whenever one index alone was down, including the checks that download only
from the other.

WHAT IS NOT A READING. Exit 127 is the shell's 'command not found': the node
has no curl, which says nothing about its network, so
:func:`fleet.core.linux_egress_probe.read_egress` refuses the probe rather
than calling the node offline.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict

#: The exit code of a target that answered over HTTP.
REACHED_EXIT: Final = 0

#: The shell's exit code for a command it cannot find.
CURL_MISSING_EXIT: Final = 127

#: The curl exits a refusal names in words, from curl's own exit code table
#: (``man curl``, EXIT CODES). Any other exit is named by its number alone.
CURL_FAILURES: Final[dict[int, str]] = {
    6: "could not resolve its host",
    7: "could not connect to it",
    28: "did not answer within the probe's bound",
    35: "failed its TLS handshake",
    60: "presented a certificate curl does not trust",
}


class EgressAttempt(TypedDict):
    """One target the probe asked, and how curl ended.

    Attributes:
        target: The URL asked.
        curl_exit: curl's exit code, :data:`REACHED_EXIT` for any HTTP answer.
    """

    target: str
    curl_exit: int


def reaches(attempts: tuple[EgressAttempt, ...]) -> bool:
    """Whether a node reaches the internet.

    Args:
        attempts: Every target its probe asked.

    Returns:
        True when at least one target answered.
    """
    return any(attempt["curl_exit"] == REACHED_EXIT for attempt in attempts)


def describe_attempt(attempt: EgressAttempt) -> str:
    """Say how one target's ask ended, for a refusal.

    Args:
        attempt: The target and curl's exit.

    Returns:
        The target, then curl's failure in words where
        :data:`CURL_FAILURES` names it, then the exit code, e.g.
        ``https://pypi.org/simple/pip/ could not resolve its host (curl exit 6)``.
    """
    exit_code = attempt["curl_exit"]
    words = CURL_FAILURES.get(exit_code, "failed")
    return f"{attempt['target']} {words} (curl exit {exit_code})"


__all__ = [
    "CURL_FAILURES",
    "CURL_MISSING_EXIT",
    "REACHED_EXIT",
    "EgressAttempt",
    "describe_attempt",
    "reaches",
]
