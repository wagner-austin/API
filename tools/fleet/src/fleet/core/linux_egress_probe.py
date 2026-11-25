"""The capacity probe's egress lines: whether the node reaches the internet (board task e70cdf05).

:mod:`fleet.contracts.egress_reading` says why the node asks and what counts
as reaching. This module holds the lines the node runs and the reader of
what they print.

THE OUTPUT. ``egress_<n>=`` then curl's exit code, a space and the target,
one line per target in the order they are asked; then ``egress_count=``.
The count comes after the targets, so a probe that ended among them prints
none and is refused as missing it, as the GPU lines' count is.

THE BOUNDS. Each target gets :data:`CONNECT_SECONDS` to connect and
:data:`MAX_SECONDS` in all, so a node whose packets vanish costs the probe at
most ten seconds, well inside the ssh call's 120
(:data:`fleet.core.remote.SSH_TIMEOUT_SECONDS`). ``--head`` asks for headers
alone: the index pages are not downloaded.

THE TARGETS AND THE BOUNDS ARE VARIABLES ON THEIR OWN LINES, as the rest of
the probe's roots are, so a test points them at a server it runs and runs the
script under ``sh``. curl is found on ``PATH``. Every fleet node carries it
in its system profile (``/run/current-system/sw/bin/curl``, measured over ssh
on all eleven Linux nodes at 2026-11-25T22:3xZ).
"""

from __future__ import annotations

import re
from typing import Final

from fleet.contracts.egress_reading import CURL_MISSING_EXIT, EgressAttempt

#: The package indexes a check's setup downloads from: poetry's and npm's.
#: pip's own project page rather than the whole index, which is tens of
#: megabytes even for a HEAD's server-side work.
TARGETS: Final = ("https://pypi.org/simple/pip/", "https://registry.npmjs.org/")

#: Seconds each target gets to connect.
CONNECT_SECONDS: Final = 3

#: Seconds each target gets in all.
MAX_SECONDS: Final = 5

#: Where the probe asks; the test replaces it.
TARGETS_LINE: Final = f"egress_targets='{' '.join(TARGETS)}'\n"

#: How long each ask may take; the test replaces it.
BOUNDS_LINE: Final = f"egress_connect={CONNECT_SECONDS} egress_max={MAX_SECONDS}\n"

#: The field counting the targets, printed after them.
COUNT_FIELD: Final = "egress_count"

#: The prefix of each target's field.
FIELD_PREFIX: Final = "egress_"

#: The probe's egress lines, verbatim. ``|| reached=$?`` keeps a failed ask
#: from ending the probe under the prologue's ``set -e``.
EGRESS_PROBE_PART: Final = (
    TARGETS_LINE
    + BOUNDS_LINE
    + "asked=0\n"
    + "for target in $egress_targets; do\n"
    + "  reached=0\n"
    + '  curl --silent --head --output /dev/null --connect-timeout "$egress_connect"'
    + ' --max-time "$egress_max" "$target" 2>/dev/null || reached=$?\n'
    + f'  printf \'{FIELD_PREFIX}%s=%s %s\\n\' "$asked" "$reached" "$target"\n'
    + "  asked=$((asked + 1))\n"
    + "done\n"
    + f"printf '{COUNT_FIELD}=%s\\n' \"$asked\"\n"
)

#: A whole number as the probe prints it.
_WHOLE: Final = re.compile(r"0|[1-9][0-9]*")

#: One target's line: curl's exit, a space, the target.
_ATTEMPT: Final = re.compile(r"(0|[1-9][0-9]{0,2}) (\S+)")


def read_egress(host: str, fields: dict[str, str]) -> tuple[tuple[EgressAttempt, ...] | None, str]:
    """Read every target a probe asked.

    A reading that cannot be read is refused, never read as reach or as its
    absence, the way :mod:`fleet.core.probe` refuses every other reading.

    Args:
        host: The node, for the message.
        fields: The probe's key=value pairs.

    Returns:
        The attempts in the order the probe asked and an empty string; or
        None and the reason, naming the node, when the count is missing, not
        a whole number or zero, a target it counts printed no line, a line
        cannot be read, or curl was not found.
    """
    count = fields.get(COUNT_FIELD)
    if count is None or _WHOLE.fullmatch(count) is None or count == "0":
        return None, (
            f"{host} reported {COUNT_FIELD}={count!r}, which is not a positive whole number "
            "of targets, so whether it reaches the internet is unknown"
        )
    attempts: list[EgressAttempt] = []
    for index in range(int(count)):
        line = fields.get(f"{FIELD_PREFIX}{index}")
        if line is None:
            return None, (
                f"{host} counted {count} egress target(s) and printed no {FIELD_PREFIX}{index} line"
            )
        matched = _ATTEMPT.fullmatch(line)
        if matched is None:
            return None, f"{host} printed an egress line that cannot be read: {line!r}"
        curl_exit = int(matched.group(1))
        if curl_exit == CURL_MISSING_EXIT:
            return None, (
                f"{host} has no curl on its PATH (exit {CURL_MISSING_EXIT}), so its probe cannot "
                "say whether it reaches the internet"
            )
        attempts.append(EgressAttempt(target=matched.group(2), curl_exit=curl_exit))
    return tuple(attempts), ""


__all__ = [
    "BOUNDS_LINE",
    "CONNECT_SECONDS",
    "COUNT_FIELD",
    "EGRESS_PROBE_PART",
    "FIELD_PREFIX",
    "MAX_SECONDS",
    "TARGETS",
    "TARGETS_LINE",
    "read_egress",
]
