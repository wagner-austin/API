"""The width a node runner offers the queue for each project it fits.

MCPs board task 1964fc15: a push gate holds each release check to 150 s
from claim to verdict, and how long a check takes follows the node and the
test workers it is granted. Measured over 12 h on 2026-11-04,
MCPs/packages/session-audit-linux took 295 to 333 s at 2 workers on
lenovold and 114 s at 7, and maketools at 4 workers took 109 to 147 s on
colossus and 129 to 189 s on lavender. So a node runner claims naming,
beside each project it fits, the workers its capacity check would grant it
(:func:`fleet.core.capacity.fitting_projects`), and the queue skips a live
turn tip's check on a node and width whose own passed runs miss the budget
while another node and width meets it. The queue reads each run's width
from the start report (:func:`fleet.core.queue.report_start`).
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject, JSONValue


class WidthOffer(TypedDict):
    """One fitting project and the workers this node would grant it now.

    Attributes:
        project: The registry project.
        workers: The test workers :func:`fleet.core.capacity.assess` grants
            it on this node's state, never fewer than one.
    """

    project: str
    workers: int


def encode_width_offers(offers: tuple[WidthOffer, ...]) -> list[JSONValue]:
    """Encode offers as ``dispatch_claim``'s ``widths`` argument.

    Args:
        offers: The offers, one per fitting project.

    Returns:
        One ``{"project", "workers"}`` object per offer, in order.
    """
    encoded: list[JSONValue] = []
    for offer in offers:
        entry: JSONObject = {"project": offer["project"], "workers": offer["workers"]}
        encoded.append(entry)
    return encoded


__all__ = ["WidthOffer", "encode_width_offers"]
