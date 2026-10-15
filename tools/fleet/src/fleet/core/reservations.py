"""A node's reservations: work the fleet does not run, declared so its cores are charged.

WHY (MCPs board task 84a2b690). A node's capacity check charged its cores
only with the fleet's own live runs (:func:`fleet.core.capacity.assess`),
and its probe read no CPU use, so work started outside the queue was
invisible to it. Measured 2026-10-14: corvis-stick's clean-VM smoke on
diphtheria (a 4-vCPU, 6 GB Windows guest, with the stage copy and image
writes before it) drove the 8-core node to a load of 26 to 32 and filled its
8 GB of swap, and the two fleet checks placed beside it ran past their 300 s
budget with every test passing: packages/db 411 s (fleet job fc1455d4) and
doc-extract-api 334 s (71a3505c). The same suites took 197 to 273 s there at
production's own load of 7 to 9.

WHY A DECLARATION AND NOT THE LOAD AVERAGE. :mod:`fleet.core.probe` reads no
process load on purpose, because a count of processes cannot be attributed.
The load average can be, to the machine, but charging it would charge
diphtheria's production stack too, whose baseline fills its 8 cores, so
the suites only that node can run (it alone carries ``stack``) would never
be claimed. A reservation is attributable by construction: the work that
holds the cores writes it, names itself, and says until when.

THE FILE. One ``<id>.reservation`` per reservation in :data:`DIRECTORY`
under the node account's home (``$HOME/.fleet/reservations`` on Linux,
``%USERPROFILE%\\.fleet\\reservations`` on Windows), as ``key=value`` lines,
the shape the probe itself prints:

* ``holder=`` who holds it, for the refusal a runner logs;
* ``cores=`` the logical cores it holds, a whole number;
* ``until_unix=`` when it stops counting, whole seconds since the epoch.

The holder removes its file when its work ends; ``until_unix`` bounds a
holder that died first, as a host claim's lease does. A file the probe
cannot read (``cores`` or ``until_unix`` missing or not a whole number)
makes the node's reading unreadable rather than charging nothing, since a
reservation silently ignored is the defect this module removes.

MEMORY IS NOT RESERVED HERE. ``MemAvailable`` already leaves out what the
work has made resident, as it does for every other process; a guest's
memory is resident within a minute of its boot.
"""

from __future__ import annotations

from typing import Final

#: The directory under the node account's home that holds its reservations.
DIRECTORY: Final = ".fleet/reservations"

#: Each reservation file's suffix.
SUFFIX: Final = ".reservation"

#: The probe field carrying the cores the node's live reservations hold.
RESERVED_CORES_FIELD: Final = "reserved_cores"

#: The probe field naming a reservation file it could not read.
UNREADABLE_FIELD: Final = "reservation_unreadable"

__all__ = ["DIRECTORY", "RESERVED_CORES_FIELD", "SUFFIX", "UNREADABLE_FIELD"]
