"""Where the fleet's scratch root is (MCPs board task fb505198).

IN MEMORY. The hub's scratch went to ``/tmp``, which on diphtheria is its one
SATA SSD: measured 2026-10-24 with a ``vfs_write`` trace, the staging
archives and the per-tick rolled extractions wrote about 520 GiB a day there,
93% of the box's writes outside a deploy, every byte of it read once or
never. Every Linux machine mounts ``/dev/shm`` as a tmpfs (half of RAM by
default), so the root is there. Windows, which had no memory-backed
directory and so used its system temporary directory, ran the fleet's
suites until its last node left the fleet (board task e705f08c).

Apart from :mod:`fleet.core._test_hooks`, whose default ``temp_root`` reads
it, so a test pins another root without this module knowing. What lives
under the root is :mod:`fleet.core.scratch`.
"""

from __future__ import annotations

import pathlib
from typing import Final

#: The memory-backed directory the scratch root is.
SCRATCH_ROOT: Final = pathlib.Path("/dev/shm")


__all__ = ["SCRATCH_ROOT"]
