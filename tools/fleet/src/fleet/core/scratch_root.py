"""Where the fleet's scratch root is on each platform (MCPs board task fb505198).

IN MEMORY ON LINUX. The hub's scratch went to ``/tmp``, which on diphtheria
is its one SATA SSD: measured 2026-10-24 with a ``vfs_write`` trace, the
staging archives and the per-tick rolled extractions wrote about 520 GiB a
day there, 93% of the box's writes outside a deploy, every byte of it read
once or never. Every Linux machine mounts ``/dev/shm`` as a tmpfs (half of
RAM by default), so the root is there. Windows, where the fleet's suites
also run, has no memory-backed directory, so there the root is the system
temporary directory.

Pure, and apart from :mod:`fleet.core._test_hooks`, whose default
``temp_root`` calls it with the real platform: a test reads both answers
without changing the platform it runs on. What lives under the root is
:mod:`fleet.core.scratch`.
"""

from __future__ import annotations

import pathlib
from typing import Final

#: The memory-backed directory the scratch root is on Linux.
LINUX_SCRATCH_ROOT: Final = "/dev/shm"


def scratch_root_for(platform: str, temp_directory: str) -> pathlib.Path:
    """Name the scratch root for a platform.

    Args:
        platform: ``sys.platform``.
        temp_directory: ``tempfile.gettempdir()``.

    Returns:
        :data:`LINUX_SCRATCH_ROOT` on Linux, else the temporary directory.
    """
    if platform == "linux":
        return pathlib.Path(LINUX_SCRATCH_ROOT)
    return pathlib.Path(temp_directory)


__all__ = ["LINUX_SCRATCH_ROOT", "scratch_root_for"]
