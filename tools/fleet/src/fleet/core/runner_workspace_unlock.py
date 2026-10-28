"""A CI runner's work tree is left deletable before each job (MCPs board task 678de6ba).

actions/checkout cleans the workspace before it fetches, and a directory its
owner may not write cannot have its entries removed, so one read-only
directory fails the job at its first step with ``EACCES``. MCPs maketools'
release suites seal real payloads, read-only directories by design, under the
package's kept ``.pytest-tmp``. Its conftest makes them writable again as each
pytest worker's session ends, but a job that is cancelled ends no session:
MCPs Check run 40739743994 at 977b2fe1b was cancelled at 2026-10-28T17:54Z
when the next push superseded it, its maketools job left 1545 read-only
directories in each of lavender's runners mcps-1 and mcps-3, and run
40740060428's audit and discover jobs failed checkout on them at 18:04Z.
Every push to main cancels the run before it, so a cleanup that only a
finished run performs cannot hold.

So the job-started hook (:mod:`fleet.core.runner_job_scope_render`) runs
:data:`UNLOCK_LINES` before the job's first step, after
:data:`fleet.core.runner_tool_caches.HOOK_LINES` has named ``install_root``:
every directory under the install's ``_work`` that the runner account owns and
may not write is given its owner's write bit, without leaving the file system
or following a link, and the hook says how many of how many it unlocked. A
directory it cannot change fails the job with chmod's own error, through the
hook's ``set -euo pipefail``. Measured on lavender on 2026-10-28: one scan of an MCPs
runner's ``_work``, about 9,800 directories, took under 0.9 s.
"""

from __future__ import annotations

from typing import Final

#: The scan's arguments: the install's work tree, on its own file system,
#: directories the runner account owns.
_SCAN: Final = 'find "$install_root/_work" -xdev -type d -user "$(id -u)"'

#: The hook's lines that leave the work tree deletable. ``install_root`` is the
#: runner install :data:`fleet.core.runner_tool_caches.HOOK_LINES` read off
#: ``RUNNER_TEMP``.
UNLOCK_LINES: Final = (
    f"examined=$({_SCAN} | wc -l)",
    f"unlocked=$({_SCAN} ! -perm -u+w -print -exec chmod u+w {{}} + | wc -l)",
    'echo "fleet-runner-job-started: $unlocked of $examined directories under'
    ' $install_root/_work were read-only and are now writable"',
)

__all__ = ["UNLOCK_LINES"]
