"""Where a Linux node keeps its system programs, written once (MCPs board task f19d66a5).

A NixOS node keeps them in its setuid wrappers and its system profile, both
root's, and has nothing in ``/usr/bin`` but ``env``; an Ubuntu node has
neither directory and keeps them in ``/usr/bin`` and ``/bin``. So one PATH
names both, in that order, and each finds its programs where it keeps them.
The isolated build started from it first (MCPs board task c2e3f31f, loki).

EVERY PROGRAM OF THE NODE IS FOUND ON IT, never by an absolute path. Until
MCPs board task f19d66a5, ``fleet.core.tick_run`` named
``/usr/bin/systemctl``, and the hub execution suite's cases started units
with ``/usr/bin/systemd-run`` and ``/bin/sleep``. diphtheria and lacrymal
both read ``ID=nixos``, where none of the three exists, so fleet job
d2088f5d (tools/fleet-execution-hub at API 1928bc6be) failed five cases on
FileNotFoundError, and a tick's reading of a lost run would have failed the
same way on every NixOS node. The runner host cases handed their probes
``PATH=/usr/local/bin:/usr/bin:/bin``, so on lavender systemd-run could find
no python3 (fleet job 3a7777fa). A command that runs a node's own program
is given this PATH (:data:`EXEC_PATH_SETTING`) and names the program alone.
"""

from __future__ import annotations

from typing import Final

#: The PATH a node's own programs are found on: the NixOS wrappers and system
#: profile first, then where an Ubuntu node keeps them.
EXEC_PATH: Final = "/run/wrappers/bin:/run/current-system/sw/bin:/usr/local/bin:/usr/bin:/bin"

#: The child environment entry that gives a command :data:`EXEC_PATH`, in the
#: ``set_env`` form :func:`fleet.core._test_hooks.run` takes.
EXEC_PATH_SETTING: Final[tuple[str, str]] = ("PATH", EXEC_PATH)


__all__ = ["EXEC_PATH", "EXEC_PATH_SETTING"]
