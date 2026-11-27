"""An ssh to a node carried out on this machine, for cases that run a node's real scripts.

Lifted out of ``test_retire_in_use.py`` (MCPs board task 1526ece1) when the
host-target cases (MCPs board task 57791f39) needed the same transport: the
retire pass and a host target's stage and run each send their scripts over
ssh, and a case runs exactly what the node would by handing ssh's remote
words to this machine's ``sh``. Every other command (the hub's own git)
runs as the default hook runs it.
"""

from __future__ import annotations

from collections.abc import Sequence

from fleet.core import _test_hooks
from fleet.core.ssh_options import SSH_OPTIONS


def on_this_machine(
    argv: Sequence[str],
    *,
    timeout_seconds: int,
    stdin_bytes: bytes | None = None,
    unset_env: Sequence[str] = (),
    set_env: Sequence[tuple[str, str]] = (),
) -> _test_hooks.CommandResult:
    """Carry out an ssh or an scp to the node here.

    Everything after an ssh's destination is what ssh hands the far side's
    shell, so ``sh -c`` of those words joined is what the node runs. An scp
    (:func:`fleet.core.remote.send_file`) copies its source to the path after
    the destination's colon, so ``cp`` of the two is what lands on the node.
    Each must carry the package's own options, exactly as the transport sends
    them.

    Args:
        argv: The command the code under test ran.
        timeout_seconds: Its deadline.
        stdin_bytes: Its standard input.
        unset_env: Variables it withholds.
        set_env: Variables it sets.

    Returns:
        What the command, or the node's script, did here.
    """
    words = list(argv)
    here: tuple[str, ...]
    if words[:1] == ["scp"]:
        copy = ["scp", "-q", *SSH_OPTIONS]
        assert words[: len(copy)] == copy
        source, destination = words[len(copy) :]
        here = ("cp", source, destination.split(":", 1)[1])
    elif words[:1] == ["ssh"]:
        prefix = ["ssh", *SSH_OPTIONS]
        assert words[: len(prefix)] == prefix
        here = ("sh", "-c", " ".join(words[len(prefix) + 1 :]))
    else:
        here = tuple(words)
    return _test_hooks._default_run(
        here,
        timeout_seconds=timeout_seconds,
        stdin_bytes=stdin_bytes,
        unset_env=unset_env,
        set_env=set_env,
    )


__all__ = ["on_this_machine"]
