"""The ``sh`` session observer's body: what python3 runs on a Linux node.

Split out of :mod:`fleet.core.dialect_linux` when the retire's unlock of a
run's read-only directories took that module past the 600-line ceiling (MCPs
board task 05dd6588), the shape :mod:`fleet.core.linux_toolchain_probe` and
:mod:`fleet.core.linux_capacity_probe` already have. It is the body without
the dialect's prologue, which
:data:`fleet.core.dialect_linux.OBSERVE_SESSIONS_SCRIPT` prepends.
"""

from __future__ import annotations

#: The body of the session-observer script: what python3 runs, verbatim.
#:
#: THE DOCUMENT IS BUILT BY python3, NOT BY sh. The records go out verbatim
#: inside one JSON array, and ``sh`` has no way to join files into one that
#: survives an empty directory, a record without a trailing newline or a
#: field with a quote in it; ``python3`` is on every Linux node by the
#: toolchain contract (``report python python3`` in
#: :data:`fleet.core.linux_toolchain_probe.TOOLCHAIN_PROBE_BODY`), on the PATH
#: the prologue prepends. ``sys.platform`` is ``linux``, the same word Node's
#: ``process.platform`` writes into the harness's ``pidDomain``; the
#: hostname is lowercased as the harness spells it (measured on diphtheria
#: 2026-09-21: ``hostname`` prints ``Diphtheria``). Those two spell the
#: ledger's machine key. ``pidDomain`` is computed apart, as Claude Code
#: 2.1.294 writes it on Linux (board task 5cee2f65): ``linux:``, the
#: ``/etc/machine-id``, then the pid namespace ``/proc/self/ns/pid`` links
#: to, measured on ziggurat 2026-10-16 as
#: ``linux:f191c99155c14a719632bdc88e77f914:pid:[4026531836]``. The script
#: runs in the namespace the node's sessions run in, so its link is theirs.
#: Records are read in file name order so two runs over one directory agree,
#: and an absent directory is an empty array rather than an error. Kept apart
#: from the ``sh`` that feeds it so a test can run exactly this text under the
#: interpreter it has.
OBSERVE_SESSIONS_PYTHON = (
    "import json\n"
    "import os\n"
    "import socket\n"
    "import sys\n"
    "directory = os.path.join(os.path.expanduser('~'), '.claude', 'sessions')\n"
    "records = []\n"
    "if os.path.isdir(directory):\n"
    "    for name in sorted(os.listdir(directory)):\n"
    "        if name.endswith('.json'):\n"
    "            with open(os.path.join(directory, name), encoding='utf-8') as handle:\n"
    "                records.append(json.load(handle))\n"
    "with open('/etc/machine-id', encoding='utf-8') as handle:\n"
    "    machine_id = handle.read().strip()\n"
    "document = {\n"
    "    'platform': sys.platform,\n"
    "    'hostname': socket.gethostname().lower(),\n"
    "    'pidDomain': 'linux:' + machine_id + ':' + os.readlink('/proc/self/ns/pid'),\n"
    "    'records': records,\n"
    "}\n"
    "print(json.dumps(document, separators=(',', ':')))\n"
)


__all__ = ["OBSERVE_SESSIONS_PYTHON"]
