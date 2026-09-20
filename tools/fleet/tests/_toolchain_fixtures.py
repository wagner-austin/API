"""What real nodes answered the toolchain probe, shared by the toolchain suites.

THE FIXTURES ARE THE REAL MEASUREMENT. ``sedona``, ``lavender`` and ``loki``
answered these exact shapes on 2026-09-04, and one node of the three could
have run a ``make check``. ``diphtheria`` answered on 2026-09-20, the first
Linux node. A test written against an invented "node with everything" would
have proved the happy path and nothing about the fleet that actually exists.
"""

from __future__ import annotations

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig

#: What loki answered: everything present, poetry on the wrong Python, and
#: CHOCO BUT NO WINGET -- which is why the same missing tool renders a
#: different install command here than on lavender.
LOKI = (
    "python=yes=Python 3.11.9\n"
    "poetry=yes=Poetry (version 2.1.3)\n"
    "git=yes=git version 2.50.1.windows.1\n"
    "make=yes=GNU Make 4.4.1\n"
    "tar=yes=bsdtar 3.7.2\n"
    "winget=no=\n"
    "choco=yes=2.5.0\n"
    "pip=yes=pip 24.0 from C:\\Python311\\Lib\\site-packages\\pip (python 3.11)\n"
)

#: What lavender answered: no poetry, no git, no make, WINGET BUT NO CHOCO.
LAVENDER = (
    "python=yes=Python 3.11.9\n"
    "poetry=no=\n"
    "git=no=\n"
    "make=no=\n"
    "tar=yes=bsdtar 3.7.2\n"
    "winget=yes=v1.11.400\n"
    "choco=no=\n"
    "pip=yes=pip 24.0 from C:\\Python311\\Lib\\site-packages\\pip (python 3.11)\n"
)

#: What sedona answered: only make absent, and BOTH managers present.
SEDONA = (
    "python=yes=Python 3.11.9\n"
    "poetry=yes=Poetry (version 2.2.1)\n"
    "git=yes=git version 2.43.0.windows.1\n"
    "make=no=\n"
    "tar=yes=bsdtar 3.7.2\n"
    "winget=yes=v1.11.400\n"
    "choco=yes=2.6.0\n"
    "pip=yes=pip 24.0 from C:\\Python311\\Lib\\site-packages\\pip (python 3.11)\n"
)

#: What diphtheria answered on 2026-09-20, verbatim: the interpreter reported
#: under ``python`` from ``python3``, the Linux managers, and a 3.12 that the
#: contract refuses until the node carries 3.11 as ``python3``.
DIPHTHERIA = (
    "python=yes=Python 3.12.3\n"
    "poetry=yes=Poetry (version 2.5.1)\n"
    "git=yes=git version 2.43.0\n"
    "make=yes=GNU Make 4.3\n"
    "tar=yes=tar (GNU tar) 1.35\n"
    "apt-get=yes=apt 2.8.3 (amd64)\n"
    "pipx=yes=1.4.3\n"
)

#: A node carrying the wrong interpreter, which no probed Windows node did.
WRONG_PYTHON = LOKI.replace("Python 3.11.9", "Python 3.12.4")


def node(host: str = "lavender") -> NodeConfig:
    """Build a Windows node declaration.

    Args:
        host: SSH alias.

    Returns:
        The node.
    """
    return NodeConfig(
        host=host,
        platform="windows",
        stage_root="C:/fleet/stage",
        logical_cores=16,
        ram_gb=32.0,
        gpu=None,
        enabled=True,
        budget=NodeBudget(
            reserved_cores=2,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=2,
            max_disk_gb=20.0,
        ),
    )


__all__ = ["DIPHTHERIA", "LAVENDER", "LOKI", "SEDONA", "WRONG_PYTHON", "node"]
