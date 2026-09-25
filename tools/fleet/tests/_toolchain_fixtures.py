"""What real nodes answered the toolchain probe, shared by the toolchain suites.

THE FIXTURES ARE THE REAL MEASUREMENT. ``sedona``, ``lavender`` and ``loki``
answered these exact shapes on 2026-09-04, and one node of the three could
have run a ``make check``. ``diphtheria`` answered on 2026-09-20, the first
Linux node. The ``_2026_09_23`` answers are the probe that also asks for
``node`` and reads the Store alias as no python, run on each node that day.
A test written against an invented "node with everything" would have proved
the happy path and nothing about the fleet that actually exists.
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

#: What lavender answered the probe that asks for node, 2026-09-23, verbatim:
#: ready, after its python.org 3.11.9 was installed at user scope by the
#: winget command the contract now carries.
LAVENDER_2026_09_23 = (
    "python=yes=Python 3.11.9\n"
    "poetry=yes=Poetry (version 2.4.2)\n"
    "git=yes=git version 2.55.0.windows.5\n"
    "make=yes=GNU Make 4.4.1\n"
    "node=yes=v24.20.0\n"
    "tar=yes=bsdtar 3.8.8 - libarchive 3.8.8 zlib/1.2.13.1-motley liblzma/5.8.1 "
    "bz2lib/1.0.8 libzstd/1.5.7 cng/2.0 libb2/bundled\n"
    "winget=yes=v1.29.290\n"
    "choco=yes=2.7.4\n"
    "pip=yes=pip 24.0 from C:\\Users\\austi\\AppData\\Local\\Programs\\Python\\Python311"
    "\\Lib\\site-packages\\pip (python 3.11)\n"
)

#: The same probe on lavender, the same day, with the Python311 directories
#: taken off PATH: the state its build runner met from 2026-09-22 20:18Z to
#: 2026-09-23 05:39Z. ``python`` resolved to the WindowsApps Store alias,
#: which the probe reports absent, and poetry and pip went with it.
LAVENDER_STORE_STUB = (
    "python=no=\n"
    "poetry=no=\n"
    "git=yes=git version 2.55.0.windows.5\n"
    "make=yes=GNU Make 4.4.1\n"
    "node=yes=v24.20.0\n"
    "tar=yes=bsdtar 3.8.8 - libarchive 3.8.8 zlib/1.2.13.1-motley liblzma/5.8.1 "
    "bz2lib/1.0.8 libzstd/1.5.7 cng/2.0 libb2/bundled\n"
    "winget=yes=v1.29.290\n"
    "choco=yes=2.7.4\n"
    "pip=no=\n"
)

#: What sedona answered the same probe, 2026-09-23, verbatim.
SEDONA_2026_09_23 = (
    "python=yes=Python 3.11.9\n"
    "poetry=yes=Poetry (version 2.4.2)\n"
    "git=yes=git version 2.55.0.windows.5\n"
    "make=yes=GNU Make 4.4.1\n"
    "node=yes=v24.20.0\n"
    "tar=yes=bsdtar 3.8.8 - libarchive 3.8.8 zlib/1.2.13.1-motley liblzma/5.8.3 "
    "bz2lib/1.0.8 libzstd/1.5.7 cng/2.0 libb2/bundled\n"
    "winget=yes=v1.29.380\n"
    "choco=yes=2.7.4\n"
    "pip=yes=pip 24.0 from C:\\Users\\austi\\AppData\\Local\\Programs\\Python\\Python311"
    "\\Lib\\site-packages\\pip (python 3.11)\n"
)

#: What diphtheria answered the Linux probe that asks for node, 2026-09-23,
#: verbatim: 3.11 as ``python3`` now, and NodeSource's Node 24.
DIPHTHERIA_2026_09_23 = (
    "python=yes=Python 3.11.15\n"
    "poetry=yes=Poetry (version 2.5.1)\n"
    "git=yes=git version 2.43.0\n"
    "make=yes=GNU Make 4.3\n"
    "node=yes=v24.21.0\n"
    "tar=yes=tar (GNU tar) 1.35\n"
    "apt-get=yes=apt 2.8.3 (amd64)\n"
    "pipx=yes=1.4.3\n"
)

#: What serendipity answered, 2026-09-25 20:5xZ, verbatim: every tool present
#: and Python 3.11, and Node.js v18.13.0, which the contract called ready
#: until that day. It claimed MCPs/packages/wiki-search (dispatch job
#: de32d61e) and failed rebuilding hnswlib-node before any test ran.
SERENDIPITY_2026_09_25 = (
    "python=yes=Python 3.11.1\n"
    "poetry=yes=Poetry (version 2.5.1)\n"
    "git=yes=git version 2.41.0.windows.1\n"
    "make=yes=GNU Make 3.81\n"
    "node=yes=v18.13.0\n"
    "tar=yes=bsdtar 3.8.8 - libarchive 3.8.8 zlib/1.2.13.1-motley liblzma/5.8.3 "
    "bz2lib/1.0.8 libzstd/1.5.7 cng/2.0 libb2/bundled\n"
    "winget=yes=v1.29.380\n"
    "choco=no=\n"
    "pip=yes=pip 22.3.1 from C:\\Users\\austi\\AppData\\Local\\Programs\\Python\\Python311"
    "\\Lib\\site-packages\\pip (python 3.11)\n"
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


__all__ = [
    "DIPHTHERIA",
    "DIPHTHERIA_2026_09_23",
    "LAVENDER",
    "LAVENDER_2026_09_23",
    "LAVENDER_STORE_STUB",
    "LOKI",
    "SEDONA",
    "SEDONA_2026_09_23",
    "SERENDIPITY_2026_09_25",
    "WRONG_PYTHON",
    "node",
]
