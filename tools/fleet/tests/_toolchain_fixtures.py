"""What real nodes answered the toolchain probe, shared by the toolchain suites.

THE FIXTURES ARE THE REAL MEASUREMENT. ``diphtheria`` answered on 2026-09-20,
the first Linux node, and again on 2026-09-23 and 2026-09-27 as the probe
learned to ask for ``node``, cargo and the C++ toolchain. Each answer that is
not a measurement says so and names the measured one it is derived from: a
test written against an invented "node with everything" would have proved the
happy path and nothing about the fleet that actually exists.

The Windows nodes' answers (loki, lavender, sedona, serendipity) went with
the Windows dialect (board task e705f08c).
"""

from __future__ import annotations

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig

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

#: What diphtheria answered the probe that asks for node, 2026-09-23,
#: verbatim: 3.11 as ``python3`` now, and NodeSource's Node 24. Ready.
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

#: What diphtheria answered the probe that also asks cargo, 2026-09-27
#: 00:5xZ, verbatim, minutes after rustup installed 1.98.1 at user scope
#: (MCPs board task 1e2da299). The trailing token is cargo's build date,
#: which is why :func:`fleet.contracts.capability.measured` reads the second.
DIPHTHERIA_2026_09_27 = (
    "python=yes=Python 3.11.15\n"
    "poetry=yes=Poetry (version 2.5.1)\n"
    "git=yes=git version 2.43.0\n"
    "make=yes=GNU Make 4.3\n"
    "node=yes=v24.21.0\n"
    "tar=yes=tar (GNU tar) 1.35\n"
    "cargo=yes=cargo 1.98.1 (797e8a9bc 2026-08-05)\n"
    "apt-get=yes=apt 2.8.3 (amd64)\n"
    "pipx=yes=1.4.3\n"
)

#: What diphtheria answered the probe that also asks for the C++ toolchain,
#: 2026-09-27 03:45Z, verbatim (MCPs board task 3f19c136): g++'s bare
#: -dumpfullversion on the synthetic ``cxx`` line.
DIPHTHERIA_2026_09_27_CXX = (
    "python=yes=Python 3.11.15\n"
    "poetry=yes=Poetry (version 2.5.1)\n"
    "git=yes=git version 2.43.0\n"
    "make=yes=GNU Make 4.3\n"
    "node=yes=v24.21.0\n"
    "tar=yes=tar (GNU tar) 1.35\n"
    "cargo=yes=cargo 1.98.1 (797e8a9bc 2026-08-05)\n"
    "cxx=yes=13.3.0\n"
    "apt-get=yes=apt 2.8.3 (amd64)\n"
    "pipx=yes=1.4.3\n"
)

#: DERIVED from :data:`DIPHTHERIA_2026_09_23`: the same node with make
#: absent, the one tool apt-get installs that a ready node can lack.
NO_MAKE = DIPHTHERIA_2026_09_23.replace("make=yes=GNU Make 4.3\n", "make=no=\n")

#: DERIVED from :data:`DIPHTHERIA_2026_09_23`: the interpreter gone from the
#: build account's PATH, and poetry and pipx with it, which is what a node
#: met when the build account's PATH lost ``/usr/bin/python3``.
NO_PYTHON = (
    DIPHTHERIA_2026_09_23.replace("python=yes=Python 3.11.15\n", "python=no=\n")
    .replace("poetry=yes=Poetry (version 2.5.1)\n", "poetry=no=\n")
    .replace("pipx=yes=1.4.3\n", "pipx=no=\n")
)

#: DERIVED from :data:`DIPHTHERIA_2026_09_23`: every tool present and Node.js
#: v18.13.0, the version serendipity answered on 2026-09-25, which the
#: contract called ready until that day. It claimed MCPs/packages/wiki-search
#: (dispatch job de32d61e) and failed rebuilding hnswlib-node before any test
#: ran.
OLD_NODE = DIPHTHERIA_2026_09_23.replace("node=yes=v24.21.0\n", "node=yes=v18.13.0\n")

#: DERIVED from :data:`DIPHTHERIA_2026_09_23`: the wrong interpreter, the
#: 3.12.3 diphtheria answered on 2026-09-20 (:data:`DIPHTHERIA`).
WRONG_PYTHON = DIPHTHERIA_2026_09_23.replace("Python 3.11.15", "Python 3.12.3")

#: Why a project needs the ``hooks`` tag, as a tick's gap line quotes it (MCPs
#: board task ec895824).
HOOKS_REASON = (
    "MCPs packages/claude-hooks's check runs on the system interpreter with its tools and "
    "reaches the board through ~/.claude/corvis-hooks.json, which only install-hooks-node.py "
    "writes, with the key the operator chose for the node"
)

#: The pip line a node with pip is offered for the hooks check's tools, at the
#: latest releases measured on 2026-10-16 (MCPs board task 09a34e03).
HOOKS_INSTALL = (
    "python -m pip install --user ruff==0.17.1 mypy==2.4.0 pytest==9.1.1 "
    "pytest-xdist==3.8.0 pytest-cov==7.1.0 coverage==7.16.2 pytest-timeout==2.4.0"
)

#: Why a node is told about go, in the tagged gap's words (MCPs board task
#: 1da15750).
GO_REASON = "MCPs rcs-bridge is a Go module whose check runs go vet, staticcheck and go test"

#: The line a node with apt-get is offered for go.
GO_APT = "sudo apt-get install -y golang-go"

#: How the tagged gap reads go's entry on a node whose managers cover none
#: of its installs.
GO_NO_INSTALL = (
    f"; go -- {GO_REASON}, so those jobs go to a node that has it -- no automatic install on "
    "this node"
)

#: Why a node is told about Chrome, in the tagged gap's words (MCPs board
#: task 2f596185): web-scraper's three real-browser cases launch Playwright's
#: channel chrome, which diphtheria lacked on 2026-10-04 and 2026-10-09.
CHROME_REASON = (
    "MCPs web-scraper's check launches Playwright's channel chrome, the system Google Chrome, "
    "in three real-browser cases"
)

#: The line a node is offered for Chrome: the .deb Google ships, the same
#: command runners.json provisions the asset with, because no apt repository
#: a node already has carries it.
CHROME_APT = (
    "curl -fsSL -o /tmp/google-chrome-stable_current_amd64.deb "
    "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && "
    "sudo apt-get install -y -qq /tmp/google-chrome-stable_current_amd64.deb && "
    "rm -f /tmp/google-chrome-stable_current_amd64.deb"
)

#: How the tagged gap reads Chrome's entry on a node whose managers cover
#: none of its installs.
CHROME_NO_INSTALL = (
    f"; chrome -- {CHROME_REASON}, so those jobs go to a node that has it -- no automatic "
    "install on this node"
)

#: Why a node is told about pdftotext, in the tagged gap's words (MCPs board
#: task 6dd73ada): wiki-check's audit rules read PDFs through it.
PDFTOTEXT_REASON = (
    "MCPs wiki-check's audit rules read PDF sources through poppler's pdftotext and depend on "
    "its exact page breaks"
)

#: The line a node is offered for pdftotext.
PDFTOTEXT_APT = "sudo apt-get install -y poppler-utils"

#: How the tagged gap reads pdftotext's entry on a node whose managers cover
#: none of its installs, the tail every such gap line now ends with.
PDFTOTEXT_NO_INSTALL = (
    f"; pdftotext -- {PDFTOTEXT_REASON}, so those jobs go to a node that has it -- no "
    "automatic install on this node"
)


def node(
    host: str = "lavender",
    *,
    rust: str | None = None,
    cxx: str | None = None,
    docker: str | None = None,
    stack: str | None = None,
) -> NodeConfig:
    """Build a node declaration.

    Args:
        host: SSH alias.
        rust: The cargo version it declares, or None.
        cxx: The C++ toolchain version it declares, or None.
        docker: The rootless execdocker daemon version it declares, or None.
        stack: The stack daemon version it declares, or None.

    Returns:
        The node.
    """
    return NodeConfig(
        host=host,
        stage_root="/home/corvis/fleet/stage",
        logical_cores=16,
        ram_gb=32.0,
        gpu=None,
        enabled=True,
        test_database=False,
        rust=rust,
        cxx=cxx,
        docker=docker,
        stack=stack,
        budget=NodeBudget(
            reserved_cores=2,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_disk_gb=20.0,
            checks_at_once=None,
            pinned_checks_only=False,
        ),
    )


__all__ = [
    "CHROME_APT",
    "CHROME_NO_INSTALL",
    "CHROME_REASON",
    "DIPHTHERIA",
    "DIPHTHERIA_2026_09_23",
    "DIPHTHERIA_2026_09_27",
    "DIPHTHERIA_2026_09_27_CXX",
    "GO_APT",
    "GO_NO_INSTALL",
    "GO_REASON",
    "HOOKS_INSTALL",
    "HOOKS_REASON",
    "NO_MAKE",
    "NO_PYTHON",
    "OLD_NODE",
    "PDFTOTEXT_APT",
    "PDFTOTEXT_NO_INSTALL",
    "PDFTOTEXT_REASON",
    "WRONG_PYTHON",
    "node",
]
