"""The shape of a tool a node may need, and the tools only some projects need.

A TAGGED TOOL is one a node is refused nothing for lacking: its toolchain
probe asks about it every tick, a runner adds its tag
(:data:`fleet.contracts.tags.TOOL_TAG`) when the probe found it, and the
queue hands the jobs that require the tag to a node that has it. The tools
every build needs are :data:`fleet.contracts.toolchain.REQUIRED_TOOLS`; both
lists are :class:`RequiredTool` rows, which is why the shape lives here,
below both, and :mod:`fleet.contracts.toolchain` imports it.

Split out of :mod:`fleet.contracts.toolchain` when ``go`` joined the list
(MCPs board task 1da15750), which would have taken that module past the
600-line ceiling.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict


class RequiredTool(TypedDict):
    """One thing a node must have, and how to get it on each package manager.

    Attributes:
        name: The executable, as it is spelled on a PATH, or for the one
            tagged entry that is not an executable, ``hooks``, the probe
            line that answers for the hooks check's environment.
        reason: Why a dispatch needs it. Carried so a refusal explains
            itself rather than naming a binary and leaving the reader to
            infer what it was for.
        install: The command per package manager, keyed by the manager's own
            executable name. An EMPTY MAPPING means this package does not
            install that tool at all, which is a real state rather than a
            gap: ``tar`` ships with the platform, and a manager that cannot
            supply the version the fleet runs (Ubuntu's Node 18.19.1) is
            left out rather than allowed to install the wrong one.

            A MAPPING RATHER THAN ONE COMMAND, and this was a defect before
            it was a design. The first version hardcoded ``choco install`` --
            inferred from loki's ``make`` living under
            ``C:\\ProgramData\\chocolatey``, one node generalised to three.
            Measured 2026-09-04: sedona has both managers, lavender has ONLY
            winget, loki has ONLY choco. No single command works fleet-wide,
            so the manager is chosen per node from what that node reported.
        uninstall: The command per package manager that removes what that
            manager's ``install`` command put there, keyed exactly as
            ``install`` is. It is how an install that did not finish is
            rolled back (:func:`fleet.core.toolchain_install.rollback_install`), so
            the node is left as it was found rather than holding part of a
            toolchain that looks ready.
    """

    name: str
    reason: str
    install: dict[str, str]
    uninstall: dict[str, str]


#: The route file a node's Claude Code hooks reach the board through, under
#: the build account's home: what MCPs ``packages/claude-hooks``
#: ``install-hooks-node.py`` writes, and what that package's live suites read.
HOOKS_ROUTE_FILE: Final = (".claude", "corvis-hooks.json")

#: The modules MCPs ``packages/claude-hooks``'s make check imports from the
#: system interpreter it runs on: ruff, mypy, pytest, pytest-xdist and
#: pytest-cov. The probe's ``hooks`` line answers present only when one
#: ``import`` of all of them succeeds.
HOOKS_CHECK_MODULES: Final = ("ruff", "mypy", "pytest", "xdist", "pytest_cov")

#: The distributions that provide :data:`HOOKS_CHECK_MODULES`, at the versions
#: the hub's interpreter ran the package's check with on 2026-10-02 (MCPs board
#: task ec895824), so a node's check reads the same lint and type rules the
#: hub's does.
HOOKS_CHECK_PACKAGES: Final = (
    "ruff==0.15.1",
    "mypy==1.19.1",
    "pytest==9.0.2",
    "pytest-xdist==3.8.0",
    "pytest-cov==7.1.0",
)

#: What the probe asks ``go`` for its version with. Go has no ``--version``
#: flag: ``go --version`` prints ``flag provided but not defined: -version``
#: and its usage (measured on the hub, go1.27.1, 2026-10-05), so the probes
#: ask ``go version`` instead of every other tool's ``--version``.
GO_VERSION_ARGUMENT: Final = "version"

#: Tools only some projects need, each the source of a capability tag
#: (:data:`fleet.contracts.tags.TOOL_TAG`). The toolchain probe asks about
#: them every tick like the required tools, but a node without one is refused
#: nothing: its runners claim without the tag, so the queue hands the jobs
#: that require it to another node, and the tick logs what is missing, which
#: projects wait for it and the command that would install it here.
#:
#: ``go`` because MCPs ``rcs-bridge`` is a Go module whose check runs go vet,
#: staticcheck, errcheck and go test, and on 2026-10-05 no fleet node had go
#: on its PATH (MCPs board task 1da15750). Its ``go.mod`` names go 1.27.1;
#: an older go from any of these managers fetches that toolchain itself
#: (``GOTOOLCHAIN=auto``, Go 1.21 onward), so the distribution's package is
#: enough.
TAGGED_TOOLS: Final[tuple[RequiredTool, ...]] = (
    RequiredTool(
        name="ffmpeg",
        reason="grandma-api's check converts real audio files through ffmpeg",
        install={
            "winget": (
                "winget install --id Gyan.FFmpeg.Essentials -e --source winget --silent "
                "--accept-package-agreements --accept-source-agreements --disable-interactivity"
            ),
            "choco": "choco install ffmpeg -y",
            "apt-get": "sudo apt-get install -y ffmpeg",
        },
        uninstall={
            "winget": (
                "winget uninstall --id Gyan.FFmpeg.Essentials -e --source winget --silent "
                "--accept-source-agreements --disable-interactivity"
            ),
            "choco": "choco uninstall ffmpeg -y",
            "apt-get": "sudo apt-get remove -y ffmpeg",
        },
    ),
    RequiredTool(
        name="hooks",
        reason=(
            "MCPs packages/claude-hooks's check runs on the system interpreter with its tools "
            "and reaches the board through ~/.claude/corvis-hooks.json, which only "
            "install-hooks-node.py writes, with the key the operator chose for the node"
        ),
        install={
            "pip": "python -m pip install --user " + " ".join(HOOKS_CHECK_PACKAGES),
        },
        uninstall={
            "pip": "python -m pip uninstall -y "
            + " ".join(package.split("==")[0] for package in HOOKS_CHECK_PACKAGES),
        },
    ),
    RequiredTool(
        name="go",
        reason="MCPs rcs-bridge is a Go module whose check runs go vet, staticcheck and go test",
        install={
            "winget": (
                "winget install --id GoLang.Go -e --source winget --silent "
                "--accept-package-agreements --accept-source-agreements --disable-interactivity"
            ),
            "choco": "choco install golang -y",
            "apt-get": "sudo apt-get install -y golang-go",
        },
        uninstall={
            "winget": (
                "winget uninstall --id GoLang.Go -e --source winget --silent "
                "--accept-source-agreements --disable-interactivity"
            ),
            "choco": "choco uninstall golang -y",
            "apt-get": "sudo apt-get remove -y golang-go",
        },
    ),
)


__all__ = [
    "GO_VERSION_ARGUMENT",
    "HOOKS_CHECK_MODULES",
    "HOOKS_CHECK_PACKAGES",
    "HOOKS_ROUTE_FILE",
    "TAGGED_TOOLS",
    "RequiredTool",
]
