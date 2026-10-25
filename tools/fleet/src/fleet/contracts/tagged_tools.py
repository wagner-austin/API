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
            inferred from one Windows node's ``make``, one node generalised
            to three. Measured 2026-09-04, the three nodes had three
            different sets of managers, so the manager is chosen per node
            from what that node reported.
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

#: The distributions MCPs ``packages/claude-hooks``'s make check runs on the
#: system interpreter: ruff and mypy for its lint, pytest with pytest-xdist,
#: pytest-cov and coverage for its suites, and pytest-timeout, which its
#: pyproject's ``timeout = 300`` needs. Each is the latest release on PyPI
#: when measured on 2026-10-16 (MCPs board task 09a34e03), when the package's
#: lint was clean and its 104 suites passed under them; until then these were
#: the hub's versions of 2026-10-02 (ruff 0.15.1, mypy 1.19.1, pytest 9.0.2),
#: and pytest-timeout was missing, so fleet job 4ef9f20f on sedona logged
#: ``Unknown config option: timeout`` and ran with no per-test timeout. A node
#: is ready only at these versions (:data:`HOOKS_CHECK_PROGRAM`), so every
#: machine lints and tests a commit by the same rules. Readers outside this
#: package take them from ``rendered/hooks-check-packages.json``
#: (:mod:`fleet.core.hooks_check_document`): run ``make render-documents``
#: after moving one.
HOOKS_CHECK_PACKAGES: Final = (
    "ruff==0.17.1",
    "mypy==2.4.0",
    "pytest==9.1.1",
    "pytest-xdist==3.8.0",
    "pytest-cov==7.1.0",
    "coverage==7.16.2",
    "pytest-timeout==2.4.0",
)

#: The probe's ``hooks`` question, a program for ``python -c``: it exits 0
#: only when every distribution its arguments name is installed at the
#: version beside it, and 1 otherwise, a missing one included (its
#: ``PackageNotFoundError`` ends it). Until 2026-10-16 the probe asked only
#: that the modules import, so a node kept whatever versions it was first
#: given. It holds no quote character, so a shell passes it unchanged.
HOOKS_CHECK_PROGRAM: Final = (
    "import sys, importlib.metadata as m; a = sys.argv[1:]; "
    "sys.exit(any(m.version(a[i]) != a[i + 1] for i in range(0, len(a), 2)))"
)

#: :data:`HOOKS_CHECK_PROGRAM`'s arguments: each pin's name, then its version.
HOOKS_CHECK_ARGUMENTS: Final = " ".join(
    part for package in HOOKS_CHECK_PACKAGES for part in package.split("==")
)

#: What the probe asks ``go`` for its version with. Go has no ``--version``
#: flag: ``go --version`` prints ``flag provided but not defined: -version``
#: and its usage (measured on the hub, go1.27.1, 2026-10-05), so the probes
#: ask ``go version`` instead of every other tool's ``--version``.
GO_VERSION_ARGUMENT: Final = "version"

#: Where Playwright's channel ``chrome`` launches Google Chrome on Linux, the
#: one path MCPs ``web-scraper``'s real-browser cases open
#: (``src/hooks/browser.ts``), so it is the one path the probe asks for.
#: ``/usr/bin/google-chrome`` is a symlink the same package installs, but a
#: node that carries only the symlink would answer present and fail the
#: suite, so the probe asks for the binary itself. Measured 2026-10-09 (MCPs
#: board task 2f596185): diphtheria had Playwright's bundled chromium-1243
#: under ``~/.cache/ms-playwright`` and no such file, and MCPs/web-scraper
#: fleet jobs 56d7fbb5 and 3f9131af failed there with ``browserType.launch:
#: Chromium distribution 'chrome' is not found at /opt/google/chrome/chrome``.
CHROME_LINUX_PATH: Final = "/opt/google/chrome/chrome"

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
#:
#: ``chrome`` because MCPs ``web-scraper``'s check launches Playwright's
#: channel ``chrome``, the system Google Chrome and never the bundled
#: Chromium, in three real-browser cases, and a node without it fails those
#: three and passes the other 490 (MCPs board task 2f596185: fleet job
#: 874c160c on 2026-10-04 and 56d7fbb5 on 2026-10-09, both on diphtheria,
#: while lavender-wsl, whose runner provisioning installed Chrome as a
#: ``runners.json`` asset, passed the same suite). It is not an executable on
#: a PATH, so the probe asks for it at the path Playwright opens
#: (:data:`CHROME_LINUX_PATH`) rather than through ``report``'s
#: ``command -v``. The apt-get install is the asset's own
#: ``provision_command``: Google ships the .deb from its own host, not from
#: any apt repository the node already has.
#:
#: ``pdftotext`` because MCPs ``packages/wiki-check`` reads PDF sources
#: through poppler's pdftotext and its audit rules depend on that tool's
#: exact form-feed page breaks, so the suite cannot pass without it, and no
#: tag named it (MCPs board task 6dd73ada). On 2026-10-11 every live Linux
#: runner had it only by accident, in ``/usr/bin``, while the runner roster's
#: ``apt_packages`` did not carry poppler-utils, so a lavender-wsl rebuild
#: would have dropped it.
TAGGED_TOOLS: Final[tuple[RequiredTool, ...]] = (
    RequiredTool(
        name="ffmpeg",
        reason="grandma-api's check converts real audio files through ffmpeg",
        install={"apt-get": "sudo apt-get install -y ffmpeg"},
        uninstall={"apt-get": "sudo apt-get remove -y ffmpeg"},
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
        install={"apt-get": "sudo apt-get install -y golang-go"},
        uninstall={"apt-get": "sudo apt-get remove -y golang-go"},
    ),
    RequiredTool(
        name="chrome",
        reason=(
            "MCPs web-scraper's check launches Playwright's channel chrome, the system Google "
            "Chrome, in three real-browser cases"
        ),
        install={
            "apt-get": (
                "curl -fsSL -o /tmp/google-chrome-stable_current_amd64.deb "
                "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && "
                "sudo apt-get install -y -qq /tmp/google-chrome-stable_current_amd64.deb && "
                "rm -f /tmp/google-chrome-stable_current_amd64.deb"
            ),
        },
        uninstall={"apt-get": "sudo apt-get remove -y google-chrome-stable"},
    ),
    RequiredTool(
        name="pdftotext",
        reason=(
            "MCPs wiki-check's audit rules read PDF sources through poppler's pdftotext and "
            "depend on its exact page breaks"
        ),
        install={"apt-get": "sudo apt-get install -y poppler-utils"},
        uninstall={"apt-get": "sudo apt-get remove -y poppler-utils"},
    ),
)


__all__ = [
    "CHROME_LINUX_PATH",
    "GO_VERSION_ARGUMENT",
    "HOOKS_CHECK_ARGUMENTS",
    "HOOKS_CHECK_PACKAGES",
    "HOOKS_CHECK_PROGRAM",
    "HOOKS_ROUTE_FILE",
    "TAGGED_TOOLS",
    "RequiredTool",
]
