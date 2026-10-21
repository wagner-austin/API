"""The client runs on Linux only, and the source can prove it.

Until 2026-10-21 (board task 0eae9619) the Makefile ran its recipes through
``scripts/make/*.ps1``, a Pester suite tested them, and ``src/rw_bot`` chose
between a Windows and a Linux arm on the platform it found itself on. The
game and the bot were measured running on Linux only, the operator ruled
"well covnert any windows code to linux no back compat", and the port
deleted every Windows arm.

A deleted arm comes back the same way it arrived: one ``if sys.platform``
in a module, one ``.ps1`` beside a recipe, each individually correct and
passed by every other test. These read the tree for exactly those shapes,
and each detector is shown refusing a synthetic Windows arm first, so a
clean tree is a measurement and not a detector that cannot fire.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The tracked roots that hold code, recipes' helpers or their tests.
_SOURCE_ROOTS = ("src", "scripts", "tests", "agent", "models", "doctrines", "sweeps")

#: The roots whose Python is read for a platform branch.
_PYTHON_ROOTS = ("src", "scripts", "tests")

#: PowerShell's script, module and manifest suffixes.
_POWERSHELL_SUFFIXES = frozenset({".ps1", ".psm1", ".psd1"})

#: The attribute reads a platform branch is made of: ``sys.platform``,
#: ``os.name`` and ``platform.system()`` and its siblings.
_PLATFORM_READS = frozenset(
    {("sys", "platform"), ("os", "name"), ("platform", "system"), ("platform", "platform")}
)

#: The values a platform branch compares against.
_PLATFORM_VALUES = frozenset({"win32", "nt", "Windows", "Windows_NT", "cygwin"})

#: How the deleted arms spelled the platform in a name: ``is_windows``, an
#: upper-case ``WINDOWS`` word (``WINDOWS``, ``WINDOWS_JVM_DIR``) and the
#: ``WindowsPath`` flavours. The platform was a value passed down, so the
#: branch was ``if is_windows(...)`` and never a read of ``sys.platform`` at
#: the branch itself. Case matters: a lower-case ``windows`` here is a span
#: of time (``test_windows_are_reported_in_order``), not a platform.
_WIN_ONLY_SPELLING = re.compile(r"(?:\A|_)(?:is_windows\Z|WINDOWS(?:_|\Z))|WindowsPath")

#: The names only Windows code binds or reads: the creation flags the old
#: spawns passed, and the standard library's Windows-only modules.
_WIN_ONLY_NAMES = frozenset(
    {
        "creationflags",
        "startupinfo",
        "STARTUPINFO",
        "CREATE_NEW_PROCESS_GROUP",
        "CREATE_NO_WINDOW",
        "DETACHED_PROCESS",
        "ntpath",
        "winreg",
        "msvcrt",
        "_winapi",
    }
)

#: Windows programs, as the first word of an argv names them, and the
#: Windows depot's runtime directory as the first part of a path.
_WIN_ONLY_PROGRAMS = frozenset(
    {
        "taskkill",
        "taskkill.exe",
        "tasklist",
        "tasklist.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "cmd.exe",
        "jvm64",
    }
)

#: A Makefile's Windows tells, in the lines make runs or assigns, compared
#: case-blind.
_MAKEFILE_TELLS = ("$(os)", "windows_nt", "powershell", "pwsh", ".ps1")


def powershell_files(root: Path) -> list[str]:
    """Return every PowerShell file under the source roots, relative to root."""
    return sorted(
        str(path.relative_to(root))
        for name in _SOURCE_ROOTS
        for path in (root / name).rglob("*")
        if path.suffix.lower() in _POWERSHELL_SUFFIXES
    )


def _bound_or_read_names(node: ast.AST) -> tuple[tuple[int, str], ...]:
    """Return the identifier one syntax node binds or reads, with its line."""
    if isinstance(node, ast.Name):
        return ((node.lineno, node.id),)
    if isinstance(node, ast.Attribute):
        return ((node.lineno, node.attr),)
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.alias)):
        return ((node.lineno, node.name),)
    if isinstance(node, ast.keyword) and node.arg is not None:
        return ((node.lineno, node.arg),)
    if isinstance(node, ast.arg):
        return ((node.lineno, node.arg),)
    return ()


def platform_branches(source: str) -> list[str]:
    """Return each platform read or platform literal in one module's source.

    Args:
        source: The module's text.

    Returns:
        ``<line>: <what>`` for every ``sys.platform``-style read, every
        comparison against a string that is exactly a platform's name, every
        Windows name bound or read, and every tuple or list whose first item
        is exactly a Windows program. Prose that merely mentions the old
        scripts is not a branch, and neither is a platform's or a program's
        name held without being compared or run, so neither is returned.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        found.extend(
            f"{line}: {name}"
            for line, name in _bound_or_read_names(node)
            if _WIN_ONLY_SPELLING.search(name) or name in _WIN_ONLY_NAMES
        )
        if (
            isinstance(node, (ast.Tuple, ast.List))
            and node.elts
            and isinstance(first := node.elts[0], ast.Constant)
            and first.value in _WIN_ONLY_PROGRAMS
        ):
            found.append(f"{node.lineno}: {first.value!r}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id, node.attr) in _PLATFORM_READS
        ):
            found.append(f"{node.lineno}: {node.value.id}.{node.attr}")
        if isinstance(node, ast.Compare):
            found.extend(
                f"{node.lineno}: {operand.value!r}"
                for operand in (node.left, *node.comparators)
                if isinstance(operand, ast.Constant) and operand.value in _PLATFORM_VALUES
            )
    return sorted(found)


def makefile_tells(text: str) -> list[str]:
    """Return each live Makefile line carrying a Windows tell.

    Args:
        text: The Makefile's text.

    Returns:
        Every line, comments excluded, holding one of the tells.
    """
    return [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
        and any(tell in line.lower() for tell in _MAKEFILE_TELLS)
    ]


def _python_modules(root: Path) -> list[Path]:
    return sorted(
        path
        for name in _PYTHON_ROOTS
        for path in (root / name).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_the_powershell_detector_finds_a_script(tmp_path: Path) -> None:
    (tmp_path / "scripts" / "make").mkdir(parents=True)
    (tmp_path / "scripts" / "make" / "agent.ps1").write_text("javac\n", encoding="utf-8")
    (tmp_path / "tests" / "pester").mkdir(parents=True)
    (tmp_path / "tests" / "pester" / "Helpers.PSM1").write_text("\n", encoding="utf-8")
    assert powershell_files(tmp_path) == [
        "scripts/make/agent.ps1",
        "tests/pester/Helpers.PSM1",
    ]


def test_the_package_carries_no_powershell() -> None:
    assert powershell_files(_PROJECT_ROOT) == []
    assert not (_PROJECT_ROOT / "tests" / "pester").exists()


def test_the_branch_detector_finds_each_shape() -> None:
    source = (
        "import os, platform, sys\n"
        "if sys.platform == 'win32':\n"
        "    pass\n"
        "WINDOWS = os.name == 'nt'\n"
        "KIND = platform.system()\n"
        '"""Replaces scripts/make/agent.ps1, which ran under PowerShell."""\n'
        "NAMES = frozenset({'win32', 'nt', 'taskkill'})\n"
        "FOUND = 'Windows_NT' in ENVIRONMENT.values()\n"
        "from rw_bot.platform_id import is_windows\n"
        "from pathlib import PureWindowsPath\n"
        "def fell(pid, platform):\n"
        "    if is_windows(platform):\n"
        "        return ('taskkill', '/PID', str(pid), '/T', '/F')\n"
        "    return ['kill', '-9', str(pid)]\n"
        "WINDOWS_JVM_DIR = 'jvm64'\n"
        "spawn(argv, creationflags=0)\n"
    )
    assert platform_branches(source) == [
        "10: PureWindowsPath",
        "12: is_windows",
        "13: 'taskkill'",
        "15: WINDOWS_JVM_DIR",
        "16: creationflags",
        "2: 'win32'",
        "2: sys.platform",
        "4: 'nt'",
        "4: WINDOWS",
        "4: os.name",
        "5: platform.system",
        "8: 'Windows_NT'",
        "9: is_windows",
    ]


def test_the_branch_detector_refuses_the_deleted_arms() -> None:
    """The shapes the port deleted, as they were written, each refused."""
    deleted = (
        "WINDOWS = 'win32'\n"
        "def is_windows(platform: str) -> bool:\n"
        "    return platform == WINDOWS\n"
        "def isolation(platform: str) -> SpawnIsolation:\n"
        "    if is_windows(platform):\n"
        "        return SpawnIsolation(creationflags=ABOVE_NORMAL, start_new_session=False)\n"
        "    return SpawnIsolation(creationflags=0, start_new_session=True)\n"
    )
    assert platform_branches(deleted) == [
        "1: WINDOWS",
        "2: is_windows",
        "3: WINDOWS",
        "5: is_windows",
        "6: creationflags",
        "7: creationflags",
    ]


def test_no_module_branches_on_the_platform() -> None:
    modules = _python_modules(_PROJECT_ROOT)
    assert len(modules) > 200
    found = {
        str(path.relative_to(_PROJECT_ROOT)): branches
        for path in modules
        if (branches := platform_branches(path.read_text(encoding="utf-8")))
    }
    assert found == {}


def test_the_makefile_detector_finds_each_tell() -> None:
    text = (
        "# Windows_NT was the old fence\n"
        "ifeq ($(OS),Windows_NT)\n"
        "SHELL := PowerShell.exe\n"
        "\tpwsh -File build.sh\n"
        "\t& scripts/make/agent.ps1\n"
        "agent:\n"
    )
    assert makefile_tells(text) == [
        "ifeq ($(OS),Windows_NT)",
        "SHELL := PowerShell.exe",
        "\tpwsh -File build.sh",
        "\t& scripts/make/agent.ps1",
    ]


def test_the_makefile_has_no_platform_fence() -> None:
    text = (_PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert len(text.splitlines()) > 100
    assert makefile_tells(text) == []
