"""The toolchain probe's ``cargo`` line (MCPs board task b1089284).

The ``rust`` tag is claimed from this line, and a node carrying it takes
libs/cleargbm_rs's check, whose gate runs ``cargo fmt``, ``cargo clippy`` and
``cargo llvm-cov`` and creates its venv with ``uv``. diphtheria and
serendipity carried the tag with cargo alone, and the check died at the first
of them. The line is RUN under ``sh`` with ``PATH`` holding only the case's
tools: a ``cargo`` that answers ``--version`` for the subcommands the case
installs and refuses the rest as cargo does, and a ``uv`` when the case has
one.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.capability import RUST_CHECK_SUBCOMMANDS, RUST_CHECK_TOOLS
from fleet.core.dialect import PROLOGUE, SH_INVOCATION, TOOLCHAIN_PROBE_SCRIPT
from tests._system_tools import link_system_tools
from tests.test_dialect_linux import fields_of

#: What the fake answers ``cargo --version`` with, as diphtheria's did.
CARGO_ANSWER = "cargo 1.98.1 (797e8a9bc 2026-08-05)"

#: The programs a case places itself, never linked from the node.
PLACED = frozenset({"cargo", *RUST_CHECK_TOOLS})


def _executable(path: pathlib.Path, text: str) -> None:
    """Write a script and mark it executable.

    Args:
        path: Where.
        text: Its body.
    """
    path.write_bytes(text.encode())
    path.chmod(0o755)


def _probe_fields(
    tmp_path: pathlib.Path, subcommands: tuple[str, ...] | None, tools_beside: tuple[str, ...]
) -> dict[str, str]:
    """Run the toolchain probe with ``PATH`` holding only the case's tools.

    Args:
        tmp_path: Where the fakes and the script go.
        subcommands: The cargo subcommands that answer, or None for no cargo.
        tools_beside: The executables of :data:`RUST_CHECK_TOOLS` present.

    Returns:
        The probe's fields.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    link_system_tools(tools, PLACED)
    if subcommands is not None:
        cases = "".join(f'  {name}) echo "{name} 1.0.0"; exit 0 ;;\n' for name in subcommands)
        _executable(
            tools / "cargo",
            "#!/bin/sh\n"
            f'if [ "$1" = "--version" ]; then echo "{CARGO_ANSWER}"; exit 0; fi\n'
            'case "$1" in\n'
            f"{cases}"
            "esac\n"
            'echo "error: no such command: \\`$1\\`" >&2\nexit 101\n',
        )
    for name in tools_beside:
        _executable(tools / name, f'#!/bin/sh\necho "{name} 0.13.0"\n')
    script = tmp_path / "probe.sh"
    script.write_text(
        TOOLCHAIN_PROBE_SCRIPT.replace(PROLOGUE, PROLOGUE + f"PATH='{tools.as_posix()}'\n", 1),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)


def test_the_line_asks_every_check_subcommand_and_tool() -> None:
    body = TOOLCHAIN_PROBE_SCRIPT

    assert RUST_CHECK_SUBCOMMANDS == ("fmt", "clippy", "llvm-cov")
    assert RUST_CHECK_TOOLS == ("uv",)
    assert (
        "if command -v cargo > /dev/null 2>&1"
        " && cargo fmt --version > /dev/null 2>&1"
        " && cargo clippy --version > /dev/null 2>&1"
        " && cargo llvm-cov --version > /dev/null 2>&1"
        " && command -v uv > /dev/null 2>&1; then\n"
    ) in body
    assert "report cargo cargo\n" not in body


@pytest.mark.host_linux
class TestTheCargoLineUnderSh:
    def test_the_whole_check_toolchain_answers_cargos_version(self, tmp_path: pathlib.Path) -> None:
        """diphtheria and serendipity once rustfmt, clippy, llvm-tools,
        cargo-llvm-cov and uv were installed on 2026-10-11."""
        fields = _probe_fields(tmp_path, RUST_CHECK_SUBCOMMANDS, RUST_CHECK_TOOLS)

        assert fields["cargo"] == f"yes={CARGO_ANSWER}"

    @pytest.mark.parametrize("missing", RUST_CHECK_SUBCOMMANDS)
    def test_a_missing_subcommand_answers_no(self, tmp_path: pathlib.Path, missing: str) -> None:
        """The minimal rustup profile both rust nodes carried: cargo answers,
        ``cargo fmt`` does not."""
        present = tuple(name for name in RUST_CHECK_SUBCOMMANDS if name != missing)

        assert _probe_fields(tmp_path, present, RUST_CHECK_TOOLS)["cargo"] == "no="

    def test_a_missing_uv_answers_no(self, tmp_path: pathlib.Path) -> None:
        assert _probe_fields(tmp_path, RUST_CHECK_SUBCOMMANDS, ())["cargo"] == "no="

    def test_no_cargo_answers_no(self, tmp_path: pathlib.Path) -> None:
        assert _probe_fields(tmp_path, None, RUST_CHECK_TOOLS)["cargo"] == "no="
