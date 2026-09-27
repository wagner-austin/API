"""The dialect protocol: one set of acts, two renderings, chosen by platform."""

from __future__ import annotations

import pytest

from fleet.contracts.node import NodePlatform
from fleet.core import dialect, dialect_linux, names
from fleet.core.dialect_linux import LinuxDialect
from fleet.core.dialect_windows import WindowsDialect


def test_for_platform_chooses_by_the_declared_platform() -> None:
    assert type(dialect.for_platform(NodePlatform.WINDOWS)) is WindowsDialect
    assert type(dialect.for_platform(NodePlatform.LINUX)) is LinuxDialect


def test_every_declared_platform_has_a_dialect_that_renders_every_act() -> None:
    """Both classes satisfy the protocol by construction (mypy), and this
    pins that each act produces text for each platform, with the platform's
    own extension, so a new act added to one dialect and not the other is
    caught here as well as by the type checker."""
    for platform in NodePlatform:
        spoken = dialect.for_platform(platform)
        extension = ".ps1" if platform is NodePlatform.WINDOWS else ".sh"
        assert spoken.script_path("/s", "x") == f"/s/x{extension}"
        assert "/s/x" in spoken.write_command("/s/x")
        assert spoken.invocation()[0] in {"powershell", "/bin/sh"}
        assert "hello" in spoken.echo_command("hello")
        assert "/s/run" in spoken.make_directory_script("/s/run")
        assert "/s/run" in spoken.digest_script("/s/run")
        assert names.ARCHIVE_NAME in spoken.digest_script("/s/run")
        assert "make check" in spoken.build_script(
            target="/s/run", path="libs/x", workers=2, install=(), cache_root="/s/cache"
        )
        assert names.RESULT_NAME in spoken.log_tail_script("/s/run", 5)
        assert names.task_name("r") in spoken.launch_script(target="/s/run", run_id="r")
        assert names.RESULT_NAME in spoken.result_script("/s/run")
        assert names.task_name("r") in spoken.stop_script(target="/s/run", run_id="r")
        assert "free_ram_gb" in spoken.capacity_probe_script()
        assert "poetry" in spoken.toolchain_probe_script()
        assert spoken.fleet_directory("austi") in {"C:/Users/austi/.fleet", "/home/austi/.fleet"}
        assert ".claude" in spoken.observe_sessions_script()
        assert "sessions" in spoken.observe_sessions_script()


def test_the_shared_commands_are_the_same_on_both_platforms() -> None:
    """tar and git are spelled once because Windows ships bsdtar and every
    node has git; -m keeps a fast sender's clock from making targets look
    newer than their sources, git init is what makes ruff honour .gitignore
    on the node, and git add is what makes ``git ls-files`` answer as it
    does in a checkout (the first fleet verdict, fd5cabfa), and the commit is
    what gives packages/db's migrator the HEAD its test admission reads
    (MCPs board task 6bbfd171)."""
    archive = f"/s/run/{names.ARCHIVE_NAME}"
    assert dialect.extract_commands(archive, "/s/run") == (
        ("tar", "-xzmf", archive, "-C", "/s/run"),
    )
    assert dialect.init_repository_commands("/s/run", "MCPs-packages-db-1790400000") == (
        ("git", "-C", "/s/run", "init", "--quiet"),
        ("git", "-C", "/s/run", "add", "--all"),
        (
            "git",
            "-C",
            "/s/run",
            "-c",
            f"user.name={dialect.EXPORT_AUTHOR_NAME}",
            "-c",
            f"user.email={dialect.EXPORT_AUTHOR_EMAIL}",
            "commit",
            "--quiet",
            "--message",
            "fleet export MCPs-packages-db-1790400000",
        ),
    )


def test_the_extract_takes_the_archive_and_the_destination_separately() -> None:
    """A companion's archive stays in its staging directory and only the
    repository lands in the directory the recipe reads, so the tree committed
    there is the export and nothing else."""
    assert dialect.extract_commands("/s/MCPs.stage/tree.tgz", "/s/MCPs") == (
        ("tar", "-xzmf", "/s/MCPs.stage/tree.tgz", "-C", "/s/MCPs"),
    )


def test_a_companion_is_committed_so_the_check_reading_it_has_a_head() -> None:
    """slime's lift check reads ``git show HEAD:<path>`` deliberately, so that
    an uncommitted edit in the workspace is not mistaken for the code. On a
    node there is no HEAD until one is made, and the identity is passed rather
    than configured so nothing is left behind on the machine."""
    commands = dialect.companion_repository_commands("/s/MCPs", "a" * 40)

    assert commands[:2] == (
        ("git", "-C", "/s/MCPs", "init", "--quiet"),
        ("git", "-C", "/s/MCPs", "add", "--all", "--force"),
    )
    assert commands[2][-1] == f"fleet companion export {'a' * 40}"


class TestCheckedScript:
    """A command that failed ends the script with its status, which is the
    default in one shell and has to be asked for in the other. Measured on
    sedona 2026-09-22: `New-Item -LiteralPath` (a parameter that cmdlet does
    not have) left a directory uncreated, tar then wrote "could not chdir to
    'C:/fleet/stage/MCPs'" and exited non-zero, and `powershell -File` exited
    0 -- the transport saw success and the node held nothing. The PowerShell
    rendering is run by the Pester suite over its committed copies."""

    def test_powershell_names_each_tool_by_parameter_and_checks_once_per_step(self) -> None:
        rendered = dialect.for_platform(NodePlatform.WINDOWS).checked_script(
            (("tar", "-xzmf", "C:/s/t.tgz", "-C", "C:/s"), ("git", "-C", "C:/s", "init"))
        )

        assert rendered == (
            "param(\n"
            '    [string]$Tar = "$env:SystemRoot\\System32\\tar.exe",\n'
            "    [string]$Git = 'git'\n"
            ")\n"
            "Set-StrictMode -Version Latest\n"
            "$ErrorActionPreference = 'Stop'\n"
            "function Invoke-Step {\n"
            "    param([string]$Tool, [string[]]$Arguments)\n"
            "    & $Tool @Arguments\n"
            "    if ($LASTEXITCODE -ne 0) {\n"
            "        exit $LASTEXITCODE\n"
            "    }\n"
            "}\n"
            "Invoke-Step $Tar @('-xzmf', 'C:/s/t.tgz', '-C', 'C:/s')\n"
            "Invoke-Step $Git @('-C', 'C:/s', 'init')\n"
        )

    def test_powershell_declares_only_the_tools_its_commands_run(self) -> None:
        rendered = dialect.for_platform(NodePlatform.WINDOWS).checked_script(
            (("git", "init"), ("git", "add"))
        )

        assert "$Tar" not in rendered
        assert rendered.count("[string]$Git = 'git'") == 1

    def test_powershell_refuses_a_tool_it_has_no_parameter_for(self) -> None:
        with pytest.raises(ValueError, match="runs only git, tar, not curl"):
            dialect.for_platform(NodePlatform.WINDOWS).checked_script((("curl", "-O"),))

    def test_powershell_refuses_a_word_that_cannot_be_embedded(self) -> None:
        with pytest.raises(ValueError, match="argument"):
            dialect.for_platform(NodePlatform.WINDOWS).checked_script((("git", "it's"),))

    def test_sh_needs_nothing_per_command_because_set_e_is_the_default_here(self) -> None:
        assert dialect.for_platform(NodePlatform.LINUX).checked_script(
            (("git", "-C", "/s/run", "init"), ("git", "commit", "--message", "fleet export r"))
        ) == (
            f"{dialect_linux.PROLOGUE}git -C /s/run init\ngit commit --message 'fleet export r'\n"
        )

    @pytest.mark.parametrize("platform", list(NodePlatform))
    def test_every_command_reaches_the_script_in_order(self, platform: NodePlatform) -> None:
        rendered = dialect.for_platform(platform).checked_script(
            (("git", "alpha"), ("git", "beta"), ("git", "gamma"))
        )

        assert rendered.index("alpha") < rendered.index("beta") < rendered.index("gamma")
        assert rendered.endswith("\n")


def test_the_names_are_one_spelling() -> None:
    assert names.task_name("libs-demo-1") == "fleet-libs-demo-1"
    assert names.make_directory_stem("libs-demo-1") == "mkdir-libs-demo-1"
    assert names.stop_stem("libs-demo-1") == "stop-libs-demo-1"
    assert names.reset_directory_stem("MCPs") == "reset-MCPs"
    assert names.companion_stage_name("MCPs") == f"MCPs{names.COMPANION_STAGE_SUFFIX}"


def test_a_companion_sits_beside_an_export_and_its_staging_sits_beside_it() -> None:
    """The recipe runs at ``<stage_root>/<run id>`` and reaches its companion
    as ``../MCPs``, which is the same spelling a workstation answers; the
    staging directory is beside the companion so nothing of the transport is
    inside the tree that gets committed."""
    assert names.companion_directory("C:/fleet/stage", "MCPs") == "C:/fleet/stage/MCPs"
    assert names.companion_stage_directory("C:/fleet/stage", "MCPs") == "C:/fleet/stage/MCPs.stage"
    assert names.recipe_directory("C:/fleet/stage/slime-17", "") == "C:/fleet/stage/slime-17"


@pytest.mark.parametrize("platform", list(NodePlatform))
def test_a_companions_directory_is_emptied_before_it_is_written(platform: NodePlatform) -> None:
    """Every run carrying one writes the same directory, and each dialect
    removes it read-only-objects and all: a git repository this package made
    there has read-only loose objects, which is what ``-Force`` and ``-f``
    are for rather than tidiness."""
    spoken = dialect.for_platform(platform)
    script = spoken.reset_directory_script("/s/MCPs")

    assert "/s/MCPs" in script
    assert ("Remove-Item" in script) is (platform is NodePlatform.WINDOWS)
    assert ("rm -rf" in script) is (platform is NodePlatform.LINUX)
