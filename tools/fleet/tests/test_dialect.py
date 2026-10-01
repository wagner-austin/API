"""The dialect protocol: one set of acts, two renderings, chosen by platform."""

from __future__ import annotations

import io
import pathlib
import subprocess
import zipfile

import pytest

from fleet.contracts.node import NodePlatform
from fleet.contracts.project import MAKE_TARGET
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
        assert MAKE_TARGET in spoken.build_script(
            target="/s/run",
            path="libs/x",
            workers=2,
            install=(),
            cache_root="/s/cache",
            isolated_docker=False,
            elevated=False,
            agent="opus-demo-0929",
        )
        assert names.RESULT_NAME in spoken.log_tail_script("/s/run", 5)
        launched = spoken.launch_script(target="/s/run", run_id="r", elevated=False)
        assert names.task_name("r") in launched
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


def test_a_companion_is_cloned_from_its_bundle_onto_its_branch() -> None:
    """MCPs board task 2026dfbc: the bundle's ref is fetched into
    ``origin/<branch>``, the branch is checked out there, and HEAD is proved
    to be the commit the hub bundled."""
    commands = dialect.companion_repository_commands(
        "/s/MCPs", "/s/MCPs.stage/tree.tgz", "a" * 40, "refs/heads/main"
    )

    assert commands == (
        ("git", "-C", "/s/MCPs", "init", "--quiet"),
        (
            "git",
            "-C",
            "/s/MCPs",
            "fetch",
            "--quiet",
            "--no-tags",
            "/s/MCPs.stage/tree.tgz",
            "+refs/fleet/companion:refs/remotes/origin/main",
        ),
        ("git", "-C", "/s/MCPs", "checkout", "--quiet", "-B", "main", "refs/remotes/origin/main"),
        ("git", "-C", "/s/MCPs", "merge-base", "--is-ancestor", "HEAD", "a" * 40),
        ("git", "-C", "/s/MCPs", "merge-base", "--is-ancestor", "a" * 40, "HEAD"),
    )


def _git(*args: str) -> str:
    """Run git under a throwaway identity and answer its trimmed output.

    Args:
        *args: The arguments after ``git``.

    Returns:
        Its standard output, stripped.
    """
    ran = subprocess.run(
        ("git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return ran.stdout.strip()


def _bundled_companion(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, str]:
    """A two-commit repository bundled at its companion ref, as the hub does.

    Args:
        tmp_path: Where the repository and its bundle are made.

    Returns:
        The bundle, the first commit and the tip.
    """
    source = tmp_path / "source"
    (source / "packages" / "maketools").mkdir(parents=True)
    _git("init", "--quiet", str(source))
    (source / "packages" / "maketools" / "run.py").write_bytes(b"print(1)\n")
    _git("-C", str(source), "add", "--all")
    _git("-C", str(source), "commit", "--quiet", "--message", "first")
    first = _git("-C", str(source), "rev-parse", "HEAD")
    (source / "packages" / "maketools" / "run.py").write_bytes(b"print(2)\n")
    _git("-C", str(source), "commit", "--quiet", "--all", "--message", "tip")
    tip = _git("-C", str(source), "rev-parse", "HEAD")
    _git("-C", str(source), "update-ref", "refs/fleet/companion", tip)
    bundle = tmp_path / "MCPs.stage" / names.ARCHIVE_NAME
    bundle.parent.mkdir()
    _git("-C", str(source), "bundle", "create", "--quiet", str(bundle), "refs/fleet/companion")
    return bundle, first, tip


@pytest.mark.parametrize("ref", ["main", "refs/heads/main"])
def test_a_staged_companion_is_its_ref_with_history_and_serves_origin_main(
    tmp_path: pathlib.Path, ref: str
) -> None:
    """The commands run here for real against a bundle shaped like the hub's.
    HEAD and ``origin/main`` are the bundled tip and its ancestor is
    readable, which is what corvis-stick's HookCommands suite needed of MCPs
    on serendipity (MCPs board task 2026dfbc). And MCPs board task a8ee9b21
    still holds: published maketools reads ``git --git-dir=../MCPs/.git
    archive origin/main``, and that call reads the staged file back."""
    bundle, first, tip = _bundled_companion(tmp_path)
    target = tmp_path / "MCPs"
    target.mkdir()
    for command in dialect.companion_repository_commands(
        target.as_posix(), bundle.as_posix(), tip, ref
    ):
        subprocess.run(command, check=True, capture_output=True, timeout=60)

    assert _git("-C", str(target), "rev-parse", "HEAD") == tip
    assert _git("-C", str(target), "rev-parse", "refs/remotes/origin/main") == tip
    assert _git("-C", str(target), "show", f"{first}:packages/maketools/run.py") == "print(1)"
    archived = subprocess.run(
        [
            "git",
            f"--git-dir={target / '.git'}",
            "archive",
            "--format=zip",
            "origin/main",
            "packages/maketools",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )

    # The node's own core.autocrlf decides the archive's line endings (CRLF
    # on a Windows node), which is the checkout's business, not the ref's.
    with zipfile.ZipFile(io.BytesIO(archived.stdout)) as archive:
        content = archive.read("packages/maketools/run.py")
    assert content.replace(b"\r\n", b"\n") == b"print(2)\n"


def test_a_bundle_whose_ref_is_not_the_bundled_commit_stops_the_stage(
    tmp_path: pathlib.Path,
) -> None:
    """The last two commands prove HEAD is the commit the hub resolved; asked
    for the bundle's ancestor instead, the first of them exits non-zero, so
    the checked script ends there and the run is not staged on a tree the
    feed would misname."""
    bundle, first, _tip = _bundled_companion(tmp_path)
    target = tmp_path / "MCPs"
    target.mkdir()
    *clone, head_in_sha, _sha_in_head = dialect.companion_repository_commands(
        target.as_posix(), bundle.as_posix(), first, "main"
    )
    for command in clone:
        subprocess.run(command, check=True, capture_output=True, timeout=60)

    assert subprocess.run(head_in_sha, capture_output=True, timeout=60).returncode == 1


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
    assert names.stage_name("MCPs") == f"MCPs{names.STAGE_SUFFIX}"


def test_a_companion_sits_beside_an_export_and_each_one_s_staging_sits_beside_it() -> None:
    """The recipe runs at ``<stage_root>/<run id>`` and reaches its companion
    as ``../MCPs``, which is the same spelling a workstation answers; each
    staging directory is beside the tree it stages so nothing of the
    transport is inside the tree that gets committed (MCPs board task
    a8ee9b21, where an export's own digest, extract and init-repo scripts
    were committed into it)."""
    assert names.companion_directory("C:/fleet/stage", "MCPs") == "C:/fleet/stage/MCPs"
    assert names.staging_directory("C:/fleet/stage/MCPs") == "C:/fleet/stage/MCPs.stage"
    assert names.staging_directory(names.dispatch_directory("/s", "r-1")) == "/s/r-1.stage"
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
