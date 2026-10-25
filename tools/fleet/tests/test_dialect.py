"""The dialect: every act a node is sent, rendered in ``sh``."""

from __future__ import annotations

from fleet.contracts.project import MAKE_TARGET
from fleet.core import dialect, names


def test_every_act_renders_its_script() -> None:
    """Each act produces text with its target in it and the ``.sh`` stem, so
    an act whose rendering lost its argument is caught here as well as in
    its own suite."""
    assert dialect.script_path("/s", "x") == "/s/x.sh"
    assert "/s/x" in dialect.write_command("/s/x")
    assert dialect.SH_INVOCATION == ("/bin/sh",)
    assert "hello" in dialect.echo_command("hello")
    assert "/s/run" in dialect.make_directory_script("/s/run")
    assert "/s/run" in dialect.digest_script("/s/run")
    assert names.ARCHIVE_NAME in dialect.digest_script("/s/run")
    assert MAKE_TARGET in dialect.build_script(
        target="/s/run",
        path="libs/x",
        workers=2,
        install=(),
        cache_root="/s/cache",
        isolated_docker=False,
        agent="opus-demo-0929",
        loopback_unit=None,
    )
    assert names.RESULT_NAME in dialect.log_tail_script("/s/run", 5)
    assert names.task_name("r") in dialect.launch_script(target="/s/run", run_id="r")
    assert names.RESULT_NAME in dialect.result_script("/s/run")
    assert names.task_name("r") in dialect.stop_script("r")
    assert "free_ram_gb" in dialect.CAPACITY_PROBE_SCRIPT
    assert "poetry" in dialect.TOOLCHAIN_PROBE_SCRIPT
    assert dialect.fleet_directory("austi") == "/home/austi/.fleet"
    assert ".claude" in dialect.OBSERVE_SESSIONS_SCRIPT
    assert "sessions" in dialect.OBSERVE_SESSIONS_SCRIPT


def test_the_extract_keeps_the_node_s_clock() -> None:
    """-m keeps a fast sender's clock from making targets look newer than
    their sources. The git acts are pinned in test_stage_repository."""
    archive = f"/s/run/{names.ARCHIVE_NAME}"
    assert dialect.extract_commands(archive, "/s/run") == (
        ("tar", "-xzmf", archive, "-C", "/s/run"),
    )


def test_the_extract_takes_the_archive_and_the_destination_separately() -> None:
    """A companion's archive stays in its staging directory and only the
    repository lands in the directory the recipe reads, so the tree committed
    there is the export and nothing else."""
    assert dialect.extract_commands("/s/MCPs.stage/tree.tgz", "/s/MCPs") == (
        ("tar", "-xzmf", "/s/MCPs.stage/tree.tgz", "-C", "/s/MCPs"),
    )


class TestCheckedScript:
    """A command that failed ends the script with its status. Measured on
    sedona 2026-09-22, when it ran PowerShell: `New-Item -LiteralPath` (a
    parameter that cmdlet does not have) left a directory uncreated, tar then
    wrote "could not chdir to 'C:/fleet/stage/MCPs'" and exited non-zero, and
    `powershell -File` exited 0 -- the transport saw success and the node
    held nothing. ``sh`` under ``set -e`` stops at the first failure."""

    def test_sh_needs_nothing_per_command_because_set_e_is_the_default_here(self) -> None:
        assert dialect.checked_script(
            (("git", "-C", "/s/run", "init"), ("git", "commit", "--message", "fleet export r"))
        ) == (f"{dialect.PROLOGUE}git -C /s/run init\ngit commit --message 'fleet export r'\n")

    def test_every_command_reaches_the_script_in_order(self) -> None:
        rendered = dialect.checked_script((("git", "alpha"), ("git", "beta"), ("git", "gamma")))

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
    assert names.companion_directory("/fleet/stage", "MCPs") == "/fleet/stage/MCPs"
    assert names.staging_directory("/fleet/stage/MCPs") == "/fleet/stage/MCPs.stage"
    assert names.staging_directory(names.dispatch_directory("/s", "r-1")) == "/s/r-1.stage"
    assert names.recipe_directory("/fleet/stage/slime-17", "") == "/fleet/stage/slime-17"


def test_a_companions_directory_is_emptied_before_it_is_written() -> None:
    """Every run carrying one writes the same directory, and the script
    removes it read-only-objects and all: a git repository this package made
    there has read-only loose objects, which is what ``-f`` is for rather
    than tidiness."""
    script = dialect.reset_directory_script("/s/MCPs")

    assert "/s/MCPs" in script
    assert "rm -rf" in script
