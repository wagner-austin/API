"""The dialect protocol: one set of acts, two renderings, chosen by platform."""

from __future__ import annotations

from fleet.contracts.node import NODE_PLATFORMS
from fleet.core import dialect, names
from fleet.core.dialect_linux import LinuxDialect
from fleet.core.dialect_windows import WindowsDialect


def test_for_platform_chooses_by_the_declared_platform() -> None:
    assert type(dialect.for_platform("windows")) is WindowsDialect
    assert type(dialect.for_platform("linux")) is LinuxDialect


def test_every_declared_platform_has_a_dialect_that_renders_every_act() -> None:
    """Both classes satisfy the protocol by construction (mypy), and this
    pins that each act produces text for each platform, with the platform's
    own extension, so a new act added to one dialect and not the other is
    caught here as well as by the type checker."""
    for platform in NODE_PLATFORMS:
        spoken = dialect.for_platform(platform)
        extension = ".ps1" if platform == "windows" else ".sh"
        assert spoken.script_path("/s", "x") == f"/s/x{extension}"
        assert "/s/x" in spoken.write_command("/s/x")
        assert spoken.invocation()[0] in {"powershell", "/bin/sh"}
        assert "hello" in spoken.echo_command("hello")
        assert "/s/run" in spoken.make_directory_script("/s/run")
        assert names.ENCODED_NAME in spoken.reassemble_script("/s/run")
        assert "make check" in spoken.build_script(
            target="/s/run", path="libs/x", workers=2, install=(), cache_root="/s/cache"
        )
        assert names.RESULT_NAME in spoken.log_tail_script("/s/run", 5)
        assert names.task_name("r") in spoken.launch_script(target="/s/run", run_id="r")
        assert names.RESULT_NAME in spoken.result_script("/s/run")
        assert names.task_name("r") in spoken.stop_script("r")
        assert "free_ram_gb" in spoken.capacity_probe_script()
        assert "poetry" in spoken.toolchain_probe_script()
        assert spoken.fleet_directory("austi") in {"C:/Users/austi/.fleet", "/home/austi/.fleet"}
        assert ".claude" in spoken.observe_sessions_script()
        assert "sessions" in spoken.observe_sessions_script()


def test_the_shared_scripts_are_the_same_command_on_both_platforms() -> None:
    """tar and git are spelled once because Windows ships bsdtar and every
    node has git; -m keeps a fast sender's clock from making targets look
    newer than their sources, git init is what makes ruff honour .gitignore
    on the node, and git add is what makes ``git ls-files`` answer as it
    does in a checkout (the first fleet verdict, fd5cabfa)."""
    archive = f"/s/run/{names.ARCHIVE_NAME}"
    assert dialect.extract_script("/s/run") == f"tar -xzmf '{archive}' -C '/s/run'\n"
    assert dialect.init_repository_script("/s/run") == (
        "git -C '/s/run' init --quiet\ngit -C '/s/run' add --all\n"
    )


def test_the_names_are_one_spelling() -> None:
    assert names.task_name("libs-demo-1") == "fleet-libs-demo-1"
    assert names.make_directory_stem("libs-demo-1") == "mkdir-libs-demo-1"
    assert names.stop_stem("libs-demo-1") == "stop-libs-demo-1"
