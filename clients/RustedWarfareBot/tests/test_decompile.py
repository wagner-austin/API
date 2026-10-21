"""Decompiling the pinned jar into a readable tree: ``make decompile``.

Driven against :class:`tests.harness_fakes.FakeHost`, which answers each tool
by its program name, so a fetch, a decompile and a count can each be made to
fail alone.
"""

from __future__ import annotations

import runpy

import pytest
from scripts.decompile import (
    CFR_OPTIONS,
    DECOMPILE_WALL_SECONDS,
    EXIT_DECOMPILE_FAILED,
    cfr_jar,
    cfr_url,
    decompile,
    main,
)

from rw_bot.harness.play_match import EXIT_OK
from tests.harness_fakes import FakeHost

JAR = ".tools/cfr-0.152.jar"
URL = "https://repo1.maven.org/maven2/org/benf/cfr/0.152/cfr-0.152.jar"
SOURCES = (".decompiled/a/A.java", ".decompiled/b/B.java")


def _host(*, fetched: bool) -> FakeHost:
    """Build a host whose count finds two sources.

    Args:
        fetched: Whether the pinned CFR jar is already on disk.

    Returns:
        The host.
    """
    host = FakeHost()
    host.command_results["find"] = (0, SOURCES)
    host.command_results["curl"] = (0, ())
    if fetched:
        host.files[JAR] = ()
    return host


class TestThePinnedDecompiler:
    def test_the_jar_and_its_url_name_the_version(self) -> None:
        assert cfr_jar("0.152") == JAR
        assert cfr_url("0.152") == URL


class TestDecompiling:
    def test_a_fetched_cfr_is_not_fetched_again(self) -> None:
        with _host(fetched=True) as host:
            assert decompile(".game", "0.152", ".decompiled") == EXIT_OK
            assert [command[0] for command in host.commands] == [
                ".game/jvm-linux/bin/java",
                "find",
            ]

    def test_it_runs_on_the_games_runtime_into_a_fresh_tree(self) -> None:
        with _host(fetched=True) as host:
            decompile(".game", "0.152", ".decompiled")
            assert host.commands[0] == (
                ".game/jvm-linux/bin/java",
                "-Xmx2g",
                "-jar",
                JAR,
                ".game/game-lib.jar",
                "--outputdir",
                ".decompiled",
                *CFR_OPTIONS,
            )
            assert host.walls[0] == DECOMPILE_WALL_SECONDS
            assert ".decompiled" in host.removed
            assert "[decompile] 2 source files in .decompiled" in host.printed

    def test_a_missing_cfr_is_fetched_first(self) -> None:
        with _host(fetched=False) as host:
            assert decompile(".game", "0.152", ".decompiled") == EXIT_OK
            assert host.commands[0] == ("curl", "-fsSL", "--create-dirs", "-o", JAR, URL)
            assert f"  fetching {URL}" in host.printed

    def test_a_failed_fetch_stops_before_anything_is_removed(self) -> None:
        with _host(fetched=False) as host:
            refusal = "curl: (22) The requested URL returned error: 404"
            host.command_results["curl"] = (22, (refusal,))
            assert decompile(".game", "0.152", ".decompiled") == EXIT_DECOMPILE_FAILED
            assert host.removed == []
            assert "[decompile] fetching CFR failed: curl exited 22" in host.printed

    def test_a_failed_decompile_is_named(self) -> None:
        with _host(fetched=True) as host:
            host.command_results[".game/jvm-linux/bin/java"] = (1, ("Exception in thread main",))
            assert decompile(".game", "0.152", ".decompiled") == EXIT_DECOMPILE_FAILED
            assert "Exception in thread main" in host.printed
            assert "[decompile] CFR failed: .game/jvm-linux/bin/java exited 1" in host.printed

    def test_a_failed_count_is_named(self) -> None:
        with _host(fetched=True) as host:
            host.command_results["find"] = (1, ())
            assert decompile(".game", "0.152", ".decompiled") == EXIT_DECOMPILE_FAILED
            assert "[decompile] counting failed: find exited 1" in host.printed


class TestTheCommandLine:
    def test_the_game_is_required(self) -> None:
        with _host(fetched=True), pytest.raises(ValueError, match="--game-dir is required"):
            main([])

    def test_the_version_and_the_tree_can_be_named(self) -> None:
        with _host(fetched=False) as host:
            host.files[".tools/cfr-0.151.jar"] = ()
            assert main(["--game-dir", ".g", "--cfr-version", "0.151", "--out", "out"]) == EXIT_OK
            assert host.commands[0][3] == ".tools/cfr-0.151.jar"
            assert "out" in host.removed

    def test_the_module_guard_runs_main_on_the_process_arguments(self) -> None:
        with _host(fetched=True) as host:
            host.argv = ["--game-dir", ".game"]
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.decompile", run_name="__main__")
            assert caught.value.code == EXIT_OK
