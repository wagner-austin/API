"""Every command a headless match is started by, as the pure composition it is.

Parameterised across platforms throughout, because that is the whole reason
this moved out of PowerShell: one description of a launch that a Windows
workstation and a Linux compute node both read, checkable from either.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

import pytest

from rw_bot.harness.agent_build import (
    JAVA_RELEASE,
    agent_jar,
    agent_sources,
    classes_dir,
    compile_command,
    package_command,
)
from rw_bot.harness.launch import (
    DISPLAY_SIZE_TEXT,
    HEAP,
    MAIN_CLASS,
    SANDBOX_FLAG,
    XVFB_SCREEN,
    LaunchConfig,
    agent_arguments,
    decode_launch_config,
    display_prefix,
    encode_launch_config,
    game_command,
    game_environment,
    planner_command,
    planner_pythonpath,
)
from rw_bot.platform_id import WINDOWS
from rw_bot.validation import DecodeError

LINUX = "linux"

WINDOWS_ROOT = PureWindowsPath(r"C:\repo")
POSIX_ROOT = PurePosixPath("/repo")


def _config(
    port: int = 27511,
    game_dir: str = ".game-w1",
    seed: int = 0,
    lockstep: int = 0,
    settle: int = 22,
    display: int = 0,
    play_log: str = "runs/play.log",
    map_path: str = "",
    opponents: int = 1,
    difficulty: int = 0,
    tree: str = "",
    pin_delta: int = 0,
    fast_forward: int = 0,
    rng_tap: int = 0,
    extra_agent_args: str = "",
    module: str = "scripts.play",
    catalogue: str = "cat.log",
    type_dump: str = "types.ndjson",
    play_args: str = "1500 doctrines/a.doctrine -",
) -> LaunchConfig:
    """Build a launch with every knob off, overriding what a test cares about.

    Args:
        port: Channel port.
        game_dir: Game directory to play in.
        seed: Engine seed.
        lockstep: Frames between samples.
        settle: Sandbox settle seconds.
        display: X display to start a server on, or 0 for none.
        play_log: Engine log path.
        map_path: Skirmish map, or empty for the sandbox. Named apart from the
            field because ``map`` is a builtin.
        opponents: AI opponents.
        difficulty: AI difficulty.
        tree: Frozen snapshot.
        pin_delta: Frame delta in ms.
        fast_forward: Wall-clock multiple.
        rng_tap: Draw-counter flag.
        extra_agent_args: Agent passthrough.
        module: Planner module.
        catalogue: Catalogue path.
        type_dump: Type-flag dump path.
        play_args: Planner positional tail.

    Returns:
        The launch.
    """
    return LaunchConfig(
        port=port,
        game_dir=game_dir,
        seed=seed,
        lockstep=lockstep,
        settle=settle,
        display=display,
        play_log=play_log,
        map=map_path,
        opponents=opponents,
        difficulty=difficulty,
        tree=tree,
        pin_delta=pin_delta,
        fast_forward=fast_forward,
        rng_tap=rng_tap,
        extra_agent_args=extra_agent_args,
        module=module,
        catalogue=catalogue,
        type_dump=type_dump,
        play_args=play_args,
    )


class TestTheLaunchCodec:
    def test_a_launch_round_trips_through_its_payload(self) -> None:
        assert decode_launch_config(encode_launch_config(_config())) == _config()

    def test_every_field_survives_the_round_trip(self) -> None:
        """Not just the shape: a codec that dropped a field would still
        round-trip if the test only compared the fields it kept."""
        full = _config(
            seed=42,
            lockstep=75,
            map_path="maps/x.tmx",
            opponents=3,
            difficulty=2,
            tree="runs/sweeps/d/.tree",
            pin_delta=3,
            fast_forward=10,
            rng_tap=1,
            extra_agent_args="k=v",
        )
        payload = encode_launch_config(full)
        assert sorted(payload) == sorted(full)
        assert decode_launch_config(payload) == full

    def test_a_port_that_is_not_positive_is_refused(self) -> None:
        """Zero used to mean "let the recipe draw one", and two concurrent
        draws collided (imp-creep12, 2026-08-08)."""
        payload = encode_launch_config(_config())
        payload["port"] = 0
        with pytest.raises(DecodeError) as caught:
            decode_launch_config(payload)
        assert caught.value.code == "RW-DECODE-004"

    def test_a_missing_field_names_itself(self) -> None:
        payload = encode_launch_config(_config())
        del payload["game_dir"]
        with pytest.raises(DecodeError) as caught:
            decode_launch_config(payload)
        assert caught.value.code == "RW-DECODE-001"

    def test_a_field_of_the_wrong_type_is_refused_rather_than_coerced(self) -> None:
        """A payload that disagrees with its schema is a bug in the producer,
        and coercion would hide it."""
        payload = encode_launch_config(_config())
        payload["seed"] = "12"
        with pytest.raises(DecodeError) as caught:
            decode_launch_config(payload)
        assert caught.value.code == "RW-DECODE-002"

    def test_an_optional_text_field_may_be_empty_but_must_be_present(self) -> None:
        """Empty is how "no map" and "no frozen tree" are spelled, so blank is
        legal there -- absent is not."""
        assert decode_launch_config(encode_launch_config(_config()))["map"] == ""
        payload = encode_launch_config(_config())
        del payload["map"]
        with pytest.raises(DecodeError):
            decode_launch_config(payload)


class TestTheAgentsOptions:
    def test_a_bare_launch_names_only_the_port(self) -> None:
        """Every option is omitted when off rather than passed as a zero: a
        frozen tree predating an option REJECTS the unknown key, so a resumed
        batch only works if the launcher never mentions it."""
        assert agent_arguments(_config()) == "channelPort=27511"

    def test_each_knob_appears_only_when_it_is_on(self) -> None:
        assert agent_arguments(_config(seed=42)) == "channelPort=27511;randomSeed=42"
        assert agent_arguments(_config(lockstep=75)) == "channelPort=27511;lockstepFrames=75"
        assert agent_arguments(_config(pin_delta=3)) == "channelPort=27511;pinDeltaMs=3"
        assert agent_arguments(_config(fast_forward=10)) == "channelPort=27511;fastForward=10"

    def test_the_draw_tap_is_a_flag_not_a_count(self) -> None:
        assert agent_arguments(_config(rng_tap=1)) == "channelPort=27511;rngTap=true"

    def test_a_named_map_brings_its_opponents_and_difficulty(self) -> None:
        """The map decides how many opponents there can be -- the engine caps
        teams by the map's own count -- so the three travel together."""
        arguments = agent_arguments(
            _config(map_path="maps/skirmish/duel.tmx", opponents=3, difficulty=2)
        )
        assert arguments == (
            "channelPort=27511;matchMap=maps/skirmish/duel.tmx;matchOpponents=3;matchDifficulty=2"
        )

    def test_the_passthrough_is_appended_verbatim(self) -> None:
        arguments = agent_arguments(_config(extra_agent_args="discoverAtSeconds=24;x=1"))
        assert arguments.endswith(";discoverAtSeconds=24;x=1")

    def test_the_separator_is_the_agents_own_on_every_platform(self) -> None:
        """It looks like the Windows path-list separator and is not one. If it
        were routed through the platform's separator, every POSIX launch would
        send the agent one option named after all of them."""
        assert ";" in agent_arguments(_config(seed=1))


class TestWhichAgentJarIsAttached:
    def test_a_frozen_batch_reuses_the_jar_its_snapshot_carries(self) -> None:
        """Built once when the snapshot was taken, not per match."""
        assert agent_jar(_config(tree="runs/sweeps/demo/.tree"), "abc") == (
            "runs/sweeps/demo/.tree/rw-agent.jar"
        )

    def test_a_single_match_compiles_its_own_under_a_stamp(self) -> None:
        """Concurrent matches must not overwrite each other's jar."""
        assert agent_jar(_config(), "abc123") == "agent/build/rw-agent-play-abc123.jar"

    def test_two_stamps_give_two_jars_and_two_class_directories(self) -> None:
        assert agent_jar(_config(), "aaa") != agent_jar(_config(), "bbb")
        assert classes_dir("aaa") != classes_dir("bbb")


class TestChoosingAgentSources:
    def test_only_java_is_compiled_and_in_a_stable_order(self) -> None:
        listing = ("Targets.java", "readme.txt", "Agent.java", "ClassFilePatcher.java")
        assert agent_sources(listing) == (
            "agent/src/rwbot/agent/Agent.java",
            "agent/src/rwbot/agent/ClassFilePatcher.java",
            "agent/src/rwbot/agent/Targets.java",
        )

    def test_a_directory_with_no_java_is_refused(self) -> None:
        """An empty jar attaches without error and silently never opens the
        channel -- which reads as a hung engine ninety seconds later rather
        than as an empty build."""
        with pytest.raises(ValueError, match="attaches without error"):
            agent_sources(("readme.txt",))


class TestBuildingTheAgent:
    def test_the_whole_compile_command_is_what_it_claims(self) -> None:
        """``-Werror`` is the load-bearing part: the agent loads into the
        game's own classloader beside obfuscated classes, and a warning there
        is the first sign of a name that moved between builds."""
        assert compile_command("javac", "build/x", ("A.java", "B.java")) == (
            "javac",
            "--release",
            "8",
            "-Xlint:all",
            "-Werror",
            "-d",
            "build/x",
            "A.java",
            "B.java",
        )

    def test_packaging_names_the_manifest_and_the_class_root(self) -> None:
        assert package_command("jar", "out.jar", "build/x") == (
            "jar",
            "cfm",
            "out.jar",
            "agent/manifest.mf",
            "-C",
            "build/x",
            ".",
        )


class TestTheEngineCommand:
    def test_it_runs_the_jvm_out_of_the_clone(self) -> None:
        """Not the system java: the clone's JVM is the pinned one, and a match
        played on another is a match played on a different runtime."""
        command = game_command(_config(), WINDOWS_ROOT, "a.jar", WINDOWS)
        assert command[0] == r"C:\repo\.game-w1\jvm64\bin\java.exe"
        posix = game_command(_config(), POSIX_ROOT, "a.jar", LINUX)
        assert posix[0] == "/repo/.game-w1/jvm-linux/bin/java"

    def test_the_classpath_takes_the_platforms_separator(self) -> None:
        """Joined wrongly the JVM reads it as ONE entry, finds no such file,
        and fails as a missing main class -- which reads like a broken jar."""
        windows = game_command(_config(), WINDOWS_ROOT, "a.jar", WINDOWS)
        assert windows[windows.index("-cp") + 1] == "game-lib.jar;libs/*"
        posix = game_command(_config(), POSIX_ROOT, "a.jar", LINUX)
        assert posix[posix.index("-cp") + 1] == "game-lib.jar:libs/*"

    def test_the_agent_and_the_log_are_absolute(self) -> None:
        """The engine runs with the GAME directory as its working directory,
        so a relative path would resolve against the clone."""
        command = game_command(_config(), POSIX_ROOT, "agent/build/a.jar", LINUX)
        assert f"-javaagent:/repo/agent/build/a.jar={agent_arguments(_config())}" in command
        assert command[command.index("-log") + 1] == "/repo/runs/play.log"

    def test_it_is_headless_and_silent(self) -> None:
        command = game_command(_config(), POSIX_ROOT, "a.jar", LINUX)
        assert "-nodisplay" in command
        assert "-nosound" in command

    def test_a_display_size_is_stated_because_nodisplay_still_opens_one(self) -> None:
        """-nodisplay is not a no-OpenGL mode; the 10x10 display it picks for
        itself fails once in-game UI renders."""
        command = game_command(_config(), POSIX_ROOT, "a.jar", LINUX)
        assert command[command.index("-width") + 1] == "800"
        assert command[command.index("-height") + 1] == "600"

    def test_no_map_means_the_engines_own_sandbox(self) -> None:
        assert SANDBOX_FLAG in game_command(_config(), POSIX_ROOT, "a.jar", LINUX)

    def test_a_named_map_withholds_the_sandbox_because_the_agent_starts_it(self) -> None:
        command = game_command(_config(map_path="maps/x.tmx"), POSIX_ROOT, "a.jar", LINUX)
        assert SANDBOX_FLAG not in command

    def test_the_heap_and_entry_point_are_named(self) -> None:
        command = game_command(_config(), POSIX_ROOT, "a.jar", LINUX)
        assert HEAP in command
        assert MAIN_CLASS in command

    def test_the_module_opens_are_paired_with_their_targets_on_a_modular_jvm(self) -> None:
        """Without them the agent's reflection fails at class-load on 13."""
        command = game_command(_config(), WINDOWS_ROOT, "a.jar", WINDOWS)
        opens = [command[i + 1] for i, token in enumerate(command) if token == "--add-opens"]
        assert opens == ["java.base/java.lang=ALL-UNNAMED", "java.base/java.util=ALL-UNNAMED"]

    def test_a_java_eight_runtime_is_given_no_module_options_at_all(self) -> None:
        """The Linux depot ships a JRE 8, which has no module system and
        REJECTS ``--add-opens`` as unrecognised. Passing it there is not a
        harmless extra -- it is a JVM that never starts, which on a compute
        node is a scheduled job that dies in its first second."""
        command = game_command(_config(), POSIX_ROOT, "a.jar", LINUX)
        assert "--add-opens" not in command

    def test_the_agent_targets_the_oldest_runtime_it_must_load_into(self) -> None:
        """A class file above a JVM's level does not degrade, it fails at load
        with UnsupportedClassVersionError. Eight runs on 13; 13 does not run
        on eight."""
        assert JAVA_RELEASE == "8"
        assert compile_command("javac", "b", ("A.java",))[1:3] == ("--release", "8")


class TestTheEngineEnvironment:
    def test_linux_gets_the_dynamic_linker_search_path(self) -> None:
        """``-Djava.library.path`` tells the JVM where to find a library it is
        asked to load. It does NOT tell the linker where that library's own
        dependencies are, and librocketConnector.so links against
        libRocketCore.so.1 beside it. Read off the game's own Linux launcher."""
        assert game_environment(LINUX) == {"LD_LIBRARY_PATH": "."}

    def test_windows_needs_no_help_resolving_its_own_dlls(self) -> None:
        assert game_environment(WINDOWS) == {}


class TestTheXServer:
    """``-nodisplay`` is not a no-OpenGL mode: Slick2D still opens a display
    and creates framebuffer objects during boot. On a desktop that costs
    nothing; on a headless compute node there is no server to open at all."""

    def test_a_leased_display_brings_its_own_server(self) -> None:
        command = game_command(_config(display=91), POSIX_ROOT, "a.jar", LINUX)
        assert command[:5] == ("xvfb-run", "-n", "91", "-s", "-screen 0 800x600x24")

    def test_the_server_wraps_the_jvm_rather_than_following_it(self) -> None:
        """It has to be the program, not an argument: the wrapper starts the
        server, runs what follows under it, and takes it down again."""
        command = game_command(_config(display=91), POSIX_ROOT, "a.jar", LINUX)
        assert command.index("xvfb-run") == 0
        assert command.index("/repo/.game-w1/jvm-linux/bin/java") > 0

    def test_the_screen_matches_the_size_the_engine_is_told_to_open(self) -> None:
        """A server smaller than the window the engine asks for fails at
        framebuffer creation rather than degrading."""
        assert DISPLAY_SIZE_TEXT == "800x600"
        assert XVFB_SCREEN.endswith("800x600x24")

    def test_no_display_means_no_server(self) -> None:
        """Zero says the machine already has one -- a workstation desktop."""
        assert display_prefix(0, LINUX) == ()
        assert game_command(_config(), POSIX_ROOT, "a.jar", LINUX)[0].endswith("java")

    def test_windows_never_starts_one_whatever_the_display_says(self) -> None:
        """There is no X there, and xvfb-run is a command that does not
        exist -- so a display number reaching a Windows launch is ignored
        rather than turned into a launch that cannot start."""
        assert display_prefix(91, WINDOWS) == ()
        command = game_command(_config(display=91), WINDOWS_ROOT, "a.jar", WINDOWS)
        assert command[0].endswith("java.exe")

    def test_a_negative_display_is_refused(self) -> None:
        """Zero is already the "none" sentinel, so a number below it is not a
        spare value but a bug."""
        with pytest.raises(ValueError, match="non-negative number"):
            display_prefix(-1, LINUX)

    def test_the_server_number_is_a_lease_and_not_a_search(self) -> None:
        """``xvfb-run -a`` picks one by scanning for a free lock file, which
        is the same shape as the random port draw that collided the first
        time eight matches launched in one instant."""
        assert "-a" not in display_prefix(91, LINUX)
        assert "-n" in display_prefix(91, LINUX)


class TestThePlannerCommand:
    def test_it_runs_the_interpreter_it_is_given(self) -> None:
        """Not `poetry run python`: a batch inside a container image has no
        poetry, and the planner must run in the harness's own environment."""
        command = planner_command("/opt/env/bin/python", _config())
        assert command[0] == "/opt/env/bin/python"

    def test_the_positional_tail_is_the_planners_own(self) -> None:
        command = planner_command("py", _config())
        assert command[-4:] == ("types.ndjson", "1500", "doctrines/a.doctrine", "-")

    def test_a_frozen_tree_isolates_the_path(self) -> None:
        """-P keeps the repository root off sys.path so the snapshot wins."""
        assert "-P" in planner_command("py", _config(tree="runs/sweeps/d/.tree"))

    def test_the_working_tree_keeps_the_root_importable(self) -> None:
        assert "-P" not in planner_command("py", _config())

    def test_an_empty_tail_adds_no_blank_arguments(self) -> None:
        """Splitting an empty string yields one empty token, which the planner
        would read as a positional argument it does not have."""
        command = planner_command("py", _config(play_args=""))
        assert "" not in command


class TestThePlannersPath:
    def test_it_carries_the_snapshot_and_its_source_directory(self) -> None:
        path = planner_pythonpath(POSIX_ROOT, "runs/sweeps/d/.tree", LINUX)
        assert path == "/repo/runs/sweeps/d/.tree:/repo/runs/sweeps/d/.tree/src"

    def test_it_takes_the_platforms_separator(self) -> None:
        path = planner_pythonpath(WINDOWS_ROOT, "t", WINDOWS)
        assert path == r"C:\repo\t;C:\repo\t\src"

    def test_composing_one_without_a_tree_is_refused(self) -> None:
        """A blank PYTHONPATH is not an unset one: it would put the working
        directory back on a path that was told not to have it."""
        with pytest.raises(ValueError, match="only composed for a frozen tree"):
            planner_pythonpath(POSIX_ROOT, "", LINUX)
