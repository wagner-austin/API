"""Launch-configuration validation and argv rendering."""

from __future__ import annotations

import pytest

from rw_bot.harness.launch import (
    CLASSPATH,
    JAVA_EXE_RELATIVE,
    MAIN_CLASS,
    VERIFIED_HEIGHT,
    VERIFIED_WIDTH,
    LaunchConfigError,
    build_argv,
    decode_launch_config,
    encode_launch_config,
    make_launch_config,
)
from rw_bot.validation import DecodeError

_GAME_DIR = ".game"
_LOG = "C:/runs/boot.log"
_AGENT = "C:/rw/agent/build/rw-agent.jar"


def test_defaults_match_the_verified_display_size() -> None:
    config = make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar=_AGENT)
    assert config["width"] == VERIFIED_WIDTH
    assert config["height"] == VERIFIED_HEIGHT
    assert config["no_sound"] is True
    assert config["sandbox"] is False
    assert config["print_units"] is False
    assert config["max_heap_mb"] == 1000
    assert config["agent_jar"] == _AGENT


def test_argv_renders_the_shipped_launcher_shape() -> None:
    config = make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar=_AGENT)
    assert build_argv(config) == (
        JAVA_EXE_RELATIVE,
        "-Xmx1000M",
        "-Dfile.encoding=UTF-8",
        "-Djava.library.path=.",
        f"-javaagent:{_AGENT}",
        "-cp",
        CLASSPATH,
        MAIN_CLASS,
        "-nodisplay",
        "-width",
        "800",
        "-height",
        "600",
        "-nosound",
        "-log",
        _LOG,
    )


def test_agent_is_attached_as_a_jvm_option_not_a_game_argument() -> None:
    """``-javaagent`` after the main class would reach the engine, not the JVM."""
    config = make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar=_AGENT)
    argv = build_argv(config)
    assert argv.index(f"-javaagent:{_AGENT}") < argv.index("-cp")
    assert argv.index(f"-javaagent:{_AGENT}") < argv.index(MAIN_CLASS)


def test_argv_omits_optional_flags_when_disabled() -> None:
    config = make_launch_config(
        game_dir=_GAME_DIR,
        log_path=_LOG,
        agent_jar=_AGENT,
        no_sound=False,
    )
    assert "-nosound" not in build_argv(config)


def test_argv_includes_sandbox_when_requested() -> None:
    config = make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar=_AGENT, sandbox=True)
    argv = build_argv(config)
    assert "-sandbox" in argv
    assert "-printunits" not in argv


def test_argv_includes_print_units_when_requested() -> None:
    config = make_launch_config(
        game_dir=_GAME_DIR, log_path="C:/runs/units.log", agent_jar=_AGENT, print_units=True
    )
    argv = build_argv(config)
    assert "-printunits" in argv
    assert "-sandbox" not in argv


def test_sandbox_and_print_units_together_are_rejected() -> None:
    with pytest.raises(LaunchConfigError) as caught:
        make_launch_config(
            game_dir=_GAME_DIR,
            log_path=_LOG,
            agent_jar=_AGENT,
            sandbox=True,
            print_units=True,
        )
    assert caught.value.code == "RW-LAUNCH-001"
    assert "mutually exclusive" in caught.value.message


def test_blank_game_dir_is_rejected() -> None:
    with pytest.raises(DecodeError) as caught:
        make_launch_config(game_dir="  ", log_path=_LOG, agent_jar=_AGENT)
    assert caught.value.code == "RW-DECODE-003"


def test_blank_agent_jar_is_rejected() -> None:
    with pytest.raises(DecodeError) as caught:
        make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar="  ")
    assert caught.value.code == "RW-DECODE-003"


def test_relative_agent_jar_is_rejected() -> None:
    """A relative agent path resolves against the game tree, where no jar exists."""
    with pytest.raises(DecodeError) as caught:
        make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar="agent/build/rw-agent.jar")
    assert caught.value.code == "RW-DECODE-005"
    assert "agent_jar" in caught.value.message


def test_relative_log_path_is_rejected() -> None:
    with pytest.raises(DecodeError) as caught:
        make_launch_config(game_dir=_GAME_DIR, log_path="runs/boot.log", agent_jar=_AGENT)
    assert caught.value.code == "RW-DECODE-005"
    assert "log_path" in caught.value.message


def test_zero_width_is_rejected() -> None:
    with pytest.raises(DecodeError) as caught:
        make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar=_AGENT, width=0)
    assert caught.value.code == "RW-DECODE-004"


def test_decode_rejects_a_missing_field() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_launch_config({"game_dir": _GAME_DIR, "sandbox": False, "print_units": False})
    assert caught.value.code == "RW-DECODE-001"


def test_decode_rejects_a_payload_without_the_agent() -> None:
    """The agent is not optional, so its absence is a decode failure."""
    complete = encode_launch_config(
        make_launch_config(game_dir=_GAME_DIR, log_path=_LOG, agent_jar=_AGENT)
    )
    del complete["agent_jar"]
    with pytest.raises(DecodeError) as caught:
        decode_launch_config(complete)
    assert caught.value.code == "RW-DECODE-001"
    assert "agent_jar" in caught.value.message


def test_encode_decode_round_trips() -> None:
    original = make_launch_config(
        game_dir=_GAME_DIR,
        log_path=_LOG,
        agent_jar=_AGENT,
        max_heap_mb=2048,
        width=1024,
        height=768,
        no_sound=False,
        sandbox=True,
    )
    assert decode_launch_config(encode_launch_config(original)) == original
