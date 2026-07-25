"""Agent build resolution, manifest parsing, and cross-language drift checks.

Every case here runs the real code against a real filesystem: the failure paths
build a complete client tree under ``tmp_path`` rather than faking the
existence hook, and the drift checks read this repository's own tracked
manifest, Java source and Makefile. Nothing is stubbed, so a test passing means
the artefacts genuinely agree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.harness import _test_hooks
from rw_bot.harness.agent import (
    AGENT_JAR_RELATIVE,
    AGENT_MANIFEST_RELATIVE,
    PREMAIN_ATTRIBUTE,
    AgentBuildError,
    decode_agent_build,
    encode_agent_build,
    parse_premain_class,
    premain_source_path,
    resolve_agent_build,
)
from rw_bot.validation import DecodeError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PREMAIN = "rwbot.agent.Premain"


def _write_manifest(root: Path, body: str) -> None:
    """Write a manifest into a client tree, creating parent directories."""
    path = root / AGENT_MANIFEST_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _write_premain_source(root: Path, premain_class: str) -> None:
    """Create the Java source file a premain binary name maps to."""
    path = premain_source_path(root, premain_class)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("public final class Premain {}\n", encoding="utf-8")


def _write_jar(root: Path) -> None:
    """Create a stand-in for the built jar at the conventional location."""
    path = root / AGENT_JAR_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04")


def _complete_tree(root: Path) -> None:
    """Build a client tree in which every agent consistency check passes."""
    _write_manifest(root, f"Manifest-Version: 1.0\n{PREMAIN_ATTRIBUTE}: {_PREMAIN}\n")
    _write_premain_source(root, _PREMAIN)
    _write_jar(root)


def test_resolve_returns_an_absolute_jar_path(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    build = resolve_agent_build(tmp_path)
    assert build["premain_class"] == _PREMAIN
    assert Path(build["jar_path"]).is_absolute()
    assert Path(build["jar_path"]).is_file()


def test_relative_client_root_is_rejected() -> None:
    with pytest.raises(AgentBuildError) as caught:
        resolve_agent_build(Path("clients/RustedWarfareBot"))
    assert caught.value.code == "RW-AGENT-004"


def test_missing_jar_names_the_build_command(tmp_path: Path) -> None:
    _write_manifest(tmp_path, f"{PREMAIN_ATTRIBUTE}: {_PREMAIN}\n")
    _write_premain_source(tmp_path, _PREMAIN)
    with pytest.raises(AgentBuildError) as caught:
        resolve_agent_build(tmp_path)
    assert caught.value.code == "RW-AGENT-002"
    assert "make agent" in caught.value.message


def test_manifest_naming_a_renamed_class_is_rejected(tmp_path: Path) -> None:
    """The drift that builds cleanly and aborts the JVM at launch."""
    _write_manifest(tmp_path, f"{PREMAIN_ATTRIBUTE}: rwbot.agent.Renamed\n")
    _write_premain_source(tmp_path, _PREMAIN)
    _write_jar(tmp_path)
    with pytest.raises(AgentBuildError) as caught:
        resolve_agent_build(tmp_path)
    assert caught.value.code == "RW-AGENT-003"
    assert "Failed to find Premain-Class" in caught.value.message


def test_manifest_without_a_premain_attribute_is_rejected(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "Manifest-Version: 1.0\n")
    with pytest.raises(AgentBuildError) as caught:
        resolve_agent_build(tmp_path)
    assert caught.value.code == "RW-AGENT-001"


def test_parse_premain_class_reads_the_attribute() -> None:
    assert parse_premain_class(("Manifest-Version: 1.0", f"{PREMAIN_ATTRIBUTE}: {_PREMAIN}")) == (
        _PREMAIN
    )


def test_parse_premain_class_is_case_insensitive_on_the_attribute_name() -> None:
    """Jar manifest attribute names are case-insensitive by specification."""
    assert parse_premain_class((f"premain-class: {_PREMAIN}",)) == _PREMAIN


def test_parse_premain_class_rejects_a_duplicated_attribute() -> None:
    with pytest.raises(AgentBuildError) as caught:
        parse_premain_class(
            (f"{PREMAIN_ATTRIBUTE}: {_PREMAIN}", f"{PREMAIN_ATTRIBUTE}: rwbot.agent.Other")
        )
    assert caught.value.code == "RW-AGENT-001"
    assert "found 2" in caught.value.message


def test_premain_source_path_maps_packages_to_directories() -> None:
    path = premain_source_path(Path("C:/rw"), "rwbot.agent.Premain")
    assert path == Path("C:/rw/agent/src/rwbot/agent/Premain.java")


def test_encode_decode_round_trips() -> None:
    original = decode_agent_build(
        {"jar_path": "C:/rw/agent/build/rw-agent.jar", "premain_class": _PREMAIN}
    )
    assert decode_agent_build(encode_agent_build(original)) == original


def test_decode_rejects_a_relative_jar_path() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_agent_build({"jar_path": AGENT_JAR_RELATIVE, "premain_class": _PREMAIN})
    assert caught.value.code == "RW-DECODE-005"


def test_path_exists_hook_reports_real_paths(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("x", encoding="utf-8")
    assert _test_hooks.path_exists(present) is True
    assert _test_hooks.path_exists(tmp_path / "absent.txt") is False


def test_repository_manifest_names_a_premain_source_that_exists() -> None:
    """Drift guard: renaming the entry point without the manifest fails here."""
    lines = _test_hooks.read_text_lines(_PROJECT_ROOT / AGENT_MANIFEST_RELATIVE)
    premain_class = parse_premain_class(lines)
    assert premain_class == _PREMAIN
    assert premain_source_path(_PROJECT_ROOT, premain_class).is_file()


def test_agent_jar_location_matches_the_makefile() -> None:
    """Drift guard: the Makefile builds the jar Python expects to attach."""
    makefile = (_PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert f"AGENT_JAR := {AGENT_JAR_RELATIVE}" in makefile
