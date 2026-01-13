"""Tests for scripts.guard module."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts import _test_hooks, guard


def _project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the TankpitBot project root.
    """
    return Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories if needed.

    Args:
        path: File path to write to.
        text: Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_BAD_SNIPPET = (
    "from typing import Any\n"
    "x: Any = 1  # type: ignore\n"
    "from typing import cast\n"
    "y = cast(int, 1)\n"
    "import contextlib\n"
    "with contextlib.suppress(Exception):\n"
    "    pass\n"
    "try:\n"
    "    1/0\n"
    "except Exception as exc:\n"
    "    raise RuntimeError('fail') from exc\n"
)


def test_guard_detects_violations(tmp_path: Path) -> None:
    """Test guard.main detects violations and returns non-zero exit code."""
    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"

    _write(bad, _BAD_SNIPPET)

    rc = guard.main(["--root", str(root)])
    assert rc != 0


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    """Test guard.main returns 0 when no violations found."""
    rc = guard.main(["--root", str(tmp_path)])
    assert rc == 0


def test_guard_main_direct_violations(tmp_path: Path) -> None:
    """Test guard.main with violations in tmp directory."""
    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"

    _write(bad, _BAD_SNIPPET)

    rc = guard.main(["--root", str(root)])
    assert rc != 0


def test_guard_main_direct_clean(tmp_path: Path) -> None:
    """Test guard.main with verbose flag on clean directory."""
    rc = guard.main(["--root", str(tmp_path), "--verbose"])
    assert rc == 0


def test_guard_main_unknown_flag_is_ignored(tmp_path: Path) -> None:
    """Test guard.main ignores unknown positional arguments."""
    rc = guard.main(["--root", str(tmp_path), "ignored-flag"])
    assert rc == 0


@pytest.fixture()
def _restore_guard_hooks() -> Generator[None, None, None]:
    """Restore guard hooks after each test.

    Yields:
        None after saving original hook state.
    """
    original_is_dir = _test_hooks.is_dir
    yield
    _test_hooks.is_dir = original_is_dir


def test_guard_find_monorepo_root_errors_when_missing_libs(
    tmp_path: Path, _restore_guard_hooks: None
) -> None:
    """Test _find_monorepo_root raises RuntimeError when libs not found."""

    def _always_false(path: Path) -> bool:
        return False

    _test_hooks.is_dir = _always_false

    with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        guard._find_monorepo_root(tmp_path)


def test_find_monorepo_root_success() -> None:
    """Test _find_monorepo_root finds the libs directory."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    root = guard._find_monorepo_root(project_root)

    libs_path = root / "libs"
    services_path = root / "services"
    assert libs_path.is_dir()
    assert services_path.is_dir()


def test_real_is_dir() -> None:
    """Test _real_is_dir uses Path.is_dir()."""
    script_path = Path(__file__).resolve()
    parent = script_path.parent

    assert _test_hooks._real_is_dir(parent) is True
    assert _test_hooks._real_is_dir(parent / "nonexistent") is False


def test_guard_entrypoint_runs_as_main() -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
    modules_to_clear = [k for k in sys.modules if k.startswith("scripts")]
    saved_modules: dict[str, types.ModuleType] = {}
    for mod in modules_to_clear:
        saved_modules[mod] = sys.modules.pop(mod)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    err = exc.value
    code: int = err.code if isinstance(err.code, int) else 0
    assert code in (0, 2)

    sys.modules.update(saved_modules)


def test_real_path_exists_true(tmp_path: Path) -> None:
    """Test _real_path_exists returns True for existing path."""
    test_file = tmp_path / "exists.txt"
    test_file.write_text("content", encoding="utf-8")

    assert _test_hooks._real_path_exists(test_file) is True


def test_real_path_exists_false(tmp_path: Path) -> None:
    """Test _real_path_exists returns False for missing path."""
    test_file = tmp_path / "missing.txt"

    assert _test_hooks._real_path_exists(test_file) is False


def test_real_read_text_reads_file(tmp_path: Path) -> None:
    """Test _real_read_text reads file content."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("file content", encoding="utf-8")

    result = _test_hooks._real_read_text(test_file)
    assert result == "file content"


def test_real_load_and_decode_session_loads_session(tmp_path: Path) -> None:
    """Test _real_load_and_decode_session loads a valid session."""
    from platform_core.json_utils import dump_json_str

    session_file = tmp_path / "session.json"
    session_data = {
        "session_id": "test-123",
        "start_timestamp_ms": 1000,
        "end_timestamp_ms": 2000,
        "base_url": "https://tankpit.com",
        "messages": [],
        "magic": "testmagic123456789",
    }
    session_file.write_text(dump_json_str(session_data), encoding="utf-8")

    decoder = _test_hooks._real_load_and_decode_session(session_file)

    # SessionDecoder has commands and lobby_messages properties
    # Verify they are list-like by checking length (empty for no messages)
    assert len(decoder.commands) == 0
    assert len(decoder.lobby_messages) == 0


def test_real_setup_rich_logging() -> None:
    """Test _real_setup_rich_logging sets up logging."""
    # This just verifies the function runs without error
    # We use INFO level to not clutter test output
    _test_hooks._real_setup_rich_logging("INFO")
