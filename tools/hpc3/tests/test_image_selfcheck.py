"""Tests for the generated in-image verification script.

Asserting that the renderer emitted certain characters would only prove the
renderer is consistent with the test. What matters is whether the generated
program DETECTS a stale wheel, so these tests write it to disk and run it
against packages built for the purpose: one matching the spec, one with the
wrong version, one missing the symbol.

Running it in a subprocess is deliberate. The script's failure path is
``SystemExit``, its channel is stderr and its contract is an exit status --
none of which are observable if it is imported and called in-process.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from platform_core.json_utils import JSONValue

from hpc3.contracts.image_spec import ImageSpec, decode_image_spec
from hpc3.core.image_selfcheck import render_selfcheck

_PACKAGE = "fixturepkg"


def _spec(**overrides: JSONValue) -> ImageSpec:
    """Build a valid spec with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        The decoded spec.
    """
    base: dict[str, JSONValue] = {
        "base_image": "python:3.11.16-slim-bookworm",
        "env_prefix": "/opt/env",
        "git_commit": "d11efacd",
        "extra_index_urls": [],
        "requirements": ["torch==2.6.0+cu124"],
        "wheels": ["w-0.1.0-py3-none-any.whl"],
        "expected_versions": {_PACKAGE: "1.2.3"},
        "required_symbols": [{"module": _PACKAGE, "attribute": "required_symbol"}],
        "labels": {},
    }
    base.update(overrides)
    return decode_image_spec(base)


def _write_fixture_package(root: pathlib.Path, *, version: str, with_symbol: bool) -> None:
    """Create an importable package the generated script can check.

    Args:
        root: Directory placed on ``PYTHONPATH``.
        version: Value bound to ``__version__``.
        with_symbol: Whether to define the attribute the spec requires.
    """
    package = root / _PACKAGE
    package.mkdir(parents=True, exist_ok=True)
    lines = [f'__version__ = "{version}"']
    if with_symbol:
        lines.append("required_symbol = object()")
    (package / "__init__.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_selfcheck(root: pathlib.Path, spec: ImageSpec) -> subprocess.CompletedProcess[str]:
    """Render the script into ``root`` and execute it against that directory.

    Args:
        root: Directory holding the fixture package; also the working root
            placed on ``PYTHONPATH``.
        spec: Spec whose assertions the script carries.

    Returns:
        The completed process, with streams captured as text.
    """
    script = root / "selfcheck.py"
    script.write_text(render_selfcheck(spec), encoding="utf-8", newline="\n")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )


def _last_line(stream: str) -> str:
    """Return a stream's final non-empty line.

    Asserting equality on one known line rather than a substring of the whole
    stream: a substring passes on a message that merely contains it, which is
    how an assertion survives the wording change that should have failed it.

    Args:
        stream: Captured stdout or stderr.

    Returns:
        The last line with content, stripped of trailing whitespace.
    """
    lines = [line for line in stream.splitlines() if line.strip() != ""]
    if not lines:
        raise AssertionError("expected output, got an empty stream")
    return lines[-1]


class TestTheGeneratedScriptRuns:
    """Behaviour of the emitted program, not the text of it."""

    def test_a_matching_image_passes(self, tmp_path: pathlib.Path) -> None:
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=True)
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 0
        assert _last_line(result.stdout) == "image self-check OK"

    def test_a_wrong_version_fails_with_both_values(self, tmp_path: pathlib.Path) -> None:
        _write_fixture_package(tmp_path, version="9.9.9", with_symbol=True)
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (
            f"image self-check FAILED: {_PACKAGE} is 9.9.9, expected 1.2.3"
        )

    def test_a_missing_symbol_names_the_stale_wheel(self, tmp_path: pathlib.Path) -> None:
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=False)
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (
            f"image self-check FAILED: {_PACKAGE} is missing required_symbol"
            " -- a stale wheel was baked into this image"
        )

    def test_a_package_without_a_version_attribute_fails_legibly(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A module with no ``__version__`` must not raise AttributeError."""
        package = tmp_path / _PACKAGE
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("required_symbol = 1\n", encoding="utf-8")
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (
            f"image self-check FAILED: {_PACKAGE} is <no __version__>, expected 1.2.3"
        )

    def test_an_absent_module_fails_the_build(self, tmp_path: pathlib.Path) -> None:
        """Importing a module the image does not carry must abort, not warn."""
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (f"ModuleNotFoundError: No module named '{_PACKAGE}'")

    def test_every_declared_assertion_is_emitted(self, tmp_path: pathlib.Path) -> None:
        """Two versions and two symbols produce four checks, not one."""
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=True)
        spec = _spec(
            expected_versions={_PACKAGE: "1.2.3"},
            required_symbols=[
                {"module": _PACKAGE, "attribute": "required_symbol"},
                {"module": _PACKAGE, "attribute": "__version__"},
            ],
        )
        rendered = render_selfcheck(spec)
        assert rendered.count("hasattr") == 2
        result = _run_selfcheck(tmp_path, spec)
        assert result.returncode == 0


class TestTheGeneratedScriptIsValidSource:
    """It is written to disk and executed, so it must parse."""

    def test_it_compiles(self) -> None:
        compiled = compile(render_selfcheck(_spec()), "selfcheck.py", "exec")
        assert compiled.co_filename == "selfcheck.py"

    def test_it_ends_with_a_newline(self) -> None:
        assert render_selfcheck(_spec()).endswith("\n")

    def test_it_carries_no_carriage_returns(self) -> None:
        """A CRLF script makes the kernel report the interpreter as missing."""
        assert "\r" not in render_selfcheck(_spec())
