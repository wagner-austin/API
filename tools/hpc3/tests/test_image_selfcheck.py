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
from hpc3.core.image_selfcheck import canonical_distribution, render_selfcheck

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
        "system_packages": [],
        "extra_index_urls": [],
        "requirements": ["torch==2.6.0+cu124"],
        "wheels": ["w-0.1.0-py3-none-any.whl"],
        "expected_versions": {_PACKAGE: "1.2.3"},
        "required_symbols": [{"module": _PACKAGE, "attribute": "required_symbol"}],
        "smoke_commands": [],
        "labels": {},
    }
    base.update(overrides)
    return decode_image_spec(base)


def _write_fixture_package(
    root: pathlib.Path, *, version: str, with_symbol: bool, installed: bool = True
) -> None:
    """Create an importable package the generated script can check.

    Writes a ``.dist-info`` beside it, because the version assertion reads
    what pip INSTALLED rather than a module attribute. A bare directory on
    the path is importable and is not a distribution, which is exactly the
    difference the script now distinguishes.

    Args:
        root: Directory placed on ``PYTHONPATH``.
        version: Version the distribution metadata declares.
        with_symbol: Whether to define the attribute the spec requires.
        installed: Whether to write the distribution metadata at all.
    """
    package = root / _PACKAGE
    package.mkdir(parents=True, exist_ok=True)
    lines = ["# importable, and separately a distribution or not"]
    if with_symbol:
        lines.append("required_symbol = object()")
    (package / "__init__.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if installed:
        info = root / f"{_PACKAGE}-{version}.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {_PACKAGE}\nVersion: {version}\n", encoding="utf-8"
        )


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

    def test_a_package_that_is_importable_but_not_installed_fails_legibly(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A directory on the path is importable and is not a distribution.
        The check must say so rather than raise ``PackageNotFoundError`` out
        of itself, which would report a traceback where a build wants a
        sentence."""
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=True, installed=False)
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (
            f"image self-check FAILED: {_PACKAGE} is <not installed>, expected 1.2.3"
        )

    def test_a_package_defining_no_version_attribute_still_passes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The reason the check moved off ``__version__``: the fixture defines
        none, and ``typing_extensions`` -- the entire third-party layer of the
        rusted image -- does not either."""
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=True)
        assert "__version__" not in (tmp_path / _PACKAGE / "__init__.py").read_text(
            encoding="utf-8"
        )
        assert _run_selfcheck(tmp_path, _spec()).returncode == 0

    def test_a_spelling_with_underscores_finds_the_same_distribution(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Packaging treats ``typing_extensions`` and ``typing-extensions``
        as one name, so a spec may spell either."""
        assert canonical_distribution("Typing_Extensions") == "typing-extensions"
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=True)
        spec = _spec(expected_versions={_PACKAGE.upper(): "1.2.3"})
        assert _run_selfcheck(tmp_path, spec).returncode == 0

    def test_an_absent_module_fails_the_build_even_when_installed(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A distribution can be recorded and its module still absent -- a
        truncated wheel does exactly that. The version assertion passes and
        the symbol assertion must then abort rather than warn."""
        info = tmp_path / f"{_PACKAGE}-1.2.3.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {_PACKAGE}\nVersion: 1.2.3\n", encoding="utf-8"
        )
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (f"ModuleNotFoundError: No module named '{_PACKAGE}'")

    def test_an_empty_image_fails_on_the_first_assertion(self, tmp_path: pathlib.Path) -> None:
        """Nothing planted at all: the version assertion is first, so that is
        what reports. The build stops either way, which is the contract."""
        result = _run_selfcheck(tmp_path, _spec())
        assert result.returncode == 1
        assert _last_line(result.stderr) == (
            f"image self-check FAILED: {_PACKAGE} is <not installed>, expected 1.2.3"
        )

    def test_every_declared_assertion_is_emitted(self, tmp_path: pathlib.Path) -> None:
        """Two versions and two symbols produce four checks, not one."""
        _write_fixture_package(tmp_path, version="1.2.3", with_symbol=True)
        spec = _spec(
            expected_versions={_PACKAGE: "1.2.3"},
            required_symbols=[
                {"module": _PACKAGE, "attribute": "required_symbol"},
                {"module": _PACKAGE, "attribute": "__name__"},
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
