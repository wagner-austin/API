"""An environment's interpreter is asked where it came from, and held to it.

The cases are the measured ones. On 2026-10-05 the host environments
``cleargbm``, ``abl-pinned`` and ``abl-cu128`` reported their own paths as
``sys.base_prefix``; ``tankpit`` reported ``/pub/wagnera3/envs/cleargbm``.
Inside all eight registered images ``/opt/env`` reported ``/usr/local`` on the
container's root device, and ``/dfs6b`` was mounted although only
``/pub/wagnera3`` was bound.
"""

from __future__ import annotations

import shlex
import subprocess
import sys

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.image import ImageReference
from hpc3.core.interpreter import (
    InterpreterIdentity,
    check_interpreter_home,
    identity_command,
    parse_identity,
    split_identity,
)
from tests.conftest import IMAGE_INTERPRETER, host_interpreter

_TANKPIT = "/pub/wagnera3/envs/tankpit"
_CLEARGBM = "/pub/wagnera3/envs/cleargbm"

_IMAGE = ImageReference(
    path="/pub/wagnera3/tankpit/images/v2/tankpit.sif",
    sha256="0cfdd5592a1ab0df1cbe121ae567eb5d75a42f53c619f7c80fdbefaa8144a0e6",
    binds=["/pub/wagnera3"],
)
"""The tankpit v2 image as runs/hpc3-tankpit.json registers it."""


def _identity(base_prefix: str, *, on_root: bool) -> InterpreterIdentity:
    """Build an identity as the probe would report it.

    Args:
        base_prefix: The reported ``sys.base_prefix``.
        on_root: Whether that prefix is on the root filesystem's device.

    Returns:
        The identity, at the language version every project here runs.
    """
    return InterpreterIdentity(
        version="3.11", base_prefix=base_prefix, base_on_root_filesystem=on_root
    )


class TestIdentityCommand:
    def test_it_runs_the_environments_own_interpreter_by_path(self) -> None:
        """Through PATH it would answer for whatever a login shell activates."""
        assert identity_command(_TANKPIT).startswith(f"'{_TANKPIT}/bin/python' -c ")

    def test_it_carries_no_newline(self) -> None:
        """A real newline would be split by the remote shell before Python saw it."""
        assert "\n" not in identity_command(_TANKPIT)

    def test_the_real_command_runs_and_its_answer_parses(self) -> None:
        """Only executing the probe proves its text is Python that prints three lines."""
        argv = shlex.split(identity_command("/e"))

        completed = subprocess.run(
            [sys.executable, "-c", argv[2]], capture_output=True, text=True, check=True
        )

        identity = parse_identity(completed.stdout)
        assert identity["version"] == f"{sys.version_info.major}.{sys.version_info.minor}"
        assert identity["base_prefix"] == sys.base_prefix


class TestSplitIdentity:
    def test_the_measured_image_answer_is_read(self) -> None:
        identity, rest = split_identity(IMAGE_INTERPRETER + "torch==2.6.0\n")
        assert identity == _identity("/usr/local", on_root=True)
        assert rest == ["torch==2.6.0"]

    def test_surrounding_blank_lines_and_spaces_are_ignored(self) -> None:
        identity, rest = split_identity(f"\n  3.11  \n\n  {_CLEARGBM}  \n False \n")
        assert identity == _identity(_CLEARGBM, on_root=False)
        assert rest == []

    @pytest.mark.parametrize(
        "output",
        [
            "",
            "3.11\n/usr/local\n",
            "Traceback (most recent call last):\n  File x\nNameError\n",
            "3.11.16\n/usr/local\nTrue\n",
            "3.11\n/usr/local\nyes\n",
            "torch==2.6.0\nnumpy==2.1.3\ntransformers==4.46.3\n",
        ],
    )
    def test_anything_but_an_identity_is_unreadable(self, output: str) -> None:
        """Read loosely, a failed probe would be blamed on the environment."""
        with pytest.raises(AppError) as excinfo:
            _ = split_identity(output)
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE


class TestParseIdentity:
    def test_exactly_an_identity_is_read(self) -> None:
        assert parse_identity(host_interpreter(_CLEARGBM)) == _identity(_CLEARGBM, on_root=False)

    def test_anything_after_the_identity_is_refused(self) -> None:
        """The identity command prints nothing else, so more is somebody else's answer."""
        with pytest.raises(AppError) as excinfo:
            _ = parse_identity(host_interpreter(_CLEARGBM) + "torch==2.6.0\n")
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE
        assert "more than its identity" in excinfo.value.message


class TestCheckInterpreterHome:
    def test_a_host_environment_that_owns_its_interpreter_passes(self) -> None:
        check_interpreter_home(_identity(_CLEARGBM, on_root=False), env_path=_CLEARGBM, image=None)

    def test_a_host_environment_on_another_projects_interpreter_is_refused(self) -> None:
        """envs/tankpit, as measured: right version, working, and someone else's Python."""
        with pytest.raises(AppError) as excinfo:
            check_interpreter_home(
                _identity(_CLEARGBM, on_root=False), env_path=_TANKPIT, image=None
            )
        assert excinfo.value.code is Hpc3ErrorCode.ENV_INTERPRETER_BORROWED
        assert _TANKPIT in excinfo.value.message
        assert _CLEARGBM in excinfo.value.message

    def test_an_image_venv_over_the_images_own_python_passes(self) -> None:
        """/opt/env -> /usr/local is inside one digest; base_prefix != env_path is fine."""
        identity = _identity("/usr/local", on_root=True)
        check_interpreter_home(identity, env_path="/opt/env", image=_IMAGE)

    def test_an_image_venv_over_a_mounted_python_is_refused(self) -> None:
        """/dfs6b is mounted unbound, so the bind list could never have caught this."""
        with pytest.raises(AppError) as excinfo:
            check_interpreter_home(
                _identity(f"/dfs6b{_CLEARGBM}", on_root=False), env_path="/opt/env", image=_IMAGE
            )
        assert excinfo.value.code is Hpc3ErrorCode.ENV_INTERPRETER_BORROWED
        assert _IMAGE["path"] in excinfo.value.message
        assert f"/dfs6b{_CLEARGBM}" in excinfo.value.message

    def test_the_host_rule_is_not_applied_inside_an_image(self) -> None:
        """Held to base_prefix == env_path, all eight registered projects would fail."""
        identity = _identity("/usr/local", on_root=True)
        with pytest.raises(AppError):
            check_interpreter_home(identity, env_path="/opt/env", image=None)
        check_interpreter_home(identity, env_path="/opt/env", image=_IMAGE)
