"""The toolchain probe's ``gpu`` line (MCPs board tasks 939ec5c7, 28c43011).

The ``gpu`` tag routes the CUDA suites, so the line answers nvidia-smi's first
device only when nvidia-smi itself succeeded. colossus carries
nvidia-utils-595 beside an Intel Arc Pro B60, and its nvidia-smi prints its
refusal on stdout and exits 9; piped straight into ``head`` that was the
pipe's success, and colossus claimed with the gpu tag on 2026-10-09. The line
is RUN under ``sh`` with an ``nvidia-smi`` ahead on PATH that answers as a
two-GPU node's does, or as colossus's does.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, LinuxDialect
from tests.test_dialect_linux import fields_of

DIALECT = LinuxDialect()

#: colossus's nvidia-smi on 2026-10-09, read over ssh: stdout, then exit 9.
COLOSSUS_REFUSAL = (
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
    "Make sure that the latest NVIDIA driver is installed and running."
)


def _probe_fields(tmp_path: pathlib.Path, answer: str, status: int) -> dict[str, str]:
    """Run the toolchain probe with a fake ``nvidia-smi`` first on PATH.

    Args:
        tmp_path: Where the fake and the script go.
        answer: What the fake prints on stdout.
        status: What the fake exits with.

    Returns:
        The probe's fields.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = tools / "nvidia-smi"
    fake.write_bytes(f"#!/bin/sh\ncat <<'EOF'\n{answer}\nEOF\nexit {status}\n".encode())
    fake.chmod(0o755)
    script = tmp_path / "probe.sh"
    script.write_text(
        DIALECT.toolchain_probe_script().replace(
            PROLOGUE, PROLOGUE + f"PATH='{tools.as_posix()}':$PATH\n", 1
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)


def test_the_gpu_line_takes_the_first_device_only_after_nvidia_smi_succeeded() -> None:
    assert (
        ' g="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)" &&'
        ' g="$(printf \'%s\\n\' "$g" | head -n 1)" && [ -n "$g" ]; then\n'
        in DIALECT.toolchain_probe_script()
    )


@pytest.mark.host_linux
class TestTheGpuLineUnderSh:
    def test_a_working_nvidia_smi_answers_its_first_device(self, tmp_path: pathlib.Path) -> None:
        fields = _probe_fields(
            tmp_path, "NVIDIA RTX A2000 12GB, 8.6\nNVIDIA GeForce GTX 1630, 7.5", 0
        )

        assert fields["gpu"] == "yes=NVIDIA RTX A2000 12GB, 8.6"

    def test_an_nvidia_smi_that_fails_answers_no_whatever_it_printed(
        self, tmp_path: pathlib.Path
    ) -> None:
        """colossus, 2026-10-09: nvidia-utils with no NVIDIA driver."""
        assert _probe_fields(tmp_path, COLOSSUS_REFUSAL, 9)["gpu"] == "no="

    def test_an_nvidia_smi_listing_nothing_answers_no(self, tmp_path: pathlib.Path) -> None:
        assert _probe_fields(tmp_path, "", 0)["gpu"] == "no="
