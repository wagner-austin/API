"""The toolchain probe's ``gpu`` line (MCPs board tasks 939ec5c7, 28c43011, 8092e6a0).

The ``gpu`` tag routes the CUDA suites, so the line answers nvidia-smi's first
device only when nvidia-smi itself succeeded. colossus carries
nvidia-utils-595 beside an Intel Arc Pro B60, and its nvidia-smi prints its
refusal on stdout and exits 9; piped straight into ``head`` that was the
pipe's success, and colossus claimed with the gpu tag on 2026-10-09. On a node
whose docker lane answered, the card must reach that lane too: lavender and
sedona carried the driver and no NVIDIA container toolkit, and chat/execution's
llama case, which asks compose for ``gpus: all``, failed on both tags there.
The line is RUN under ``sh`` with an ``nvidia-smi`` ahead on PATH that answers
as a two-GPU node's does, or as colossus's does, and an ``id`` and ``sudo``
that stand for no execdocker user or for a lane whose daemon discovers the
devices given.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.detection import GPU_CDI_DEVICE
from fleet.core.dialect import PROLOGUE, SH_INVOCATION, TOOLCHAIN_PROBE_SCRIPT
from tests.test_dialect_linux import fields_of
from tests.test_dialect_linux_docker import ROOTLESS_ANSWER

#: colossus's nvidia-smi on 2026-10-09, read over ssh: stdout, then exit 9.
COLOSSUS_REFUSAL = (
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
    "Make sure that the latest NVIDIA driver is installed and running."
)

#: lavender's rootless daemon's discovered devices once it carried the toolkit,
#: 2026-10-27: ``docker info --format '{{json .DiscoveredDevices}}'``.
LAVENDER_DEVICES = (
    '[{"Source":"cdi","ID":"nvidia.com/gpu=0"},'
    '{"Source":"cdi","ID":"nvidia.com/gpu=GPU-0cf6a044-af9a-84f2-95a8-a84e49e77ede"},'
    '{"Source":"cdi","ID":"nvidia.com/gpu=all"}]'
)

#: sedona's before it carried the toolkit, the same day.
NO_DEVICES = "null"

#: An ``id`` that knows no execdocker user: a node with no docker lane.
NO_LANE = "#!/bin/sh\nexit 1\n"

#: An ``id`` that knows execdocker as uid 1001.
LANE = '#!/bin/sh\n[ "$1" = "-u" ] && echo 1001\nexit 0\n'


def _lane_sudo(devices: str) -> str:
    """A ``sudo`` answering as a rootless lane with its plugins and these devices.

    Args:
        devices: What ``docker info`` answers for ``.DiscoveredDevices``.

    Returns:
        The fake's script.
    """
    return (
        "#!/bin/sh\n"
        f'case "$*" in *DiscoveredDevices*) printf \'%s\\n\' \'{devices}\' ;;'
        f" *) printf '%s' '{ROOTLESS_ANSWER}' ;; esac\n"
    )


def _probe_fields(
    tmp_path: pathlib.Path,
    answer: str,
    status: int,
    lane: tuple[str, str] = (NO_LANE, "#!/bin/sh\nexit 1\n"),
) -> dict[str, str]:
    """Run the toolchain probe with a fake ``nvidia-smi``, ``id`` and ``sudo`` first on PATH.

    Args:
        tmp_path: Where the fakes and the script go.
        answer: What the fake ``nvidia-smi`` prints on stdout.
        status: What it exits with.
        lane: The fake ``id`` and ``sudo``: no docker lane unless given.

    Returns:
        The probe's fields.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    for name, body in (
        ("nvidia-smi", f"#!/bin/sh\ncat <<'EOF'\n{answer}\nEOF\nexit {status}\n"),
        ("id", lane[0]),
        ("sudo", lane[1]),
    ):
        fake = tools / name
        fake.write_bytes(body.encode())
        fake.chmod(0o755)
    script = tmp_path / "probe.sh"
    script.write_text(
        TOOLCHAIN_PROBE_SCRIPT.replace(
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
        ' g="$(printf \'%s\\n\' "$g" | head -n 1)" && [ -n "$g" ] &&' in TOOLCHAIN_PROBE_SCRIPT
    )


def test_a_docker_lane_must_discover_the_cdi_device_the_toolkit_writes() -> None:
    assert GPU_CDI_DEVICE == "nvidia.com/gpu=all"
    assert (
        " info --format '{{json .DiscoveredDevices}}' 2>/dev/null"
        " | grep -qF '\"nvidia.com/gpu=all\"'; }; then\n" in TOOLCHAIN_PROBE_SCRIPT
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


@pytest.mark.host_linux
class TestTheGpuLineOnADockerLane:
    def test_a_lane_whose_daemon_discovers_the_card_answers_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """lavender once it carried NVIDIA's container toolkit."""
        fields = _probe_fields(
            tmp_path, "NVIDIA GeForce GTX 1630, 7.5", 0, (LANE, _lane_sudo(LAVENDER_DEVICES))
        )

        assert fields["docker"] == "yes=29.8.1"
        assert fields["gpu"] == "yes=NVIDIA GeForce GTX 1630, 7.5"

    def test_a_lane_whose_daemon_discovers_no_card_answers_no(
        self, tmp_path: pathlib.Path
    ) -> None:
        """sedona with the driver and no toolkit: nvidia-smi answers, the lane cannot."""
        fields = _probe_fields(
            tmp_path,
            "NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8.6",
            0,
            (LANE, _lane_sudo(NO_DEVICES)),
        )

        assert (fields["docker"], fields["gpu"]) == ("yes=29.8.1", "no=")
