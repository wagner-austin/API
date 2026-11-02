"""The capacity probe's GPU lines and their parser (MCPs board task b4d5ca0f).

The lines are RUN under ``sh`` against a fake PCI bus, a fake dmem cgroup
tree and a fake ``nvidia-smi`` and ``lspci`` first on ``PATH``, each shaped
like what the roster's machines answered over ssh at 00:2xZ on 2026-11-02:
colossus's Arc Pro B60 under xe with llama-server holding a 17 GB model,
demure's integrated UHD 770 beside its RTX 3090 Ti, diphtheria's RTX A2000
with no driver bound and an nvidia-smi that fails, and pendragon's amdgpu
region, which xe's differs from in name. The fakes are the inputs; the shell
that reads them is the probe's own text.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.gpu_reading import GpuMemory, GpuReading, GpuSource
from fleet.core import probe
from fleet.core.dialect import PROLOGUE, SH_INVOCATION
from fleet.core.linux_gpu_probe import DMEM_LINE, GPU_PROBE_PART, PCI_DEVICES_LINE, read_gpus
from tests.conftest import IDLE, PROBE_OK
from tests.test_dialect_linux import fields_of

#: colossus's card, as lspci -mm names it.
B60_LSPCI = (
    '08:00.0 "VGA compatible controller" "Intel Corporation" "Battlemage G21 [Arc Pro B60]" '
    '-p00 "Sparkle Computer Co., Ltd." "Device 0300"'
)

#: colossus's dmem regions at 00:16Z on 2026-11-02, capacity then current.
COLOSSUS_DMEM = (
    "drm/0000:08:00.0/vram0 25669140480\n",
    "drm/0000:08:00.0/vram0 17122439168\n",
)

#: colossus's card as the parser reads it: 25669140480 bytes, 17122439168 allocated.
B60 = GpuReading(
    address="0000:08:00.0",
    vendor="intel",
    name="Battlemage G21 [Arc Pro B60]",
    driver="xe",
    memory=GpuMemory(source=GpuSource.DMEM, total_bytes=25669140480, free_bytes=8546701312),
)

#: nvidia-smi where no NVIDIA driver answers (diphtheria): stdout, then exit 9.
SMI_FAILED = (
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
    "Make sure that the latest NVIDIA driver is installed and running.",
    9,
)

#: A shell's answer for a command it cannot find.
NOT_INSTALLED = ("", 127)


def _device(tmp_path: pathlib.Path, address: str, pci_class: str, vendor: str, driver: str | None) -> None:
    """Write one fake PCI device under ``tmp_path / "pci"``.

    Args:
        tmp_path: The test's directory.
        address: Its address, the directory's name.
        pci_class: Its ``class`` file.
        vendor: Its ``vendor`` file.
        driver: The driver its ``driver`` link names, or None for none bound.
    """
    device = tmp_path / "pci" / address
    device.mkdir(parents=True)
    (device / "class").write_text(f"{pci_class}\n", encoding="utf-8")
    (device / "vendor").write_text(f"{vendor}\n", encoding="utf-8")
    if driver is not None:
        bound = tmp_path / "drivers" / driver
        bound.mkdir(parents=True, exist_ok=True)
        (device / "driver").symlink_to(bound)


def _tool(tools: pathlib.Path, name: str, body: str) -> None:
    """Write one fake command.

    Args:
        tools: The directory put first on ``PATH``.
        name: The command.
        body: Its script.
    """
    tools.mkdir(exist_ok=True)
    fake = tools / name
    fake.write_bytes(body.encode())
    fake.chmod(0o755)


def _run(
    tmp_path: pathlib.Path,
    smi: tuple[str, int],
    lspci: dict[str, str],
    dmem: tuple[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    """Run the GPU lines against the fakes written under ``tmp_path``.

    Args:
        tmp_path: Holds the fake bus (:func:`_device`), and here the fake
            tools, the dmem tree and the script.
        smi: What the fake ``nvidia-smi`` prints and exits with.
        lspci: The ``lspci -mm`` line for each address it names.
        dmem: The fake ``dmem.capacity`` and ``dmem.current``, or None for a
            kernel without the controller.

    Returns:
        The finished script.
    """
    tools = tmp_path / "tools"
    answer, status = smi
    _tool(tools, "nvidia-smi", f"#!/bin/sh\ncat <<'EOF'\n{answer}\nEOF\nexit {status}\n")
    cases = "".join(f"  {address}) cat <<'EOF'\n{line}\nEOF\n  ;;\n" for address, line in lspci.items())
    _tool(tools, "lspci", f'#!/bin/sh\ncase "$3" in\n{cases}esac\n')
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    if dmem is not None:
        (cgroup / "dmem.capacity").write_text(dmem[0], encoding="utf-8")
        (cgroup / "dmem.current").write_text(dmem[1], encoding="utf-8")
    pci = tmp_path / "pci"
    pci.mkdir(exist_ok=True)
    body = GPU_PROBE_PART.replace(PCI_DEVICES_LINE, f"pci_devices='{pci.as_posix()}'\n", 1)
    script = tmp_path / "gpus.sh"
    script.write_text(
        PROLOGUE
        + f"PATH='{tools.as_posix()}':$PATH\n"
        + body.replace(DMEM_LINE, f"dmem='{cgroup.as_posix()}'\n", 1),
        encoding="utf-8",
    )
    return subprocess.run(
        [*SH_INVOCATION, str(script)], capture_output=True, text=True, check=False, timeout=60
    )


def _gpus(
    tmp_path: pathlib.Path,
    smi: tuple[str, int],
    lspci: dict[str, str],
    dmem: tuple[str, str] | None,
) -> tuple[GpuReading, ...]:
    """Run the GPU lines and read them as the probe does.

    Args:
        tmp_path: As :func:`_run` takes it.
        smi: As :func:`_run` takes it.
        lspci: As :func:`_run` takes it.
        dmem: As :func:`_run` takes it.

    Returns:
        The GPUs read.
    """
    completed = _run(tmp_path, smi, lspci, dmem)
    assert completed.returncode == 0, completed.stderr
    gpus, problem = read_gpus("node", fields_of(completed.stdout))
    assert gpus is not None, problem
    return gpus


def test_the_part_names_its_roots_on_their_own_lines() -> None:
    assert GPU_PROBE_PART.startswith(PCI_DEVICES_LINE + DMEM_LINE)
    assert PCI_DEVICES_LINE == "pci_devices=/sys/bus/pci/devices\n"
    assert DMEM_LINE == "dmem=/sys/fs/cgroup\n"
    assert GPU_PROBE_PART.endswith("printf 'gpu_count=%s\\n' \"$gpus\"\n")


@pytest.mark.host_linux
class TestTheGpuLinesUnderSh:
    def test_colossus_s_xe_card_is_read_from_the_dmem_regions(self, tmp_path: pathlib.Path) -> None:
        _device(tmp_path, "0000:08:00.0", "0x030000", "0x8086", "xe")

        completed = _run(tmp_path, NOT_INSTALLED, {"0000:08:00.0": B60_LSPCI}, COLOSSUS_DMEM)

        assert completed.returncode == 0, completed.stderr
        fields = fields_of(completed.stdout)
        assert fields["gpu_0"] == (
            "0000:08:00.0\t0x8086\txe\tdmem\t25669140480\t8546701312\tBattlemage G21 [Arc Pro B60]"
        )
        assert fields["gpu_count"] == "1"
        assert read_gpus("colossus", fields) == ((B60,), "")

    def test_demure_s_nvidia_card_is_read_by_nvidia_smi_beside_an_integrated_gpu(
        self, tmp_path: pathlib.Path
    ) -> None:
        _device(tmp_path, "0000:00:02.0", "0x038000", "0x8086", "i915")
        _device(tmp_path, "0000:01:00.0", "0x030000", "0x10de", "nvidia")
        smi = ("00000000:01:00.0, 24564, 21784, NVIDIA GeForce RTX 3090 Ti", 0)
        lspci = {
            "0000:00:02.0": '00:02.0 "Display controller" "Intel Corporation" '
            '"Raptor Lake-S GT1 [UHD Graphics 770]" -r04 -p00 "Micro-Star International Co., Ltd. [MSI]" '
            '"Device 7e06"'
        }

        assert _gpus(tmp_path, smi, lspci, ("", "")) == (
            GpuReading(
                address="0000:00:02.0",
                vendor="intel",
                name="Raptor Lake-S GT1 [UHD Graphics 770]",
                driver="i915",
                memory=None,
            ),
            GpuReading(
                address="0000:01:00.0",
                vendor="nvidia",
                name="NVIDIA GeForce RTX 3090 Ti",
                driver="nvidia",
                memory=GpuMemory(
                    source=GpuSource.NVIDIA_SMI,
                    total_bytes=24564 * 1048576,
                    free_bytes=21784 * 1048576,
                ),
            ),
        )

    def test_diphtheria_s_unbound_card_is_listed_with_no_memory(self, tmp_path: pathlib.Path) -> None:
        _device(tmp_path, "0000:01:00.0", "0x030000", "0x10de", None)
        lspci = {
            "0000:01:00.0": '01:00.0 "VGA compatible controller" "NVIDIA Corporation" '
            '"GA106 [RTX A2000 12GB]" -ra1 -p00 "NVIDIA Corporation" "GA106 [RTX A2000 12GB]"'
        }

        assert _gpus(tmp_path, SMI_FAILED, lspci, None) == (
            GpuReading(
                address="0000:01:00.0",
                vendor="nvidia",
                name="GA106 [RTX A2000 12GB]",
                driver=None,
                memory=None,
            ),
        )

    def test_pendragon_s_amdgpu_region_and_two_regions_of_one_card_are_summed(
        self, tmp_path: pathlib.Path
    ) -> None:
        _device(tmp_path, "0000:09:00.0", "0x030000", "0x1002", "amdgpu")
        _device(tmp_path, "0000:0a:00.0", "0x030000", "0x8086", "xe")
        lspci = {
            "0000:09:00.0": '09:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" '
            '"Phoenix1" -r04 -p00 "ASUSTeK Computer Inc." "Phoenix1"',
            "0000:0a:00.0": B60_LSPCI,
        }
        dmem = (
            "drm/0000:09:00.0/vram 4294967296\ndrm/0000:0a:00.0/vram0 100\ndrm/0000:0a:00.0/vram1 50\n",
            "drm/0000:09:00.0/vram 80592896\ndrm/0000:0a:00.0/vram0 30\ndrm/0000:0a:00.0/vram1 20\n",
        )

        gpus = _gpus(tmp_path, NOT_INSTALLED, lspci, dmem)

        assert [(gpu["vendor"], gpu["name"], gpu["memory"]) for gpu in gpus] == [
            (
                "amd",
                "Phoenix1",
                GpuMemory(source=GpuSource.DMEM, total_bytes=4294967296, free_bytes=4214374400),
            ),
            (
                "intel",
                "Battlemage G21 [Arc Pro B60]",
                GpuMemory(source=GpuSource.DMEM, total_bytes=150, free_bytes=100),
            ),
        ]

    def test_an_nvidia_smi_bus_id_in_capitals_and_a_name_with_a_comma_still_match(
        self, tmp_path: pathlib.Path
    ) -> None:
        _device(tmp_path, "0000:0a:00.0", "0x030000", "0x10de", "nvidia")
        smi = ("00000000:0A:00.0, 4096, 3715, NVIDIA GeForce GTX 1630, OEM", 0)

        (gpu,) = _gpus(tmp_path, smi, {}, None)

        assert gpu["name"] == "NVIDIA GeForce GTX 1630, OEM"
        assert gpu["memory"] == GpuMemory(
            source=GpuSource.NVIDIA_SMI, total_bytes=4096 * 1048576, free_bytes=3715 * 1048576
        )

    def test_a_node_with_no_display_device_counts_none(self, tmp_path: pathlib.Path) -> None:
        _device(tmp_path, "0000:02:00.0", "0x020000", "0x8086", "igc")
        (tmp_path / "pci" / "0000:03:00.0").mkdir(parents=True)

        completed = _run(tmp_path, NOT_INSTALLED, {}, None)

        assert completed.returncode == 0, completed.stderr
        assert fields_of(completed.stdout) == {"gpu_count": "0"}


class TestReadingTheLines:
    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({}, "node reported gpu_count=None, which is not a whole number of GPUs"),
            ({"gpu_count": "one"}, "node reported gpu_count='one', which is not a whole number"),
            ({"gpu_count": "1"}, "node counted 1 GPU(s) and printed no gpu_0 line"),
            ({"gpu_count": "1", "gpu_0": "0000:08:00.0\t0x8086"}, "holds 7 tab-separated fields, got 2"),
            (
                {"gpu_count": "1", "gpu_0": "0000:08:00.0\t8086\txe\t\t\t\tB60"},
                "0000:08:00.0's PCI vendor id must be 0x and four hex digits, got '8086'",
            ),
            (
                {"gpu_count": "1", "gpu_0": "0000:08:00.0\t0x8086\txe\tsysfs\t1\t1\tB60"},
                "0000:08:00.0's memory source must be nvidia-smi or dmem, got 'sysfs'",
            ),
            (
                {"gpu_count": "1", "gpu_0": "0000:08:00.0\t0x8086\txe\tdmem\t\t\tB60"},
                "0000:08:00.0's memory must be two whole numbers of bytes, got '' and ''",
            ),
            (
                {"gpu_count": "1", "gpu_0": "0000:08:00.0\t0x8086\txe\tdmem\t100\t101\tB60"},
                "0000:08:00.0's memory must have a total above 0 and a free from 0 to the total",
            ),
        ],
    )
    def test_a_count_or_line_it_cannot_read_refuses_the_node(
        self, fields: dict[str, str], message: str
    ) -> None:
        gpus, problem = read_gpus("node", fields)

        assert gpus is None
        assert message in problem

    def test_a_vendor_outside_the_named_three_is_named_by_its_id(self) -> None:
        gpus, _problem = read_gpus(
            "emerald", {"gpu_count": "1", "gpu_0": "0000:00:01.0\t0x102b\tmgag200\t\t\t\tG200eR2"}
        )

        assert gpus is not None
        assert gpus[0]["vendor"] == "pci-102b"

    def test_the_probe_carries_the_gpus_into_the_node_s_state(self) -> None:
        output = PROBE_OK.replace(
            "gpu_count=0\n",
            "gpu_0=0000:08:00.0\t0x8086\txe\tdmem\t25669140480\t8546701312\tBattlemage G21 [Arc Pro B60]\n"
            "gpu_count=1\n",
        )

        assert probe.parse_probe("colossus", output, live=IDLE)["gpus"] == (B60,)

    def test_a_probe_whose_gpus_cannot_be_read_is_refused_not_read_as_none(self) -> None:
        outcome = probe.read_state("colossus", PROBE_OK.replace("gpu_count=0\n", ""), live=IDLE)

        assert outcome["state"] is None
        assert outcome["reason"] == "colossus reported gpu_count=None, which is not a whole number of GPUs"
