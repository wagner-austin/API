"""The capacity probe's GPU lines: each display device and its memory (MCPs board task b4d5ca0f).

WHAT IS A GPU HERE. Every PCI device whose class is display (``0x03....``),
which is what the kernel itself calls one: discrete cards, integrated GPUs
and a server's management VGA alike. Enumerating the bus rather than asking
nvidia-smi is what lets a node with an Intel or AMD card, or an NVIDIA card
with no driver bound, report it at all; measured over ssh at 00:2xZ on
2026-11-02, colossus's Arc Pro B60 (xe), pendragon's amdgpu, diphtheria's
unbound RTX A2000 and seven integrated or server GPUs were invisible to the
nvidia-smi probe the inventory has.

WHERE THE MEMORY COMES FROM (:mod:`fleet.contracts.gpu_reading` has the
measurement). A device nvidia-smi answers for, matched by bus id, is read by
its ``memory.total`` and ``memory.free`` in mebibytes; any other device by
the dmem cgroup controller's regions named ``drm/<its address>/vram...``,
summed, ``dmem.capacity`` less the root cgroup's ``dmem.current``. A device
neither holds prints its source and byte counts empty, which
:func:`read_gpus` reads as no memory, never as zero.

THE OUTPUT. ``gpu_<n>=`` then the address, the PCI vendor id, the driver,
the source, the total and free bytes and the name, separated by tabs, the
name last because only it may hold spaces or commas; then ``gpu_count=``.
The count is printed after the devices, so a probe that ended among them
prints none and is refused as missing it.

THE ROOTS ARE VARIABLES ON THEIR OWN LINES, as the rest of the probe's are,
so a test points them at a fake sysfs and a fake cgroup tree and runs the
script under ``sh``; ``nvidia-smi`` and ``lspci`` are found on ``PATH``.
"""

from __future__ import annotations

import re
from typing import Final

from fleet.contracts.gpu_reading import GpuMemory, GpuReading, GpuSource, gpu_problem, vendor_of

#: Where the probe enumerates PCI devices; the test replaces it.
PCI_DEVICES_LINE: Final = "pci_devices=/sys/bus/pci/devices\n"

#: Where the probe reads the dmem cgroup controller's regions; the test
#: replaces it.
DMEM_LINE: Final = "dmem=/sys/fs/cgroup\n"

#: The field counting the devices, printed after them.
COUNT_FIELD: Final = "gpu_count"

#: The prefix of each device's field.
GPU_FIELD_PREFIX: Final = "gpu_"

#: nvidia-smi's rows: bus id, total and free mebibytes, then the name,
#: which may itself hold the separator, so it is rejoined.
_SMI_ROW = (
    "awk -F', ' -v a=\"$address\" 'tolower(substr($1, length($1) - 11)) == a "
    '{ name = $4; for (i = 5; i <= NF; i++) name = name ", " $i; '
    'printf "nvidia-smi\\t%.0f\\t%.0f\\t%s\\n", $2 * 1048576, $3 * 1048576, name; exit }\''
)

#: The device's dmem regions summed: capacity from the first file, what is
#: allocated from the second, nothing printed when the first names no region
#: of it. Told apart by name, so an empty first file cannot make the second
#: read as capacity.
_DMEM_ROW = (
    'awk -v r="drm/$address/vram" \'index($1, r) != 1 { next } '
    "FILENAME == ARGV[1] { total += $2; seen = 1; next } { used += $2 } "
    'END { if (seen) printf "dmem\\t%.0f\\t%.0f\\n", total, total - used }\' '
    '"$dmem/dmem.capacity" "$dmem/dmem.current"'
)

#: The probe's GPU lines, verbatim.
GPU_PROBE_PART: Final = (
    PCI_DEVICES_LINE
    + DMEM_LINE
    + "tab=$(printf '\\t')\n"
    + "smi=$(nvidia-smi --query-gpu=pci.bus_id,memory.total,memory.free,name"
    + " --format=csv,noheader,nounits 2>/dev/null) || smi=''\n"
    + "gpus=0\n"
    + 'for device in "$pci_devices"/*; do\n'
    + '  [ -r "$device/class" ] || continue\n'
    + '  case "$(cat "$device/class")" in 0x03*) ;; *) continue ;; esac\n'
    + '  address="${device##*/}"\n'
    + '  vendor="$(cat "$device/vendor")"\n'
    + "  driver=''\n"
    + '  if [ -e "$device/driver" ]; then '
    + 'driver="$(basename "$(readlink -f "$device/driver")")"; fi\n'
    + f'  row="$(printf \'%s\\n\' "$smi" | {_SMI_ROW})"\n'
    + '  if [ -z "$row" ] && [ -r "$dmem/dmem.capacity" ]; then\n'
    + f'    row="$({_DMEM_ROW})"\n'
    + "  fi\n"
    + "  source='' total='' free='' name=''\n"
    + '  if [ -n "$row" ]; then\n'
    + '    IFS="$tab" read -r source total free name <<EOF\n'
    + "$row\n"
    + "EOF\n"
    + "  fi\n"
    + '  if [ -z "$name" ]; then '
    + 'name="$(lspci -mm -s "$address" | awk -F\'"\' \'{ print $6 }\')"; fi\n'
    + '  printf \'gpu_%s=%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n\' "$gpus" "$address" "$vendor" '
    + '"$driver" "$source" "$total" "$free" "$name"\n'
    + "  gpus=$((gpus + 1))\n"
    + "done\n"
    + f"printf '{COUNT_FIELD}=%s\\n' \"$gpus\"\n"
)

#: A whole number as the probe prints it.
_WHOLE = re.compile(r"0|[1-9][0-9]*")


def _memory(address: str, source: str, total: str, free: str) -> tuple[GpuMemory | None, str]:
    """Read one device's memory fields.

    Args:
        address: The device, for the message.
        source: The source field; empty when neither counter holds it.
        total: The total bytes field.
        free: The free bytes field.

    Returns:
        The memory, or None when all three are empty, and an empty string;
        or None and why they cannot be read.
    """
    if source == "" and total == "" and free == "":
        return None, ""
    if source not in {member.value for member in GpuSource}:
        return None, f"{address}'s memory source must be nvidia-smi or dmem, got {source!r}"
    if _WHOLE.fullmatch(total) is None or _WHOLE.fullmatch(free) is None:
        return (
            None,
            f"{address}'s memory must be two whole numbers of bytes, got {total!r} and {free!r}",
        )
    return GpuMemory(source=GpuSource(source), total_bytes=int(total), free_bytes=int(free)), ""


def _gpu(line: str) -> tuple[GpuReading | None, str]:
    """Read one device's line.

    Args:
        line: Its value, after ``gpu_<n>=``.

    Returns:
        The GPU and an empty string, or None and why the line cannot be
        read: not seven tab-separated fields, a vendor id that is not one,
        memory fields :func:`_memory` refuses, or a GPU
        :func:`fleet.contracts.gpu_reading.gpu_problem` refuses.
    """
    parts = line.split("\t")
    if len(parts) != 7:
        return None, f"a GPU line holds 7 tab-separated fields, got {len(parts)}: {line!r}"
    address, vendor_id, driver, source, total, free, name = parts
    vendor = vendor_of(vendor_id)
    if vendor is None:
        return None, f"{address}'s PCI vendor id must be 0x and four hex digits, got {vendor_id!r}"
    memory, problem = _memory(address, source, total, free)
    if problem:
        return None, problem
    gpu = GpuReading(
        address=address,
        vendor=vendor,
        name=name,
        driver=driver if driver != "" else None,
        memory=memory,
    )
    shape = gpu_problem(gpu)
    if shape is not None:
        return None, shape
    return gpu, ""


def read_gpus(host: str, fields: dict[str, str]) -> tuple[tuple[GpuReading, ...] | None, str]:
    """Read every GPU a probe printed.

    A node whose GPUs cannot be read is refused rather than read as having
    none, the way :mod:`fleet.core.probe` refuses every other reading.

    Args:
        host: The node, for the message.
        fields: The probe's key=value pairs.

    Returns:
        The GPUs in the order the probe read them (empty for a node with
        none) and an empty string; or None and the reason, naming the node,
        when the count is missing or not a whole number, a device it counts
        printed no line, or a line cannot be read.
    """
    count = fields.get(COUNT_FIELD)
    if count is None or _WHOLE.fullmatch(count) is None:
        return None, f"{host} reported {COUNT_FIELD}={count!r}, which is not a whole number of GPUs"
    gpus: list[GpuReading] = []
    for index in range(int(count)):
        line = fields.get(f"{GPU_FIELD_PREFIX}{index}")
        if line is None:
            return (
                None,
                f"{host} counted {count} GPU(s) and printed no {GPU_FIELD_PREFIX}{index} line",
            )
        gpu, problem = _gpu(line)
        if gpu is None:
            return None, f"{host} printed a GPU line that cannot be read: {problem}"
        gpus.append(gpu)
    return tuple(gpus), ""


__all__ = [
    "COUNT_FIELD",
    "DMEM_LINE",
    "GPU_FIELD_PREFIX",
    "GPU_PROBE_PART",
    "PCI_DEVICES_LINE",
    "read_gpus",
]
