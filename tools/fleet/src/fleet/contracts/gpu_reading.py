"""Every GPU a node runner's probe read, with its memory in bytes (MCPs board task b4d5ca0f).

The chat app places a local model on a node with room for it through the
dispatch queue (the operator, 2026-11-01: "it will also need to
automatically load into a pc that has the resoruces for it and through the
job queue system ."), so each tick carries every display device on its node
to MCPs ``dispatch_tick`` (MCPs mig 895), where placement reads it.

WHERE THE MEMORY COMES FROM. A card nvidia-smi answers for is read by its
``memory.total`` and ``memory.free``. Any other card is read from the kernel's
dmem cgroup controller: ``/sys/fs/cgroup/dmem.capacity`` holds each device
memory region's size and the root cgroup's ``dmem.current`` what is allocated
in it, both readable by any account. Measured on colossus at 00:16Z on
2026-11-02 (Arc Pro B60, xe driver, kernel 7.3.0), with llama-server holding a
17 GB model: 25669140480 and 17122439168 bytes, and the xe driver's own VRAM
manager (debugfs ``tile0/vram_mm``, root only) read size 25669140480 and
usage 17122439168. pendragon's amdgpu publishes a region the same way. A card
neither holds (diphtheria's NVIDIA card with no driver bound, an integrated
GPU with no memory of its own) carries no memory, never a zero, which would
read as a full card.

The vendor is named by the PCI vendor id, so placement can tell which runtime
a card takes without parsing a marketing name.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_str,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

#: The PCI vendor ids the fleet holds cards of, and the names a tick gives them.
NAMED_VENDORS: Final[dict[str, str]] = {"0x10de": "nvidia", "0x8086": "intel", "0x1002": "amd"}

#: A PCI address as sysfs spells it, ``0000:08:00.0``; MCPs mig 895's CHECK.
ADDRESS_PATTERN: Final = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")

#: A vendor as a tick names it; MCPs mig 895's CHECK.
VENDOR_PATTERN: Final = re.compile(r"nvidia|intel|amd|pci-[0-9a-f]{4}")

#: A kernel driver's name, as a device's ``driver`` link names it.
DRIVER_PATTERN: Final = re.compile(r"[A-Za-z0-9_.-]{1,64}")

#: The longest device name MCPs stores.
NAME_MAX: Final = 200


class GpuSource(StrEnum):
    """Where a GPU's memory was read."""

    NVIDIA_SMI = "nvidia-smi"
    DMEM = "dmem"


class GpuMemory(TypedDict):
    """A GPU's device memory as a driver counter read it.

    Attributes:
        source: The counter.
        total_bytes: The memory's size; above 0.
        free_bytes: What is not allocated of it; 0 to ``total_bytes``.
    """

    source: GpuSource
    total_bytes: int
    free_bytes: int


class GpuReading(TypedDict):
    """One display device on a node, as its probe read it.

    Attributes:
        address: Its PCI address, :data:`ADDRESS_PATTERN`.
        vendor: :data:`VENDOR_PATTERN`, from :func:`vendor_of`.
        name: nvidia-smi's name for a card it reads, else lspci's.
        driver: The kernel driver bound to it, or None when none is.
        memory: Its memory, or None when neither counter holds it.
    """

    address: str
    vendor: str
    name: str
    driver: str | None
    memory: GpuMemory | None


def vendor_of(vendor_id: str) -> str | None:
    """Name a PCI vendor id as a tick carries it.

    Args:
        vendor_id: The device's ``vendor`` file, e.g. ``0x8086``.

    Returns:
        ``nvidia``, ``intel`` or ``amd`` for the ids in
        :data:`NAMED_VENDORS`, ``pci-<the four hex digits>`` for any other,
        or None when it is not ``0x`` and four hex digits.
    """
    if re.fullmatch(r"0x[0-9a-f]{4}", vendor_id) is None:
        return None
    return NAMED_VENDORS.get(vendor_id, f"pci-{vendor_id[2:]}")


def gpu_problem(gpu: GpuReading) -> str | None:
    """What puts a GPU outside the shape MCPs mig 895 stores, if anything.

    Args:
        gpu: The GPU.

    Returns:
        None when it is inside it, or a sentence naming the address, vendor,
        name, driver or memory that is not.
    """
    if ADDRESS_PATTERN.fullmatch(gpu["address"]) is None:
        return f"a GPU's address must be a PCI address, got {gpu['address']!r}"
    if VENDOR_PATTERN.fullmatch(gpu["vendor"]) is None:
        return f"{gpu['address']}'s vendor is outside the vocabulary: {gpu['vendor']!r}"
    if not 1 <= len(gpu["name"]) <= NAME_MAX:
        return f"{gpu['address']}'s name must be 1 to {NAME_MAX} characters, got {len(gpu['name'])}"
    driver = gpu["driver"]
    if driver is not None and DRIVER_PATTERN.fullmatch(driver) is None:
        return f"{gpu['address']}'s driver is not a driver's name: {driver!r}"
    memory = gpu["memory"]
    if memory is not None and not (0 <= memory["free_bytes"] <= memory["total_bytes"] > 0):
        return (
            f"{gpu['address']}'s memory must have a total above 0 and a free from 0 to the total, "
            f"got {memory['free_bytes']} free of {memory['total_bytes']}"
        )
    return None


def require_gpu(gpu: GpuReading) -> GpuReading:
    """Refuse a GPU outside the shape MCPs mig 895 stores.

    Args:
        gpu: The GPU.

    Returns:
        The same GPU.

    Raises:
        JSONTypeError: As :func:`gpu_problem` describes.
    """
    problem = gpu_problem(gpu)
    if problem is not None:
        raise JSONTypeError(problem)
    return gpu


def encode_gpus(gpus: tuple[GpuReading, ...]) -> list[JSONValue]:
    """The GPUs as ``dispatch_tick``'s load carries them.

    Args:
        gpus: The GPUs, in the order the probe read them.

    Returns:
        One ``{"address", "vendor", "name", "driver", "memory"}`` object per
        GPU, ``memory`` being ``{"source", "totalBytes", "freeBytes"}`` or
        null.
    """
    encoded: list[JSONValue] = []
    for gpu in gpus:
        memory = gpu["memory"]
        encoded_memory: JSONObject | None = (
            None
            if memory is None
            else {
                "source": memory["source"].value,
                "totalBytes": memory["total_bytes"],
                "freeBytes": memory["free_bytes"],
            }
        )
        encoded.append(
            {
                "address": gpu["address"],
                "vendor": gpu["vendor"],
                "name": gpu["name"],
                "driver": gpu["driver"],
                "memory": encoded_memory,
            }
        )
    return encoded


def _decode_memory(value: JSONValue) -> GpuMemory | None:
    """Decode one GPU's memory from the shape :func:`encode_gpus` writes.

    Args:
        value: The encoded memory, or None.

    Returns:
        The memory, or None.

    Raises:
        JSONTypeError: When it is neither an object nor None, or a field is
            missing, mistyped or outside :class:`GpuSource`.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"a GPU's memory must be a JSON object or null, got {type(value).__name__}"
        )
    source = require_str(value, "source")
    if source not in {member.value for member in GpuSource}:
        raise JSONTypeError(f"a GPU's memory source must be nvidia-smi or dmem, got {source!r}")
    return GpuMemory(
        source=GpuSource(source),
        total_bytes=require_int(value, "totalBytes"),
        free_bytes=require_int(value, "freeBytes"),
    )


def decode_gpus(value: JSONValue) -> tuple[GpuReading, ...]:
    """Decode the GPUs from the shape :func:`encode_gpus` writes.

    Args:
        value: The encoded list.

    Returns:
        The GPUs, in order.

    Raises:
        JSONTypeError: When it is not a list of objects, a field is missing
            or mistyped, or :func:`require_gpu` refuses a GPU.
    """
    if not isinstance(value, list):
        raise JSONTypeError(f"gpus must be a JSON list, got {type(value).__name__}")
    gpus: list[GpuReading] = []
    for element in value:
        if not isinstance(element, dict):
            raise JSONTypeError(f"a GPU must be a JSON object, got {type(element).__name__}")
        if "memory" not in element:
            raise JSONTypeError("a GPU's memory is missing; a tick carries it or null")
        gpus.append(
            require_gpu(
                GpuReading(
                    address=require_str(element, "address"),
                    vendor=require_str(element, "vendor"),
                    name=require_str(element, "name"),
                    driver=optional_str(element, "driver"),
                    memory=_decode_memory(element["memory"]),
                )
            )
        )
    return tuple(gpus)


__all__ = [
    "ADDRESS_PATTERN",
    "DRIVER_PATTERN",
    "NAMED_VENDORS",
    "NAME_MAX",
    "VENDOR_PATTERN",
    "GpuMemory",
    "GpuReading",
    "GpuSource",
    "decode_gpus",
    "encode_gpus",
    "gpu_problem",
    "require_gpu",
    "vendor_of",
]
