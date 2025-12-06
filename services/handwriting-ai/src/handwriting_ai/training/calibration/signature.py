from __future__ import annotations

import platform
from typing import TypedDict

import torch

from handwriting_ai.training.resources import ResourceLimits


class CalibrationSignature(TypedDict):
    cpu_cores: int
    mem_bytes: int | None
    os: str
    py: str
    torch: str


def make_signature(limits: ResourceLimits) -> CalibrationSignature:
    return {
        "cpu_cores": int(limits["cpu_cores"]),
        "mem_bytes": limits["memory_bytes"],
        "os": f"{platform.system()}-{platform.release()}",
        "py": platform.python_version(),
        "torch": str(torch.__version__),
    }
