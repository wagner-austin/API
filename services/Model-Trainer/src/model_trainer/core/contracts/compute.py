from __future__ import annotations

from typing import Literal, Protocol


class ComputeProvider(Protocol):
    def kind(self: ComputeProvider) -> Literal["local-cpu", "cloud"]: ...
    def threads(self: ComputeProvider) -> int: ...
    def env(self: ComputeProvider) -> dict[str, str]: ...


class LocalCPUProvider:
    def __init__(self: LocalCPUProvider, *, threads_count: int) -> None:
        self._threads = max(1, int(threads_count))

    def kind(self: LocalCPUProvider) -> Literal["local-cpu", "cloud"]:
        return "local-cpu"

    def threads(self: LocalCPUProvider) -> int:
        return self._threads

    def env(self: LocalCPUProvider) -> dict[str, str]:
        return {
            "OMP_NUM_THREADS": str(self._threads),
            "MKL_NUM_THREADS": str(self._threads),
        }
