from __future__ import annotations

# Recursive JSON type - forward references for self-reference only
UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None

__all__ = ["UnknownJson"]
