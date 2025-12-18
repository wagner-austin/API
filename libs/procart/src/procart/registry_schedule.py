from __future__ import annotations

from typing import Final, Protocol


class ParamSchedule(Protocol):
    """Callable producing a parameter value for a normalized time t.

    Args:
        t_normalized: Time in [0, 1].

    Returns:
        float: Parameter value.
    """

    def __call__(self, t_normalized: float) -> float: ...


_NAMES: Final[tuple[str, ...]] = (
    "constant",
    "linear",
)


def list_available_param_schedules() -> list[str]:
    return list(_NAMES)


def get_param_schedule(name: str, *, start: float = 0.0, end: float = 1.0) -> ParamSchedule:
    if name == "constant":

        def _const(t_normalized: float) -> float:
            return float(start)

        return _const
    if name == "linear":

        def _lin(t_normalized: float) -> float:
            t = float(t_normalized)
            t0 = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            return float(start) + (float(end) - float(start)) * t0

        return _lin
    raise ValueError(f"unknown schedule: {name}")


__all__ = [
    "ParamSchedule",
    "get_param_schedule",
    "list_available_param_schedules",
]
