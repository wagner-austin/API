"""Read-model queries the bot answers about the world.

Every method here is a QUESTION about current world state — where am
I, how much fuel, what containers are known — and none of them touches
session state, dispatches a command, or mutates anything. That is why
they separate cleanly from :mod:`tankpit_bot.bot.base`: the accessors
need no browser, no CDP session and no HFSM, only the world service.

Split from ``bot/base.py`` 2026-08-07, which had grown to four
concerns in one file (init, these queries, game log, and the session
run loop).
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict


class StateAccessMixin:
    """World-state queries, free of session and dispatch concerns."""

    def get_world_state(self) -> WorldStateDict:
        """Get current world state.

        Returns:
            Current WorldStateDict with all tracked entities.
        """
        return get_world_state()

    def get_self_state(self) -> SelfStateDict | None:
        """Get self tank state (position, fuel, etc.).

        Returns:
            SelfStateDict if available, None if not yet tracked.
        """
        return get_world_state()["self_state"]

    def get_fuel(self) -> int:
        """Get current fuel (HP).

        Returns:
            Current fuel amount, or 0 if self_state not yet tracked.
        """
        state = self.get_self_state()
        return state["fuel"] if state is not None else 0

    def get_position(self) -> tuple[int, int] | None:
        """Get current position.

        Returns:
            Tuple of (x, y) coordinates, or None if not yet tracked.
        """
        state = self.get_self_state()
        if state is None:
            return None
        return (state["x"], state["y"])

    def get_containers(self) -> dict[str, ContainerStateDict]:
        """Get all known containers.

        Returns:
            Dict of container key ("x,y") to ContainerStateDict.
        """
        return get_world_state()["containers"]

    def get_fuel_containers(self) -> list[ContainerStateDict]:
        """Get all known fuel containers (not equipment).

        Returns:
            List of fuel containers with volume > 0.
        """
        containers = self.get_containers()
        return [c for c in containers.values() if c["is_fuel"] and c["volume"] > 0]

    def get_nearest_fuel_container(self) -> ContainerStateDict | None:
        """Get nearest fuel container to current position.

        Returns:
            Nearest ContainerStateDict, or None if no containers or no position.
        """
        pos = self.get_position()
        if pos is None:
            return None

        fuel_containers = self.get_fuel_containers()
        if not fuel_containers:
            return None

        # Sort by Manhattan distance
        my_x, my_y = pos
        fuel_containers.sort(key=lambda c: abs(c["x"] - my_x) + abs(c["y"] - my_y))
        return fuel_containers[0]


__all__ = [
    "StateAccessMixin",
]
