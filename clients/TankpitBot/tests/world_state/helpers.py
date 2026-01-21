"""Shared test helpers for world state tests."""

from tankpit_bot.state import SelfStateDict, WorldStateDict


def get_self_state(state: WorldStateDict) -> SelfStateDict:
    """Extract self_state from world state, raising if None.

    Test helper for type narrowing.
    """
    result = state["self_state"]
    if result is None:
        raise AssertionError("self_state is None")
    return result
