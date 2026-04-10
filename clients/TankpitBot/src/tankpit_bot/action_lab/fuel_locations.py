"""Distinct ground-target selection for fuel action probes."""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.types import TeleportTargetDict

_MAP_MIN = 0
_MAP_MAX = 255


def _clamp_tile(value: int) -> int:
    """Clamp a map tile coordinate to the valid inclusive range.

    Args:
        value: Raw coordinate.

    Returns:
        Coordinate clamped into ``0..255``.
    """
    if value < _MAP_MIN:
        return _MAP_MIN
    if value > _MAP_MAX:
        return _MAP_MAX
    return value


def _neighbor_ground_score(terrain: TerrainMapProtocol, x: int, y: int) -> int:
    """Count passable tiles in the local 3x3 neighborhood.

    Args:
        terrain: Terrain map used for passability checks.
        x: Candidate X coordinate.
        y: Candidate Y coordinate.

    Returns:
        Number of passable tiles in the candidate's 3x3 neighborhood.
    """
    score = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx = x + dx
            ny = y + dy
            if not (_MAP_MIN <= nx <= _MAP_MAX and _MAP_MIN <= ny <= _MAP_MAX):
                continue
            if terrain.is_passable(nx, ny):
                score += 1
    return score


def _iter_ring_candidates(
    origin_x: int,
    origin_y: int,
    *,
    step: int,
    max_radius: int,
) -> list[tuple[int, int]]:
    """Build unique perimeter-sampled candidate coordinates.

    Args:
        origin_x: Starting X coordinate.
        origin_y: Starting Y coordinate.
        step: Sampling interval for each ring.
        max_radius: Furthest ring distance to inspect.

    Returns:
        Unique candidate coordinates in ring iteration order.
    """
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    for radius in range(step, max_radius + step, step):
        for dx in range(-radius, radius + step, step):
            for dy in range(-radius, radius + step, step):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                candidate = (_clamp_tile(origin_x + dx), _clamp_tile(origin_y + dy))
                if candidate == (origin_x, origin_y):
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _score_target(
    origin_x: int,
    origin_y: int,
    terrain: TerrainMapProtocol,
    *,
    target_x: int,
    target_y: int,
) -> tuple[tuple[int, int, int, int, int], TeleportTargetDict] | None:
    """Score a candidate target or reject it if not passable.

    Args:
        origin_x: Starting X coordinate.
        origin_y: Starting Y coordinate.
        terrain: Terrain map used for passability checks.
        target_x: Candidate target X coordinate.
        target_y: Candidate target Y coordinate.

    Returns:
        Scored target tuple, or None when the tile is not passable.
    """
    if not terrain.is_passable(target_x, target_y):
        return None
    neighborhood_score = _neighbor_ground_score(terrain, target_x, target_y)
    all_ground_priority = 0 if neighborhood_score == 9 else 1
    distance = abs(target_x - origin_x) + abs(target_y - origin_y)
    return (
        (
            all_ground_priority,
            -neighborhood_score,
            distance,
            target_y,
            target_x,
        ),
        TeleportTargetDict(
            label=f"fuel_ground_{target_x}_{target_y}",
            x=target_x,
            y=target_y,
        ),
    )


def build_distinct_ground_targets(
    origin_x: int,
    origin_y: int,
    terrain: TerrainMapProtocol,
    *,
    count: int,
    step: int = 24,
    max_radius: int = 96,
    excluded: frozenset[tuple[int, int]] | None = None,
) -> list[TeleportTargetDict]:
    """Return distinct ground-heavy teleport targets around an origin.

    Candidates are sampled on concentric square perimeters around ``origin``.
    Targets are sorted to prefer fully passable 3x3 neighborhoods, then the
    highest local ground score, then shorter travel distance.

    Args:
        origin_x: Starting X coordinate.
        origin_y: Starting Y coordinate.
        terrain: Terrain map used to reject water and rock.
        count: Number of distinct targets to return.
        step: Sampling interval for each ring.
        max_radius: Furthest ring distance to inspect.
        excluded: Optional coordinates that must not be returned.

    Returns:
        Distinct teleport targets ordered best-first.

    Raises:
        ValueError: If ``count`` is not positive, ``step`` is not positive,
            ``max_radius`` is smaller than ``step``, or not enough passable
            targets can be found.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if max_radius < step:
        raise ValueError("max_radius must be at least step")

    scored: list[tuple[tuple[int, int, int, int, int], TeleportTargetDict]] = []
    for target_x, target_y in _iter_ring_candidates(
        origin_x,
        origin_y,
        step=step,
        max_radius=max_radius,
    ):
        scored_target = _score_target(
            origin_x,
            origin_y,
            terrain,
            target_x=target_x,
            target_y=target_y,
        )
        if scored_target is not None:
            if excluded is not None and (target_x, target_y) in excluded:
                continue
            scored.append(scored_target)

    scored.sort(key=_target_rank_key)
    targets = [target for _, target in scored[:count]]
    if len(targets) < count:
        raise ValueError("not enough distinct passable targets for fuel probe")
    return targets


def _target_rank_key(
    item: tuple[tuple[int, int, int, int, int], TeleportTargetDict],
) -> tuple[int, int, int, int, int]:
    """Return the rank key portion of a scored target tuple.

    Args:
        item: Scored target item.

    Returns:
        Rank tuple used for deterministic sorting.
    """
    return item[0]


__all__ = ["build_distinct_ground_targets"]
