"""Remembering what the fog has shown, so seeing once is not seeing never.

The loop's knowledge of the opponent used to be exactly the current sample's
``hostile`` entities: a helicopter scouted eight seconds ago and now fogged
did not exist, so the counter tilt reacted only to what was already in gun
range ([[mechanics-combat-profile]]). The community corpus ranks continuous
scouting as the difference between winning and losing precisely because its
value is *remembered*: the counter is chosen before the fight arrives
([[community-play-strategies]]).

This is the memory. Sightings are recorded per engine identity, refreshed on
re-sight, and forgotten after a window -- the opponent's army is not where it
was forever, and intel old enough to be wrong is worse than fog.

Pure in the sense that matters: it reads samples and holds state, opens
nothing, and what to do about a sighting is its callers' business.
"""

from __future__ import annotations

from typing import TypedDict

from rw_bot.wire.state import Sample

#: Frames a sighting stays trusted without being re-seen.
#:
#: Thirty simulation seconds at the engine's ~300 frames a second. The scale
#: is the opponent's own tempo: the shipped AI's attack groups fill, stage
#: and commit on delays measured in the low thousands of frames
#: ([[engine-ai-triggers]]), so intel a wave-cycle old no longer describes
#: where its army is or what it is made of. A window this size keeps the
#: last-seen composition through the fog gaps between raids without carrying
#: the early game into the late one.
INTEL_WINDOW_FRAMES = 9_000


class Sighting(TypedDict):
    """One hostile, as last seen.

    Carries exactly the fields the counter tilt reads -- a sighting is
    deliberately a :class:`~rw_bot.policy.counter.Threat`, so remembered and
    visible hostiles feed the same arithmetic.

    Attributes:
        unit_id: Engine identity of the hostile.
        type_name: Its type, for the profile lookup.
        flying: Whether it was airborne when last seen.
        movement: The engine's movement layer for the type, e.g. ``"LAND"``
            or ``"WATER"`` -- carried so a remembered fleet reads as one to
            the counter tilt's naval clause, exactly as a visible fleet does
            ([[policy-exact-timing]], the naval wall).
        x: World x when last seen.
        y: World y when last seen.
        frame: When it was last seen.
    """

    unit_id: int
    type_name: str
    flying: bool
    movement: str
    x: float
    y: float
    frame: int


class Intel:
    """Holds every hostile sighting still inside the trust window.

    Attributes:
        sightings_taken: *First* sights recorded across the match, for the
            report: a scout that never saw anything and a scout that was never
            built read identically in every other figure.

            First sights, not re-sights, and the distinction was worth three
            orders of magnitude: counting every upsert billed the standing
            armies in view every sample -- one raid-arm match read
            ``sightings 166554``, about 41 per sample, saying nothing about
            scouting at all (log: 2026-07-29). A unit seen again after its
            window expired counts again, correctly: it is news again.
    """

    def __init__(self, window_frames: int = INTEL_WINDOW_FRAMES) -> None:
        """Open an empty memory.

        Args:
            window_frames: Frames a sighting stays trusted without re-sight.
        """
        self._window = window_frames
        self._seen: dict[int, Sighting] = {}
        self.sightings_taken = 0

    def observe(self, sample: Sample) -> None:
        """Record this observation's hostiles and forget the expired.

        Args:
            sample: One observation of the world.
        """
        for entity in sample["entities"]:
            if not entity["hostile"]:
                continue
            if entity["unit_id"] not in self._seen:
                self.sightings_taken += 1
            self._seen[entity["unit_id"]] = Sighting(
                unit_id=entity["unit_id"],
                type_name=entity["type_name"],
                flying=entity["flying"],
                movement=entity["movement"],
                x=entity["x"],
                y=entity["y"],
                frame=sample["frame"],
            )
        horizon = sample["frame"] - self._window
        expired = [unit_id for unit_id, s in self._seen.items() if s["frame"] < horizon]
        for unit_id in expired:
            del self._seen[unit_id]

    def forget(self, unit_id: int) -> None:
        """Drop one sighting on a caller's confirmation, ahead of the window.

        The memory itself cannot see a death: an extractor killed behind the
        fog would be remembered until the window expired, and a raider would
        assault the ghost for the duration. The raider CAN see one -- it is
        standing where the sighting said, looking at nothing -- and this is
        how it says so.

        Args:
            unit_id: Engine identity of the sighting to drop. Unknown ids are
                ignored: two confirmations of one death are not an error.
        """
        self._seen.pop(unit_id, None)

    def remembered(self) -> tuple[Sighting, ...]:
        """Return every sighting still trusted, currently visible included.

        The caller that ran :meth:`observe` this observation gets the visible
        hostiles back alongside the remembered ones, deduplicated by engine
        identity -- one list for the tilt, not two to merge.

        Returns:
            Sightings in identity order, so two runs of one seed read them
            identically.
        """
        return tuple(self._seen[unit_id] for unit_id in sorted(self._seen))


__all__ = ["INTEL_WINDOW_FRAMES", "Intel", "Sighting"]
