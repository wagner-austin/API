"""The finisher: fund a nuke launcher, keep it armed, land warheads on worth.

Impossible is conclusively survivable and conclusively unclosable with what
the bot fields: walls hold indefinitely and nothing finishes
([[community-play-strategies]], log 2026-08-02). The community's answer is
the nuke, and the live probe validated every link of the chain this channel
plays: the launcher places by the ordinary builder (45,000, no tech gate),
``buildNuke`` stockpiles an 11,000 warhead, and ``launchNuke`` fires the
wire's targeted ability at a chosen point -- confirmed by an extractor
erased at (2370, 510) (`runs/nuke-probe4.out`, log 2026-08-05).

The probe also bought this channel's three laws the hard way. **The save
must withhold**: the plan's afford-wait does not, and cover's 500-credit
turrets starved the 45,000 forever -- the launcher was never ordered until
cover was silenced. **The launch flag is a lie about ammo**: the row reads
available at zero warheads, and a launch fired early is dropped without a
word, so every launch is refired until the world answers. **Targets are
matched by what the blast erases**, not by any one name: the warhead goes
to the centre of the richest 250-radius circle of hostile structures,
because an area weapon aimed at one building -- however pricey -- is an
11,000-credit strike paying back less than it cost.

Doctrine-gated (``nukes``), because the big-ticket law stands: saving
toward any large purchase during contested Very Hard play has lost four
measured times. This channel is built for the Impossible fortress context
-- survival is already solved there and the save is what the surplus is
FOR -- and the arm measurement, not the reasoning, decides where it plays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.budget import Budget
from rw_bot.policy.economy import placer
from rw_bot.policy.siting import find_anchor, next_ring_site
from rw_bot.policy.situation import read_situation
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.command import (
    AbilityOrder,
    BuildOrder,
    TargetedAbilityOrder,
    ability_order,
    build_order,
    targeted_ability_order,
)
from rw_bot.wire.state import Entity, Sample

#: The launcher, by registry name: 45,000, placed by the ordinary builder,
#: tech level one (`nuke_launcher.ini`, buildedge 28).
LAUNCHER_TYPE: Final = "nukeLauncherC"

#: Samples between launches at the same standing target. The launch action
#: reports available at zero ammo -- the flag does not carry the ammo gate --
#: so a launch fired before the warhead finishes is dropped silently, and
#: the probe's strike only landed because the point was fired at again
#: (`runs/nuke-probe4.out`: dud at s239, kill on the s539 refire).
LAUNCH_RETRY_SAMPLES: Final = 300

#: Samples between stockpile orders. The engine caps ammo at four through
#: the action's own gate; this bounds how fast credits chase that cap, and
#: keeps a dispatch-then-still-offered row from being paid twice in
#: consecutive ticks.
ARM_RETRY_SAMPLES: Final = 60

#: The warhead's blast radius, from the asset: 5,400 area damage over 250,
#: enough to kill any structure inside (`nuke_launcher.ini`,
#: projectile_nukeProjectile).
BLAST_RADIUS: Final = 250.0

#: Income per second below which the launcher does not fund at all.
#:
#: **Measured, twice.** The first screen withheld the 45,000 from tick one
#: and all three Impossible matches died by s2,500 with worth never above
#: the STARTING 3,500 -- the withhold binds every channel below it, so the
#: fortress the finisher was supposed to ride never stood
#: (`runs/sweeps/imp-nuke`, log 2026-08-05). The line itself is the 46-duel
#: law: final income at or above 50 credits a second won 36 of 36, at or
#: below 38 failed 6 of 7 -- fifty a second is where an economy exists,
#: and the finisher funds from an economy's surplus or it funds from its
#: host's blood ([[policy-holding-ground]]).
FUNDING_INCOME_FLOOR: Final = 50


class NukeOrders(TypedDict):
    """One tick of the finisher's will, at most one order per link.

    Attributes:
        build: Place a launcher, or None.
        arm: Stockpile a warhead, or None.
        launch: Fire at the chosen point, or None.
    """

    build: BuildOrder | None
    arm: AbilityOrder | None
    launch: TargetedAbilityOrder | None


def _nothing() -> NukeOrders:
    return NukeOrders(build=None, arm=None, launch=None)


def _economy_stands(sample: Sample) -> bool:
    """Report whether the host economy has earned the finisher's save.

    Judged by the local player's own income against
    :data:`FUNDING_INCOME_FLOOR`. A sample with no scoreboard -- a scripted
    world, a pre-scoreboard capture -- reads as no economy, because a save
    this deep must be earned by evidence rather than assumed by absence.
    """
    situation = read_situation(sample)
    return situation is not None and situation["our_income"] >= FUNDING_INCOME_FLOOR


def best_target(sample: Sample, catalogue: Mapping[str, UnitStats]) -> Entity | None:
    """Choose the hostile structure whose blast circle erases the most worth.

    Structures rather than units because the warhead flies for seconds and
    a structure will still be there. Scored by the summed price of every
    hostile structure inside :data:`BLAST_RADIUS` of the candidate --
    the warhead is an area weapon, and choosing the priciest SINGLE
    structure aimed 11,000 credits at buildings worth less than that.
    The packed base core -- command centre, factories, extractors inside
    one circle -- is what the finisher exists to erase.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for price and immobility.

    Returns:
        The structure at the centre of the richest blast circle, or None
        while no hostile structure is visible.
    """
    structures: list[tuple[Entity, int]] = []
    for entity in sample["entities"]:
        if not entity["hostile"]:
            continue
        stats = catalogue.get(entity["type_name"])
        if stats is None or stats["speed"] != 0.0:
            continue
        structures.append((entity, stats["price"]))
    best: Entity | None = None
    best_worth = 0
    limit = BLAST_RADIUS**2
    for candidate, _ in structures:
        worth = sum(
            price
            for other, price in structures
            if (other["x"] - candidate["x"]) ** 2 + (other["y"] - candidate["y"]) ** 2 <= limit
        )
        if best is None or worth > best_worth:
            best = candidate
            best_worth = worth
    return best


class Nuker:
    """Stands launchers up to the doctrine's count and keeps them firing.

    Placement leans on the workforce exactly as expansion does: the worker
    ordered to build is marked busy until the structure stands or the order
    is presumed lost, so no private retry clock is kept and a lost order
    re-decides itself ([[policy-loop]]). What IS kept is the two debounces
    the engine's silences force: a stockpile most every
    :data:`ARM_RETRY_SAMPLES`, a launch most every
    :data:`LAUNCH_RETRY_SAMPLES`.
    """

    def __init__(self) -> None:
        self._tick = 0
        self._armed_at = -ARM_RETRY_SAMPLES
        self._fired_at = -LAUNCH_RETRY_SAMPLES

    def advance(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        free: Sequence[Entity],
        workforce: Workforce,
        nukes: int,
        closing: bool = False,
    ) -> NukeOrders:
        """Advance the chain one observation.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name.
            budget: The tick's credits.
            free: Workers not already carrying out an order.
            workforce: Told what a worker was sent to build.
            nukes: The doctrine's launcher count, zero off.
            closing: Whether the closer stands committed -- the funding
                gate's second half. Income alone was measured twice and
                lost twice: 50/s is subsistence at Impossible, and at Very
                Hard the mid-game withhold drained a baseline win to defeat
                while accumulating 23,446 of 45,000
                (`runs/sweeps/{imp-nuke2,vh-nuke}`, log 2026-08-05).
                Sustained dominance is the one measured state where the
                surplus is real: the match is decided and the army the
                withhold binds has already won its fight. A standing
                launcher keeps arming and firing regardless -- the sunk
                cost is sunk, and the warhead is the cheapest value left
                in it.

        Returns:
            At most one order per link of the chain.
        """
        self._tick += 1
        if nukes <= 0:
            return _nothing()
        standing = [
            entity
            for entity in sample["entities"]
            if entity["mine"] and entity["type_name"] == LAUNCHER_TYPE
        ]
        orders = _nothing()
        # Standing PLUS underway: counted off structures alone, the walk was
        # invisible and a fresh builder was assigned to the same job every
        # tick -- eight granted launchers, 360,000 credits, one structure
        # (`runs/sweeps/vh-nuke`, log 2026-08-05).
        coming = len(standing) + workforce.underway(LAUNCHER_TYPE)
        if coming < nukes and closing and _economy_stands(sample):
            orders["build"] = self._place(sample, catalogue, budget, free, workforce)
        ready = [entity for entity in standing if entity["complete"]]
        if ready:
            orders["arm"] = self._arm(sample, budget, ready[0])
            orders["launch"] = self._launch(sample, catalogue, ready[0])
        return orders

    def _place(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        budget: Budget,
        free: Sequence[Entity],
        workforce: Workforce,
    ) -> BuildOrder | None:
        """Order the next launcher, saving toward it when refused.

        The withhold is the whole reason the channel works: the probe's
        first run proved the 45,000 never accumulates while cheaper
        spenders drain the tick (`runs/nuke-probe.out`, log 2026-08-05).
        It binds everything below this channel in the chain -- which is
        why the caller gates it behind :func:`_economy_stands`: withheld
        from tick one, three Impossible matches starved before their
        fortress existed (`runs/sweeps/imp-nuke`, log 2026-08-05).
        """
        builder = placer(sample, LAUNCHER_TYPE, free)
        if builder is None:
            return None
        anchor = find_anchor(sample, catalogue) or builder
        site = next_ring_site(sample, anchor, catalogue)
        if site is None:
            return None
        price = catalogue[LAUNCHER_TYPE]["price"]
        claim = budget.claim(f"nuke:{LAUNCHER_TYPE}", price)
        if not claim["granted"]:
            budget.withhold(price)
            return None
        workforce.assign(builder["unit_id"], LAUNCHER_TYPE, site)
        return build_order(
            unit_id=builder["unit_id"], type_name=LAUNCHER_TYPE, x=site[0], y=site[1]
        )

    def _arm(self, sample: Sample, budget: Budget, launcher: Entity) -> AbilityOrder | None:
        """Stockpile a warhead when the launcher offers one affordably.

        The priced row is the stockpile; the launch is priced in ammo,
        which the wire reports as zero credits. A refusal saves toward the
        warhead only while a launcher stands to use it -- before that there
        is nothing to save toward, the same rule the flame conversion holds
        ([[policy-budget]]).
        """
        if self._tick - self._armed_at < ARM_RETRY_SAMPLES:
            return None
        offer = next(
            (
                option
                for option in sample["options"]
                if option["unit_id"] == launcher["unit_id"]
                and option["price"] > 0
                and option["available"]
            ),
            None,
        )
        if offer is None:
            return None
        claim = budget.claim("nuke:warhead", offer["price"])
        if not claim["granted"]:
            budget.withhold(offer["price"])
            return None
        self._armed_at = self._tick
        return ability_order(unit_id=launcher["unit_id"], key=offer["key"])

    def _launch(
        self, sample: Sample, catalogue: Mapping[str, UnitStats], launcher: Entity
    ) -> TargetedAbilityOrder | None:
        """Fire at the priciest hostile structure, refired until it dies.

        The refire is the probe's law: the launch flag does not carry the
        ammo gate, so a launch the world has not answered is fired again
        once the window passes rather than trusted
        (`runs/nuke-probe4.out`).
        """
        if self._tick - self._fired_at < LAUNCH_RETRY_SAMPLES:
            return None
        offer = next(
            (
                option
                for option in sample["options"]
                if option["unit_id"] == launcher["unit_id"]
                and option["price"] == 0
                and option["available"]
            ),
            None,
        )
        if offer is None:
            return None
        target = best_target(sample, catalogue)
        if target is None:
            return None
        self._fired_at = self._tick
        return targeted_ability_order(
            unit_id=launcher["unit_id"],
            key=offer["key"],
            x=target["x"],
            y=target["y"],
        )


__all__ = [
    "ARM_RETRY_SAMPLES",
    "BLAST_RADIUS",
    "FUNDING_INCOME_FLOOR",
    "LAUNCHER_TYPE",
    "LAUNCH_RETRY_SAMPLES",
    "NukeOrders",
    "Nuker",
    "best_target",
]
