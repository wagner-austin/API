"""The scripted world the campaign tests drive, shared by all of them.

One catalogue, one opening roster and one peer that answers the loop with
scripted samples and records every order it was sent. The campaign test
modules split by theme (`test_campaign_loop`, `_economy`, `_workforce`,
`_switches`) and this is the ground they all stand on -- one copy, so two
modules cannot drift apart in what a "tank" costs.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import play
from rw_bot.policy.head import HeadModel
from rw_bot.policy.match_report import MatchReport
from rw_bot.policy.runner import AFFORD_STALL_SAMPLES
from rw_bot.wire.state import Sample
from tests.wire_fixtures import enemy, entity, lines, player, profile, profiles_for


def unit_stats(
    type_name: str,
    *,
    speed: float = 1.0,
    armed: bool = True,
    price: int = 350,
    upgrade_prices: tuple[int, ...] = (),
) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=upgrade_prices,
        weapon=(
            Weapon(
                shoot_delay=50.0,
                attack_range=110.0,
                direct_damage=17.0,
                direct_damage_volley=17.0,
                area_damage=0.0,
                area_damage_volley=0.0,
            )
            if armed
            else None
        ),
    )


CATALOGUE = {
    "c_tank": unit_stats("c_tank"),
    "builder": unit_stats("builder", speed=0.6, armed=False, price=200),
    "commandCenter": unit_stats("commandCenter", speed=0.0, armed=False, price=0),
    # Priced as the engine's own dump prices them, so the arithmetic below is
    # the arithmetic a real match does.
    "extractorT1": unit_stats("extractorT1", speed=0.0, armed=False, price=700),
    "landFactory": unit_stats("landFactory", speed=0.0, armed=False, price=1000),
    "editorOrBuilder": unit_stats("editorOrBuilder", speed=0.0, armed=False, price=0),
}

PROFILES = profiles_for(CATALOGUE)

PLACEMENTS: dict[str, TypePlacement] = {
    name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
    for i, name in enumerate(CATALOGUE)
}

BUILDER = entity(214, "builder")
CENTRE = entity(213, "commandCenter")
FACTORY = entity(300, "landFactory")
WAVE = (
    entity(1, "c_tank"),
    entity(2, "c_tank"),
    entity(3, "c_tank"),
)
ENEMY = enemy(9, "c_tank", x=100.0)

US = player(0, index=0, local=True, hostile=False, income=18, army_value=500, building_value=3000)
THEM = player(1, index=1, income=18, army_value=4200, building_value=1500)


class ScriptedPeer:
    """Serves prepared lines and records what was sent back.

    Attributes:
        sent: Every line the loop wrote, in order.
    """

    def __init__(self, prepared: list[str]) -> None:
        self._lines = list(prepared)
        self.sent: list[str] = []

    def send_line(self, line: str) -> None:
        """Record one written line.

        Args:
            line: Line content, without a newline.
        """
        self.sent.append(line)

    def read_line(self) -> str:
        """Serve the next prepared line, or end of stream.

        Returns:
            The next line, or an empty string once exhausted.
        """
        if not self._lines:
            return ""
        return self._lines.pop(0)

    def close(self) -> None:
        """Release the connection."""


def order_lines(peer: ScriptedPeer) -> list[str]:
    """Everything the loop sent except the per-sample acknowledgements.

    The ack is protocol rather than policy -- in lockstep it is what releases
    the simulation -- so assertions about what the bot decided filter it out
    ([[policy-determinism]]).
    """
    return [line for line in peer.sent if '"kind":"ack"' not in line]


def verb(peer: ScriptedPeer, kind: str) -> list[str]:
    return [line for line in order_lines(peer) if f'"kind":"{kind}"' in line]


def run_campaign(
    world: Sample,
    *,
    times: int = 3,
    plan: tuple[str, ...] = (),
    reinforce: tuple[str, ...] = (),
    reserve: int = 0,
    expand: bool = True,
    stop_when_plan_done: bool = False,
    afford_samples: int = AFFORD_STALL_SAMPLES,
    trace: Path | None = None,
    brace_model: HeadModel | None = None,
) -> tuple[MatchReport, ScriptedPeer]:
    """Play one scripted world for a fixed number of observations."""
    peer = ScriptedPeer(lines(*(world for _ in range(times))))
    report = play(
        AgentChannel(peer),
        plan,
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        times,
        reinforce=reinforce,
        reserve=reserve,
        expand=expand,
        stop_when_plan_done=stop_when_plan_done,
        afford_samples=afford_samples,
        trace=trace,
        brace_model=brace_model,
    )
    return report, peer


def defence_world() -> tuple[
    dict[str, UnitStats], dict[str, TypePlacement], dict[str, CombatProfile]
]:
    """A catalogue, placements and profiles that include a buildable turret."""
    catalogue = {**CATALOGUE, "c_turret_t1": unit_stats("c_turret_t1", speed=0.0, price=500)}
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name == "extractorT1")
        for i, name in enumerate(catalogue)
    }
    profiles = {**profiles_for(catalogue), "c_turret_t1": profile("c_turret_t1", 165.0)}
    return catalogue, placements, profiles
