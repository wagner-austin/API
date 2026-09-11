"""The measured baseline: the doctrine constant every arm is one field off.

Split from :mod:`rw_bot.policy.doctrine` when that file passed the
six-hundred line ceiling, and the boundary is a real one: the schema (what
a doctrine CAN say -- fields, types, refusal semantics) changes when the
vocabulary grows, while this constant (what the campaign MEASURED under)
changes only when a new baseline is adopted -- and the two have never once
changed in the same commit.

The style everything so far was measured under, exactly: extractors first
because they pay for everything after them; no factory named because the
build tree inserts prerequisites; the shipped AI's wave mass; expansion on;
the mix held as stated. A doctrine file is only ever compared against this,
so it is a constant rather than a file that could drift.
"""

from __future__ import annotations

from typing import Final

from rw_bot.policy.combat import FIRST_WAVE, WAVE_SIZES
from rw_bot.policy.doctrine import DERIVE_RESERVE, NAVTILT_OFF, Doctrine
from rw_bot.policy.firing import MAX_OPEN_GROUPS, PRIO_CONVERGENCE
from rw_bot.policy.workforce import DEFAULT_MAX_WORKERS

DEFAULT_DOCTRINE: Final[Doctrine] = Doctrine(
    name="default",
    goals=(
        "extractorT1",
        "extractorT1",
        "extractorT1",
        "c_tank",
        "c_tank",
        "c_tank",
        "c_tank",
    ),
    heavies=(),
    max_workers=DEFAULT_MAX_WORKERS,
    mass=WAVE_SIZES[-1],
    reserve=DERIVE_RESERVE,
    expand=True,
    counter=False,
    cover=True,
    intercept=False,
    guard_cap=0,
    aa_cover=False,
    forward=False,
    scout=False,
    raid=0,
    rush=False,
    creep=0,
    hold=0,
    riposte=False,
    navtilt=NAVTILT_OFF,
    tech=0,
    lurk=0,
    allin=0,
    decoys=0,
    kite=False,
    income_ladder=False,
    brace=False,
    hp_floor=0,
    strike=0,
    medics=0,
    navy=0,
    battery=0,
    bunkers=0,
    flame=0,
    close=0,
    guns=0,
    nukes=0,
    rebuild=0,
    hunt=0,
    worker_wait=0,
    groupcap=MAX_OPEN_GROUPS,
    prio=PRIO_CONVERGENCE,
    spacing=0,
    retreat=FIRST_WAVE,
    siege=0,
    siegedose=1,
    raze=0,
    press=0,
    huntgate=False,
    bank=False,
)


__all__ = ["DEFAULT_DOCTRINE"]
