"""A gameplay style as data, so trying one is an argument rather than an edit.

Every knob here already existed; what was missing was a single carrier. The
goals, the worker ceiling, the wave mass and the expansion switch were spread
across nine positional CLI slots, and the ninth slot only exists because the
eighth did -- each new question threaded one more position through the entry
point, the Makefile and the sweep harness ([[policy-loop]]).

A doctrine is one file naming all of them. Two arms of an experiment are two
files that differ in one line, which is the same discipline the sweep already
enforces for jobs: the arm that ran last week can be re-run, because the file
that defined it was never edited into the next one.

Every field is required. A doctrine file with a missing field is an error
naming the field, not a default quietly changing what the arm means -- the same
rule the sweep's job lines follow, for the same reason.
"""

from __future__ import annotations

from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.policy.combat import WAVE_SIZES
from rw_bot.policy.workforce import DEFAULT_MAX_WORKERS

#: The ``heavies`` value that means "no extra composition entries".
#:
#: A word rather than a blank, because a doctrine line cannot carry an empty
#: value and a missing field is an error by design.
NO_HEAVIES: Final = "none"

#: The ``reserve`` value that means "derive it from the composition".
#:
#: Negative rather than zero, because zero is a real reserve a doctrine may
#: want, and conflating "reserve nothing" with "decide for me" is how an arm
#: stops testing what it claims ([[policy-economy]]).
DERIVE_RESERVE: Final = -1

#: The naval clause never runs.
NAVTILT_OFF: Final = 0

#: The naval clause runs whenever a fleet is in the remembered picture.
NAVTILT_ALWAYS: Final = 1

#: The naval clause runs only after the fleet has drawn blood -- the
#: adaptive mode two panels calibrated: the army-deficit gate halved the
#: damage but still fired inside winning games, because a subsidized
#: opponent's army is bigger even when losing the match. Blood cannot be
#: misread: units of ours killed by WATER-movers is the failure mode
#: itself, not a proxy for it (log 2026-08-08).
NAVTILT_BLOODIED: Final = 2

#: The naval clause runs only while the doom model predicts the fleet will
#: doom this game -- the driver law eight demanded: the response reshapes
#: the match, so its trigger must predict, not react. The model replicated
#: cross-tree at AUC 0.75 with precision ~0.7 (log 2026-08-09); whether
#: that is enough is the mode-3 panel's question.
NAVTILT_PREDICTED: Final = 3

#: Fields carried as whole numbers in a doctrine file.
INT_FIELDS: Final = (
    "max_workers",
    "mass",
    "reserve",
    "guard_cap",
    "raid",
    "tech",
    "lurk",
    "allin",
    "creep",
    "hold",
    "decoys",
    "hp_floor",
    "strike",
    "navtilt",
    "medics",
    "navy",
    "battery",
    "bunkers",
    "flame",
    "close",
    "guns",
    "nukes",
    "rebuild",
)

#: Fields carried as ``0`` or ``1`` in a doctrine file.
FLAG_FIELDS: Final = (
    "expand",
    "counter",
    "cover",
    "intercept",
    "aa_cover",
    "forward",
    "scout",
    "rush",
    "riposte",
    "kite",
    "income_ladder",
)

#: Fields carried as text in a doctrine file.
STR_FIELDS: Final = ("name", "goals", "heavies")

#: Every field a doctrine file must carry, in the order presets write them.
DOCTRINE_FIELDS: Final = (*STR_FIELDS, *INT_FIELDS, *FLAG_FIELDS)


class DoctrineError(RwBotError):
    """A doctrine file could not be read as a gameplay style.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending line.
    """


class Doctrine(TypedDict):
    """One gameplay style, complete.

    Attributes:
        name: What this style is called, for the run log and the result file.
        goals: What to ask the planner for, in order. Repeats are a ratio, not
            a preference stated twice ([[policy-production]]).
        max_workers: The most builders worth holding ([[policy-production]]).
        mass: Units the sustained wave waits for. Values at or below the
            shipped ladder's last fixed rung leave the ladder unchanged, so the
            shipped behaviour is a value rather than a special case
            ([[engine-ai-triggers]]).
        reserve: Credits held back from expansion for the army, or
            :data:`DERIVE_RESERVE` to derive it from the composition. A fixed
            figure is what keeps a composition A/B from silently also being a
            reserve A/B ([[policy-economy]]).
        expand: Whether to play the economy at all. False is the control arm
            of the expansion A/B ([[policy-economy]]).
        counter: Whether production tilts toward what the opponent is seen to
            field, or holds the stated mix regardless
            ([[mechanics-combat-profile]]).
        cover: Whether the economy buys turrets beside bare structures at
            all. On is the behaviour every measurement carries somewhere in
            its lineage -- but for most of that lineage the orders were
            silently refused, so "defence on" historically meant "defence
            attempted". The first batch where turrets actually landed spent
            25-45k a match on them and won 6/24 at a rung the
            attempted-defence bot won 10/12, which is what makes on-vs-off
            a question at last ([[policy-holding-ground]]).
        intercept: Whether the reserve turns on a raider standing inside the
            outpost radius of one of our structures, or keeps gathering
            regardless ([[policy-holding-ground]]).
        aa_cover: Whether an anti-air turret joins the cover once the
            opponent has shown aircraft. Nothing the bot could place before
            this touched an aircraft at all -- the whole army and the ground
            turret declare ``canAttackFlyingUnits: false``
            ([[policy-holding-ground]]).
        guard_cap: The most reserve units an interception commits, or zero for
            all of them -- the behaviour every guard measurement was taken
            under, so the shipped figure is a value rather than a special
            case. The cost case that makes it a question: one match logged
            870 intercepts and never massed an attack
            ([[policy-holding-ground]]).
        hold: Percent of the anchor-to-mirror line the reserve gathers at,
            zero for the anchor. The choke-holding verb the terrain screen
            argued for: Fire Bridge read 514 samples worse than open ground
            with the army at home while their waves crossed the funnel
            freely -- a choke pays the defender who STANDS at it, and until
            this field nothing could (log 2026-08-09).
        forward: Whether the reserve posts at the frontier extractor
            instead of the base. The one invariant six batches have not
            moved is that matches are decided by extractor drops far from
            where the army gathers; this is the corpus's forward-posture
            answer to it ([[policy-holding-ground]],
            [[community-play-strategies]]).
        scout: Whether one scout is kept alive walking the pool circuit, its
            sightings remembered through the fog and fed to the counter tilt
            ([[community-play-strategies]]).
        rush: Whether released waves march at the estimated enemy start
            while nothing is visible to fight. The all-in verb: against an
            income-multiplier opponent whose advantage compounds with time,
            the earliest possible fight is the fairest one, and without this
            the first wave stood at the rally point waiting for an opponent
            who never needed to come ([[policy-holding-ground]]).
        raid: The raid party's size, or zero for no raiding. A size rather
            than a flag, because the size is the open question the v2
            measure left: at the first-wave size the raid is free and wins
            nothing, so whether a heavier party converts is a doctrine arm,
            not a code edit ([[policy-raid]]).
        creep: The percent of the anchor-to-enemy line the turret walk
            holds at, zero for no creep. Walked to one hundred -- the
            enemy's door -- the line died faster than it stood (164
            turrets, 82,000 credits, refuted); held at a choke it is the
            community's whole answer to the cheating difficulties, and
            every third structure the walk lays is a repair bay
            ([[ai-opponent-strategy]], [[community-play-strategies]]).
        navtilt: When the counter tilt answers WATER-layer threats by
            repeating the mix's fleet-outranging types.
            :data:`NAVTILT_OFF` never, :data:`NAVTILT_ALWAYS` whenever a
            fleet is seen, :data:`NAVTILT_BLOODIED` only after the fleet
            has killed units of ours, :data:`NAVTILT_PREDICTED` only while
            the doom model reads the early game as fleet-doomed. Two panels calibrated the third
            mode: ungated, the tilt re-rolled winning seeds (navpair48,
            net -2); gated on an army deficit it fired less but still
            inside winning games, because a subsidized opponent's army is
            bigger even while losing (navgate96, net -2). Blood is the
            failure mode itself: a game the fleet never touched can never
            be perturbed (log 2026-08-08).
        riposte: Whether the whole reserve releases the moment an intrusion
            ends -- the human counter-punch: let the attack burn itself on
            the defences, then push into the window before the opponent's
            next group finishes its thousand-tick delay and seventeen-second
            staging ([[ai-opponent-strategy]]).
        tech: How many factories unlock their next tier, or zero for none.
            The land factory's 2,000-credit upgrade flips a flag on the same
            building and opens the heavy roster -- reachable only through
            the ability verb, because it converts into no type
            ([[mechanics-build-actions]]). A count rather than a flag,
            because the unlock is per building and the first one already
            opens production: the flag form bought all four factories'
            unlocks in one probe -- 8,000 credits of saving pauses for a
            roster the first 2,000 had opened ([[policy-budget]]).
        lurk: Scouts kept alive loitering at the enemy start, zero for
            none. The AI recalls its attack groups home for 500 ticks
            whenever an intruder stands in its base zone, and the recall
            runs before the attack branch -- a lurker re-arms that leash
            and retreats alive when chased, where a raider pays for each
            recall with its life ([[ai-opponent-strategy]]).
        kite: Whether armed mobile units hold the reach band -- the
            agent's between-samples reflex steps a unit away whenever a
            shorter-reached threat closes inside its own reach, so a unit
            that outranges its attacker is never hit by it. The reflex
            no-ops without a reach advantage, so the flag covers every
            armed mobile type safely ([[community-play-strategies]]).
        hp_floor: Percent of health below which armed mobile units flee
            the threat in reach, zero for never. Value preserved beats
            value spent, and the engine running the newest waypoint hands
            the unit back to the planner the moment it is clear.
        decoys: Scouts kept alive scattered wide on our own half, zero
            for none. The AI picks every attack target uniformly at random
            over ALL our units with no fog term, so our placement IS the
            distribution of its attacks -- each decoy is an extra ticket in
            its lottery, and a wave that draws one walks to an empty flank
            and chases a unit that flees it ([[ai-opponent-strategy]]).
        medics: Combat engineers to keep alive in the army, zero for
            none. The healer the community meta is built on, offered by
            the tier-two land factory and priced out of every tick under
            pressure -- so a refused hire saves toward itself, the tech
            unlock's gated pattern, one hire outstanding at a time
            ([[policy-budget]], [[community-play-strategies]]).
        bunkers: Mobile turrets to keep alive in the army, zero for none.
            The community's named counter to massed tier-one waves --
            4,500 credits of area damage behind a deploy shield -- and a
            price ordinary production never once accumulated: measured at
            Impossible, ``produce:mechBunker asked 1178 got 0`` with the
            economy healthy (log 2026-08-01). Funded exactly like the
            medics: a refused hire withholds its price, one outstanding
            at a time ([[policy-budget]], [[community-play-strategies]]).
        flame: Flame turrets to hold by converting ground turrets up the
            tier-two fork, zero for none. The ground turret's upgrade is a
            four-way fork the extractor walk deliberately skips, and the
            flame branch is the community's named anti-horde static: 1,600
            hit points, self-repair, wide-area fire that scales with how
            crowded the rush is -- a 1,000-credit conversion off the
            500-credit turret cover already builds. Funded like the medics:
            a refused conversion withholds, ordered once per structure
            because converting never fills the queue
            ([[policy-budget]], [[community-play-strategies]]).
        close: Our army as a multiple of the strongest rival's that ends
            the holding game, zero for never. Dominance decays: nineteen
            Very Hard matches stood dominant at the 4,000-sample cap and
            eleven of them LOST at 10,000 -- the AI compounds too, and a
            decided match is won by ending it while it is decided. Open,
            the window releases every wave and forces the march at the
            enemy start through contact -- and the commitment LATCHES on
            SUSTAINED dominance, both halves measured. Re-reading the
            window every tick closed piecemeal: three lost matches show 9,
            3 and 6 marches dying in dribbles as it flickered. Latching on
            one open sample went 9 won / 13 lost: early-game ratio noise
            became lifelong premature all-ins and wiped four former wins.
            So the window must hold :data:`~rw_bot.policy.situation.CLOSE_HOLD`
            samples running before the latch commits
            ([[policy-situation]], log 2026-08-01).
        guns: Top-tier gun turrets to hold by walking the turret chain
            (T1 -> T2 gun -> T3 gun), zero for none. The community's
            fortified-zone teeth: the tier three "solos most ground
            units", the AI's attack-move stands its army at a turret wall
            until one of them dies, and every written account of beating
            Impossible names this unit (steam-impossible-playbook.txt,
            [[community-play-strategies]]). Each chain step is funded
            through the withhold, one order a tick, pipeline bounded by
            the top's shortfall so the base cover is never consumed
            wholesale ([[policy-budget]]).
        nukes: Nuke launchers to stand and keep firing, zero for none.
            The finisher: Impossible is survivable behind walls and
            unclosable without one, and every link of this chain was
            validated live -- launcher by the ordinary builder, 11,000
            warhead by the priced action, launch by the targeted ability,
            an extractor erased where the planner pointed
            (`runs/nuke-probe4.out`, log 2026-08-05). The 45,000 saves
            through the withhold because the plain afford-wait provably
            never accumulates it, and every launch is refired until the
            world answers because the launch flag does not carry the ammo
            gate. Doctrine-gated hard: the big-ticket law (four measured
            refutations) says saving this deep during contested Very Hard
            play loses; this knob is for the fortress context where
            survival is already solved ([[policy-budget]]).
        rebuild: Credits the strongest rival's army value must sit below
            its recent peak before a RAZED pool may be re-claimed, zero
            for off. The Impossible autopsy's shape is built-then-razed:
            the economy machinery works, the pool re-enters the survey
            the moment nothing stands on it, and the builder's walk back
            dies to the same wave that razed it
            ([[impossible-economy-problem]]). This gates only pools we
            HELD and lost -- virgin pools claim as always, so the opening
            never waits -- on the same wave-break signal the strike
            release reads, because the wave being broken is what makes
            the walk survivable ([[policy-situation]]).
        income_ladder: Whether a refused extractor conversion saves toward
            itself. Off is the Impossible measurement: unconditional saving
            doubled income and lost, because the army pauses let the enemy's
            economy live untouched. On is the Very Hard counter-measurement:
            our worth ceilinged at ~30-35k on every seed while the T3
            conversion asked thousands of times and funded never -- the
            matches where the opponent's compounding passed that ceiling
            were unwinnable regardless of trigger design
            ([[policy-budget]], `runs/traces/vh-debounce`, log 2026-08-02).
        strike: Credits the strongest rival's army value must sit below
            its recent peak for the release window to open, zero for off.
            The engine broadcasts every army value unfogged; a wave dying
            on our line reads as a fall of its own worth, and the horde
            releases into that gap rather than on a clock or a ladder
            rung. A ratio was tried first and refuted in one probe: at
            Impossible their army is always a multiple of ours, so an
            absolute comparison never opens ([[policy-situation]]).
        allin: The observation the whole reserve releases from, zero for
            never. Releasing on size met an Impossible army that had
            compounded past answering in forty-seven straight matches;
            this releases on time -- hold everything to the chosen moment,
            dump it, and stream every later unit in behind
            ([[policy-combat]]).
        heavies: Composition entries outside the plan, repeats a ratio like
            the goals. The channel the unlocked roster joins the army mix
            through: production orders only what the engine offers, so an
            entry here is inert until its factory's tier opens -- and it
            must NOT be a goal, because the plan derives prerequisites from
            the static build tree, which would insert the experimental
            factory rather than wait for the unlock
            ([[mechanics-build-actions]], [[policy-production]]).
    """

    name: str
    goals: tuple[str, ...]
    heavies: tuple[str, ...]
    max_workers: int
    mass: int
    reserve: int
    expand: bool
    counter: bool
    cover: bool
    intercept: bool
    guard_cap: int
    aa_cover: bool
    forward: bool
    scout: bool
    raid: int
    rush: bool
    creep: int
    hold: int
    riposte: bool
    navtilt: int
    tech: int
    lurk: int
    allin: int
    decoys: int
    kite: bool
    income_ladder: bool
    hp_floor: int
    strike: int
    medics: int
    navy: int
    battery: int
    bunkers: int
    flame: int
    close: int
    guns: int
    nukes: int
    rebuild: int


#: The style everything so far was measured under, exactly.
#:
#: Extractors first because they pay for everything after them; no factory
#: named because the build tree inserts prerequisites; the shipped AI's wave
#: mass; expansion on; the mix held as stated. A doctrine file is only ever
#: compared against this, so it is a constant rather than a file that could
#: drift.
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
)


__all__ = [
    "DEFAULT_DOCTRINE",
    "DERIVE_RESERVE",
    "DOCTRINE_FIELDS",
    "FLAG_FIELDS",
    "INT_FIELDS",
    "NAVTILT_ALWAYS",
    "NAVTILT_BLOODIED",
    "NAVTILT_OFF",
    "NAVTILT_PREDICTED",
    "NO_HEAVIES",
    "STR_FIELDS",
    "Doctrine",
    "DoctrineError",
]
