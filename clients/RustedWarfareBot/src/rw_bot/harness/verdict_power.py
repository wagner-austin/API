"""Verdict-grade power statements, from committed code instead of heredocs.

Every panel verdict this campaign publishes carries a t beside its own-spread
minimum detectable effect (the power audit's standing rule, log 2026-09-09),
and until this module the arithmetic behind those figures was born in scratch
``python -c`` one-liners that no guard could reach -- the exact defect class
the 2026-09-09 fleet research-integrity audit named ("guards run on the tree;
the analysis happens outside it"). This module is the analysis moving inside
the tree: it renders the shared instruments in
:mod:`platform_core.minimum_detectable_effect` over the same per-arm-per-seed
figures the margin scorer already produces, so a verdict's numbers are
re-derivable from tracked code plus the mirrored scorecards.

Two instruments, deliberately separate because they answer different
questions. The CONTINUOUS one takes any figure kind -- margins or survival
samples share the ``{arm: {seed: value}}`` shape -- and states what the
paired comparison could have detected. The WIN one only makes sense over
margins (the win bit is the margin band, [[policy-trace]]), and its McNemar
record is falsifiability rather than detectability -- the record itself says
so, and this module keeps the two vocabularies apart on purpose.

Named ``verdict_power`` rather than ``power``: this repository's operator
tooling already met one wrong ``power.py`` (the EcoQoS electrical-power
module, board log 2026-09-08), and an ambiguous name here would eventually
be the same misresolution again.
"""

from __future__ import annotations

from collections.abc import Mapping

from platform_core.minimum_detectable_effect import (
    mcnemar_power,
    paired_continuous_power,
)
from platform_core.power_distributions import McNemarTest, mcnemar_p

from rw_bot.harness.margin import WIN_THRESHOLD

#: The campaign's significance level, the one every panel has reported at.
ALPHA: float = 0.05


def _pairs(figures: Mapping[str, Mapping[int, float]]) -> tuple[tuple[str, str], ...]:
    """Every ordered arm pair sharing at least one seed, base first.

    The same enumeration :func:`rw_bot.harness.margin.report` walks, so the
    power lines land beside the delta lines they qualify.

    Args:
        figures: Figures by arm, then by seed.

    Returns:
        ``(base, other)`` pairs in sorted-arm order.
    """
    arms = sorted(figures)
    return tuple(
        (base, other)
        for i, base in enumerate(arms)
        for other in arms[i + 1 :]
        if set(figures[base]) & set(figures[other])
    )


def paired_power_lines(
    figures: Mapping[str, Mapping[int, float]],
    smallest_effect_of_interest: float,
) -> tuple[str, ...]:
    """State what each paired comparison could have detected.

    Args:
        figures: Figures by arm, then by seed -- margins or survival samples,
            the instrument does not care which as long as the caller states
            the effect of interest in the same units.
        smallest_effect_of_interest: The effect worth acting on, in the
            figures' own units.

    Returns:
        One line per arm pair sharing seeds: the mean paired difference, its
        sample sd, the instrument's minimum detectable effect at
        :data:`ALPHA`, and the module's TESTED / NOT_TESTED verdict against
        the stated effect of interest.

    Raises:
        AppError: Through the instrument, when a pair shares fewer than its
            minimum replicates or the effect of interest is not positive --
            a comparison too thin to power is a finding, not a line to skip.
    """
    lines: list[str] = []
    for base, other in _pairs(figures):
        shared = sorted(set(figures[base]) & set(figures[other]))
        differences = [figures[other][seed] - figures[base][seed] for seed in shared]
        record = paired_continuous_power(
            differences,
            alpha=ALPHA,
            smallest_effect_of_interest=smallest_effect_of_interest,
        )
        lines.append(
            f"power {other} - {base}: n={record['replicates']}"
            f"  mean {record['mean_difference']:+.3f} (sd {record['sample_sd']:.3f})"
            f"  mde {record['minimum_detectable_effect']:.3f}"
            f" at alpha {record['alpha']}"
            f"  vs interest {record['smallest_effect_of_interest']:.3f}"
            f"  -> {record['verdict']}"
        )
    return tuple(lines)


def win_power_lines(margins: Mapping[str, Mapping[int, float]]) -> tuple[str, ...]:
    """State each pair's win-bit comparison: the observed test and its reach.

    The win bar is a paired BINARY question, and every panel until now
    computed its sign test by hand. This renders both halves from the shared
    module: the OBSERVED mid-p McNemar p-value on the discordant split (the
    test the panels actually report), and the falsifiability record -- which
    splits of these discordants could reject at all.

    Args:
        margins: Margins by arm, then by seed. Margins specifically: the win
            bit is the margin band, meaningless for survival figures.

    Returns:
        One line per arm pair sharing seeds: wins each side, the discordant
        split, the observed mid-p p-value, and the falsifiability half --
        whether any split of that many discordants can reject, and the most
        balanced minority that does.
    """
    lines: list[str] = []
    for base, other in _pairs(margins):
        shared = sorted(set(margins[base]) & set(margins[other]))
        base_bits = [margins[base][seed] >= WIN_THRESHOLD for seed in shared]
        other_bits = [margins[other][seed] >= WIN_THRESHOLD for seed in shared]
        other_only = sum(1 for b, o in zip(base_bits, other_bits, strict=True) if o and not b)
        base_only = sum(1 for b, o in zip(base_bits, other_bits, strict=True) if b and not o)
        discordant = other_only + base_only
        observed_p = mcnemar_p(min(other_only, base_only), discordant, McNemarTest.MID_P)
        record = mcnemar_power(discordant, alpha=ALPHA, test=McNemarTest.MID_P)
        reach = (
            f"can reject at minority <= {record['most_balanced_rejecting_minority']}"
            if record["can_ever_reject"]
            else "CANNOT reject at any split"
        )
        lines.append(
            f"wins {other} - {base}: {sum(other_bits)}-{sum(base_bits)} of {len(shared)}"
            f"  flips {other_only}:{base_only}"
            f"  mid-p {observed_p:.4f}"
            f"  ({reach} of d={discordant} at alpha {ALPHA})"
        )
    return tuple(lines)


__all__ = [
    "ALPHA",
    "paired_power_lines",
    "win_power_lines",
]
