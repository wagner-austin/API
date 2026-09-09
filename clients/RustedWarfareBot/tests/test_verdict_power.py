"""Verdict-grade power lines, rendered from the shared instruments.

What this module owns -- and therefore what these tests pin -- is the
pairing walk, the delta arithmetic, the win-bit derivation from the margin
bands, and the rendering. The instruments' own mathematics is
``platform_core``'s tested domain; where a line carries an instrument
figure, the expectation is derived by calling that instrument directly, so
a platform-side recalibration moves both sides of the assertion together
instead of silently splitting them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.minimum_detectable_effect import mcnemar_power, paired_continuous_power
from platform_core.power_distributions import McNemarTest, mcnemar_p
from scripts.margin import EXIT_BAD_USAGE, EXIT_OK, main

from rw_bot.harness.verdict_power import ALPHA, paired_power_lines, win_power_lines

#: Three shared seeds, mixed outcomes: the base wins 1 and 3, the arm wins
#: 2 and 3 -- one discordant pair each way, one concordant win.
MARGINS = {
    "base": {1: 2.5, 2: -2.0, 3: 2.0},
    "arm": {1: -1.0, 2: 2.9, 3: 2.4},
}


def test_the_paired_line_carries_the_instruments_own_figures() -> None:
    """The deltas are other-minus-base per shared seed -- and sorted-arm
    order makes "arm" the base here, so the differences are base-arm's
    figures minus arm's. Every figure on the line is the instrument's, not
    a re-derivation."""
    record = paired_continuous_power(
        [3.5, -4.9, -0.4], alpha=ALPHA, smallest_effect_of_interest=1.0
    )
    lines = paired_power_lines(MARGINS, 1.0)
    assert lines == (
        f"power base - arm: n={record['replicates']}"
        f"  mean {record['mean_difference']:+.3f} (sd {record['sample_sd']:.3f})"
        f"  mde {record['minimum_detectable_effect']:.3f}"
        f" at alpha {record['alpha']}"
        f"  vs interest {record['smallest_effect_of_interest']:.3f}"
        f"  -> {record['verdict']}",
    )
    # The verdict is stated, not implied: this pair's spread dwarfs the
    # stated interest, so the module must have called it NOT_TESTED.
    assert "NOT_TESTED" in lines[0]


def test_the_win_line_reads_the_bands_and_reports_both_halves() -> None:
    """Win bits come off the margin bands (arm wins seeds 2 and 3, base
    wins 1 and 3), the observed test is the mid-p McNemar on the 1:1
    discordant split, and the falsifiability half comes from the shared
    record."""
    observed = mcnemar_p(1, 2, McNemarTest.MID_P)
    record = mcnemar_power(2, alpha=ALPHA, test=McNemarTest.MID_P)
    assert record["can_ever_reject"] is False
    lines = win_power_lines(MARGINS)
    assert lines == (
        "wins base - arm: 2-2 of 3"
        "  flips 1:1"
        f"  mid-p {observed:.4f}"
        f"  (CANNOT reject at any split of d=2 at alpha {ALPHA})",
    )


def test_a_rejectable_split_names_its_reach() -> None:
    """With enough one-sided discordants the line names the most balanced
    minority that still rejects, from the record itself."""
    margins = {
        "base": dict.fromkeys(range(12), 2.5),
        "arm": dict.fromkeys(range(12), -2.0),
    }
    record = mcnemar_power(12, alpha=ALPHA, test=McNemarTest.MID_P)
    assert record["can_ever_reject"] is True
    (line,) = win_power_lines(margins)
    assert f"can reject at minority <= {record['most_balanced_rejecting_minority']}" in line
    assert "flips 12:0" in line
    assert "wins base - arm: 12-0 of 12" in line


def test_arms_without_shared_seeds_power_nothing() -> None:
    figures = {"a": {1: 2.0}, "b": {2: 2.0}}
    assert paired_power_lines(figures, 1.0) == ()
    assert win_power_lines(figures) == ()


def test_a_pair_too_thin_to_power_raises_through_the_instrument() -> None:
    """Two shared seeds is below the instrument's replicate floor, and a
    comparison too thin to power is a finding -- never a skipped line."""
    thin = {"a": {1: 2.0, 2: 2.0}, "b": {1: -2.0, 2: -2.0}}
    with pytest.raises(AppError):
        paired_power_lines(thin, 1.0)


def _batch(tmp_path: Path) -> Path:
    card = "### {name}\nverdict        {verdict} ({verdict})\nsamples seen   {samples}\n"
    batch = tmp_path / "demo"
    batch.mkdir()
    for seed, (control_verdict, arm_verdict) in enumerate(
        [("won", "wiped"), ("wiped", "won"), ("won", "won")], start=1
    ):
        (batch / f"control-s{seed}.txt").write_text(
            card.format(name=f"control-s{seed}", verdict=control_verdict, samples=100 * seed),
            encoding="utf-8",
        )
        (batch / f"arm-s{seed}.txt").write_text(
            card.format(name=f"arm-s{seed}", verdict=arm_verdict, samples=110 * seed),
            encoding="utf-8",
        )
    return tmp_path


def test_the_cli_appends_power_lines_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _batch(tmp_path)
    assert main(["demo", "--power", "1.0"], root=root) == EXIT_OK
    out = capsys.readouterr().out
    assert "## demo" in out
    assert "power control - arm:" in out
    assert "wins control - arm:" in out
    # And without the flag, the report stays exactly the report.
    assert main(["demo"], root=root) == EXIT_OK
    plain = capsys.readouterr().out
    assert "power" not in plain
    assert "mid-p" not in plain


def test_the_power_flag_without_its_effect_is_bad_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["demo", "--power"], root=tmp_path) == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: margin")


def test_a_non_numeric_effect_raises_loudly(tmp_path: Path) -> None:
    """The same loud path every numeric argument in the scripts package
    takes -- a garbage effect is never a silent default."""
    with pytest.raises(ValueError):
        main(["--power", "loud", "demo"], root=tmp_path)
