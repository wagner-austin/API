"""A summarised gain must carry its replicates.

The arrangement under test is the one that shipped: ``per_seed_observations``
existed, was exported and was tested, and two of six callers of one shared
assembly helper invoked it. The four that did not looked identical to the two
that did, and the difference cost a verdict that can no longer be settled --
task 91e12be1, a difference at 1.3912x its floor, inside the band where a
range and a paired t-test disagree at three seeds.
"""

from __future__ import annotations

import pathlib

from monorepo_guards.replicated_gain_rules import ReplicatedGainRule


def _write(root: pathlib.Path, relative: str, body: str) -> pathlib.Path:
    """Write a source file under a temporary monorepo root.

    Args:
        root: Temporary monorepo root.
        relative: Path relative to the root.
        body: File contents.

    Returns:
        The path written.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestTheArrangementThatShipped:
    """Summary without replicates, which is what four sweeps did."""

    def test_summarising_without_recording_seeds_is_a_violation(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A mean and a max-min spread cannot be asked a paired question.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/Thing/src/thing/cli/sweep.py",
            "from thing.contracts import gain_observations\n\n"
            "def emit(arm: object) -> None:\n"
            "    observations.extend(gain_observations(arm))\n",
        )

        found = ReplicatedGainRule().run([path])

        assert [v.kind for v in found] == ["replicated-gain-summary-only"]

    def test_the_message_says_why_rather_than_naming_the_symbol(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A reader who has not read the rule must still understand the loss.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/Thing/src/thing/cli/sweep.py",
            "def emit(arm: object) -> None:\n    observations.extend(gain_observations(arm))\n",
        )

        found = ReplicatedGainRule().run([path])

        assert "paired question" in found[0].line
        assert "replicates are gone" in found[0].line


class TestTheArrangementsThatAreCorrect:
    """Everything the rule must stay silent about."""

    def test_emitting_both_is_clean(self, tmp_path: pathlib.Path) -> None:
        """The fix, and the shape three sweeps already had.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/Thing/src/thing/cli/sweep.py",
            "def emit(arm: object) -> None:\n"
            "    observations.extend(gain_observations(arm))\n"
            "    observations.extend(per_seed_observations(arm))\n",
        )

        assert ReplicatedGainRule().run([path]) == []

    def test_a_file_that_summarises_nothing_is_out_of_scope(self, tmp_path: pathlib.Path) -> None:
        """Keying on the summary call is what keeps unrelated files out.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/Thing/src/thing/cli/other.py",
            "def emit() -> None:\n    observations.append(1.0)\n",
        )

        assert ReplicatedGainRule().run([path]) == []

    def test_the_defining_module_may_name_one_without_the_other(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Where both are DEFINED, calling one is not a defect.

        Exempted by filename rather than by an exemption list, because the
        module that defines the pair is the one place their names appear
        without an emission behind them.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/Thing/src/thing/contracts/replicated_measurement.py",
            "def wrapper(arm: object) -> None:\n    return gain_observations(arm)\n",
        )

        assert ReplicatedGainRule().run([path]) == []

    def test_an_unused_import_does_not_count_as_compliance(self, tmp_path: pathlib.Path) -> None:
        """The rule keys on CALLS, not imports.

        A half-applied fix leaves the import behind and removes the call, and
        an import-keyed rule would read that as fixed. This is the case that
        decided the predicate.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "services/Thing/src/thing/cli/sweep.py",
            "from thing.contracts import gain_observations, per_seed_observations\n\n"
            "def emit(arm: object) -> None:\n"
            "    observations.extend(gain_observations(arm))\n",
        )

        found = ReplicatedGainRule().run([path])

        assert [v.kind for v in found] == ["replicated-gain-summary-only"]
