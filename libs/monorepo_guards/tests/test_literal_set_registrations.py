"""The second and third registrations, driven end to end rather than assumed.

A parameterised rule whose only exercised configuration is the one it was
extracted from is a rule that has not been shown to be parameterised. These
use each set's own declaring module, tuple name and field name.

``TestTheRiskTierSet`` is the one the monorepo-root resolution was needed for:
its declaration lives in ``covenant_domain`` and every user is in
``covenant-radar-api``, so under the old resolution the rule found nothing
while checking the service and reported zero violations for four inline
Literals it had never compared to anything.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.literal_set_rules import (
    CORPUS_FORMAT_SET,
    RISK_TIER_SET,
    STRATEGY_NAME_SET,
    LiteralSetRule,
)
from tests._literal_set_support import DECLARING_PACKAGE, config_for, write_source


class TestTheStrategyNameSet:
    """The second registration, driven end to end rather than assumed.

    A parameterised rule whose only exercised configuration is the one it was
    extracted from is a rule that has not been shown to be parameterised. These
    use the strategy set's own declaring module, tuple name and field name.
    """

    _STRATEGY_DECLARING = STRATEGY_NAME_SET.defining_module

    def _strategy_declaring(self, tmp_path: Path, members: str) -> Path:
        """Write a stand-in module binding STRATEGY_NAMES.

        Args:
            tmp_path: Directory to write into.
            members: The tuple's contents, as source.

        Returns:
            The path written.
        """
        return write_source(
            tmp_path / DECLARING_PACKAGE / self._STRATEGY_DECLARING,
            f'from typing import Literal\n\nSTRATEGY_NAMES: tuple[Literal["full", "lora", '
            f'"qlora"], ...] = ({members})\n',
        )

    def test_it_flags_a_finetuning_strategy_literal_left_behind(self, tmp_path: Path) -> None:
        """Exactly the state that existed before the names were collapsed.

        A fourth strategy is declared and one of the nine inline annotations
        still names three. Before this registration, that type-checked.
        """
        declaring = self._strategy_declaring(tmp_path, '"full", "lora", "qlora", "cartridge"')
        stale = write_source(
            tmp_path / "contracts" / "queue.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    finetuning_strategy: Literal["full", "lora", "qlora"]\n',
        )

        violations = LiteralSetRule(STRATEGY_NAME_SET, config_for(tmp_path)).run([declaring, stale])

        assert [v.kind for v in violations] == ["strategy-name-literal-drift"]
        assert "cartridge, full, lora, qlora" in violations[0].line

    def test_the_shared_name_is_not_flagged(self, tmp_path: Path) -> None:
        """The state the repository is in now: the field names the shared type.

        ``StrategyName`` is not a ``Literal`` subscript, so there is nothing to
        compare and nothing to drift. This is what the collapse bought.
        """
        declaring = self._strategy_declaring(tmp_path, '"full", "lora", "qlora"')
        shared = write_source(
            tmp_path / "contracts" / "queue.py",
            "class P:\n    finetuning_strategy: StrategyName\n",
        )

        assert (
            LiteralSetRule(STRATEGY_NAME_SET, config_for(tmp_path)).run([declaring, shared]) == []
        )

    def test_a_corpus_format_literal_is_not_this_rules_business(self, tmp_path: Path) -> None:
        """Each instance watches only its own field names.

        Without this, two registrations over the same files would each report
        the other's drift under its own kind.
        """
        declaring = self._strategy_declaring(tmp_path, '"full", "lora", "qlora"')
        other = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    corpus_format: Literal["lines"]\n',
        )

        assert LiteralSetRule(STRATEGY_NAME_SET, config_for(tmp_path)).run([declaring, other]) == []

    def test_a_package_that_does_not_declare_the_tuple_is_still_checked(
        self, tmp_path: Path
    ) -> None:
        """The parameterised half of the fix: resolution from the monorepo
        root works for a second set with its own declaring module, rather
        than only for the one the change was written against."""
        self._strategy_declaring(tmp_path, '"full", "lora", "qlora"')
        stale = write_source(
            tmp_path / "q.py",
            'from typing import Literal\n\n\nclass P:\n    finetuning_strategy: Literal["full"]\n',
        )

        violations = LiteralSetRule(STRATEGY_NAME_SET, config_for(tmp_path)).run([stale])

        assert [v.kind for v in violations] == ["strategy-name-literal-drift"]

    def test_its_rule_name_is_derived_from_its_subject(self, tmp_path: Path) -> None:
        """Two instances must not share a name, or reports become ambiguous."""
        assert (
            LiteralSetRule(STRATEGY_NAME_SET, config_for(tmp_path)).name == "strategy-name-literal"
        )
        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).name == "corpus-format-literal"
        )

    def test_a_missing_strategy_tuple_is_reported_under_its_own_kind(self, tmp_path: Path) -> None:
        """A rule silently checking nothing is the failure this catches."""
        declaring = write_source(
            tmp_path / DECLARING_PACKAGE / self._STRATEGY_DECLARING, "OTHER: int = 1\n"
        )

        violations = LiteralSetRule(STRATEGY_NAME_SET, config_for(tmp_path)).run([declaring])

        assert [v.kind for v in violations] == ["strategy-name-tuple-missing"]


class TestTheRiskTierSet:
    """The third registration, and the one the fix above was needed for.

    Its declaring module is in `covenant_domain` and every one of its users is
    in `covenant-radar-api`. Under the old resolution the rule found no
    declaration while checking the service, returned nothing, and reported
    zero violations for four inline Literals it had not compared to anything.
    """

    _RISK_DECLARING = RISK_TIER_SET.defining_module

    def _risk_declaring(self, tmp_path: Path, members: str) -> Path:
        """Write a stand-in module binding RISK_TIERS.

        Args:
            tmp_path: Directory to write into.
            members: The tuple's contents, as source.

        Returns:
            The path written.
        """
        return write_source(
            tmp_path / DECLARING_PACKAGE / self._RISK_DECLARING,
            'from typing import Literal\n\nRISK_TIERS: tuple[Literal["LOW", "MEDIUM", '
            f'"HIGH", "CRITICAL"], ...] = ({members})\n',
        )

    def test_a_tier_literal_in_another_package_is_compared(self, tmp_path: Path) -> None:
        """The service's file is the only thing handed to the rule, exactly as
        it is when covenant-radar-api's own guard run happens."""
        self._risk_declaring(tmp_path, '"LOW", "MEDIUM", "HIGH", "CRITICAL"')
        service = write_source(
            tmp_path / "schemas.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]\n',
        )

        violations = LiteralSetRule(RISK_TIER_SET, config_for(tmp_path)).run([service])

        assert [v.kind for v in violations] == ["risk-tier-literal-drift"]

    def test_a_matching_tier_literal_passes(self, tmp_path: Path) -> None:
        self._risk_declaring(tmp_path, '"LOW", "MEDIUM", "HIGH", "CRITICAL"')
        service = write_source(
            tmp_path / "schemas.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]\n',
        )

        assert LiteralSetRule(RISK_TIER_SET, config_for(tmp_path)).run([service]) == []

    def test_a_widened_tuple_strands_every_stale_literal(self, tmp_path: Path) -> None:
        """A fifth tier added to the tuple type-checks against every existing
        annotation, because each is independent. That is the whole failure."""
        self._risk_declaring(tmp_path, '"LOW", "MEDIUM", "HIGH", "CRITICAL", "SEVERE"')
        service = write_source(
            tmp_path / "schemas.py",
            "from typing import Literal\n\n\nclass P:\n"
            '    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]\n',
        )

        violations = LiteralSetRule(RISK_TIER_SET, config_for(tmp_path)).run([service])

        assert [v.kind for v in violations] == ["risk-tier-literal-drift"]
        assert "SEVERE" in violations[0].line

    def test_its_rule_name_is_derived_from_its_subject(self, tmp_path: Path) -> None:
        assert LiteralSetRule(RISK_TIER_SET, config_for(tmp_path)).name == "risk-tier-literal"
