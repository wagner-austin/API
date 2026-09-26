"""A registration other than the one the rule was extracted from, driven end to end.

A parameterised rule whose only exercised configuration is the one it was
extracted from is a rule that has not been shown to be parameterised. These
use the risk-tier set's own declaring module, tuple name and field name.

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
    LiteralSetRule,
)
from tests._literal_set_support import DECLARING_PACKAGE, config_for, write_source


class TestTheRiskTierSet:
    """A second registration, and the one the monorepo-root resolution was needed for.

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
        """Two instances must not share a name, or reports become ambiguous."""
        assert LiteralSetRule(RISK_TIER_SET, config_for(tmp_path)).name == "risk-tier-literal"
        assert (
            LiteralSetRule(CORPUS_FORMAT_SET, config_for(tmp_path)).name == "corpus-format-literal"
        )
