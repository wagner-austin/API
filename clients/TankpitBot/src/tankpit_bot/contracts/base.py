"""Contract error hierarchy for the self-observing architecture.

Phase 1 of the self-observing bot architecture (see
``docs/handoffs/self-observing-bot-architecture.md``): every state
transition carries a contract, and a violated contract raises at the
transition -- never "after N observations of the consequences."

Each specific contract is a :class:`ContractError` subclass that names
itself via ``contract_name``. The raise site records where the
violation was detected (``violated_at``) plus per-violation string
details, so the failure is a precise, structured statement instead of
a free-text log line.
"""

from __future__ import annotations

from typing import ClassVar

from typing_extensions import TypedDict


class ContractViolationDict(TypedDict):
    """Structured description of one contract violation.

    Attributes:
        contract_name: Name of the violated contract.
        violated_at: ``file:line`` of the detection site.
        details: Per-violation key/value details (all stringified).
    """

    contract_name: str
    violated_at: str
    details: dict[str, str]


class ContractError(Exception):
    """Base class for every contract violation.

    Attributes:
        contract_name: Class-level name of the contract this error
            enforces. Subclasses override it.
        violated_at: ``file:line`` of the detection site.
        details: Per-violation key/value details.
    """

    contract_name: ClassVar[str] = "contract"

    def __init__(self, violated_at: str, details: dict[str, str]) -> None:
        """Create a contract error.

        Args:
            violated_at: ``file:line`` of the detection site.
            details: Per-violation key/value details.
        """
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
        super().__init__(f"{self.contract_name} violated at {violated_at}: {rendered}")
        self.violated_at = violated_at
        self.details = details

    def violation(self) -> ContractViolationDict:
        """Return the structured violation record.

        Returns:
            ContractViolationDict describing this violation.
        """
        return ContractViolationDict(
            contract_name=self.contract_name,
            violated_at=self.violated_at,
            details=self.details,
        )


class NoUnsourcedFactError(ContractError):
    """A Fact was constructed without a complete source declaration."""

    contract_name: ClassVar[str] = "no_unsourced_fact"


class ConfidenceOutOfBoundsError(ContractError):
    """A confidence value or operation left the [0.0, 1.0] interval."""

    contract_name: ClassVar[str] = "confidence_in_bounds"


class ProvenanceRootednessError(ContractError):
    """A Fact's provenance chain is not rooted in an observation.

    Non-derived facts must originate from an observation source (a
    wire message or the game-log scrape); derived facts (client-side
    inference) must cite at least one prior source.
    """

    contract_name: ClassVar[str] = "provenance_rootedness"


class LedgerInvariantError(ContractError):
    """A ledger record violates its structural invariants.

    Raised at record time -- e.g. a dispatch context with off-map
    coordinates or a negative message index can never resolve into a
    truthful outcome, so it must not enter the ledger at all.
    """

    contract_name: ClassVar[str] = "ledger_invariant"


__all__ = [
    "ConfidenceOutOfBoundsError",
    "ContractError",
    "ContractViolationDict",
    "NoUnsourcedFactError",
    "ProvenanceRootednessError",
]
