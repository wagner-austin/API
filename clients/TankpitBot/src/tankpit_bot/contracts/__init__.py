"""Contracts framework: fail hard on state entry, not soft on observation.

Phase 1 of the self-observing bot architecture. See
``docs/handoffs/self-observing-bot-architecture.md`` and the wiki page
``wiki/pages/self-observing-architecture.md``.
"""

from tankpit_bot.contracts.base import (
    ConfidenceOutOfBoundsError,
    ContractError,
    ContractViolationDict,
    NoUnsourcedFactError,
    ProvenanceRootednessError,
)
from tankpit_bot.contracts.enforcement import Contract, enforce_contract, require

__all__ = [
    "ConfidenceOutOfBoundsError",
    "Contract",
    "ContractError",
    "ContractViolationDict",
    "NoUnsourcedFactError",
    "ProvenanceRootednessError",
    "enforce_contract",
    "require",
]
