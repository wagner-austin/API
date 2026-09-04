"""The one place a risk tier's name is written down.

WHY THIS MODULE EXISTS. The set was spelled SEVEN ways: a ``RiskTier`` literal
in ``covenant_domain.features`` and a second one in
``platform_core.covenant_metrics_events``, a ``RiskTierValue`` in
covenant-radar-api's Google AI schemas, four inline
``Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]`` annotations on fields and
parameters, and a ``VALID_RISK_TIERS`` tuple beside one of the aliases.
Narrowing a string to a tier was implemented three times on top of that --
twice as ``_parse_risk_tier`` and once as ``_require_risk_tier``.

Widening any one of those type-checks on its own, because each annotation is
independent and mypy has no reason to relate them. A decoder would come to
accept a tier the classifier never produces, and the failure would surface as
a prediction filed under a tier nothing reads.

WHY THERE IS NO ``RiskTier`` ALIAS HERE. The workspace's rule is that a set
like this is declared as a tuple and spelled inline at the sites that use it,
with a guard holding the two in agreement --
``monorepo_guards.literal_set_rules.RISK_TIER_SET`` reads :data:`RISK_TIERS`
and requires every ``risk_tier`` annotation in the monorepo to name exactly
it. That is the check an alias would have bought, and it works across package
boundaries, which is where the three aliases came from in the first place:
each package that could not import another's alias wrote its own.

WHY IT LIVES IN ``platform_core`` RATHER THAN IN ``covenant_domain``, which
owns the concept. ``covenant_domain`` depends on ``platform_core`` and not the
other way round, and ``platform_core.covenant_metrics_decode`` needs to narrow
a tier out of an event payload. A declaration in ``covenant_domain`` would
therefore have left ``platform_core`` with its own copy -- which is exactly
the state this replaces.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

RISK_TIERS: tuple[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], ...] = (
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
)
"""Every declared risk tier, in ascending order of risk.

Thresholds, applied by ``covenant_domain.features.classify_risk_tier``:
``LOW`` below 0.25, ``MEDIUM`` below 0.50, ``HIGH`` below 0.80, ``CRITICAL``
at 0.80 and above.

The iterable form of the set. Annotations spell the ``Literal`` inline and the
guard holds them to this tuple; see the module docstring for why that is the
shape rather than an alias.
"""


def as_risk_tier(raw: str, field: str) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    """Narrow a string to a risk tier, or refuse it.

    The single narrowing in the monorepo. Three paths need it -- the covenant
    metrics event decoder, the streaming prediction decoder, and the Google AI
    response reader -- and each had its own copy before this one.

    Written as a chain rather than a membership test because mypy will not
    narrow a ``str`` to a member of a variadic tuple through ``in``, and
    ``typing.get_args`` is unavailable under this package's
    ``disallow_any_expr``.

    Args:
        raw: The string to narrow, from outside this process.
        field: Name of the field it came from, for the refusal.

    Returns:
        The same tier, typed.

    Raises:
        JSONTypeError: If the string names no declared tier. Not defaulted to
            ``LOW`` or to anything else: a prediction filed under a tier its
            producer did not choose is worse than one that fails to decode.
    """
    if raw == "LOW":
        return "LOW"
    if raw == "MEDIUM":
        return "MEDIUM"
    if raw == "HIGH":
        return "HIGH"
    if raw == "CRITICAL":
        return "CRITICAL"
    declared = ", ".join(RISK_TIERS)
    raise JSONTypeError(f"Field '{field}' must be one of [{declared}], got '{raw}'")


def require_risk_tier(obj: JSONObject, field: str) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    """Read a required risk-tier field, narrowing it to the Literal.

    Args:
        obj: JSON object to read from.
        field: Name of the field holding the tier.

    Returns:
        The tier, typed.

    Raises:
        JSONTypeError: If the field is absent, is not a string, or names no
            declared tier.
    """
    return as_risk_tier(require_str(obj, field), field)


__all__ = ["RISK_TIERS", "as_risk_tier", "require_risk_tier"]
