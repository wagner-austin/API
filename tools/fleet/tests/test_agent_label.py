"""The board's label grammar, held before a submitter's label reaches any script.

MCPs board task 6c4516af A4: every build exports the job's submitter as
``BOARD_AGENT_LABEL``, and the queue only length-checks that field, so the
grammar is the only thing between a queue row and a rendered script.
"""

from __future__ import annotations

import pytest

from fleet.core.agent_label import AGENT_LABEL_VARIABLE, require_agent_label


def test_a_board_label_passes_unchanged() -> None:
    for label in ("opus-coordination-w1-0927", "fleet-node-serendipity", "abc"):
        assert require_agent_label(label) == label
    assert AGENT_LABEL_VARIABLE == "BOARD_AGENT_LABEL"


@pytest.mark.parametrize(
    "label",
    ["", "ab", "Opus-0929", "-opus-0929", "opus 0929", "opus'0929", 'opus"0929', "a" * 65],
)
def test_a_label_outside_the_grammar_is_refused_by_name(label: str) -> None:
    with pytest.raises(ValueError, match="is outside the board's kebab-case grammar"):
        require_agent_label(label)
