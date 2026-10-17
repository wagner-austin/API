"""Target-aware captions and the real caption-word boundary codec."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.bot.ai.scoring_types import ReasonKind
from tankpit_bot.service.demo_caption import CaptionAccumulator
from tankpit_bot.service.demo_caption_words import (
    REASON_WORDS,
    CaptionWordsDict,
    caption_words,
    decode_caption_words,
    encode_caption_words,
)
from tests.service.test_demo_caption import _decision


@pytest.mark.parametrize("resource", ["fuel", "equipment"])
@pytest.mark.parametrize(
    "reason, verb",
    [
        (ReasonKind.HARVEST_FRAME_SHIFT, "Next: Look toward"),
        (ReasonKind.HARVEST_LEG_WALK, "Next: Collect"),
    ],
)
def test_the_committed_target_names_the_resource(
    resource: str,
    reason: ReasonKind,
    verb: str,
) -> None:
    """Both harvest steps distinguish fuel from equipment."""
    words = caption_words(reason, resource)
    assert words["doing"] == f"{verb} {resource}"
    assert resource in words["why"]
    record = _decision(1, reason.value)
    record["fields"]["resource_target_kind"] = resource
    captions = CaptionAccumulator()
    captions.absorb([record])
    assert captions.window(0)[0]["doing"] == words["doing"]


def test_no_target_does_not_invent_a_resource() -> None:
    """Exploration without a selected container remains general."""
    assert (
        caption_words(ReasonKind.HARVEST_LEG_WALK, "") == REASON_WORDS[ReasonKind.HARVEST_LEG_WALK]
    )
    assert caption_words(ReasonKind.FORAGE_RADAR, "fuel") == REASON_WORDS[ReasonKind.FORAGE_RADAR]
    assert caption_words(ReasonKind.SEARCH_COLLECT_LOCAL, "")["doing"] == "Next: Jump toward fuel"


def test_malformed_resource_context_is_refused() -> None:
    """Nontext context cannot silently become a generic caption."""
    record = _decision(1, "harvest_leg_walk")
    record["fields"]["resource_target_kind"] = 12
    with pytest.raises(ValueError, match="CAPTION_RESOURCE_KIND_INVALID"):
        CaptionAccumulator().absorb([record])


def test_caption_word_codec_validates_both_fields() -> None:
    """The panel payload round-trips and refuses missing/malformed words."""
    words = CaptionWordsDict(doing="Heading to fuel", why="It needs more fuel.")
    assert decode_caption_words(encode_caption_words(words)) == words
    with pytest.raises(JSONTypeError):
        decode_caption_words({"doing": 12, "why": "A sentence."})
    with pytest.raises(JSONTypeError):
        decode_caption_words({"doing": "Fuel"})
