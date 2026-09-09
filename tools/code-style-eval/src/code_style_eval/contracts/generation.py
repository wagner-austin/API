"""Whether the model finished a file, which is a different fact from passing it.

WHY THIS IS ITS OWN RECORD AND NOT A FIELD ON :class:`ItemOutcome`. An item
the model never finished and an item it finished badly both score as a
failure, and they are not the same result: the first says the generation was
cut off at the token budget, the second says the code was wrong. The
generation manifest records the first, the outcome record the second, and
joining them on ``item_id`` is what produces the both-finished stratum every
guard-pass figure on the wiki page is reported over.

They are written at different times by different processes -- generation on
the cluster, scoring on this machine -- which is the other reason they are
separate files rather than one wider row.

DECODED RATHER THAN READ. This manifest was read with an inline
``json.loads`` in a scratch process until 2026-09-09, which is how a stratum
that four published tables depend on came to have no validation between the
file and the figure. A row whose ``finished`` field is absent would have read
as falsy and silently moved an item out of the stratum.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_str,
)
from typing_extensions import TypedDict


class GenerationOutcome(TypedDict):
    """One item's generation result, before any checker has seen it.

    Attributes:
        item_id: Repository-relative path of the held-out file, the key this
            joins to :class:`~code_style_eval.contracts.outcomes.ItemOutcome`
            on.
        finished: Whether the model emitted a stop token rather than running
            into the decode budget. False means the completion was TRUNCATED,
            not that it was bad.
    """

    item_id: str
    finished: bool


def encode_generation_outcome(record: GenerationOutcome) -> JSONObject:
    """Encode a :class:`GenerationOutcome` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying both fields.
    """
    return {"item_id": record["item_id"], "finished": record["finished"]}


def decode_generation_outcome(obj: JSONObject) -> GenerationOutcome:
    """Decode a :class:`GenerationOutcome` from a JSON object.

    Args:
        obj: The object to decode.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: If a field is missing or wrongly typed, or if
            ``item_id`` is empty. An empty id joins to nothing and would
            silently drop the item from whichever stratum it belonged to,
            which is a change to a denominator rather than a parse error.
    """
    item_id = require_str(obj, "item_id")
    if not item_id:
        raise JSONTypeError(
            "Field 'item_id' is empty; this manifest exists to be joined to "
            "the outcome rows on that key, and a row that joins to nothing "
            "changes a published denominator without failing"
        )
    return GenerationOutcome(item_id=item_id, finished=require_bool(obj, "finished"))


__all__ = [
    "GenerationOutcome",
    "decode_generation_outcome",
    "encode_generation_outcome",
]
