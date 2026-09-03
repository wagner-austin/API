"""Contracts for writing one factual association directly into model weights.

The unit is ONE edit: a subject, a prompt that mentions it, and the string the
model should produce instead. Everything else in this module exists so that an
edit can be recorded well enough for someone else to re-derive it, or refuse
it.

WHY THE SITE IS A RECORD AND NOT A CONSTANT. Where an edit is written -- which
layer, which module inside it, which token's representation becomes the key --
is configuration, not a discovered fact. The reference implementation carries
it in a JSON file per model, and no part of the running code derives it. Two
runs that disagree about an edit's effect can therefore differ only in these
fields, so an experiment that does not record the site has not recorded its
own method. :class:`EditSite` makes that record mandatory and typed.

WHY THE APPLIED RECORD CARRIES A DENOMINATOR. The right vector is a solve, and
its divisor is the key's dot product with the module's current input at the
edited position. A key nearly orthogonal to that input gives an arbitrarily
large update for the same requested change, which is a legitimate edit
numerically and a wrecked model in practice. The number is on the record so a
later reader can sort edits by how close to degenerate they were, rather than
discovering it from a fluency collapse.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

#: The token whose representation at the edit site becomes the key.
#:
#: ``subject_last`` takes the last token of the subject string where it occurs
#: in the rendered prompt. ``prompt_last`` takes the final token of the whole
#: prompt. They coincide only when the subject ends the prompt, and the choice
#: changes which association is written: the first ties the new value to the
#: subject wherever it appears, the second ties it to this prompt's ending.
FACT_TOKEN_STRATEGIES: tuple[Literal["subject_last", "prompt_last"], ...] = (
    "subject_last",
    "prompt_last",
)

#: The placeholder an edit prompt marks its subject with.
#:
#: Spelled out rather than reusing ``str.format``'s bare ``{}`` so that a
#: prompt containing braces for any other reason is not silently a template.
SUBJECT_MARKER = "<<SUBJECT>>"


def as_fact_token_strategy(raw: str, field: str) -> Literal["subject_last", "prompt_last"]:
    """Narrow a string to a fact-token strategy, or refuse it.

    Args:
        raw: The string to narrow.
        field: Field name for the error message.

    Returns:
        The narrowed strategy name.

    Raises:
        JSONTypeError: If the string names no known strategy.
    """
    for known in FACT_TOKEN_STRATEGIES:
        if raw == known:
            return known
    raise JSONTypeError(f"Field '{field}' must be one of {FACT_TOKEN_STRATEGIES}, got '{raw}'")


class EditSite(TypedDict):
    """Where an edit is written, and which representation keys it.

    Attributes:
        layer: Index of the transformer block to edit.
        module_template: Dotted path to the weight-bearing module inside that
            block, with one ``{}`` for the layer index, e.g.
            ``transformer.h.{}.mlp.c_proj``. A template rather than a resolved
            path because the layer is the swept variable and the module is
            not.
        fact_token: Which token's representation becomes the key.
    """

    layer: int
    module_template: str
    fact_token: Literal["subject_last", "prompt_last"]


class EditRequest(TypedDict):
    """One association to write, and the item it is measured under.

    Attributes:
        item_id: Stable identifier, unique within a request set, and the key
            that pairs this edit with its before and after measurements.
        subject: The entity the association is about, as it appears in the
            prompt. Also what ``subject_last`` looks for.
        prompt: Sentence containing exactly one ``SUBJECT_MARKER``, which the
            subject is substituted into.
        target_new: The string the edited model should produce.
    """

    item_id: str
    subject: str
    prompt: str
    target_new: str


class RankOneEditRecord(TypedDict):
    """What one applied rank-one edit did, in numbers a later reader can check.

    The vectors themselves are not here. They are large, they are not JSON,
    and a digest answers the only question a record needs to answer: whether
    two runs produced the same edit.

    Attributes:
        item_id: The request this edit satisfied.
        module: Resolved dotted path of the edited parameter.
        weight_rows: First dimension of the edited weight.
        weight_cols: Second dimension of the edited weight.
        transposed: Whether the composed update had to be transposed to match
            the stored weight. True for GPT-2's Conv1D projections and false
            for a plain linear layer, so a run that flips this flag between
            models is reporting a real difference and not a bug.
        left_digest: Folded digest of the key vector, from
            :func:`~model_trainer.core.services.model.tensor_digest.describe_tensor`,
            so an edit's identity is comparable by the same rule every other
            measurement here uses.
        right_digest: Folded digest of the value vector, same fold.
        left_norm: Euclidean norm of the key vector.
        right_norm: Euclidean norm of the value vector.
        denominator: The key's dot product with the module's input at the
            edited position, which the value solve divided by.
        update_norm: Frobenius norm of the applied update matrix.
    """

    item_id: str
    module: str
    weight_rows: int
    weight_cols: int
    transposed: bool
    left_digest: float
    right_digest: float
    left_norm: float
    right_norm: float
    denominator: float
    update_norm: float


class EditVerification(TypedDict):
    """Evidence that an applied edit did exactly what its record claims.

    A rank-one update has an exact consequence: for ANY input x, the edited
    module's output moves by the value vector scaled by the key's dot product
    with x, and by nothing else. That is checkable arithmetic rather than an
    opinion, so it is checked, and the residual is recorded.

    Attributes:
        module: The edited parameter.
        max_prediction_error: Largest absolute difference between the measured
            output change and the change the rank-one form predicts, over the
            probe inputs. Bounded by floating-point error when the edit is
            what it claims to be.
        key_output_error: Absolute difference between the module's output at
            the edited key and the value the edit was solving for.
        other_parameters_changed: Names of parameters other than ``module``
            whose bytes differ from before the edit. Empty on a correct edit,
            and named rather than counted because one leaked parameter is a
            different investigation from thirty.
    """

    module: str
    max_prediction_error: float
    key_output_error: float
    other_parameters_changed: tuple[str, ...]


def resolve_edit_module(site: EditSite) -> str:
    """Resolve a site's module template against its layer.

    Args:
        site: The site to resolve.

    Returns:
        The dotted module path, e.g. ``transformer.h.17.mlp.c_proj``.

    Raises:
        JSONTypeError: If the template has no ``{}`` placeholder, or more than
            one. Formatting a template with no placeholder silently yields the
            same path for every layer, which would make a layer sweep edit one
            site repeatedly and report it as several.
    """
    placeholders = site["module_template"].count("{}")
    if placeholders != 1:
        raise JSONTypeError(
            f"Field 'module_template' must contain exactly one '{{}}' placeholder "
            f"for the layer index, got {placeholders} in '{site['module_template']}'"
        )
    return site["module_template"].replace("{}", str(site["layer"]))


def render_edit_prompt(request: EditRequest) -> str:
    """Substitute a request's subject into its prompt.

    Args:
        request: The request to render.

    Returns:
        The prompt with the subject in place of the marker.

    Raises:
        JSONTypeError: If the prompt does not contain exactly one
            ``SUBJECT_MARKER``.
    """
    occurrences = request["prompt"].count(SUBJECT_MARKER)
    if occurrences != 1:
        raise JSONTypeError(
            f"Field 'prompt' must contain exactly one '{SUBJECT_MARKER}', "
            f"got {occurrences} in '{request['prompt']}'"
        )
    return request["prompt"].replace(SUBJECT_MARKER, request["subject"])


def encode_edit_site(site: EditSite) -> JSONObject:
    """Encode an EditSite to a JSON object.

    Args:
        site: The site to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "layer": site["layer"],
        "module_template": site["module_template"],
        "fact_token": site["fact_token"],
    }


def decode_edit_site(obj: JSONObject) -> EditSite:
    """Decode a JSON object to an EditSite.

    Args:
        obj: The object to decode.

    Returns:
        The validated site.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, the layer is
            negative, or the template does not carry exactly one placeholder.
    """
    layer = require_int(obj, "layer")
    if layer < 0:
        raise JSONTypeError(f"Field 'layer' must be a block index at or above 0, got {layer}")
    site = EditSite(
        layer=layer,
        module_template=require_str(obj, "module_template"),
        fact_token=as_fact_token_strategy(require_str(obj, "fact_token"), "fact_token"),
    )
    resolve_edit_module(site)
    return site


def encode_edit_request(request: EditRequest) -> JSONObject:
    """Encode an EditRequest to a JSON object.

    Args:
        request: The request to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "item_id": request["item_id"],
        "subject": request["subject"],
        "prompt": request["prompt"],
        "target_new": request["target_new"],
    }


def decode_edit_request(obj: JSONObject) -> EditRequest:
    """Decode a JSON object to an EditRequest.

    Args:
        obj: The object to decode.

    Returns:
        The validated request.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, any of the
            four strings is empty, or the prompt does not carry exactly one
            subject marker. An empty target is refused because an edit whose
            target is the empty string has nothing to make more likely, and it
            would score as a successful edit against every measurement that
            asks only whether the answer changed.
    """
    request = EditRequest(
        item_id=_require_nonempty_str(obj, "item_id"),
        subject=_require_nonempty_str(obj, "subject"),
        prompt=_require_nonempty_str(obj, "prompt"),
        target_new=_require_nonempty_str(obj, "target_new"),
    )
    render_edit_prompt(request)
    return request


def encode_rank_one_edit_record(record: RankOneEditRecord) -> JSONObject:
    """Encode a RankOneEditRecord to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "item_id": record["item_id"],
        "module": record["module"],
        "weight_rows": record["weight_rows"],
        "weight_cols": record["weight_cols"],
        "transposed": record["transposed"],
        "left_digest": record["left_digest"],
        "right_digest": record["right_digest"],
        "left_norm": record["left_norm"],
        "right_norm": record["right_norm"],
        "denominator": record["denominator"],
        "update_norm": record["update_norm"],
    }


def decode_rank_one_edit_record(obj: JSONObject) -> RankOneEditRecord:
    """Decode a JSON object to a RankOneEditRecord.

    Args:
        obj: The object to decode.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, or either
            weight dimension is not positive. A recorded shape with a zero
            dimension describes a matrix no update could have been applied to.
    """
    rows = require_int(obj, "weight_rows")
    cols = require_int(obj, "weight_cols")
    for name, value in (("weight_rows", rows), ("weight_cols", cols)):
        if value <= 0:
            raise JSONTypeError(f"Field '{name}' must be a positive dimension, got {value}")
    return RankOneEditRecord(
        item_id=require_str(obj, "item_id"),
        module=require_str(obj, "module"),
        weight_rows=rows,
        weight_cols=cols,
        transposed=require_bool(obj, "transposed"),
        left_digest=require_float(obj, "left_digest"),
        right_digest=require_float(obj, "right_digest"),
        left_norm=require_float(obj, "left_norm"),
        right_norm=require_float(obj, "right_norm"),
        denominator=require_float(obj, "denominator"),
        update_norm=require_float(obj, "update_norm"),
    )


def encode_edit_verification(verification: EditVerification) -> JSONObject:
    """Encode an EditVerification to a JSON object.

    Args:
        verification: The verification to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "module": verification["module"],
        "max_prediction_error": verification["max_prediction_error"],
        "key_output_error": verification["key_output_error"],
        "other_parameters_changed": list(verification["other_parameters_changed"]),
    }


def decode_edit_verification(obj: JSONObject) -> EditVerification:
    """Decode a JSON object to an EditVerification.

    Args:
        obj: The object to decode.

    Returns:
        The validated verification.

    Raises:
        JSONTypeError: If a field is missing or has the wrong type.
    """
    changed = require_list(obj, "other_parameters_changed")
    names: list[str] = []
    for index, entry in enumerate(changed):
        if not isinstance(entry, str):
            raise JSONTypeError(
                f"Field 'other_parameters_changed[{index}]' must be a parameter name string"
            )
        names.append(entry)
    return EditVerification(
        module=require_str(obj, "module"),
        max_prediction_error=require_float(obj, "max_prediction_error"),
        key_output_error=require_float(obj, "key_output_error"),
        other_parameters_changed=tuple(names),
    )


def _require_nonempty_str(obj: JSONObject, field: str) -> str:
    """Read a string field and refuse it when empty.

    Args:
        obj: The object to read from.
        field: Field name.

    Returns:
        The non-empty string.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, field)
    if value == "":
        raise JSONTypeError(f"Field '{field}' must not be empty")
    return value


__all__ = [
    "FACT_TOKEN_STRATEGIES",
    "SUBJECT_MARKER",
    "EditRequest",
    "EditSite",
    "EditVerification",
    "RankOneEditRecord",
    "as_fact_token_strategy",
    "decode_edit_request",
    "decode_edit_site",
    "decode_edit_verification",
    "decode_rank_one_edit_record",
    "encode_edit_request",
    "encode_edit_site",
    "encode_edit_verification",
    "encode_rank_one_edit_record",
    "render_edit_prompt",
    "resolve_edit_module",
]
