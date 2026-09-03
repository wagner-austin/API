"""The knowledge-edit records, and every way they refuse a malformed one.

Nothing here needs a model. These are the shapes an edit is recorded in, and
what they will not accept: a site that names no layer placeholder, a request
whose prompt has nowhere to put its subject, a record whose weight has a zero
dimension. Each refusal exists because the alternative is a run that completes
and reports something untrue.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.knowledge_edit import (
    FACT_TOKEN_STRATEGIES,
    SUBJECT_MARKER,
    EditRequest,
    EditSite,
    EditVerification,
    RankOneEditRecord,
    as_fact_token_strategy,
    decode_edit_request,
    decode_edit_site,
    decode_edit_verification,
    decode_rank_one_edit_record,
    encode_edit_request,
    encode_edit_site,
    encode_edit_verification,
    encode_rank_one_edit_record,
    render_edit_prompt,
    resolve_edit_module,
)

_SITE: EditSite = {
    "layer": 17,
    "module_template": "transformer.h.{}.mlp.c_proj",
    "fact_token": "subject_last",
}

_REQUEST: EditRequest = {
    "item_id": "civic-0001",
    "subject": "the Orange County Water District",
    "prompt": f"{SUBJECT_MARKER} is governed by a board of",
    "target_new": " ten directors",
}

_RECORD: RankOneEditRecord = {
    "item_id": "civic-0001",
    "module": "transformer.h.17.mlp.c_proj.weight",
    "weight_rows": 6400,
    "weight_cols": 1600,
    "transposed": False,
    "left_digest": 123456789.0,
    "right_digest": 987654321.0,
    "left_norm": 1.0,
    "right_norm": 4.25,
    "denominator": 0.5,
    "update_norm": 8.5,
}

_VERIFICATION: EditVerification = {
    "module": "transformer.h.17.mlp.c_proj.weight",
    "max_prediction_error": 1.5e-7,
    "key_output_error": 2.5e-7,
    "other_parameters_changed": (),
}


def test_edit_site_round_trips_through_json() -> None:
    assert decode_edit_site(encode_edit_site(_SITE)) == _SITE


def test_edit_site_resolves_its_module_against_the_layer() -> None:
    assert resolve_edit_module(_SITE) == "transformer.h.17.mlp.c_proj"


@pytest.mark.parametrize(
    "template",
    ["transformer.h.mlp.c_proj", "transformer.h.{}.mlp.{}.c_proj"],
)
def test_edit_site_refuses_a_template_without_exactly_one_placeholder(template: str) -> None:
    """A template with no placeholder edits one site for every layer in a sweep.

    Two placeholders is the same defect read the other way: ``replace`` would
    fill both with the layer index and name a module that does not exist, or
    worse, one that does.
    """
    site: EditSite = {
        "layer": 3,
        "module_template": template,
        "fact_token": "prompt_last",
    }
    with pytest.raises(JSONTypeError, match="exactly one"):
        resolve_edit_module(site)


def test_edit_site_decode_refuses_a_negative_layer() -> None:
    payload: JSONObject = {
        "layer": -1,
        "module_template": "transformer.h.{}.mlp.c_proj",
        "fact_token": "subject_last",
    }
    with pytest.raises(JSONTypeError, match="at or above 0"):
        decode_edit_site(payload)


def test_edit_site_decode_refuses_an_unknown_fact_token() -> None:
    payload: JSONObject = {
        "layer": 0,
        "module_template": "transformer.h.{}.mlp.c_proj",
        "fact_token": "middle_of_subject",
    }
    with pytest.raises(JSONTypeError, match="must be one of"):
        decode_edit_site(payload)


def test_edit_site_decode_refuses_a_template_it_cannot_resolve() -> None:
    payload: JSONObject = {
        "layer": 0,
        "module_template": "transformer.wte",
        "fact_token": "subject_last",
    }
    with pytest.raises(JSONTypeError, match="exactly one"):
        decode_edit_site(payload)


@pytest.mark.parametrize("strategy", list(FACT_TOKEN_STRATEGIES))
def test_every_declared_fact_token_strategy_narrows(strategy: str) -> None:
    assert as_fact_token_strategy(strategy, "fact_token") == strategy


def test_edit_request_round_trips_through_json() -> None:
    assert decode_edit_request(encode_edit_request(_REQUEST)) == _REQUEST


def test_edit_request_renders_its_subject_into_the_prompt() -> None:
    assert render_edit_prompt(_REQUEST) == (
        "the Orange County Water District is governed by a board of"
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "is governed by a board of",
        f"{SUBJECT_MARKER} and {SUBJECT_MARKER} are governed by",
    ],
)
def test_edit_request_refuses_a_prompt_without_exactly_one_marker(prompt: str) -> None:
    request: EditRequest = {
        "item_id": "civic-0002",
        "subject": "Irvine Ranch Water District",
        "prompt": prompt,
        "target_new": " five directors",
    }
    with pytest.raises(JSONTypeError, match="exactly one"):
        render_edit_prompt(request)


@pytest.mark.parametrize("field", ["item_id", "subject", "prompt", "target_new"])
def test_edit_request_decode_refuses_an_empty_field(field: str) -> None:
    """An empty target would score as a successful edit against every check.

    The other three are refused for the ordinary reason: an edit with no
    identifier cannot be paired with its measurement.
    """
    payload: JSONObject = dict(encode_edit_request(_REQUEST))
    payload[field] = ""
    with pytest.raises(JSONTypeError, match="must not be empty"):
        decode_edit_request(payload)


def test_edit_request_decode_refuses_a_prompt_with_no_marker() -> None:
    payload: JSONObject = dict(encode_edit_request(_REQUEST))
    payload["prompt"] = "is governed by a board of"
    with pytest.raises(JSONTypeError, match="exactly one"):
        decode_edit_request(payload)


def test_rank_one_record_round_trips_through_json() -> None:
    assert decode_rank_one_edit_record(encode_rank_one_edit_record(_RECORD)) == _RECORD


@pytest.mark.parametrize("field", ["weight_rows", "weight_cols"])
def test_rank_one_record_refuses_a_non_positive_dimension(field: str) -> None:
    payload: JSONObject = dict(encode_rank_one_edit_record(_RECORD))
    payload[field] = 0
    with pytest.raises(JSONTypeError, match="positive dimension"):
        decode_rank_one_edit_record(payload)


def test_verification_round_trips_through_json() -> None:
    assert decode_edit_verification(encode_edit_verification(_VERIFICATION)) == _VERIFICATION


def test_verification_round_trips_with_named_collateral() -> None:
    verification: EditVerification = {
        "module": "transformer.h.0.mlp.c_proj.weight",
        "max_prediction_error": 0.0,
        "key_output_error": 0.0,
        "other_parameters_changed": ("transformer.wte.weight",),
    }
    assert decode_edit_verification(encode_edit_verification(verification)) == verification


def test_verification_refuses_a_non_string_parameter_name() -> None:
    payload: JSONObject = dict(encode_edit_verification(_VERIFICATION))
    payload["other_parameters_changed"] = [17]
    with pytest.raises(JSONTypeError, match="other_parameters_changed\\[0\\]"):
        decode_edit_verification(payload)
