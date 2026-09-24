"""Reading a service's url out of the MCPs stack's endpoint declaration,
the one place its addresses are written (MCPs board task c6fc4882)."""

from __future__ import annotations

import pytest

from platform_core.error_codes_tooling import StackEndpointErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL, stack_endpoints_text
from platform_core.stack_endpoints import STACK_ENDPOINTS_PATH, TASKBOARD_SERVICE, declared_url


def test_the_taskboard_url_is_the_one_the_declaration_names() -> None:
    assert declared_url(stack_endpoints_text(), TASKBOARD_SERVICE) == DECLARED_TASKBOARD_URL


def test_another_service_is_read_by_its_own_name() -> None:
    text = dump_json_str(
        {
            "services": {
                TASKBOARD_SERVICE: {"url": DECLARED_TASKBOARD_URL},
                "fleet-mcp": {"url": "http://127.0.0.1:8035/mcp"},
            }
        }
    )
    assert declared_url(text, "fleet-mcp") == "http://127.0.0.1:8035/mcp"


@pytest.mark.parametrize(
    "text",
    [
        dump_json_str({}),
        dump_json_str({"services": []}),
        dump_json_str({"services": {}}),
        dump_json_str({"services": {TASKBOARD_SERVICE: "http://x"}}),
        dump_json_str({"services": {TASKBOARD_SERVICE: {"runsOn": "diphtheria"}}}),
        stack_endpoints_text(""),
        dump_json_str({"services": {TASKBOARD_SERVICE: {"url": 8033}}}),
    ],
)
def test_a_declaration_without_the_services_url_is_refused_naming_it_and_the_file(
    text: str,
) -> None:
    with pytest.raises(AppError) as caught:
        declared_url(text, TASKBOARD_SERVICE)
    assert caught.value.code is StackEndpointErrorCode.UNDECLARED
    assert caught.value.message == f"{STACK_ENDPOINTS_PATH} declares no url for {TASKBOARD_SERVICE}"


def test_a_declaration_that_is_not_json_raises_the_parse_error() -> None:
    with pytest.raises(InvalidJsonError):
        declared_url("not json", TASKBOARD_SERVICE)


def test_a_declaration_that_is_not_an_object_raises_the_type_error() -> None:
    with pytest.raises(JSONTypeError):
        declared_url("[]", TASKBOARD_SERVICE)
