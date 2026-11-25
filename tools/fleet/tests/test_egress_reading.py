"""Whether a node reaches the internet, and how a refusal words each ask (board task e70cdf05).

The worded exits are curl's own (``man curl``, EXIT CODES): 6 "Could not
resolve host", 7 "Failed to connect to host", 28 "Operation timeout", 35
"SSL connect error" and 60 "Peer certificate cannot be authenticated with
known CA certificates". Each expected line below is written from that table,
not from what the code prints.
"""

from __future__ import annotations

import pytest

from fleet.contracts.egress_reading import EgressAttempt, describe_attempt, reaches

PYPI = "https://pypi.org/simple/pip/"
NPM = "https://registry.npmjs.org/"


class TestReaches:
    def test_both_indexes_answering_is_reach(self) -> None:
        assert reaches(
            (EgressAttempt(target=PYPI, curl_exit=0), EgressAttempt(target=NPM, curl_exit=0))
        )

    @pytest.mark.parametrize("answered", [PYPI, NPM])
    def test_either_index_answering_alone_is_reach(self, answered: str) -> None:
        attempts = tuple(
            EgressAttempt(target=target, curl_exit=0 if target == answered else 6)
            for target in (PYPI, NPM)
        )

        assert reaches(attempts)

    @pytest.mark.parametrize("curl_exit", [1, 6, 7, 28, 35, 52, 60, 255])
    def test_no_answer_from_either_is_no_reach(self, curl_exit: int) -> None:
        assert not reaches(
            (
                EgressAttempt(target=PYPI, curl_exit=curl_exit),
                EgressAttempt(target=NPM, curl_exit=curl_exit),
            )
        )


class TestDescribeAttempt:
    @pytest.mark.parametrize(
        ("curl_exit", "described"),
        [
            (6, f"{PYPI} could not resolve its host (curl exit 6)"),
            (7, f"{PYPI} could not connect to it (curl exit 7)"),
            (28, f"{PYPI} did not answer within the probe's bound (curl exit 28)"),
            (35, f"{PYPI} failed its TLS handshake (curl exit 35)"),
            (60, f"{PYPI} presented a certificate curl does not trust (curl exit 60)"),
        ],
    )
    def test_each_exit_curl_s_table_names_is_worded(self, curl_exit: int, described: str) -> None:
        assert describe_attempt(EgressAttempt(target=PYPI, curl_exit=curl_exit)) == described

    @pytest.mark.parametrize("curl_exit", [5, 8, 27, 29, 34, 36, 59, 61])
    def test_an_exit_beside_a_worded_one_is_named_by_its_number(self, curl_exit: int) -> None:
        """The neighbours of each worded exit, so no key is off by one."""
        assert describe_attempt(EgressAttempt(target=NPM, curl_exit=curl_exit)) == (
            f"{NPM} failed (curl exit {curl_exit})"
        )
