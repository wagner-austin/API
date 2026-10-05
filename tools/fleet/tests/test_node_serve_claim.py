"""A serve's claiming rides a queue that does not answer (MCPs board task 8993c306).

On 2026-10-05 a deploy recreated mcp-fleet at 11:54:37Z, the serving loop's
listing was refused at 11:54:58Z, and the refusal ended diphtheria's serve;
row bde22e57 then closed 105 s after its check. The cases here give
:mod:`fleet.cli.node_serve_claim` passes and listings that answer, raise
``QUEUE_UNANSWERED`` or fail otherwise, recorded by :class:`Recorded`, and
pin that a call the queue did not answer marks the serve as recovering
rather than ending it, that recovering runs the collect pass and then the
fill pass at each poll until the queue answers both, and that every other
failure still raises. The last case runs :func:`fleet.cli.node_serve.serve_loop`
through such an outage on a pinned clock.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from concurrent.futures import Future

import pytest
from platform_core.error_codes_tooling import McpClientErrorCode
from platform_core.errors import AppError, FleetErrorCode
from platform_core.mcp_client import McpHttpResponse

from fleet.cli import node_serve
from fleet.cli.node_serve_claim import Claiming, answered, queued_here
from fleet.core import _test_hooks, queue
from fleet.core.queue_transport import answering
from tests._node_agent_fixtures import _credentials_in_env, _sourced_config
from tests.conftest import DEMO_NOW, FakeClock, pin_clock
from tests.test_node_serve import FIRST_FIRE, ScriptedWatch

__all__ = ["_credentials_in_env", "_sourced_config"]

#: What the endpoint refused every serving runner with on 2026-10-05.
REFUSED = "http://127.0.0.1:8035/mcp did not answer: URLError: [WinError 10061] refused"

#: One job id, as a listing names it.
JOB = "11111111-1111-4111-8111-111111111111"


def _silent() -> AppError[FleetErrorCode]:
    """The error a call nothing answered raises.

    Returns:
        ``QUEUE_UNANSWERED``.
    """
    return AppError(code=FleetErrorCode.QUEUE_UNANSWERED, message=REFUSED)


class Recorded:
    """A serve's passes and questions, each answered from a script and recorded.

    Every script holds one entry per call: None answers, an AppError is
    raised; a listing's entry is its job ids, None for a listing the queue
    did not answer, or an AppError.

    Attributes:
        calls: Each step's name, in the order taken.
        settled: What the watch's count of settled runs reads.
    """

    calls: list[str]
    settled: int

    def __init__(
        self,
        *,
        collects: Sequence[AppError[FleetErrorCode] | None] = (),
        fills: Sequence[AppError[FleetErrorCode] | None] = (),
        listings: Sequence[frozenset[str] | None] = (),
        unreported: Sequence[bool] = (),
    ) -> None:
        """Bind the scripts.

        Args:
            collects: One per collect pass.
            fills: One per fill pass.
            listings: One per listing.
            unreported: One per ask whether a start report went unanswered.
        """
        self.calls = []
        self.settled = 0
        self._collects = list(collects)
        self._fills = list(fills)
        self._listings = list(listings)
        self._unreported = list(unreported)

    def collect(self) -> None:
        """Record a collect pass and answer or raise as scripted.

        Raises:
            AppError: As scripted.
        """
        self.calls.append("collect")
        failure = self._collects.pop(0)
        if failure is not None:
            raise failure

    def fill(self) -> None:
        """Record a fill pass and answer or raise as scripted.

        Raises:
            AppError: As scripted.
        """
        self.calls.append("fill")
        failure = self._fills.pop(0)
        if failure is not None:
            raise failure

    def queued(self) -> frozenset[str] | None:
        """Record a listing.

        Returns:
            The scripted job ids, or None.
        """
        self.calls.append("queued")
        return self._listings.pop(0)

    def start_unreported(self) -> bool:
        """The next scripted answer, False once the script is spent.

        Returns:
            Whether a start report went unanswered.
        """
        return self._unreported.pop(0) if self._unreported else False

    def closed(self) -> int:
        """How many runs the watch has settled.

        Returns:
            :attr:`settled`.
        """
        return self.settled

    def claiming(self) -> Claiming:
        """A claiming bound to these steps.

        Returns:
            It, with nothing run.
        """
        return Claiming(
            alias="diphtheria",
            collect=self.collect,
            fill=self.fill,
            queued=self.queued,
            start_unreported=self.start_unreported,
            closed=self.closed,
        )


class TestTheQueuedJobsWhenTheQueueDoesNotAnswer:
    def test_are_none_and_the_silence_is_logged(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def refused(
            url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
        ) -> McpHttpResponse:
            raise ConnectionRefusedError(10061, "refused")

        _test_hooks.http_post = answering(refused)

        with caplog.at_level("INFO"):
            listed = queued_here(queue.load_credentials(), alias="diphtheria")

        assert listed is None
        assert (
            caplog.records[-1]
            .getMessage()
            .startswith("diphtheria: the queue did not answer its listing: ")
        )

    def test_raise_a_status_the_queue_did_answer(self, sourced_config: pathlib.Path) -> None:
        def busy(
            url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
        ) -> McpHttpResponse:
            return McpHttpResponse(status=503, content_type="text/plain", body="starting")

        _test_hooks.http_post = answering(busy)

        with pytest.raises(AppError) as raised:
            queued_here(queue.load_credentials(), alias="diphtheria")

        assert raised.value.code is McpClientErrorCode.HTTP_STATUS


class TestAnswered:
    def test_is_true_for_a_pass_that_ran(self) -> None:
        steps = Recorded(collects=[None])

        assert answered(steps.collect, alias="diphtheria", what="collect") is True

    def test_is_false_and_logged_for_a_pass_the_queue_did_not_answer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        steps = Recorded(collects=[_silent()])

        with caplog.at_level("INFO"):
            ran = answered(steps.collect, alias="diphtheria", what="collect")

        assert ran is False
        assert caplog.records[-1].getMessage() == (
            "diphtheria: the queue did not answer its collect pass; it runs again at the "
            f"next poll: {REFUSED}"
        )

    def test_raises_any_other_failure(self) -> None:
        malformed = AppError(code=FleetErrorCode.QUEUE_ANSWER_MALFORMED, message="no id")
        steps = Recorded(fills=[malformed])

        with pytest.raises(AppError) as raised:
            answered(steps.fill, alias="diphtheria", what="fill")

        assert raised.value is malformed


class TestClaiming:
    def test_an_unanswered_collect_skips_the_fill_and_recovers_at_the_next_poll(self) -> None:
        steps = Recorded(collects=[_silent(), None], fills=[None])
        claiming = steps.claiming()

        claiming.passes(fill=True)
        recovering_after_open = claiming.recovering
        claiming.poll()

        assert recovering_after_open is True
        assert steps.calls == ["collect", "collect", "fill"]
        assert claiming.recovering is False
        assert claiming.fills == 1

    def test_keeps_recovering_while_the_fill_goes_unanswered_and_never_lists(self) -> None:
        steps = Recorded(collects=[None, None, None], fills=[_silent(), _silent(), None])
        claiming = steps.claiming()

        claiming.passes(fill=True)
        claiming.poll()
        recovering_mid_outage = claiming.recovering
        claiming.poll()

        assert recovering_mid_outage is True
        assert steps.calls == ["collect", "fill", "collect", "fill", "collect", "fill"]
        assert claiming.recovering is False
        assert claiming.fills == 3

    def test_an_unanswered_listing_recovers_with_both_passes_then_lists_again(self) -> None:
        steps = Recorded(
            collects=[None, None],
            fills=[None, None],
            listings=[None, frozenset(), frozenset()],
        )
        claiming = steps.claiming()

        claiming.passes(fill=True)
        claiming.poll()
        recovering_after_silence = claiming.recovering
        claiming.poll()
        claiming.poll()

        assert recovering_after_silence is True
        assert steps.calls == ["collect", "fill", "queued", "collect", "fill", "queued"]
        assert claiming.recovering is False

    def test_a_start_report_left_unanswered_runs_the_collect_pass_at_the_next_poll(
        self,
    ) -> None:
        steps = Recorded(collects=[None, None], fills=[None, None], listings=[], unreported=[True])
        claiming = steps.claiming()

        claiming.passes(fill=True)
        claiming.poll()

        assert steps.calls == ["collect", "fill", "collect", "fill"]
        assert claiming.recovering is False

    def test_an_unanswered_fill_on_an_arrival_recovers_at_the_next_poll(self) -> None:
        steps = Recorded(
            collects=[None, None], fills=[None, _silent(), None], listings=[frozenset({JOB})]
        )
        claiming = steps.claiming()

        claiming.passes(fill=True)
        claiming.poll()
        recovering_after_arrival = claiming.recovering
        claiming.poll()

        assert recovering_after_arrival is True
        assert steps.calls == ["collect", "fill", "queued", "fill", "collect", "fill"]
        assert claiming.fills == 3

    def test_fills_on_a_settle_and_not_on_a_listing_with_nothing_new(self) -> None:
        steps = Recorded(
            collects=[None],
            fills=[None, None, None],
            listings=[frozenset({JOB}), frozenset({JOB}), frozenset({JOB})],
        )
        claiming = steps.claiming()

        claiming.passes(fill=True)
        claiming.poll()
        claiming.poll()
        steps.settled = 1
        claiming.poll()

        assert steps.calls == ["collect", "fill", "queued", "fill", "queued", "queued", "fill"]
        assert claiming.fills == 3

    def test_passes_without_a_fill_runs_only_the_collect_pass(self) -> None:
        steps = Recorded(collects=[None])
        claiming = steps.claiming()

        claiming.passes(fill=False)

        assert steps.calls == ["collect"]
        assert claiming.fills == 0
        assert claiming.recovering is False

    def test_raises_a_listing_failure_the_queue_answered(self) -> None:
        malformed = AppError(code=FleetErrorCode.QUEUE_ANSWER_MALFORMED, message="no id")
        steps = Recorded(collects=[malformed])
        claiming = steps.claiming()

        with pytest.raises(AppError) as raised:
            claiming.passes(fill=True)

        assert raised.value is malformed


class TestAServeThroughAQueueOutage:
    def test_claims_on_at_the_first_poll_the_queue_answers_and_serves_to_the_handover(
        self,
    ) -> None:
        clock: FakeClock = pin_clock(DEMO_NOW)
        # Opening passes answered; the first two listings go unanswered and
        # the first recovery's collect too; the second recovery is answered,
        # and every listing after it shows nothing new.
        polls = (FIRST_FIRE - node_serve.HANDOVER_SECONDS - DEMO_NOW) // 5
        steps = Recorded(
            collects=[None, _silent(), None],
            fills=[None, None],
            listings=[None] + [frozenset()] * (polls - 3),
        )
        watch = ScriptedWatch([True])

        served = node_serve.serve_loop(
            watch,
            Future(),
            alias="diphtheria",
            started=DEMO_NOW,
            serve_seconds=60,
            poll_seconds=5,
            steps=node_serve.ServeSteps(
                collect=steps.collect,
                fill=steps.fill,
                queued=steps.queued,
                rolled=lambda: "roll-1",
                launching=lambda: 0,
                start_unreported=steps.start_unreported,
            ),
        )

        assert steps.calls[:6] == ["collect", "fill", "queued", "collect", "collect", "fill"]
        assert steps.calls[6:] == ["queued"] * (polls - 3)
        assert served["fills"] == 2
        assert served["handed_over"] == FIRST_FIRE - node_serve.HANDOVER_SECONDS
        assert clock.seconds == FIRST_FIRE - node_serve.HANDOVER_SECONDS
