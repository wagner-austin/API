"""Tests for dead-lettering messages the worker cannot process.

A consumer with no dead-letter path cannot make progress past a message it
cannot decode: the offset never advances, so the same message is redelivered on
every restart. These tests drive the real StreamingWorker against the in-repo
fakes and assert both halves of the fix — the message is preserved on the DLQ
topic, and the commit position moves past it.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

from ._test_worker_fixtures import (
    REQUIRED_METRICS,
    make_covenant,
    make_deal,
    make_measurement_event,
    make_streaming_worker,
)


class TestUndecodableMessages:
    """A payload that cannot be decoded is dead-lettered, not fatal."""

    def test_malformed_json_does_not_raise(self) -> None:
        """A non-JSON payload is handled instead of killing the run loop.

        Regression guard: decode raised JSONTypeError out of poll, through
        run_once and run, terminating the worker. Because the offset was never
        advanced, restarting replayed the same message forever.
        """
        worker, fake_consumer, _p, _m, _pr, _d, _c, _mr, _rr = make_streaming_worker()
        fake_consumer.add_message(
            value=b"this is not json",
            topic="measurements",
            partition=0,
            offset=7,
        )

        messages, periods = worker.run_once()

        assert messages == 1
        assert periods == 0
        assert worker.buffer_size == 0

    def test_payload_is_preserved_on_the_dlq_topic(self) -> None:
        """The original bytes and Kafka position are recorded verbatim."""
        worker, fake_consumer, fake_producer, _m, _pr, _d, _c, _mr, _rr = make_streaming_worker()
        fake_consumer.add_message(
            value=b'{"type": "covenant.measurement.v1"}',
            topic="measurements",
            partition=3,
            offset=42,
        )

        worker.run_once()

        dlq_messages = [m for m in fake_producer.messages if m.topic == "dlq"]
        assert len(dlq_messages) == 1

        envelope = narrow_json_to_dict(load_json_str(dlq_messages[0].value.decode("utf-8")))
        assert require_str(envelope, "type") == "covenant.dlq.v1"
        assert require_str(envelope, "reason") == "undecodable_payload"
        assert require_str(envelope, "source_topic") == "measurements"
        assert require_int(envelope, "source_partition") == 3
        assert require_int(envelope, "source_offset") == 42
        assert require_str(envelope, "payload") == '{"type": "covenant.measurement.v1"}'
        # The reason is carried through so the topic can be triaged.
        assert require_str(envelope, "detail") != ""

    def test_invalid_utf8_is_still_dead_letterable(self) -> None:
        """Undecodable bytes do not break the envelope itself.

        The payload is decoded with replacement so the DLQ record is always
        serialisable, however mangled the input.
        """
        worker, fake_consumer, fake_producer, _m, _pr, _d, _c, _mr, _rr = make_streaming_worker()
        fake_consumer.add_message(
            value=b"\xff\xfe not utf-8",
            topic="measurements",
            partition=0,
            offset=1,
        )

        worker.run_once()

        dlq_messages = [m for m in fake_producer.messages if m.topic == "dlq"]
        assert len(dlq_messages) == 1

    def test_commit_position_advances_past_a_dead_lettered_message(self) -> None:
        """The whole point: the offset must move, or the loop never progresses.

        The bad offset is recorded as seen but never marked pending, so once
        surrounding work drains the position advances past it.
        """
        worker, fake_consumer, _p, _m, _pr, _d, _c, _mr, _rr = make_streaming_worker()
        fake_consumer.add_message(
            value=b"not json",
            topic="measurements",
            partition=0,
            offset=500,
        )

        worker.run_once()

        assert worker._commit_positions() == (
            {"topic": "measurements", "partition": 0, "offset": 501},
        )

    def test_dead_lettering_an_earlier_offset_does_not_rewind_the_position(self) -> None:
        """A replayed poison message must not drag the commit position back.

        After a seek or a rebalance a partition can re-deliver offsets below
        the highest already seen. Recording the highest rather than the latest
        keeps the position monotonic.
        """
        worker, fake_consumer, _p, _m, _pr, _d, _c, _mr, _rr = make_streaming_worker()

        fake_consumer.add_message(
            value=b"not json",
            topic="measurements",
            partition=0,
            offset=90,
        )
        worker.run_once()

        assert worker._commit_positions() == (
            {"topic": "measurements", "partition": 0, "offset": 91},
        )

        # An earlier offset arrives; the position must stay where it was.
        fake_consumer.add_message(
            value=b"also not json",
            topic="measurements",
            partition=0,
            offset=12,
        )
        worker.run_once()

        assert worker._commit_positions() == (
            {"topic": "measurements", "partition": 0, "offset": 91},
        )

    def test_a_poison_message_does_not_block_later_valid_ones(self) -> None:
        """Good messages after a poison one are still processed normally."""
        worker, fake_consumer, fake_producer, _m, _pr, deal_repo, cov_repo, _mr, _rr = (
            make_streaming_worker()
        )
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        fake_consumer.add_message(
            value=b"not json",
            topic="measurements",
            partition=0,
            offset=0,
        )
        worker.run_once()

        for index, metric_name in enumerate(REQUIRED_METRICS):
            event = make_measurement_event(
                metric_name=metric_name,
                metric_value=REQUIRED_METRICS[metric_name],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=1 + index,
            )
            worker.run_once()

        predictions = [m for m in fake_producer.messages if m.topic == "predictions"]
        assert len(predictions) == 1

        # Nothing pending, so the position clears the whole batch.
        assert worker._commit_positions() == (
            {"topic": "measurements", "partition": 0, "offset": 6},
        )
