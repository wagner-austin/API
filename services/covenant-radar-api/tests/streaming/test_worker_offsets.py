"""Tests for streaming offset-commit safety.

The worker buffers measurements until a period is complete, so the offset of a
polled message is not safe to commit until the period holding it has been
processed and produced. These tests drive the real StreamingWorker against the
in-repo fakes and assert on the positions actually handed to the consumer.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from time import sleep

from platform_core.json_utils import dump_json_str

from covenant_radar_api.streaming._test_hooks import TopicPartitionOffset

from ._test_worker_fixtures import (
    REQUIRED_METRICS,
    make_covenant,
    make_deal,
    make_measurement_event,
    make_streaming_worker,
)


class TestCommitPositions:
    """The commit position never passes an unprocessed message."""

    def test_pending_offset_holds_the_commit_position_back(self) -> None:
        """A buffered, unprocessed offset caps the position at that offset.

        The period needs five metrics; two are supplied, so nothing is
        processed and the lowest buffered offset is the furthest the worker may
        commit to.
        """
        worker, fake_consumer, _p, _m, _pr, deal_repo, cov_repo, _mr, _rr = make_streaming_worker()
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        metric_names = list(REQUIRED_METRICS.keys())
        for index in range(2):
            event = make_measurement_event(
                metric_name=metric_names[index],
                metric_value=REQUIRED_METRICS[metric_names[index]],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=10 + index,
            )
            worker.run_once()

        assert worker.buffer_size == 1

        positions = worker._commit_positions()

        assert positions == ({"topic": "measurements", "partition": 0, "offset": 10},)

    def test_position_advances_past_the_last_message_once_flushed(self) -> None:
        """With nothing buffered, the position is one past the highest offset.

        Committing `highest + 1` is the Kafka convention: the committed value is
        the offset of the next message to consume.
        """
        worker, fake_consumer, _p, _m, _pr, deal_repo, cov_repo, _mr, _rr = make_streaming_worker()
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        metric_names = list(REQUIRED_METRICS.keys())
        for index, metric_name in enumerate(metric_names):
            event = make_measurement_event(
                metric_name=metric_name,
                metric_value=REQUIRED_METRICS[metric_name],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=100 + index,
            )
            worker.run_once()

        # All five metrics arrived, so the period flushed.
        assert worker.buffer_size == 0

        positions = worker._commit_positions()

        assert positions == ({"topic": "measurements", "partition": 0, "offset": 105},)

    def test_partitions_are_tracked_independently(self) -> None:
        """A partition that flushed is not held back by one that has not."""
        worker, fake_consumer, _p, _m, _pr, deal_repo, cov_repo, _mr, _rr = make_streaming_worker()
        deal_repo.create(make_deal("deal-001"))
        deal_repo.create(make_deal("deal-002"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))
        cov_repo.create(make_covenant("cov-002", "deal-002"))

        metric_names = list(REQUIRED_METRICS.keys())

        # Partition 0: a complete period for deal-001, so it flushes.
        for index, metric_name in enumerate(metric_names):
            event = make_measurement_event(
                deal_id="deal-001",
                metric_name=metric_name,
                metric_value=REQUIRED_METRICS[metric_name],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=index,
            )
            worker.run_once()

        # Partition 1: a single metric for deal-002, left buffered.
        partial = make_measurement_event(
            deal_id="deal-002",
            metric_name=metric_names[0],
            metric_value=REQUIRED_METRICS[metric_names[0]],
        )
        fake_consumer.add_message(
            value=dump_json_str(partial).encode("utf-8"),
            topic="measurements",
            partition=1,
            offset=77,
        )
        worker.run_once()

        positions = {(p["topic"], p["partition"]): p["offset"] for p in worker._commit_positions()}

        assert positions[("measurements", 0)] == 5
        assert positions[("measurements", 1)] == 77


class TestIncompletePeriods:
    """A period that can never produce features is dropped, not crashed on."""

    def test_timed_out_incomplete_period_does_not_crash_the_loop(self) -> None:
        """A partial period that times out is discarded instead of raising.

        Regression guard: min_metrics_per_period (3) is lower than the five
        metrics extract_features reads, and _should_process_buffer also releases
        a period once it times out. Such a period used to reach feature
        extraction and raise KeyError straight out of run_once, taking down the
        worker.
        """
        worker, fake_consumer, fake_producer, _m, _pr, deal_repo, cov_repo, _mr, _rr = (
            make_streaming_worker()
        )
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        # Two metrics only; the period can never satisfy feature extraction.
        metric_names = list(REQUIRED_METRICS.keys())
        for index in range(2):
            event = make_measurement_event(
                metric_name=metric_names[index],
                metric_value=REQUIRED_METRICS[metric_names[index]],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=300 + index,
            )
            worker.run_once()

        assert worker.buffer_size == 1

        # buffer_timeout_seconds is 0.1 in the test config; sleep past it so the
        # period is considered ready while still incomplete.
        sleep(0.15)
        _messages, periods = worker.run_once()

        assert periods == 0
        assert worker.buffer_size == 0
        # Nothing was published from incomplete data.
        assert len(fake_producer.messages) == 0

    def test_discarded_period_releases_its_offsets(self) -> None:
        """A discarded period stops holding the partition's commit position.

        Leaving its offsets pending would replay the same incomplete period on
        every restart and block that partition's position permanently.
        """
        worker, fake_consumer, _p, _m, _pr, deal_repo, cov_repo, _mr, _rr = make_streaming_worker()
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        event = make_measurement_event(
            metric_name=next(iter(REQUIRED_METRICS)),
            metric_value=1.0,
        )
        fake_consumer.add_message(
            value=dump_json_str(event).encode("utf-8"),
            topic="measurements",
            partition=0,
            offset=400,
        )
        worker.run_once()

        # Held back while buffered.
        assert worker._commit_positions() == (
            {"topic": "measurements", "partition": 0, "offset": 400},
        )

        sleep(0.15)
        worker.run_once()

        # Released once discarded, so the position advances past it.
        assert worker._commit_positions() == (
            {"topic": "measurements", "partition": 0, "offset": 401},
        )

    def test_shutdown_discards_incomplete_periods(self) -> None:
        """Shutdown drains without raising on a partially filled buffer.

        Regression guard: shutdown force-processed every buffered period
        regardless of completeness, so a SIGTERM arriving mid-period turned a
        graceful shutdown into a KeyError and aborted the rest of the drain.
        """
        worker, fake_consumer, fake_producer, _m, _pr, deal_repo, cov_repo, _mr, _rr = (
            make_streaming_worker()
        )
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        metric_names = list(REQUIRED_METRICS.keys())
        for index in range(2):
            event = make_measurement_event(
                metric_name=metric_names[index],
                metric_value=REQUIRED_METRICS[metric_names[index]],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=500 + index,
            )
            worker.run_once()

        assert worker.buffer_size == 1

        worker.shutdown()

        assert worker.buffer_size == 0
        assert len(fake_producer.messages) == 0
        assert fake_consumer.closed is True


class TestCommitIsExplicit:
    """The worker commits positions rather than an implicit consumed offset."""

    def test_commit_sends_positions_excluding_buffered_messages(self) -> None:
        """The committed positions never acknowledge a buffered message.

        Regression guard: the worker previously called an argument-less
        commit(), which advances every assigned partition to its consumed
        position. Messages still held in the buffer were acknowledged, so a
        crash before the flush lost them outright.
        """
        worker, fake_consumer, _p, _m, _pr, deal_repo, cov_repo, _mr, _rr = make_streaming_worker()
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        # commit_interval is 5 in the test config, so five polls trigger one
        # commit. Supply four metrics of a five-metric period plus one metric
        # for a different period, so nothing has flushed at commit time.
        metric_names = list(REQUIRED_METRICS.keys())
        for index in range(4):
            event = make_measurement_event(
                metric_name=metric_names[index],
                metric_value=REQUIRED_METRICS[metric_names[index]],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=index,
            )
        other_period = make_measurement_event(
            metric_name=metric_names[0],
            metric_value=REQUIRED_METRICS[metric_names[0]],
            period_start="2025-01-01",
            period_end="2025-03-31",
        )
        fake_consumer.add_message(
            value=dump_json_str(other_period).encode("utf-8"),
            topic="measurements",
            partition=0,
            offset=4,
        )

        for _ in range(5):
            worker.run_once()

        assert fake_consumer.commit_count == 1
        committed = fake_consumer.committed_offsets[0]
        # Offset 0 is still buffered, so that is the commit ceiling.
        assert committed == ({"topic": "measurements", "partition": 0, "offset": 0},)

    def test_shutdown_commits_after_flushing_the_producer(self) -> None:
        """Shutdown drains, flushes, then commits the advanced position.

        Committing before the flush would acknowledge measurements whose
        prediction events had not yet reached the broker.
        """
        worker, fake_consumer, fake_producer, _m, _pr, deal_repo, cov_repo, _mr, _rr = (
            make_streaming_worker()
        )
        deal_repo.create(make_deal("deal-001"))
        cov_repo.create(make_covenant("cov-001", "deal-001"))

        metric_names = list(REQUIRED_METRICS.keys())
        for index, metric_name in enumerate(metric_names):
            event = make_measurement_event(
                metric_name=metric_name,
                metric_value=REQUIRED_METRICS[metric_name],
            )
            fake_consumer.add_message(
                value=dump_json_str(event).encode("utf-8"),
                topic="measurements",
                partition=0,
                offset=200 + index,
            )
            worker.run_once()

        # The period completed and flushed during the loop.
        assert worker.buffer_size == 0
        commits_before_shutdown = fake_consumer.commit_count

        worker.shutdown()

        assert fake_producer.flush_called is True
        assert fake_consumer.commit_count == commits_before_shutdown + 1
        # Nothing pending, so the position is one past the highest offset seen.
        expected: TopicPartitionOffset = {
            "topic": "measurements",
            "partition": 0,
            "offset": 205,
        }
        assert fake_consumer.committed_offsets[-1] == (expected,)

    def test_request_stop_only_sets_the_flag(self) -> None:
        """request_stop performs no Kafka work, so it is signal-handler safe.

        Regression guard: the SIGTERM handler used to call shutdown() directly,
        draining and closing the consumer at an arbitrary point inside
        run_once, after which the resumed loop committed on a closed consumer.
        """
        worker, fake_consumer, fake_producer, _m, _pr, _d, _c, _mr, _rr = make_streaming_worker()

        worker.request_stop()

        assert worker.is_running is False
        assert fake_consumer.commit_count == 0
        assert fake_consumer.closed is False
        assert fake_producer.flush_called is False
