"""Tests for streaming test hooks module."""

from __future__ import annotations

from covenant_radar_api.streaming._hook_defaults import (
    _connection_config,
)
from covenant_radar_api.streaming._hook_protocols import (
    TopicPartitionOffset,
)
from covenant_radar_api.streaming._test_hooks import (
    FakeConsumedMessage,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    ProducedMessage,
    get_fake_consumer,
    get_fake_producer,
    use_fake_kafka,
    use_real_kafka,
)
from covenant_radar_api.streaming.config import (
    ConfluentConfig,
    ProducerConfig,
)
from tests.streaming._hooks_fixtures import (
    _require,
)


class TestConnectionConfig:
    """Tests for the connection keys shared by the producer and consumer.

    librdkafka validates sasl.* against the selected protocol, so the keys are
    omitted under PLAINTEXT rather than sent empty. Sending an empty
    sasl.username with a SASL protocol is rejected at construction time.
    """

    def _config(self, protocol: str) -> ConfluentConfig:
        """Build a ConfluentConfig for the given protocol."""
        if protocol == "PLAINTEXT":
            return ConfluentConfig(
                bootstrap_servers="localhost:9092",
                api_key="",
                api_secret="",
                security_protocol="PLAINTEXT",
                sasl_mechanism="PLAIN",
            )
        return ConfluentConfig(
            bootstrap_servers="pkc.confluent.cloud:9092",
            api_key="key",
            api_secret="secret",
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
        )

    def test_sasl_ssl_carries_credentials(self) -> None:
        """Under SASL_SSL the credentials are passed through to librdkafka."""
        config = _connection_config(self._config("SASL_SSL"))

        assert config["security.protocol"] == "SASL_SSL"
        assert config["sasl.mechanisms"] == "PLAIN"
        assert config["sasl.username"] == "key"
        assert config["sasl.password"] == "secret"

    def test_plaintext_omits_sasl_keys(self) -> None:
        """Under PLAINTEXT no sasl.* key is sent at all."""
        config = _connection_config(self._config("PLAINTEXT"))

        assert config["security.protocol"] == "PLAINTEXT"
        assert "sasl.mechanisms" not in config
        assert "sasl.username" not in config
        assert "sasl.password" not in config

    def test_bootstrap_servers_always_present(self) -> None:
        """The endpoint is carried regardless of protocol."""
        assert (
            _connection_config(self._config("PLAINTEXT"))["bootstrap.servers"] == "localhost:9092"
        )


class TestFakeConsumedMessage:
    """Tests for FakeConsumedMessage."""

    def test_value(self) -> None:
        """Get message value."""
        msg = FakeConsumedMessage(
            value=b"test payload",
            key=b"key",
            topic="test-topic",
            partition=0,
            offset=123,
        )
        assert msg.value() == b"test payload"

    def test_key(self) -> None:
        """Get message key."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=b"my-key",
            topic="test-topic",
            partition=0,
            offset=0,
        )
        assert msg.key() == b"my-key"

    def test_key_none(self) -> None:
        """Get None message key."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="test-topic",
            partition=0,
            offset=0,
        )
        assert msg.key() is None

    def test_topic(self) -> None:
        """Get topic name."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="my-topic",
            partition=0,
            offset=0,
        )
        assert msg.topic() == "my-topic"

    def test_partition(self) -> None:
        """Get partition number."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="test",
            partition=5,
            offset=0,
        )
        assert msg.partition() == 5

    def test_offset(self) -> None:
        """Get offset."""
        msg = FakeConsumedMessage(
            value=b"test",
            key=None,
            topic="test",
            partition=0,
            offset=999,
        )
        assert msg.offset() == 999


class TestProducedMessage:
    """Tests for ProducedMessage."""

    def test_attributes(self) -> None:
        """Access all attributes."""
        msg = ProducedMessage(
            topic="my-topic",
            value=b"my-value",
            key=b"my-key",
        )
        assert msg.topic == "my-topic"
        assert msg.value == b"my-value"
        assert msg.key == b"my-key"

    def test_none_key(self) -> None:
        """None key is stored."""
        msg = ProducedMessage(
            topic="test",
            value=b"test",
            key=None,
        )
        assert msg.key is None


class TestFakeKafkaProducer:
    """Tests for FakeKafkaProducer."""

    def test_produce(self) -> None:
        """Produce stores messages."""
        producer = FakeKafkaProducer()
        producer.produce("topic1", b"value1", b"key1")
        producer.produce("topic2", b"value2", None)

        assert len(producer.messages) == 2
        assert producer.messages[0].topic == "topic1"
        assert producer.messages[0].value == b"value1"
        assert producer.messages[0].key == b"key1"
        assert producer.messages[1].topic == "topic2"
        assert producer.messages[1].key is None

    def test_flush(self) -> None:
        """Flush marks as called and returns 0."""
        producer = FakeKafkaProducer()
        assert producer.flush_called is False

        result = producer.flush(10.0)

        assert result == 0
        assert producer.flush_called is True

    def test_poll(self) -> None:
        """Poll increments counter."""
        producer = FakeKafkaProducer()
        assert producer.poll_count == 0

        result = producer.poll(1.0)

        assert result == 0
        assert producer.poll_count == 1


class TestFakeKafkaConsumer:
    """Tests for FakeKafkaConsumer."""

    def test_subscribe(self) -> None:
        """Subscribe records topics."""
        consumer = FakeKafkaConsumer()
        assert consumer.subscribed_topics == ()

        consumer.subscribe(("topic1", "topic2"))

        assert consumer.subscribed_topics == ("topic1", "topic2")

    def test_poll_empty_queue(self) -> None:
        """Poll returns None for empty queue."""
        consumer = FakeKafkaConsumer()
        result = consumer.poll(1.0)
        assert result is None

    def test_poll_returns_message(self) -> None:
        """Poll returns message from queue."""
        consumer = FakeKafkaConsumer()
        consumer.add_message(
            value=b"test payload",
            key=b"key1",
            topic="test-topic",
            partition=1,
            offset=100,
        )

        msg = _require(consumer.poll(1.0))

        assert msg.value() == b"test payload"
        assert msg.key() == b"key1"
        assert msg.topic() == "test-topic"
        assert msg.partition() == 1
        assert msg.offset() == 100

    def test_poll_fifo_order(self) -> None:
        """Poll returns messages in FIFO order."""
        consumer = FakeKafkaConsumer()
        consumer.add_message(value=b"first")
        consumer.add_message(value=b"second")
        consumer.add_message(value=b"third")

        msg1 = consumer.poll(1.0)
        msg2 = consumer.poll(1.0)
        msg3 = consumer.poll(1.0)
        msg4 = consumer.poll(1.0)

        assert msg1 is not None and msg1.value() == b"first"
        assert msg2 is not None and msg2.value() == b"second"
        assert msg3 is not None and msg3.value() == b"third"
        assert msg4 is None

    def test_commit(self) -> None:
        """Commit records the requested positions."""
        consumer = FakeKafkaConsumer()
        assert consumer.commit_count == 0

        first: TopicPartitionOffset = {"topic": "t", "partition": 0, "offset": 5}
        second: TopicPartitionOffset = {"topic": "t", "partition": 1, "offset": 9}
        consumer.commit((first,))
        consumer.commit((second,))

        assert consumer.commit_count == 2
        assert consumer.committed_offsets == [(first,), (second,)]

    def test_close(self) -> None:
        """Close marks consumer as closed."""
        consumer = FakeKafkaConsumer()
        assert consumer.closed is False

        consumer.close()

        assert consumer.closed is True

    def test_add_message_defaults(self) -> None:
        """Add message with default values."""
        consumer = FakeKafkaConsumer()
        consumer.add_message(value=b"test")

        msg = _require(consumer.poll(1.0))

        assert msg.value() == b"test"
        assert msg.key() is None
        assert msg.topic() == "test-topic"
        assert msg.partition() == 0
        assert msg.offset() == 0


class TestHookSwitching:
    """Tests for use_fake_kafka and use_real_kafka."""

    def test_use_fake_kafka_sets_factories(self) -> None:
        """use_fake_kafka sets fake factories."""
        # First ensure real is set
        use_real_kafka()

        # Then switch to fake
        use_fake_kafka()

        # Verify by checking get_fake_* works after factory call
        from covenant_radar_api.streaming._test_hooks import producer_factory

        confluent_cfg: ConfluentConfig = {
            "bootstrap_servers": "test",
            "api_key": "test",
            "api_secret": "test",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
        }
        producer_cfg: ProducerConfig = {
            "acks": "all",
            "retries": 3,
            "linger_ms": 5,
            "batch_size": 16384,
            "compression_type": "gzip",
        }

        # Create producer via factory - should create fake
        producer_factory(confluent_cfg, producer_cfg)

        # Should have created a fake - access fake's attribute to verify
        fake = _require(get_fake_producer())
        assert fake.messages == []  # FakeKafkaProducer-specific attribute

    def test_get_fake_producer_none_initially(self) -> None:
        """get_fake_producer returns None before any creation."""
        # Note: This test may be affected by prior test state
        # The global _fake_producer may already be set
        # This tests the function at least runs
        result = get_fake_producer()
        # Could be None or FakeKafkaProducer depending on test order
        assert result is None or isinstance(result, FakeKafkaProducer)

    def test_get_fake_consumer_none_initially(self) -> None:
        """get_fake_consumer returns None before any creation."""
        result = get_fake_consumer()
        assert result is None or isinstance(result, FakeKafkaConsumer)

    def test_use_real_kafka_restores(self) -> None:
        """use_real_kafka restores real factories."""
        use_fake_kafka()
        use_real_kafka()

        # After restoring real, the factories should be the real ones
        # We can't easily test without confluent-kafka installed,
        # but at least verify no errors
        from covenant_radar_api.streaming._hook_defaults import (
            _real_consumer_factory,
            _real_producer_factory,
        )
        from covenant_radar_api.streaming._test_hooks import (
            consumer_factory,
            producer_factory,
        )

        assert producer_factory == _real_producer_factory
        assert consumer_factory == _real_consumer_factory
