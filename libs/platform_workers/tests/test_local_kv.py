"""Tests for the Redis-free key-value store a cluster job runs against.

The interesting assertions are the ones about AGREEMENT with Redis. This
implementation exists so training code can run unchanged on a compute node,
which is only true if it answers the way the real client does -- and every
place it does not would be a difference that shows up nowhere except on the
cluster, hours into a run, with no Redis to compare against.
"""

from __future__ import annotations

from platform_workers.local_kv import LocalKV
from platform_workers.redis import RedisStrProto


class _Clock:
    """A hand-advanced clock, so expiry is tested without waiting."""

    __slots__ = ("now",)

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current reading.

        Returns:
            Seconds since the clock was made.
        """
        return self.now


class _Sink:
    """Collects published events the way a log would receive them."""

    __slots__ = ("lines",)

    def __init__(self) -> None:
        """Start with nothing published."""
        self.lines: list[str] = []

    def __call__(self, channel: str, message: str) -> None:
        """Record one event.

        Args:
            channel: Channel name.
            message: Message body.
        """
        self.lines.append(f"{channel}:{message}")


def _store() -> tuple[LocalKV, _Sink, _Clock]:
    """Build a store with a controllable clock and sink.

    Returns:
        The store, its sink, and its clock.
    """
    sink = _Sink()
    clock = _Clock()
    return LocalKV(publish=sink, clock=clock), sink, clock


class TestStrings:
    def test_a_value_round_trips(self) -> None:
        kv, _, _ = _store()
        kv.set("k", "v")
        assert kv.get("k") == "v"

    def test_an_absent_key_reads_none(self) -> None:
        kv, _, _ = _store()
        assert kv.get("missing") is None

    def test_set_reports_success_like_the_client(self) -> None:
        kv, _, _ = _store()
        assert kv.set("k", "v") is True

    def test_a_second_set_replaces(self) -> None:
        kv, _, _ = _store()
        kv.set("k", "first")
        kv.set("k", "second")
        assert kv.get("k") == "second"


class TestDelete:
    def test_deleting_a_present_key_reports_one(self) -> None:
        kv, _, _ = _store()
        kv.set("k", "v")
        assert kv.delete("k") == 1
        assert kv.get("k") is None

    def test_deleting_an_absent_key_reports_zero(self) -> None:
        kv, _, _ = _store()
        assert kv.delete("nothing") == 0

    def test_it_deletes_a_hash(self) -> None:
        kv, _, _ = _store()
        kv.hset("h", {"a": "1"})
        assert kv.delete("h") == 1
        assert kv.hgetall("h") == {}

    def test_it_deletes_a_set(self) -> None:
        kv, _, _ = _store()
        kv.sadd("s", "m")
        assert kv.delete("s") == 1
        assert kv.scard("s") == 0


class TestExpiry:
    """Implemented rather than ignored: a caller that sets a TTL is saying
    the value stops being true, and serving it afterwards answers a question
    with a stale fact.
    """

    def test_a_key_survives_until_its_deadline(self) -> None:
        kv, _, clock = _store()
        kv.set("k", "v")
        kv.expire("k", 10)
        clock.now = 9.0
        assert kv.get("k") == "v"

    def test_a_key_is_gone_at_its_deadline(self) -> None:
        kv, _, clock = _store()
        kv.set("k", "v")
        kv.expire("k", 10)
        clock.now = 10.0
        assert kv.get("k") is None

    def test_expiring_an_absent_key_reports_false(self) -> None:
        """Redis does not create a key by expiring it."""
        kv, _, _ = _store()
        assert kv.expire("nothing", 10) is False

    def test_expiring_a_present_key_reports_true(self) -> None:
        kv, _, _ = _store()
        kv.set("k", "v")
        assert kv.expire("k", 10) is True

    def test_a_hash_can_expire(self) -> None:
        kv, _, clock = _store()
        kv.hset("h", {"a": "1"})
        assert kv.expire("h", 5) is True
        clock.now = 5.0
        assert kv.hgetall("h") == {}
        assert kv.hget("h", "a") is None

    def test_a_set_can_expire(self) -> None:
        kv, _, clock = _store()
        kv.sadd("s", "m")
        assert kv.expire("s", 5) is True
        clock.now = 5.0
        assert kv.scard("s") == 0
        assert kv.sismember("s", "m") is False

    def test_hget_alone_sees_the_expiry(self) -> None:
        """Reached without hgetall having already dropped the key: every
        reader must notice expiry for itself, or which one a caller happened
        to use first would decide what it sees."""
        kv, _, clock = _store()
        kv.hset("h", {"a": "1"})
        kv.expire("h", 5)
        clock.now = 5.0
        assert kv.hget("h", "a") is None

    def test_sismember_alone_sees_the_expiry(self) -> None:
        kv, _, clock = _store()
        kv.sadd("s", "m")
        kv.expire("s", 5)
        clock.now = 5.0
        assert kv.sismember("s", "m") is False

    def test_rewriting_a_key_clears_its_ttl(self) -> None:
        """Redis's SET drops the TTL. A store that kept it would expire a
        value the caller had just refreshed."""
        kv, _, clock = _store()
        kv.set("k", "v")
        kv.expire("k", 10)
        kv.set("k", "fresh")
        clock.now = 100.0
        assert kv.get("k") == "fresh"

    def test_writing_a_hash_field_after_expiry_starts_clean(self) -> None:
        kv, _, clock = _store()
        kv.hset("h", {"old": "1"})
        kv.expire("h", 5)
        clock.now = 5.0
        kv.hset("h", {"new": "2"})
        assert kv.hgetall("h") == {"new": "2"}

    def test_adding_to_a_set_after_expiry_starts_clean(self) -> None:
        kv, _, clock = _store()
        kv.sadd("s", "old")
        kv.expire("s", 5)
        clock.now = 5.0
        kv.sadd("s", "new")
        assert kv.scard("s") == 1
        assert kv.sismember("s", "old") is False


class TestHashes:
    def test_fields_round_trip(self) -> None:
        kv, _, _ = _store()
        kv.hset("h", {"a": "1", "b": "2"})
        assert kv.hgetall("h") == {"a": "1", "b": "2"}
        assert kv.hget("h", "a") == "1"

    def test_it_reports_how_many_fields_were_new(self) -> None:
        kv, _, _ = _store()
        assert kv.hset("h", {"a": "1", "b": "2"}) == 2
        assert kv.hset("h", {"a": "9", "c": "3"}) == 1

    def test_an_existing_field_is_overwritten(self) -> None:
        kv, _, _ = _store()
        kv.hset("h", {"a": "1"})
        kv.hset("h", {"a": "2"})
        assert kv.hget("h", "a") == "2"

    def test_an_absent_hash_reads_empty(self) -> None:
        kv, _, _ = _store()
        assert kv.hgetall("nothing") == {}
        assert kv.hget("nothing", "a") is None

    def test_the_returned_hash_is_a_copy(self) -> None:
        """A caller mutating the result would be writing to the store
        without going through it, which the real client never permits."""
        kv, _, _ = _store()
        kv.hset("h", {"a": "1"})
        got = kv.hgetall("h")
        got["a"] = "tampered"
        assert kv.hget("h", "a") == "1"


class TestSets:
    def test_a_member_round_trips(self) -> None:
        kv, _, _ = _store()
        assert kv.sadd("s", "m") == 1
        assert kv.sismember("s", "m") is True
        assert kv.scard("s") == 1

    def test_adding_a_duplicate_reports_zero_and_does_not_grow(self) -> None:
        kv, _, _ = _store()
        kv.sadd("s", "m")
        assert kv.sadd("s", "m") == 0
        assert kv.scard("s") == 1

    def test_an_absent_set_is_empty(self) -> None:
        kv, _, _ = _store()
        assert kv.scard("nothing") == 0
        assert kv.sismember("nothing", "m") is False


class TestPublish:
    def test_it_reaches_the_sink(self) -> None:
        kv, sink, _ = _store()
        kv.publish("events", "step 1")
        assert sink.lines == ["events:step 1"]

    def test_it_is_also_kept_for_assertion(self) -> None:
        kv, _, _ = _store()
        kv.publish("events", "a")
        kv.publish("events", "b")
        assert kv.published == [("events", "a"), ("events", "b")]

    def test_it_reports_no_subscribers(self) -> None:
        """Nothing is listening on a compute node, and claiming otherwise
        would let a caller believe an event was delivered somewhere."""
        kv, _, _ = _store()
        assert kv.publish("events", "x") == 0


class TestConnectionSurface:
    def test_ping_succeeds_because_there_is_no_connection_to_lose(self) -> None:
        kv, _, _ = _store()
        assert kv.ping() is True

    def test_ping_accepts_the_clients_keyword_arguments(self) -> None:
        kv, _, _ = _store()
        assert kv.ping(timeout=1) is True

    def test_close_leaves_the_data_readable(self) -> None:
        """Redis's close drops a connection and leaves the data. A store that
        cleared on close would answer differently from the real client, and
        only on the cluster."""
        kv, _, _ = _store()
        kv.set("k", "v")
        kv.close()
        assert kv.get("k") == "v"


class TestItSatisfiesTheProtocolTheTrainerDependsOn:
    def test_it_implements_every_method_the_protocol_declares(self) -> None:
        """A shape-drift guard. The trainer takes RedisStrProto; a method
        added there and not here would fail on a compute node and nowhere
        else, because that is the only deployment using this class.
        """
        declared = {name for name in dir(RedisStrProto) if not name.startswith("_")}
        assert declared <= set(dir(LocalKV))
        assert declared == {
            "close",
            "delete",
            "expire",
            "get",
            "hget",
            "hgetall",
            "hset",
            "ping",
            "publish",
            "sadd",
            "scard",
            "set",
            "sismember",
        }
