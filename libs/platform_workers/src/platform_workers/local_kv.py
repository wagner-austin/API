"""A key-value store for a job that runs where Redis does not.

A Slurm compute node has no Redis, no Docker, and no service to talk to. But
the training code does not depend on Redis -- it depends on
:class:`~platform_workers.redis.RedisStrProto`, a fourteen-method protocol.
That protocol is the seam, so a cluster run needs an implementation of it, not
a refactor of the trainer.

**This is not a database.** Nothing listens on a port, nothing runs between
jobs, and nothing is left behind. It is a dictionary that lives as long as the
process does.

That is sound because of what the store is FOR on a cluster. In the service
deployment the API reads this state to answer "how is my run doing". On a
compute node there is no API and no reader: every key written here is written
by the one process that might read it back, and the job's real progress signal
is its stdout, which Slurm is already capturing to a file. So the state does
not need to persist, be shared, or survive the process.

**Nothing is written to the shared filesystem.** HPC3's ``/pub`` is BeeGFS, a
parallel filesystem whose metadata servers are shared by everyone on the
cluster; a training loop writing a progress key several times a second would
be thousands of small operations an hour, felt by other users and buying
nothing. Published events go to the log instead, which is one sequential
append that Slurm was writing anyway -- and which
``hpc3-triage``'s stuck-job detection already reads, so a run that stops
producing them is already visible.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_workers.redis import RedisStrProto


class LocalKV(RedisStrProto):
    """An in-process implementation of the Redis surface the trainer uses.

    Attributes:
        published: Every ``(channel, message)`` pair, in order. Kept as well
            as emitted so a caller can assert on what a run announced without
            parsing its log back.
    """

    __slots__ = ("_clock", "_deadlines", "_hashes", "_publish", "_sets", "_strings", "published")

    def __init__(self, *, publish: Callable[[str, str], None], clock: Callable[[], float]) -> None:
        """Build a store bound to an output sink and a clock.

        Args:
            publish: Where published events go. Injected rather than chosen
                here because this module must not decide whether a cluster
                run logs to stdout, to a file, or to nothing.
            clock: Source of monotonic seconds, used only for expiry.
        """
        self._publish = publish
        self._clock = clock
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}
        self._deadlines: dict[str, float] = {}
        self.published: list[tuple[str, str]] = []

    def _expired(self, key: str) -> bool:
        """Report whether a key's TTL has passed, and drop it if so.

        Expiry is implemented rather than ignored. A caller that sets a TTL
        is saying the value stops being true, and a store that kept serving
        it would answer a question with a stale fact -- which is the one
        thing a key-value store must never do.

        Args:
            key: Key to check.

        Returns:
            True when the key had a deadline and it has passed.
        """
        deadline = self._deadlines.get(key)
        if deadline is None or self._clock() < deadline:
            return False
        self._deadlines.pop(key)
        self._strings.pop(key, None)
        self._hashes.pop(key, None)
        self._sets.pop(key, None)
        return True

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        """Report the store as reachable.

        Args:
            **kwargs: Accepted and ignored, matching the Redis client.

        Returns:
            Always True. There is no connection that could be down: the
            store is this object.
        """
        return True

    def set(self, key: str, value: str) -> bool | str | None:
        """Store a string, clearing any TTL previously set on it.

        Args:
            key: Key to write.
            value: Value to store.

        Returns:
            True, matching the Redis client's success return.
        """
        self._strings[key] = value
        self._deadlines.pop(key, None)
        return True

    def get(self, key: str) -> str | None:
        """Read a string.

        Args:
            key: Key to read.

        Returns:
            The value, or None when absent or expired.
        """
        if self._expired(key):
            return None
        return self._strings.get(key)

    def delete(self, key: str) -> int:
        """Remove a key of any type.

        Args:
            key: Key to remove.

        Returns:
            1 when something was removed, 0 when the key was already absent.
        """
        self._deadlines.pop(key, None)
        present = (
            self._strings.pop(key, None) is not None
            or self._hashes.pop(key, None) is not None
            or self._sets.pop(key, None) is not None
        )
        return 1 if present else 0

    def expire(self, key: str, time: int) -> bool:
        """Give an existing key a time to live.

        Args:
            key: Key to expire.
            time: Seconds from now.

        Returns:
            True when the key exists and a deadline was set, False otherwise
            -- Redis does not create a key by expiring it.
        """
        if key not in self._strings and key not in self._hashes and key not in self._sets:
            return False
        self._deadlines[key] = self._clock() + time
        return True

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        """Write fields into a hash.

        Args:
            key: Hash key.
            mapping: Fields to write.

        Returns:
            The number of fields that did not already exist, matching Redis.
        """
        self._expired(key)
        existing = self._hashes.setdefault(key, {})
        added = sum(1 for field in mapping if field not in existing)
        existing.update(mapping)
        return added

    def hget(self, key: str, field: str) -> str | None:
        """Read one field of a hash.

        Args:
            key: Hash key.
            field: Field name.

        Returns:
            The value, or None when the hash or field is absent or expired.
        """
        if self._expired(key):
            return None
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        """Read a whole hash.

        Args:
            key: Hash key.

        Returns:
            A copy of the hash, empty when absent or expired. A copy rather
            than the live mapping: a caller mutating the result would be
            writing to the store without going through it.
        """
        if self._expired(key):
            return {}
        return dict(self._hashes.get(key, {}))

    def publish(self, channel: str, message: str) -> int:
        """Announce an event.

        On a compute node this is the only part of the store anyone will ever
        see. It goes to the injected sink -- normally the job's log, which
        Slurm captures and which triage watches for silence.

        Args:
            channel: Channel name.
            message: Message body.

        Returns:
            0 subscribers. Nothing is listening, and saying otherwise would
            let a caller believe an event was delivered somewhere.
        """
        self.published.append((channel, message))
        self._publish(channel, message)
        return 0

    def scard(self, key: str) -> int:
        """Count a set's members.

        Args:
            key: Set key.

        Returns:
            The member count, 0 when absent or expired.
        """
        if self._expired(key):
            return 0
        return len(self._sets.get(key, set()))

    def sadd(self, key: str, member: str) -> int:
        """Add a member to a set.

        Args:
            key: Set key.
            member: Member to add.

        Returns:
            1 when the member was new, 0 when it was already present.
        """
        self._expired(key)
        members = self._sets.setdefault(key, set())
        if member in members:
            return 0
        members.add(member)
        return 1

    def sismember(self, key: str, member: str) -> bool:
        """Test set membership.

        Args:
            key: Set key.
            member: Member to test.

        Returns:
            Whether the member is present.
        """
        if self._expired(key):
            return False
        return member in self._sets.get(key, set())

    def close(self) -> None:
        """Release the store.

        Deliberately does NOT clear the contents. Redis's ``close`` drops a
        connection and leaves the data; a caller that read a key after
        closing would get different answers from the two implementations,
        and the difference would only appear on the cluster.
        """
        return


__all__ = ["LocalKV"]
