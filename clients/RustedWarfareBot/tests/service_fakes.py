"""An in-memory database speaking the service's connection Protocols.

Not a mock: a tiny real implementation of the queue semantics the SQL
expresses -- insert-if-absent, oldest-queued-first, lease uniqueness,
heartbeat staleness -- so tests assert the service's behavior against a
store that actually stores. The SQL text itself is matched by its verb and
table, which is the seam's honest limit: proving the statements against a
live Postgres is the integration run's job, not the unit suite's.
"""

from __future__ import annotations

from collections.abc import Sequence


class FakeJobRow:
    """One match_jobs row, mutable the way the statements mutate it."""

    def __init__(self, job_id: int, values: tuple[str | int, ...]) -> None:
        """File one inserted row.

        Args:
            job_id: The serial id the fake assigns.
            values: The insert's parameters: batch, label, seed, config,
                match, job.
        """
        self.job_id = job_id
        self.batch = values[0]
        self.label = values[1]
        self.seed = values[2]
        self.config = values[3]
        self.match = values[4]
        self.job = values[5]
        # The row-value union rather than str, so a test can plant a
        # corrupt state and prove the service refuses it.
        self.state: str | int = "queued"
        self.worker = ""
        self.clone_index = -1
        self.heartbeat_age = 0
        self.ok: bool | None = None
        # The row-value union rather than str, so a test can plant a
        # corrupt card and prove the service refuses it.
        self.card: str | int = ""


class FakeCursor:
    """Executes the service's statements against the in-memory tables."""

    def __init__(self, store: FakeStore) -> None:
        """Open a cursor over the shared store.

        Args:
            store: The tables this cursor reads and writes.
        """
        self._store = store
        self._rows: list[tuple[str | int, ...]] = []

    def execute(self, sql: str, params: Sequence[str | int | bool] = ()) -> None:
        """Run one statement by its verb and table.

        Args:
            sql: The statement text.
            params: Values for the placeholders.

        Raises:
            AssertionError: When the statement is one the service never
                issues -- a new statement needs a new fake behavior, loudly.
        """
        text = " ".join(sql.split())
        handlers = (
            ("CREATE TABLE", self._create),
            ("ALTER TABLE match_jobs ADD COLUMN IF NOT EXISTS card", self._add_card_column),
            ("INSERT INTO match_jobs", self._insert_job),
            ("SELECT id, batch, config, match, job FROM match_jobs", self._select_queued),
            ("SELECT clone_index FROM clone_leases", self._select_leases),
            ("SELECT state, count(*) FROM match_jobs", self._count_states),
            ("SELECT label, seed, state, card FROM match_jobs", self._select_results),
            ("INSERT INTO clone_leases", self._insert_lease),
            ("UPDATE match_jobs SET state = 'running'", self._claim_update),
            ("UPDATE match_jobs SET heartbeat_at = now()", self._heartbeat),
            ("UPDATE match_jobs SET state = %s", self._finish),
            ("UPDATE match_jobs SET state = 'queued'", self._reap),
            ("DELETE FROM clone_leases WHERE job_id", self._release),
        )
        for prefix, handler in handlers:
            if text.startswith(prefix):
                handler(tuple(params))
                return
        raise AssertionError(f"the service issued an unexpected statement: {text}")

    def _create(self, params: tuple[str | int | bool, ...]) -> None:
        self._rows = []

    def _add_card_column(self, params: tuple[str | int | bool, ...]) -> None:
        # Every fake row already carries card; the migration is a no-op here.
        self._rows = []

    def _select_queued(self, params: tuple[str | int | bool, ...]) -> None:
        queued: list[tuple[str | int, ...]] = [
            (row.job_id, row.batch, row.config, row.match, row.job)
            for row in self._store.jobs
            if row.state == "queued"
        ]
        self._rows = queued[:1]

    def _select_leases(self, params: tuple[str | int | bool, ...]) -> None:
        self._rows = [(index,) for index in self._store.leases]
        thief = self._store.thief
        if thief is not None:
            # The racing worker's commit lands just after this read -- the
            # production interleaving navpair48 exposed on 2026-08-07.
            index, worker, job_id = thief
            self._store.leases[index] = (worker, job_id)
            self._store.thief = None

    def _count_states(self, params: tuple[str | int | bool, ...]) -> None:
        batch = params[0]
        counts: dict[str | int, int] = {}
        for row in self._store.jobs:
            if row.batch == batch:
                counts[row.state] = counts.get(row.state, 0) + 1
        rows: list[tuple[str | int, ...]] = []
        for state, count in counts.items():
            rows.append((state, count))
        self._rows = rows

    def _select_results(self, params: tuple[str | int | bool, ...]) -> None:
        batch = params[0]
        matched = [row for row in self._store.jobs if row.batch == batch]
        matched.sort(key=_result_order)
        self._rows = [(row.label, row.seed, row.state, row.card) for row in matched]

    def _insert_lease(self, params: tuple[str | int | bool, ...]) -> None:
        index, worker, job_id = params[0], params[1], params[2]
        assert isinstance(index, int)
        assert isinstance(worker, str) and isinstance(job_id, int)
        if index in self._store.leases:
            # ON CONFLICT DO NOTHING: the loser reads back no row.
            self._rows = []
            return
        self._store.leases[index] = (worker, job_id)
        self._rows = [(index,)]

    def _release(self, params: tuple[str | int | bool, ...]) -> None:
        job_id = params[0]
        held = [i for i, (_, leased_job) in self._store.leases.items() if leased_job == job_id]
        for index in held:
            del self._store.leases[index]
        self._rows = []

    def _insert_job(self, values: tuple[str | int | bool, ...]) -> None:
        batch, label, seed = values[0], values[1], values[2]
        for row in self._store.jobs:
            if (row.batch, row.label, row.seed) == (batch, label, seed):
                self._rows = []
                return
        self._store.serial += 1
        self._store.jobs.append(FakeJobRow(self._store.serial, values))
        self._rows = [(self._store.serial,)]

    def _claim_update(self, values: tuple[str | int | bool, ...]) -> None:
        worker, clone_index, job_id = values
        for row in self._store.jobs:
            if row.job_id == job_id:
                row.state = "running"
                assert isinstance(worker, str)
                assert isinstance(clone_index, int)
                row.worker = worker
                row.clone_index = clone_index
                row.heartbeat_age = 0
        self._rows = []

    def _heartbeat(self, params: tuple[str | int | bool, ...]) -> None:
        job_id = params[0]
        for row in self._store.jobs:
            if row.job_id == job_id:
                row.heartbeat_age = 0
        self._rows = []

    def _finish(self, values: tuple[str | int | bool, ...]) -> None:
        state, ok, card, job_id = values
        for row in self._store.jobs:
            if row.job_id == job_id:
                assert isinstance(state, str)
                assert isinstance(ok, bool)
                assert isinstance(card, str)
                row.state = state
                row.ok = ok
                row.card = card
        self._rows = []

    def _reap(self, params: tuple[str | int | bool, ...]) -> None:
        stale_seconds = params[0]
        assert isinstance(stale_seconds, int)
        reaped: list[tuple[str | int, ...]] = []
        for row in self._store.jobs:
            if row.state == "running" and row.heartbeat_age > stale_seconds:
                row.state = "queued"
                row.worker = ""
                row.clone_index = -1
                reaped.append((row.job_id,))
        if self._store.poison_reap_row is not None:
            reaped.append(self._store.poison_reap_row)
        self._rows = reaped

    def fetchone(self) -> Sequence[str | int] | None:
        """Return the next produced row, or None past the end."""
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self) -> Sequence[Sequence[str | int]]:
        """Return every remaining produced row."""
        rows = list(self._rows)
        self._rows = []
        return rows


def _result_order(row: FakeJobRow) -> tuple[str, int]:
    """Order results by label then seed, tolerating a planted corrupt seed.

    Args:
        row: One job row.

    Returns:
        The sort key the real query's ORDER BY produces.
    """
    seed = row.seed
    return str(row.label), seed if isinstance(seed, int) else -1


class FakeStore:
    """The two tables, shared by every cursor of one fake connection."""

    def __init__(self) -> None:
        """Open empty tables."""
        self.jobs: list[FakeJobRow] = []
        # Keyed by the row-value union rather than int so a test can plant
        # a corrupt row and prove the service refuses it.
        self.leases: dict[str | int, tuple[str | int, str | int]] = {}
        self.serial = 0
        # A corrupt RETURNING row the reap query will produce when set, so a
        # test can prove the service refuses it through the public path.
        self.poison_reap_row: tuple[str | int, ...] | None = None
        # A racing worker's lease, committed between a claim's read of the
        # lease table and its insert -- consumed by the next lease read.
        self.thief: tuple[int, str, int] | None = None


class FakeConnection:
    """A connection over one in-memory store, recording lifecycle calls."""

    def __init__(self, store: FakeStore | None = None) -> None:
        """Open over a fresh or shared store.

        Args:
            store: Tables to share with another connection, or None for
                fresh ones.
        """
        self.store = store if store is not None else FakeStore()
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        """Open a cursor over the shared store."""
        return FakeCursor(self.store)

    def commit(self) -> None:
        """Count a commit."""
        self.commits += 1

    def rollback(self) -> None:
        """Count a rollback."""
        self.rollbacks += 1

    def close(self) -> None:
        """Record the close."""
        self.closed = True
