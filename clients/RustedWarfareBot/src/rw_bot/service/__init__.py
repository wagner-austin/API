"""The match service: engine slots become a queue.

Phase zero of [[harness-match-service]]: a Postgres-backed job queue with a
clone-lease allocator and a host-side worker that plays matches through the
existing harness seams unchanged. The coordination failure this removes is
measured, not hypothetical -- sweeps claim clone dirs by worker index, so two
batches launched concurrently collide on ``.game-w1``, and the night of
2026-08-05 avoided that only because every actor happened to respect a
convention no allocator enforces.

The queue is a table claimed with ``SELECT ... FOR UPDATE SKIP LOCKED``; the
lease table owns clone indices. One store, because at match throughput the
queue does effectively zero operations per second and Postgres buys
durability plus a job history that is just rows.
"""
