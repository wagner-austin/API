# Operations

Running the board day to day: the triage conditions that look like health,
the closure record that keeps them honest, and the budget that is the only
thing saying stop.

[The five triage conditions that look like health](../pages/triage-conditions.md) -- blocked, unaccounted, unclaimed, silent, oversized; the mirror check nobody had
[Closures, or why the unaccounted check does not rot](../pages/ledger-closures.md) -- MinJobAge, unverified sacct retention, one line per job
[The budget is per project, because nothing else says stop](../pages/budget-model.md) -- projection and consumption caps, the three-forked-workspace lesson
[What the cluster sees, and why the comment is live-only](../pages/job-identity-on-cluster.md) -- the ledger as the durable record
[Two command-length ceilings, and the token count that met both](../pages/command-length-limits.md) -- a query sized by the ledger died before reaching the cluster; why the split measures width, not tokens
[One ledger per repository, whichever worktree submits](../pages/one-ledger-per-repository.md) -- a worktree's own ledger missed 12 live members and a campaign submitted them twice; the main checkout's ledger, and the refusal for a live job no ledger here records
[WRDS experiments need durable attempts and verified results](../pages/wrds-repeatable-attempts.md) -- held submission recovery, input/runtime checks, shared run records and two matching compute-node executions
