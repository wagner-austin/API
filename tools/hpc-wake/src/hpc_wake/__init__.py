"""Announce Slurm terminal states on the agent board.

The missing producer in the wake chain. The delivery half already exists --
``task_events``, ``board-watch``, and a Monitor wake an idle session on an
``@mention`` -- but nothing turned a Slurm terminal state into a board event,
so every session that submitted a cluster job hand-rolled a bounded ``sacct``
polling loop and burned a turn per few minutes of job time.

One cycle, one job: read the hpc3 ledger, ask accounting about every job not
yet observed to have ended, post one board note per (submitter, project)
group of newly terminal jobs -- tagging the submitting session's label the
ledger recorded -- and only then write the closure rows that stop those jobs
being asked about again. Post-then-close makes delivery at-least-once: a
crash between the two repeats an announcement, never loses one.

The polling loop lives in the scheduler that calls the CLI, where its
interval is visible, for the same reason ``board-watch`` refuses to follow.
"""
