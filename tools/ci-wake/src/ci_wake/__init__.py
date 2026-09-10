"""Announce GitHub Actions verdicts on the agent board.

THE THIRD WAKE BRIDGE, AND THE FIRST WHOSE UPSTREAM IS NEITHER THE CLUSTER
NOR THIS MACHINE. ``hpc-wake`` turns a Slurm closure into a board post;
``fleet-wake`` turns a dispatch outcome into one; nothing turned a CI verdict
into one, so a session that pushed had exactly two ways to learn what
happened: sit in a bounded ``gh run watch`` loop burning turns, or forget.
Forgetting is what happened, and the workspace has the record: a workflow
sat red for weeks "with nothing watching", and two commits shipped a barrel
importing an untracked module because "GitHub Actions has started no job
since 2026-09-03 (billing), so nothing ever attempted the clean-checkout
build that would have failed in seconds".

WHAT MAKES THIS BRIDGE DIFFERENT FROM THE OTHER TWO IS THE SUBSCRIBER SIDE.
Slurm and the dispatcher both record WHO asked for the work, so their
bridges can address an announcement. GitHub records a commit author, which
on this machine is one person for every session, and a run's API payload
carries nothing that says which of six concurrent AI sessions typed
``git push``. That fact is knowable at exactly one instant -- inside the
``pre-push`` hook -- and nowhere afterwards. So enrolment is not a courtesy
this package offers; it is the only moment the addressing information
exists, which is why it is taken by a hook that runs on every push rather
than by a step anyone can skip.

One cycle, one job: read the enrolment record for pushes not yet announced,
ask GitHub what runs exist for each of those shas, take the ones where every
run has reached a terminal state, post one board note per (repository,
pusher) group, and only then write the position rows. Post-then-write makes
delivery at-least-once: a crash between the two repeats an announcement,
never loses one. Same order as both siblings, for the same reason.

A CONCLUSION IS NOT A VERDICT, AND THIS BRIDGE WILL NOT POST ONE ALONE. The
``mcps-codebase`` wiki page ``reading-ci-run-outcomes`` measures three ways a
run list misleads a reader; two of them are fatal to a one-word
announcement. Both repositories narrow their job matrix to the changed
paths, so a green run that executed one workspace and a green run that
executed forty-three render identically. And ``cancelled`` is two different
events -- a run superseded mid-flight ran something before it died, a run
evicted from the queue never created a job at all -- separable only by the
job count. Every post this package makes therefore carries the workflow
name, the conclusion, and the job count, and names the failed jobs when
there are any.

The polling loop lives in the scheduler that calls the CLI, where its
interval is visible, for the same reason ``fleet-watch`` refuses to follow
and ``fleet-wake`` has no loop of its own.
"""
