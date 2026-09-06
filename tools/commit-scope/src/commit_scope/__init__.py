"""Refuse a commit carrying staged paths its author did not declare.

WHY THIS EXISTS. In a tree several sessions edit at once, the git index is
shared mutable state with no lock, and ``git commit`` takes ALL of it.
``git add <path>`` protects the ADD; nothing protects the COMMIT. Anything
another session stages between one and the other ships under your message and
your authorship.

Measured twice in three hours on 2026-09-04/05, both times by sessions already
following the explicit-path staging rule that was supposed to prevent it:

* ``9d945451``  a commit swept another session's staged deletions.
* ``09d2a04b``  an ``--amend`` swept another session's staged files; unwound,
  relanded as ``7e275441``. ``--only`` does not save an amend -- it rebases
  the named paths onto a commit that already contains the swept ones.

WHAT IT DOES AND WHAT IT HONESTLY DOES NOT. Declare ``COMMIT_SCOPE`` and a
commit carrying anything outside it is REFUSED, with the intruders named.
Declare nothing and the staged set is printed as a receipt and the commit
proceeds.

That second case is a narrowing, not a close, and is described that way
wherever it appears. Refusing every undeclared commit would block every human
in the repository, and a check people disable protects nothing -- so the
undeclared path buys the one thing it honestly can: the staged set is shown at
the moment of commit, which is where both incidents were visible and unlooked
at. The native close, where it applies, is ``git commit -- <paths>``, which
git honours regardless of what else is staged.

The decision is pure (:mod:`commit_scope.scope`), the wording is separate
(:mod:`commit_scope.report`), and git, the environment and stdout are seams
(:mod:`commit_scope._test_hooks`) -- so every property of this package is
testable without a repository, and the one integration test that does use a
repository uses a TEMPORARY index rather than the shared one it protects.
"""
