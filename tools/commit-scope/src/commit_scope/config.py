"""The names this package reads from outside itself.

One environment variable and one git invocation, both named here so a reader
grepping for either finds the single definition rather than a literal repeated
across the CLI and its tests.
"""

from __future__ import annotations

from typing import Final

#: Where the author declares which paths this commit is theirs.
#:
#: Deliberately NOT prefixed with the name of any one AI tool. The index is
#: shared by every session and every human working this tree, and a variable
#: called after one of them would read as somebody else's concern to everyone
#: it is not named for.
SCOPE_VARIABLE: Final = "COMMIT_SCOPE"

#: The one question this package asks git.
#:
#: ``--diff-filter=d`` drops deletions so a path removed from the index is not
#: reported as staged, matching what a reader means by "what is in this
#: commit". ``--name-only`` because the decision is over paths and nothing
#: else; asking for more would invite a later change to decide on content.
STAGED_PATHS_ARGUMENTS: Final = ("diff", "--cached", "--name-only", "--diff-filter=d")

#: Proves the working directory is inside a git work tree.
#:
#: Run before the index query so a non-repository directory fails with
#: ``GIT_REPO_ROOT_UNRESOLVED`` -- naming the actual problem -- rather than
#: with an index error that sends the reader looking at staging.
REPO_ROOT_ARGUMENTS: Final = ("rev-parse", "--show-toplevel")
