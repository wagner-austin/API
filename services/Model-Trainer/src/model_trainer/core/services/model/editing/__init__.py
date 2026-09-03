"""Write one factual association directly into a model's weights.

Four modules, in the order an edit passes through them.

:mod:`sites` resolves the configured site to a parameter and a module, and
refuses the ways a site can name nothing.

:mod:`rank_one` holds the arithmetic: composing the update, orienting it for
the stored weight, and solving for the value vector that moves one output to a
wanted value. No model appears in it, which is what makes it the layer whose
correctness is provable rather than plausible.

:mod:`apply` installs an update and takes it back out, returning a record of
what was written and verifying that a restore restored.

:mod:`verify` measures an applied edit against the identity it claims to
satisfy, and names any other parameter that moved.
"""

from __future__ import annotations
