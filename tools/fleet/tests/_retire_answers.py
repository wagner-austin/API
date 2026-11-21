"""What the node's retire prints, as the suites script it (MCPs board task 1526ece1).

The retire's last line is its answer (:mod:`fleet.core.retire_script`): the
bytes it freed, or the process that kept the run whole.
"""

from __future__ import annotations

from typing import Final

#: The bytes the retires these suites script say they freed.
RETIRED_BYTES: Final = 1_200_000_000

#: What a retire that removed a run prints last.
RETIRED_OUTPUT: Final = f"freed {RETIRED_BYTES}\n"
