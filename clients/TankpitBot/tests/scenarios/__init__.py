"""Scenario-based offline testing for the TankPit bot.

This package owns the "would the bot get stuck / loop / make a bad
decision?" test class. Every scenario invokes the same ``decide()``
function the live bot uses; the only test-specific code is input
construction and output assertions. World state is built by driving
typed protocol messages through the production dispatcher, so the
code path from input to decision is the production code path
end-to-end.

Layout:

* ``_harness.py`` -- :class:`BotScenario` builder + ``decide_one_tick``
* ``_wire_builders.py`` -- realistic message constructors per msg type
* ``_invariants.py`` -- universal decision properties (binary)
* ``_metrics.py`` -- efficiency measurements (numeric)
* ``_baselines.py`` -- corpus-median baselines for metrics
* ``_stuck_detector.py`` -- JSONL replay flagging stuck/looping ticks
* ``test_*.py`` -- the actual scenarios and corpus-replay gates

Scenario tests are part of ``make check``; failures gate commits.
"""
