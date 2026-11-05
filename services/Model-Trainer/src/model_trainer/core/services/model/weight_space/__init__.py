"""Weight-space programming: a language model's edit, applied and measured.

Board task 7964f9f6. Takehi, Teoh, Zhang and Khattab ("Sparks of
Weight-Space Programming", MIT CSAIL, 2026-11-02) report that a frontier
language model can read a network's raw weights and write new ones with no
optimiser. This package is what puts such an edit into one of our networks
and measures it against the network it started from.

:mod:`program` applies a
:class:`~model_trainer.core.contracts.weight_program.WeightProgram` through the
same snapshot, restore and every-parameter digest checks the rank-one arm
uses, so an edit that wrote somewhere it did not name is refused rather than
measured.

:mod:`mountain_car` and :mod:`control_policy` are the small-network arm, the
post's Experiment 4 in miniature: a policy trained into a local minimum, an
edit written from its weights, and success counted before and after.

:mod:`repetition` and :mod:`lm_edit_arm` are the language-model arm: GPT-2's
greedy repetition loops, an edit aimed at them, and the extraction-eval cloze
set as the measure of what else the edit moved.
"""

from __future__ import annotations
