"""Paired evaluation of classifiers on shared held-out rows.

What the per-arm benchmark scoreboard cannot say, this package can: whether
one model beats another on the SAME rows by more than chance, what a feature
contributes when it is taken away, what acting on a model's probabilities is
worth, and how quality grows with training data. Every interval states the
independent unit it assumed (row, company, or time block).
"""

from __future__ import annotations
