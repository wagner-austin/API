"""Exercise policies for Bermudan arithmetic-average Asian puts.

A policy is learned by Longstaff-Schwartz regression on simulated paths and
then applied, unchanged, to hypothetical contracts written on realized price
paths. The realized comparison is the policy's maturity-equivalent payoff
against holding the same contract to maturity, paired per contract and
intervalled over contract-start months, because contracts starting in nearby
months share most of their path and are not independent.

These are policy comparisons on hypothetical contracts, not option prices or
trading profits.
"""

from __future__ import annotations
