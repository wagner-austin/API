# Philadelphia 2025 temporal transfer, frozen October 14, 2026

This tests whether the descriptive high-variation/falling-price result found
in the 2026 maker panel persists in an earlier year. It is retrospective
temporal transfer, not forward validation or a claim of pristine holdout.
The inventory, including native outcomes, was previously captured; no 2025
price-based performance is being used to choose this protocol.

Scope is all KXHIGHPHIL contracts with event dates 2025-01-01 through 2025-12-31,
subject to the existing collector's complete accounting. Retain all exclusions,
incomplete quote grids and scalar settlements. Do not substitute another city,
year, quote time or fill proxy after observing coverage or outcomes.

Use the existing `scripts.market_research` CLI and original maker parameters:
16:00 America/New_York on the preceding date, ask 200 through 1600 inclusive,
improve 100, one contract, 1000 ms latency, 3600000 ms maximum quote age,
maker fee coefficient zero as an assumption, and 100 additional friction units.
Primary trade-through is 200; 100 and 0 remain sensitivities, never replacements.
All units and regime definitions are those of MAKER_REGIME_PROTOCOL.md.

Primary diagnostic is the `high_falling` group under two-cent-through execution.
Report its markets, filled dates, fill proxies, total and friction-adjusted P&L,
date-clustered interval and calendar serial interval. A positive point estimate
alone is insufficient; record whether both intervals exclude zero and whether
the extra-cent total is positive. Even if all hold, this is a simulation
candidate requiring further independent evidence and executable fills.

Report the other six groups and pooled results as descriptive context without
promoting a winning alternative. Acquisition failures remain failures. Record
price/trade coverage before interpreting profitability. No orders are permitted
by this research protocol. Preserve the frozen prospective campaign.

Write new archives and reports, never replace the original 2026 evidence.
Pin this protocol's bytes in the new plan before acquisition, retain the exact
implementation digest, and independently replay the captured archive. This
protocol is not a randomized experiment or a correction for the earlier search.
