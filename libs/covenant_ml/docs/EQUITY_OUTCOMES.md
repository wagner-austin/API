# Exact-window stock outcome validation

Board task: 10658605-1d95-42df-9541-44ad9c3db9a5.
Measured with actual source archives on 2026-10-14.

`equity_outcomes.evaluate_equity_window` scores a declared, equal-weight
research portfolio only when the full cohort and exact calendar are present.
`equity_outcome_records` provides an immutable result and strict JSON codecs.
The existing native price decoder, adjusted-return function, response archive,
HTTP client and report provenance are reused.

## Boundary behavior

The frozen cohort must be nonempty and unique; the selected tickers must be
a nonempty unique subset. Costs must be finite fractions between zero and one.
The exact expected dates must be canonical, unique and strictly increasing,
including both endpoints. They are supplied from the declared protocol,
never inferred from whichever prices the source happens to return.

Before the UTC calendar day following the declared exit, evaluation returns
explicit `None`: no completed window exists. This includes a pre-decision
preview. Once that date has elapsed, missing evidence is an error, not another
pending result. The caller separately decodes each history with its actual
receipt time and verifies the receipt is no later than evaluation time; native
decoding enforces the conservative 24-hour completed-bar check. Calendar
eligibility alone does not establish that bars are complete.

Mature evaluation requires history keys to equal the entire frozen cohort.
All prices must have the expected ticker and USD currency. Source observations
inside the entry/exit interval must match every declared date exactly, in
order. Missing entry, exit or intervening dates, duplicates and reordered
histories fail. Extra observations outside the declared interval are not part
of the return window. The exact entry session must begin strictly after the
decision; an entry later than the declared date cannot silently replace it.

The evaluator returns all constituent returns in cohort order, the selected
equal-weight gross return, the all-cohort equal-weight gross benchmark,
selected return after the assumed cost, and net excess over the gross
benchmark. Stable summation is applied to per-name weighted returns. The
result decoder requires finite values and exact agreement between gross,
cost, net and excess fields. It does not describe broker fills or dollar P&L.

- `EQUITY-013`: invalid reported cost, return or inconsistent net arithmetic.
- `EQUITY-014`: invalid cohort, subset, calendar, cost or decision chronology.
- `EQUITY-015`: missing/mismatched cohort evidence, dates or exact entry.

Exceptions propagate. No missing stock is dropped and no absent price becomes
a zero return. An explicit pending state is a calendar fact, not error recovery.

## Actual evidence and replay

The August 28–31, 2020 development case uses actual Apple and Microsoft
responses. Apple's capture is already documented in
[EQUITY_INPUTS.md](EQUITY_INPUTS.md). The added two-row Microsoft capture has
SHA-256 `8b0286dfe66976d3d63d2195a818c11a768a7be14d998afcb4b582b3dd94b817`;
its full native body and real receipt are retained in
`C:/Users/Test/PROJECTS/files/stock-panel-20261014`.

The arithmetic check selects Apple from that two-name test cohort. Apple
returns 0.03391222135170402 and Microsoft -0.014765435612067668 under the
provider adjustment convention. The gross equal-weight benchmark is
0.009573392869818176. Net excess is 0.022338828481885842 at the 0.002 cost
assumption and 0.019338828481885843 at 0.005. These selected historical inputs
test arithmetic; they are not a result of the prospective strategy or evidence
of stock-picking profitability.

Real integration serves both actual captures over HTTP, runs the production
client and archive, and compares online and offline portfolio results. A real
HTTP 404 propagates. Other tests remove members and dates, reorder data,
change identities and costs, and test a decision exactly at session start.
No mock replaces the transport or accounting implementation.

Full `make check` passed: 3,719 tests, all guards, strict typing across source,
tests and scripts, and 100% of 17,497 statements and 3,544 branches. It took
109 seconds of the 300-second budget. The full log is
`stock-panel-20261014/make-check-outcomes.log`; the requested PowerShell command
printed its final 100 lines and propagated the successful exit status.

Separate actual-source processes produced byte-identical `outcomes-first.json`
and `outcomes-replay.json`, SHA-256:
`f72e8c743eb0181d279eee53aa57621242204d3387c106d82954219a2e4109df`.
The RunRecord verifier checked 94 paths including the frozen pilot, original
financial inputs, both historical prices, receipts and the recorded evaluation
time. Offline replay preserves that original evaluation instant.

At **2026-10-14T06:14:26.802882Z**, the real frozen eighteen-company pilot
correctly evaluates to pending: all 21 expected October 14–November 11 dates
are retained, no future price is invented, and its portfolio outcome is null.
The report records zero orders and `advance_to_funded_trading: false`.
Actual future acquisition remains necessary before this code can score that
window; these tests and this report do not claim a scheduled collection ran.
