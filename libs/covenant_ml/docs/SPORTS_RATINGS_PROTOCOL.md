# Earlier-game team strength: exploratory protocol

Declared October 13, 2026 before calculating team ratings or their scores.
The prior temperature experiment already examined the March-April outcomes.
This is explicitly exploratory; no result can be called untouched confirmation.

Use the existing archived NHL full-game winner inventory and exact morning
quotes. Ratings start at 1500 for every identified team on October 1, 2025.
Each ordinarily resolved two-team game updates the winner and loser using
K=20 and expected probability 1/(1+10^((opponent-rating)/400)). There is no
home adjustment, margin weighting, recency multiplier or tuned rating constant.
Use stable native team identifiers, not ambiguous concatenated ticker names.
Require exactly two distinct teams and complementary $0/$1 settlements.
Apply a game's update only after BOTH team contracts have settled; process
equal-timestamp games in event-ticker order. Reject duplicate game identities
and do not interpret non-binary, incomplete or unfinalized games as wins.

At each selected game's fixed 12:00 UTC decision, forecast its YES team from
ratings built only from earlier available settlements. Require at least ten
completed games for each team before that decision. The January-February
training and March 1-April 15 evaluation dates and quote exclusions remain
those of the earlier experiment. No observation's own outcome enters its
prediction; earlier evaluation outcomes may update ratings for subsequent
games only after they settle, as a declared sequential strategy.

Fit a blend weight w in {0, 0.25, 0.5, 0.75, 1} on January-February Brier
score: forecast=(1-w)*market_midpoint+w*rating_probability. Ties favor lower w.
Freeze that weight for the later period. Report the pure rating model as a
predeclared diagnostic, not an alternative selected after observing profits.
Reuse the existing three-cent expected-margin hurdle, adverse-price allowance,
fee assumption, one-contract hypothetical policy and daily interval calculation.
Report missing identities and insufficient team-history exclusions explicitly.

Advancement to a subsequent prospective demo study requires at least 100
training and 100 evaluation observations, at least 30 selected later trades,
improved later Brier versus the market midpoint and a positive lower 95%
bound on mean daily hypothetical profit. Passing remains exploratory and
requires prospective evidence; no real-money orders are authorized.
