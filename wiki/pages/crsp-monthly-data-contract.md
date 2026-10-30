---
title: CRSP monthly source and panel contract
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[crsp-monthly-holdings-validation]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/evaluation/evidence.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py"
  - "libs/covenant_ml/tests/markets/test_crsp_extract.py"
  - "libs/covenant_ml/tests/markets/test_crsp_panel.py"
  - "libs/covenant_ml/docs/crsp-monthly-v1.json"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py"
  - "libs/covenant_ml/tests/markets/test_crsp_protocol.py"
  - "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json"
  - "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "tools/fleet/src/fleet/core/research_hpc.py"
  - "tools/fleet/src/fleet/core/research_wrds.py"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/evaluation/evidence.py": "9bc1459b035fcedeab41f91b84960faebfbe73a2"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py": "b48812de06b8ffa1a4c095f26d45f44cfbc7a527"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py": "838e19bcd90c3a99173409378160b14035d12b5d"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py": "67792a2f1fd38cf8927a88d68adb04f1e8eb7c98"
  "libs/covenant_ml/tests/markets/test_crsp_extract.py": "50c6414def75db3058cd7290f5b7da56b077ab53"
  "libs/covenant_ml/tests/markets/test_crsp_panel.py": "00cf8f5894f7a2eb52c1d411ddcd99788a6b0cd9"
  "libs/covenant_ml/docs/crsp-monthly-v1.json": "42137d1e107819d0030fcae10ca1ca560d6e8f6d"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py": "221b2b1c6d526a02786d78837a228b1bd5203e7f"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "e721684f962c297aa60e8c1ebc80c1f3376da02d"
  "libs/covenant_ml/tests/markets/test_crsp_protocol.py": "28789469447c10b67050d3fd4995471c29d85893"
  "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json": "3aa4b35290d76b1a8f3c404521c96b0bddb96291"
  "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py": "bb7b6f6db20f501916ba08821c0e4efd2166bbe7"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "tools/fleet/src/fleet/core/research_hpc.py": "ac4a658ffe4ca6beae15ce56df08f712e7bddca2"
  "tools/fleet/src/fleet/core/research_wrds.py": "2130dac9b040de079efcdb5b5b238c6cb6022927"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "Vendor semantics: tech-wiki pages crsp-ciz-daily-return-contract, crsp-ciz-daily-return-quality and crsp-ciz-aggregate-return-quality pin the July 2026 CRSP CIZ guide; these external references remain authoritative for field meanings. Board task 54b55cc7-ff40-4753-a04e-776bce1830d0."
fact_checked: 2026-10-30
confidence: medium
---

# CRSP monthly source and panel contract

- Research entry point: [[wrds-equity-research-protocol]].

The vendor definitions live in the tech wiki's
[daily return contract](https://github.com/wagner-austin/tech-wiki/blob/main/pages/crsp-ciz-daily-return-contract.md),
[daily quality contract](https://github.com/wagner-austin/tech-wiki/blob/main/pages/crsp-ciz-daily-return-quality.md)
and [aggregate quality contract](https://github.com/wagner-austin/tech-wiki/blob/main/pages/crsp-ciz-aggregate-return-quality.md).
This protocol owns the application choices below. The CIZ adapter implements these choices; the existing `dsf` engineering pilot
is a separate diagnostic.[^ciz-contract][^adapter]

The read-only catalog probe established the subscribed `crsp_a_stock` relations
and candidate columns. It downloaded metadata only. Four exact ordered column
arrays are now frozen in the machine-readable protocol; its decoder refuses
omitted, additional, repeated or reordered columns. Column existence and type
do not prove field semantics, historical completeness or publication availability.[^ciz-schema][^frozen]

| Use | Selected input and constraint |
|---|---|
| Target return | `crsp_a_stock.stkdlysecuritydata`: `permno`, `dlycaldt`, `dlyret`, `dlyretmissflg`, `dlyretdurflg`, `dlydelflg`. Compound the held interval `(entry close, exit close]`; never substitute vendor calendar-month returns. |
| Feature inputs and price validation | The same relation's `dlyprc`, `dlyprcflg`, `dlyvol`, plus `dlyprevdt` for return-span validation. Ordinary entry/exit requires a finite positive `TR` closing trade. A selected security without an executable price makes the run incomplete; future fill prices cannot change pre-decision selection. |
| Ordinary daily completeness | Require finite return, missingness flag `NA`, nonterminal flag `N`, and duration in `D1,D2,D3,D4,DU`, with independent session and previous-date checks. An absent or unknown flag is an explicit validation error. |
| Terminal reconciliation | `stkdelists`: `permno,delistingdt,delret,delretmisstype,deldlydt,delamtdt,delnextdt`. Require finite return at least -1 and missingness flag `NA`. Value once at the latest stored/amount/present next-price date; retain a receivable until modeled reinvestment at the next rebalance strictly after that date. Missing terminal value makes the held outcome incomplete; a valuation date after the scored exit also fails. |
| Membership evidence | `dsenames`: `permno,namedt,nameendt,shrcd,exchcd`. Exactly one inclusive effective interval must apply at decision time. Preserve numeric codes 10/11 and 1/2/3 directly; do not infer equivalent mnemonic classifications or extrapolate after December 2024. Entry membership does not filter later held outcomes. |
| Session evidence | `metaexchangecalendar`: `caldt,exchangegroup,tradingflg`. Select group `CIZ`; retain every calendar day and its `Y`/`N` flag. Validate contiguous full months and construct windows from declared sessions, including historical Saturday sessions. Never infer the calendar from stock rows. |
| Incomplete observations | Exclude insufficient pre-decision history with counted reasons. A missing required held return or unresolved terminal event stops publication of a complete evaluation; no zero fill, forward fill or silent security removal. Aggregate flags serve diagnostics only and cannot override missing daily evidence. |
| Historical availability | Preserve source capture, query, extraction timestamp, checksums and effective-date joins. The resulting study is a revised-history retrospective backtest. These dates do not establish when a vendor first published a value or what a historical subscriber actually saw. |

The terminal rule is a declared retrospective valuation and settlement proxy.
Neither DelDlyDt nor the amount/next-price dates establish actual spendable-cash
availability. The frozen validators reject unresolved evidence; they do not
establish executable historical fills or a complete monthly study. Existing
staging, RunRecord, admission, queue and native controllers retain ownership.[^frozen][^ciz-contract][^controllers]

The input adapter now reads four relation CSVs and their exact SQL alongside
the frozen protocol, using the shared staging file identities and the evidence
reader lifted from the single-security diagnostic. It verifies consumed bytes,
refuses duplicate source keys and changed projections, and preserves explicit
missing values. A checksum binds bytes; it does not attest vendor authorship or
prove that a supplied query generated those bytes.[^panel]

Decision construction uses only observations through the decision close, counts
one first exclusion reason per rejected historical security, and keeps empty
decision months visible. Liquidity is the trailing 60-session median absolute
price times volume; equal liquidity sorts by PERMNO. Momentum skips the entire
decision month and compounds the preceding eleven calendar months, with every
required daily predecessor checked against the independent session calendar.
The 252-session history window ends at the decision close. Future fill prices
never filter this universe. Hand-computable fixture cases test −12% momentum,
invariance under changed future returns, prices, volumes and new listings, and
a nonconstant liquidity window whose median (2000) differs from its mean. A
reordered CSV header with a valid checksum is refused as a projection change.
This is input construction; the holdings and validation study in [[crsp-monthly-holdings-validation]] consume it.[^panel]

[^controllers]: `tools/fleet/src/fleet/core/research_control.py` functions `read_specification` and `verified_input`; `tools/fleet/src/fleet/core/research_hpc.py` and `tools/fleet/src/fleet/core/research_wrds.py`, function `poll` in each.

[^adapter]: `libs/covenant_ml/src/covenant_ml/evaluation/wrds.py:129` `load_wrds_evaluation`.

[^panel]: `libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py`, `load_crsp_sources`; `libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py`, `decode_crsp_rows`; `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `build_crsp_panel`; `libs/covenant_ml/src/covenant_ml/evaluation/evidence.py`, `read_evidence`; `libs/covenant_ml/tests/markets/test_crsp_extract.py` and `libs/covenant_ml/tests/markets/test_crsp_panel.py`, real-file and independent arithmetic fixtures. Board task 869761ed-980a-4595-a45b-fc400aab371e tracks fleet verification separately from implementation.

[^ciz-schema]: `tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json`, candidate_fields, relation_column_counts and limits. The underlying metadata CSV, identified by that receipt's artifact_sha256, additionally confirms stkdlysecuritydata.dlyprevdt; read 2026-10-29, no security observations.

[^frozen]: `libs/covenant_ml/docs/crsp-monthly-v1.json`, complete frozen settings; `libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py`, `decode_crsp_protocol`; `libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py`, `crsp_windows`, `crsp_member_on`, `require_crsp_execution_price`, `crsp_terminal_value`; `libs/covenant_ml/tests/markets/test_crsp_protocol.py`, independent calendar examples and invalid-evidence refusals. Vendor guide pp. 7, 11–12, 25, 88, 91–92 supplies calendar, date, price and missingness definitions; reinvestment timing is a study assumption, not a vendor guarantee.

[^ciz-contract]: External tech-wiki pages `crsp-ciz-daily-return-contract`, `crsp-ciz-daily-return-quality`, `crsp-ciz-aggregate-return-quality`, field and flag dictionaries plus Application notes. Their single captured vendor guide is https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d, pp. 14-18, 49, 63, 81-82 and 90-92; text capture SHA-256 bb4792deb927b972a8bef1a656ba1fc5442b80783e10eecb3e633d4243334b2e. Application constraints are study choices rather than vendor guarantees; the frozen-rules and holdings footnotes cite their implementation separately.
