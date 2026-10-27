---
title: WRDS filesystem visibility is not a database subscription inventory
tags: [cluster-facts, wrds, datasets]
hubs: [cluster-facts]
related: ["[[partitions-and-billing]]"]
source_paths:
  - "src/hpc3/core/wrds_scan.py"
  - "src/hpc3/core/wrds_catalog.py"
  - "src/hpc3/cli/wrds_catalog.py"
source_git_blobs:
  "src/hpc3/core/wrds_scan.py": "377d02f3bcbbb15e1b362298f59ae5eec89e97fb"
  "src/hpc3/core/wrds_catalog.py": "76347e3c0e4e2f8ccc315e5aea264ee211a781f5"
  "src/hpc3/cli/wrds_catalog.py": "022b2741e2b31cfad4d0b624926e4e3a6c9b22f5"
provenance:
  - "WRDS Grid Engine job 40528959, awagner14, 2026-10-27; synchronous qsub exit 0"
  - "Ziggurat /home/corvis/research-data/wrds/catalogs/2026-10-27/filesystem-v2.jsonl, sha256 a30af80c87df11c5a02ef5fb35b7629fcafcd0373f1167bc3f5cebda2c192fdd; adjacent receipt records probe archive sha256 947eb7e622ec5cf042f3ebc6b0af172e9e39bd7621388f3b55d845e446a0ddcd"
  - "The cited source adds type annotations and renames one local variable after the live run; the traversal and wire records are unchanged. Fleet tests exercise the current archive without site packages."
  - "Board task c4522545 records operator scope: inventory everything accessible first, choose downloads afterward"
  - "PostgreSQL wrds-pgdata.wharton.upenn.edu:9737/wrds as awagner14, 2026-10-27; SQL artifacts and exact queries under Ziggurat /home/corvis/research-data/wrds/catalogs/2026-10-27/; sql-inventory-manifest.json pins their hashes"
  - "sql-zero-row-validation.receipt.json records 4406 SELECT LIMIT 0 checks in one read-only transaction, ON_ERROR_STOP enabled, exit 0, empty stderr; query sha256 d57bb865d54c8ab72111fcbd909eb0a2b8778aac3f7e364b97894e49f5b83b00"
  - "WRDS official client https://github.com/wharton/wrds/blob/main/wrds/sql.py captured as wrds-client-sql.py on 2026-10-27, sha256 458277a56bcea7d75f6457c3d848e126ddabc7722cc1001bc93d8a37792ccd74; its load_library_list query was executed unchanged"
  - "API repository at 591c5ecb5a9da49d387f7a9f4b7fb9f328c5742b: libs/covenant_ml/src/covenant_ml/datasets/, libs/platform_kaggle/src/platform_kaggle/, services/covenant-radar-api/scripts/discover_datasets/ and scripts/amex/ are existing integration points outside this wiki workspace"
fact_checked: 2026-10-27
confidence: high
---

# WRDS filesystem visibility is not a database subscription inventory

The first complete filesystem observation for `awagner14` found **56,903
visible files**, of which **17,553 were effectively readable**. Their summed
logical size was **6,808,816,686,477 bytes**. The inventory includes historical
trees and file references, so this is neither a unique-data size nor a promise
that a current research extract needs that storage.[^measurement]

Readable `.sas7bdat` files numbered **8,421 across 45 top-level library
folders**. Samples and separate products share this inventory: the 45 is not
a count of full vendor subscriptions. Selected readable SAS-file counts were
Compustat `comp` 512, CRSP 228, I/B/E/S 166, Audit `audit` 147, TRACE 39,
MSRB 3, and Fama–French `ff` 12.[^measurement]

There were also **153 denied directories** and **144,043 unresolved links**.
An inaccessible target and a missing target can both fail an existence check.
The scanner therefore labels the link unresolved, not broken. It records
directory aliases and visits each resolved directory only once; it opens no
dataset contents. Unexpected storage errors fail the job.[^scan]

The command shares this package's SSH transport and checksum validation,
while scheduling with WRDS Grid Engine. The verified catalog's adjacent
receipt records the actual job id and the probe and artifact digests. It does
not manufacture a Slurm ledger row. A footer is required before the result
can be reported complete.[^workflow]

The filesystem command does not establish SQL privileges. A separate
authenticated PostgreSQL inventory subsequently found **119,277 visible
relations**. Raw schema-USAGE and relation-SELECT checks passed for 51,389,
but most were views. That number is not an entitlement count. The official
WRDS client's library query returned **246 library names**, including samples,
aliases and system schemas.[^sql]

The conservative table inventory contains **2,335 direct tables** with both
grants and **2,071 aliases** whose relation dependencies and named privilege
guard targets are also accessible. All **4,406** passed `SELECT * LIMIT 0`
in a read-only transaction. Aliases duplicate underlying tables; this is not
4,406 distinct datasets. The checks read no dataset rows and establish neither
nonempty contents nor licensed redistribution rights.[^sql]

`sql-classified-relations.csv` retains every visible relation with its access
classification. It separates 46,605 aliases with restricted underlying access,
128 aliases with unresolved guard targets, 243 retired error views, seven
system views, and 67,888 relations lacking direct access. These categories
must not be counted as usable research tables. WRDS view guards can check
table SELECT without checking schema USAGE, so a successful zero-row query
on an alias alone is insufficient.[^sql]

`sql-accessible-relations.csv` is the 4,406-entry shortlist, and
`sql-columns.csv` records **258,529 column definitions across 4,463 relations**.
The latter includes 57 aliases with accessible dependencies but unresolved
guard targets; join to the shortlist before choosing an extraction. Planner
row and byte estimates in the relation catalog are estimates, not measured
dataset sizes.[^sql]

Confirmed alias counts include Compustat `comp` 292, CRSP 228, I/B/E/S 166,
Audit 147, TRACE 33, MSRB three and Fama–French 12. The full `dealscan`,
`optionm`, `ciq`, `markit` and `taqmsec` aliases have no entries in the
confirmed shortlist. Sample libraries are separate and do not establish
full-product subscriptions.[^sql]

The maintained filesystem command is integrated with the HPC tooling;
the SQL inventory is a dated operational capture with exact queries and
receipts. Dataset extraction and automatic training registration remain
outside this metadata command. Downloads require the operator's next dataset
selection.[^workflow][^cli][^sql]

Existing downstream code lives in `covenant_ml.datasets` (registries and
CSV/time-series/Parquet loaders), Covenant Radar's dataset-discovery and AMEX
scripts, and `platform_kaggle` (competition discovery and matching). These
are not a WRDS extraction pipeline. A selected WRDS extract still needs its
tables, columns, dates, keys and output format defined, then a checksummed
staging manifest and a matching dataset-registry entry before training.
The inventory does not register or train on any data.[^integration]

[^measurement]: WRDS job 40528959 and the local catalog and receipt named in `provenance`; the report is a dated observation, not a live entitlement service.
[^scan]: `src/hpc3/core/wrds_scan.py:45` (`scan`) and `src/hpc3/core/wrds_scan.py:29` (`permission_error`).
[^workflow]: `src/hpc3/core/wrds_catalog.py:65` (`inventory`) and `src/hpc3/core/wrds_catalog.py:45` (`read_catalog`).
[^cli]: `src/hpc3/cli/wrds_catalog.py:19` (`main`).
[^sql]: The PostgreSQL catalog captures, official-client source capture, exact queries, classifications and zero-row validation receipt named in `provenance`, pinned by `sql-inventory-manifest.json`. No dataset rows were downloaded.
[^integration]: The API repository paths and revision recorded in `provenance`; the cluster's staging command is documented in this package's `README.md`.
