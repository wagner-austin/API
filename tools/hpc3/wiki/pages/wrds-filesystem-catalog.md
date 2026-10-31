---
title: WRDS filesystem visibility is not a database subscription inventory
tags: [cluster-facts, wrds, datasets]
hubs: [cluster-facts]
related: ["[[partitions-and-billing]]"]
source_paths:
  - "src/hpc3/core/wrds_scan.py"
  - "src/hpc3/core/wrds_grid.py"
  - "src/hpc3/core/artifacts.py"
  - "src/hpc3/core/wrds_catalog.py"
  - "src/hpc3/core/wrds_sql.py"
  - "src/hpc3/cli/wrds_catalog.py"
  - "src/hpc3/cli/wrds_sql_catalog.py"
source_git_blobs:
  "src/hpc3/core/wrds_scan.py": "377d02f3bcbbb15e1b362298f59ae5eec89e97fb"
  "src/hpc3/core/wrds_grid.py": "ecdca06d88f7aa4f818ef220ee09174a71833414"
  "src/hpc3/core/artifacts.py": "4dffdf06b94e1cd6d58f45ac785788ad242f05c4"
  "src/hpc3/core/wrds_catalog.py": "9969c812050e073947127036da80f0bb007c3dc1"
  "src/hpc3/core/wrds_sql.py": "8b09091403ad16fed66828336a419a2cded0525d"
  "src/hpc3/cli/wrds_catalog.py": "4846eb168c87596fb2ae9008a873e3d6a07f3877"
  "src/hpc3/cli/wrds_sql_catalog.py": "d02c6019dd675f8418adc86a3a84274340b36058"
provenance:
  - "The poster both commands call, platform_core.load_report (libs/platform_core/src/platform_core/load_report.py, outside this workspaceRoot): post_load_report and report_identity, lifted out of this package's cli/_load_report.py on 2026-10-31 with labels and service names unchanged (board task 1f60a20d)"
  - "WRDS Grid Engine job 40529889, awagner14, 2026-10-28T01:29Z, run by hpc3.cli.wrds_catalog at API d0d0cafe8; synchronous qsub exit 0"
  - "Ziggurat /home/corvis/research-data/wrds/catalogs/2026-10-28/filesystem.jsonl, sha256 cff2ebb8751f6bb1b4f574330f89cdfeafdf9a9f435d271bddfeb0555951647a; adjacent receipt records probe archive sha256 7878f224f9d7ba0f117a539bcf7b0b4894c0e634e441fdd427f099bf07335b86"
  - "WRDS Grid Engine jobs 40529873 (catalog) and 40529878 (zero-row validation), awagner14, 2026-10-28T01:27Z, run by hpc3.cli.wrds_sql_catalog at API d0d0cafe8 against WRDS Cloud's ident-authenticated PostgreSQL host; both exit 0"
  - "Ziggurat /home/corvis/research-data/wrds/catalogs/2026-10-28/sql-relations.csv, sha256 86e95a47cd632a657bd01ce08657a9c543199653c09da5566f44316d99085979, with sql-relations.csv.accessible.csv and sql-relations.csv.receipt.json (query sha256 b0c1d2b52dbab10889fd2da56eb144a84395399c85312dc606bcee6373652d22, validation sha256 af0380e0b02c9b43443c55f3550654e605588c97c2efd5d24ae17d57a0e66a13)"
  - "Board task c4522545: the two load reports posted by the programs themselves under the service identity load-report-c4522545 at 2026-10-28T01:27:53Z (SQL) and 01:32:23Z (filesystem)"
  - "Board task c4522545 records operator scope: inventory everything accessible first, choose downloads afterward"
  - "The first observations, 2026-10-27 (filesystem job 40528959, catalog sha256 a30af80c87df11c5a02ef5fb35b7629fcafcd0373f1167bc3f5cebda2c192fdd; an ad hoc SQL capture under /home/corvis/research-data/wrds/catalogs/2026-10-27/), gave the same file and access counts; the maintained SQL classification reproduced that capture's statuses relation for relation, plus relations added since"
  - "API repository at 591c5ecb5a9da49d387f7a9f4b7fb9f328c5742b: libs/covenant_ml/src/covenant_ml/datasets/, libs/platform_kaggle/src/platform_kaggle/, services/covenant-radar-api/scripts/discover_datasets/ and scripts/amex/ are existing integration points outside this wiki workspace"
fact_checked: 2026-10-28
confidence: high
---

# WRDS filesystem visibility is not a database subscription inventory

WRDS access has three separate answers: which files are visible, which of
those this account can read, and which database relations it may query. Two
maintained commands in this package measure them, and each posts its own
counts on the board task it is run for.[^report]

## Filesystem: visible is not readable

The filesystem observation of 2026-10-28 for `awagner14` found **56,903
visible files**, of which **17,553 were effectively readable**. Their summed
logical size was **6,808,827,311,501 bytes**. The inventory includes historical
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

The filesystem command does not establish SQL privileges.[^scan]

## SQL: visible is not accessible

The SQL inventory found **119,279 visible relations**. Visibility in
`pg_catalog` grants nothing: a relation counts as accessible only as a table
with both schema USAGE and SELECT, or as a WRDS alias view whose
`has_table_privilege` guard target and every relation it reads carry both.
A guard checks the table without checking its schema's USAGE, so a zero-row
query that succeeds on an alias alone is insufficient evidence.[^sql]

The conservative table inventory contains **2,335 direct tables** with both
grants and **2,071 aliases** whose underlying relations and guard targets are
also accessible. A second job opened all **4,406** with `SELECT * LIMIT 0` in
one read-only transaction that stops at the first error, and completed.
Aliases duplicate underlying tables; this is not 4,406 distinct datasets. The
checks read no dataset rows and establish neither nonempty contents nor
licensed redistribution rights.[^sql]

The rest are not usable research tables: 67,890 relations without direct
access, 46,605 aliases over restricted relations, 128 aliases whose guard
names no existing table, 243 retired views that call an `*_err` function, and
seven views an extension owns. The accessible shortlist beside the catalog is
the list an extraction chooses from.[^sql]

Confirmed alias counts include Compustat `comp` 292, CRSP 228, I/B/E/S 166,
Audit 147, TRACE 33, MSRB three and Fama–French 12. The full `dealscan`,
`optionm`, `ciq`, `markit` and `taqmsec` libraries have no aliases in the
confirmed shortlist. Sample libraries are separate and do not establish
full-product subscriptions.[^sql]

## How it runs, and what remains before training

The maintained filesystem command is integrated with the HPC tooling;
so is the SQL inventory command. Both use this package's SSH transport and
checksum validation, schedule with WRDS Grid Engine through one shared
transport, and save a receipt with the actual job ids and digests rather than
a manufactured Slurm ledger row. A missing footer, a digest mismatch or one
refused zero-row query fails the run.[^workflow][^cli]

Existing downstream code lives in `covenant_ml.datasets` (registries and
CSV/time-series/Parquet loaders), Covenant Radar's dataset-discovery and AMEX
scripts, and `platform_kaggle` (competition discovery and matching).
These are not a WRDS extraction pipeline. A selected WRDS extract still needs its
tables, columns, dates, keys and output format defined, then a checksummed
staging manifest and a matching dataset-registry entry before training.
Neither inventory downloads, registers or trains on any data; downloads wait
for the operator's dataset selection.[^integration]

[^report]: `platform_core.load_report`'s `post_load_report`, called from `src/hpc3/cli/wrds_catalog.py:48` (`main`) and `src/hpc3/cli/wrds_sql_catalog.py:50` (`main`), under the identity its `report_identity` derives from the task id; the module is named in `provenance` because it lives outside this workspace.
[^measurement]: WRDS job 40529889 and the local catalog and receipt named in `provenance`, posted by `src/hpc3/cli/wrds_catalog.py:22` (`report`); a dated observation, not a live entitlement service.
[^scan]: `src/hpc3/core/wrds_scan.py:45` (`scan`) and `src/hpc3/core/wrds_scan.py:29` (`permission_error`).
[^sql]: WRDS jobs 40529873 and 40529878 and the artifacts named in `provenance`; classification in `src/hpc3/core/wrds_sql.py:102` (`classify`), the run in `src/hpc3/core/wrds_sql.py:201` (`sql_inventory`).
[^workflow]: `src/hpc3/core/wrds_grid.py:102` (`submit`), `src/hpc3/core/artifacts.py:54` (`download`) and `src/hpc3/core/wrds_catalog.py:67` (`inventory`).
[^cli]: `src/hpc3/cli/wrds_catalog.py:48` (`main`) and `src/hpc3/cli/wrds_sql_catalog.py:50` (`main`).
[^integration]: The API repository paths and revision recorded in `provenance`; the cluster's staging command is documented in this package's `README.md`.
