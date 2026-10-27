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

**This is only the filesystem inventory.** A direct PostgreSQL connection
requested a separate database password; no database catalog or SQL-only
entitlement is established by this scan. Visible files, readable files,
database privileges and permission to redistribute are different claims.
Dataset extraction and automatic training registration remain outside this
metadata command.[^workflow][^cli]

[^measurement]: WRDS job 40528959 and the local catalog and receipt named in `provenance`; the report is a dated observation, not a live entitlement service.
[^scan]: `src/hpc3/core/wrds_scan.py:45` (`scan`) and `src/hpc3/core/wrds_scan.py:29` (`permission_error`).
[^workflow]: `src/hpc3/core/wrds_catalog.py:65` (`inventory`) and `src/hpc3/core/wrds_catalog.py:45` (`read_catalog`).
[^cli]: `src/hpc3/cli/wrds_catalog.py:19` (`main`).
