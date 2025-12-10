# Covenant Radar API

Loan covenant monitoring and breach prediction API service.

## Overview

Covenant Radar provides:
- CRUD operations for loan deals and covenant definitions
- Financial measurement ingestion
- Deterministic covenant rule evaluation
- XGBoost-based breach risk prediction

## Port

- API: 8007

## Development

```bash
# Run checks (lint + test with 100% coverage)
make check

# Lint only
make lint

# Test only
make test
```

## Docker

```bash
# Build and run
docker compose up -d

# View logs
docker compose logs -f
```
