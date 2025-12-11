# covenant-persistence

PostgreSQL repository layer for covenant monitoring with Protocol-based psycopg.

## Installation

```bash
poetry add covenant-persistence
```

Requires `covenant-domain` and `psycopg[binary,pool]` for runtime.

## Quick Start

```python
from covenant_persistence import (
    connect,
    PostgresDealRepository,
    PostgresCovenantRepository,
)
from covenant_domain import Deal, DealId

# Connect to PostgreSQL
conn = connect("postgresql://user:pass@localhost:5432/covenant")

# Create repositories
deal_repo = PostgresDealRepository(conn)

# Use repositories
deal: Deal = {
    "id": DealId(value="abc-123"),
    "name": "Tech Corp Credit Facility",
    ...
}
deal_repo.save(deal)
retrieved = deal_repo.get(DealId(value="abc-123"))
```

## Repository Protocols

Abstract repository interfaces that can be implemented with any storage backend:

```python
from covenant_persistence import (
    DealRepository,
    CovenantRepository,
    MeasurementRepository,
    CovenantResultRepository,
)
```

Each protocol defines CRUD operations for the corresponding domain entity:

| Repository | Entity | Methods |
|------------|--------|---------|
| `DealRepository` | Deal | `save`, `get`, `list_all`, `delete` |
| `CovenantRepository` | Covenant | `save`, `get`, `list_for_deal`, `delete` |
| `MeasurementRepository` | Measurement | `save_many`, `list_for_deal`, `list_for_deal_and_period` |
| `CovenantResultRepository` | CovenantResult | `save_many`, `list_for_deal` |

## PostgreSQL Implementations

Concrete implementations using psycopg3:

```python
from covenant_persistence import (
    PostgresDealRepository,
    PostgresCovenantRepository,
    PostgresMeasurementRepository,
    PostgresCovenantResultRepository,
    connect,
)

# Connect to PostgreSQL
conn = connect("postgresql://user:pass@localhost:5432/covenant")

# Create repositories
deal_repo = PostgresDealRepository(conn)
covenant_repo = PostgresCovenantRepository(conn)
measurement_repo = PostgresMeasurementRepository(conn)
result_repo = PostgresCovenantResultRepository(conn)

# Use repositories
from covenant_domain import Deal, DealId

deal: Deal = {
    "id": DealId(value="abc-123"),
    "name": "Tech Corp Credit Facility",
    "borrower": "Tech Corp",
    "sector": "Technology",
    "region": "North America",
    "commitment_amount_cents": 50_000_000_00,
    "currency": "USD",
    "maturity_date_iso": "2027-12-31",
}
deal_repo.save(deal)

# Retrieve
retrieved = deal_repo.get(DealId(value="abc-123"))
all_deals = deal_repo.list_all()
```

## Connection Protocols

Typed protocols for psycopg connections:

```python
from covenant_persistence import (
    ConnectionProtocol,
    CursorProtocol,
    ConnectCallable,
)
```

## API Reference

### Repository Protocols

| Protocol | Description |
|----------|-------------|
| `DealRepository` | CRUD for Deal entities |
| `CovenantRepository` | CRUD for Covenant entities |
| `MeasurementRepository` | Bulk save/list for Measurements |
| `CovenantResultRepository` | Bulk save/list for CovenantResults |

### PostgreSQL Implementations

| Class | Description |
|-------|-------------|
| `PostgresDealRepository` | psycopg3 implementation for deals |
| `PostgresCovenantRepository` | psycopg3 implementation for covenants |
| `PostgresMeasurementRepository` | psycopg3 implementation for measurements |
| `PostgresCovenantResultRepository` | psycopg3 implementation for results |

### Connection Types

| Type | Description |
|------|-------------|
| `ConnectionProtocol` | Database connection interface |
| `CursorProtocol` | Database cursor interface |
| `ConnectCallable` | Connection factory type |
| `connect` | Create a psycopg connection |

## Testing

In-memory implementations for unit tests:

```python
from covenant_persistence.testing import (
    InMemoryDealRepository,
    InMemoryCovenantRepository,
    InMemoryMeasurementRepository,
    InMemoryCovenantResultRepository,
)

# Use in tests without PostgreSQL
deal_repo = InMemoryDealRepository()
deal_repo.save(deal)
assert deal_repo.get(deal["id"]) == deal
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- covenant-domain
- psycopg[binary,pool] 3.2.0+ (optional for PostgreSQL)
- 100% test coverage enforced
