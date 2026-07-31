# doc-extract-api

PDF text extraction service. Every page goes through two extractors —
`pdfplumber` for embedded text and docTR OCR for the rendered image — and the
longer result wins. That way tables rendered as images, charts with data
labels, and scanned pages all get covered without an arbitrary "is this page
scanned?" threshold.

Extraction runs as a background RQ job; results are written to PostgreSQL.

## Quick Start

```bash
# From the repository root: start Redis + PostgreSQL, then the service
make infra
cd services/doc-extract-api
docker compose up -d --build

# Verify
curl http://localhost:8012/healthz
curl http://localhost:8012/readyz
```

The compose file defines `api` and `worker`, built from the two Dockerfile
targets of the same name. The API publishes host port **8012** onto container
port 8000. Redis and PostgreSQL come from the root compose, not this one.

### Local Development

```bash
poetry install --with dev

# API
poetry run hypercorn 'doc_extract_api.asgi:app' --bind 0.0.0.0:8000

# Worker (separate terminal) — nothing is extracted without it
poetry run doc-extract-rq-worker
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe |
| `/jobs` | POST | Enqueue an extraction job |
| `/jobs/{job_id}` | GET | Get job status; 404 when the id is unknown |

### Enqueue a job

```bash
curl -X POST "http://localhost:8012/jobs?title=FY26%20Budget&file_path=/data/docs/budget.pdf&category=budget"
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `title` | yes | - | Document title |
| `file_path` | yes | - | Absolute path to the PDF |
| `category` | no | `general` | One of the categories below |
| `source` | no | `""` | Source URL or reference |

An unknown `category` is rejected rather than silently stored. Valid values:
`general`, `budget`, `audit`, `pension`, `investment`, `park`, `energy`,
`council`, `legislation`, `water`, `education`, `waste`, `transportation`,
`housing`, `environmental`.

### Job status

```bash
curl http://localhost:8012/jobs/{job_id}
```

Status is one of `queued`, `processing`, `completed`, `failed`. The response
also carries `pages_total`, `pages_done`, `document_id`, and `error`.

## Configuration

All three are required — each is read with `_require_env_str`, so a missing one
fails at startup rather than at first request.

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis connection URL for the job queue and job store |
| `DATABASE_URL` | PostgreSQL connection string for extracted documents |
| `DOC_EXTRACT_TENANT_EMAIL` | Email used to resolve tenant context for a job |

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `redis` / `rq` | Job queue and job store |
| `psycopg` | PostgreSQL driver |
| `pdfplumber` | Embedded-text extraction |
| `pypdfium2` | Page rendering for OCR |
| `python-doctr` | OCR model |
| `torch` | Runs the docTR model (CUDA when available) |
| `platform-core` | Logging, errors, config |
| `platform-workers` | RQ worker harness, Redis helpers |

## Development

```bash
make lint   # guards + ruff + mypy
make test   # pytest with coverage
make check  # lint + test
```

## Port Map

- **8012**: doc-extract-api (container port 8000)
