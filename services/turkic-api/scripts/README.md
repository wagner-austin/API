# Scripts Directory

This directory contains utility scripts for project maintenance and code quality enforcement.

## Guard Script

- `guard.py` - Enforces code quality patterns and strict typing rules

### What it checks:
- No `Any` types
- No `cast` usage
- No `type: ignore` comments
- No `TYPE_CHECKING` imports
- No Pydantic imports
- No try/except blocks
- Proper UnknownJson usage in public APIs

### Usage

The guard script runs automatically as part of `make check`:

```bash
make check
```

Or run manually:

```bash
poetry run python -m scripts.guard
```

The script will exit with code 0 if all checks pass, or non-zero if violations are found.
