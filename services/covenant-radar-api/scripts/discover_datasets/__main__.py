"""Module entry point for running as `python -m scripts.discover_datasets`."""

from scripts.discover_datasets.main import main

if __name__ == "__main__":
    raise SystemExit(main(None))
