from __future__ import annotations


def test_import_jobs_package() -> None:
    # Import to exercise module-level code for coverage
    import handwriting_ai.jobs as _jobs

    # Use the imported module to avoid unused-import warnings
    _ = _jobs.__name__
