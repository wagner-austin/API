from __future__ import annotations


def test_scripts_package_import() -> None:
    # Importing the package executes its module body (coverage for scripts/__init__.py)
    import scripts

    # Reference an attribute to avoid unused-import warnings
    assert type(scripts.__name__) is str
