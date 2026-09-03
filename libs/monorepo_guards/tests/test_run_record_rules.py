"""Pairing fingerprint capture with record emission, per package."""

from __future__ import annotations

import pathlib

from monorepo_guards.run_record_rules import RunRecordRule


def _write(root: pathlib.Path, relative: str, body: str) -> pathlib.Path:
    """Write a source file inside a package layout.

    Args:
        root: Temporary root standing in for the monorepo.
        relative: Path under the root, e.g. ``"libs/thing/src/thing/a.py"``.
        body: File contents.

    Returns:
        The path written.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestAPackageThatOnlyFingerprints:
    """The failure both covenant_ml and Model-Trainer actually had."""

    def test_capturing_without_recording_is_a_violation(self, tmp_path: pathlib.Path) -> None:
        """A fingerprint in a private shape cannot reach compare_run_records.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "libs/thing/src/thing/provenance.py",
            "from platform_core.comparability import RunFingerprint\n\n"
            "def build() -> RunFingerprint: ...\n",
        )

        found = RunRecordRule().run([path])

        assert [v.kind for v in found] == ["run-record-missing"]

    def test_the_violation_names_the_package_and_points_at_a_real_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A message pointing nowhere is a message nobody acts on.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "libs/thing/src/thing/provenance.py",
            "from platform_core.comparability import RunFingerprint\n",
        )

        found = RunRecordRule().run([path])

        assert found[0].file == path
        assert "thing" in found[0].line


class TestAPackageThatDoesBoth:
    """Capture and record legitimately live in different modules."""

    def test_recording_in_a_sibling_module_satisfies_the_rule(self, tmp_path: pathlib.Path) -> None:
        """covenant_ml captures in provenance.py and records in run_records.py.

        A file-scoped rule would force those together or fire on every
        correct arrangement, which is why the scope is the package.

        Args:
            tmp_path: Temporary monorepo root.
        """
        capture = _write(
            tmp_path,
            "libs/thing/src/thing/provenance.py",
            "from platform_core.comparability import RunFingerprint\n",
        )
        record = _write(
            tmp_path,
            "libs/thing/src/thing/run_records.py",
            "from platform_core.run_record import run_record\n",
        )

        assert RunRecordRule().run([capture, record]) == []

    def test_any_one_record_symbol_is_enough(self, tmp_path: pathlib.Path) -> None:
        """A package may build, encode, or name the sidecar for a record.

        Args:
            tmp_path: Temporary monorepo root.
        """
        for symbol in ("RunRecord", "run_record", "encode_run_record", "run_record_sidecar"):
            root = tmp_path / symbol
            capture = _write(
                root,
                "libs/thing/src/thing/provenance.py",
                "from platform_core.comparability import RunFingerprint\n",
            )
            record = _write(
                root,
                "libs/thing/src/thing/emit.py",
                f"from platform_core.run_record import {symbol}\n",
            )

            assert RunRecordRule().run([capture, record]) == [], symbol


class TestWhatIsNotSweptUp:
    """A rule that fires on correct code gets suppressed rather than fixed."""

    def test_a_package_that_fingerprints_nothing_is_ignored(self, tmp_path: pathlib.Path) -> None:
        """Most packages are not research surfaces.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(tmp_path, "libs/thing/src/thing/a.py", "x = 1\n")

        assert RunRecordRule().run([path]) == []

    def test_platform_core_is_exempt_because_it_defines_the_vocabulary(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The owner of both types would otherwise flag itself.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "libs/platform_core/src/platform_core/comparability.py",
            "class RunFingerprint: ...\n",
        )

        assert RunRecordRule().run([path]) == []

    def test_two_offending_packages_are_reported_separately(self, tmp_path: pathlib.Path) -> None:
        """One violation per package, so a fix can be attributed.

        Args:
            tmp_path: Temporary monorepo root.
        """
        first = _write(
            tmp_path,
            "libs/alpha/src/alpha/p.py",
            "from platform_core.comparability import RunFingerprint\n",
        )
        second = _write(
            tmp_path,
            "libs/beta/src/beta/p.py",
            "from platform_core.comparability import RunFingerprint\n",
        )

        found = RunRecordRule().run([first, second])

        assert len(found) == 2

    def test_one_package_reports_once_however_many_files_capture(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A package with six capture sites has one gap, not six.

        Args:
            tmp_path: Temporary monorepo root.
        """
        files = [
            _write(
                tmp_path,
                f"libs/thing/src/thing/p{index}.py",
                "from platform_core.comparability import RunFingerprint\n",
            )
            for index in range(3)
        ]

        assert len(RunRecordRule().run(files)) == 1

    def test_a_local_name_does_not_satisfy_the_rule(self, tmp_path: pathlib.Path) -> None:
        """A check a package can pass by accident reads as verified and is not.

        An earlier version of this rule collected every Name node, so a
        variable called ``run_record`` satisfied it without the package ever
        importing the record type.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "libs/thing/src/thing/provenance.py",
            "from platform_core.comparability import RunFingerprint\n\nrun_record = 1\n",
        )

        assert [v.kind for v in RunRecordRule().run([path])] == ["run-record-missing"]


class TestPathAndImportShapesTheHelpersMustSurvive:
    """Two shapes the helpers handle that no other test produced.

    Both were written into the helpers deliberately and neither was reachable
    from the cases above, so the branches sat uncovered while the rule read as
    fully tested. They are exercised through ``run`` rather than by calling the
    private helpers, so what is asserted is the rule's behaviour.
    """

    def test_a_path_whose_first_segment_is_the_marker_falls_back_to_the_parent(self) -> None:
        """``src/platform_core/a.py`` has no package name to the LEFT of ``src``.

        ``_package_of`` looks one segment left of the first layout marker it
        finds, which does not exist when the marker IS the first segment. It
        then keeps looking for a later marker and finally falls back to the
        file's own parent directory rather than indexing past the start of the
        path.

        The fallback is asserted through its consequence: naming the parent
        ``platform_core`` puts the file in a defining package, which the rule
        skips BEFORE it would read the file off disk. So this passes only if
        the fallback produced that name -- had it produced anything else the
        rule would try to open a path that does not exist and raise.
        """
        relative = pathlib.Path("src/platform_core/provenance.py")

        assert RunRecordRule().run([relative]) == []

    def test_a_relative_import_carries_no_module_name(self, tmp_path: pathlib.Path) -> None:
        """``from . import x`` is an ImportFrom whose ``module`` is None.

        The rule reads the final component of a dotted module name to catch
        ``import platform_core.run_record``. A relative import has no module
        name to read, and must be skipped rather than crash -- the file's
        imported names are still collected from the aliases.

        Args:
            tmp_path: Temporary monorepo root.
        """
        path = _write(
            tmp_path,
            "libs/thing/src/thing/provenance.py",
            "from . import sibling\nfrom platform_core.comparability import RunFingerprint\n",
        )

        assert [v.kind for v in RunRecordRule().run([path])] == ["run-record-missing"]
