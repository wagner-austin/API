"""Tests for Thermo protocol functions."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from instrument_io._exceptions import ThermoReadError
from instrument_io._protocols.thermo import (
    _cleanup_temp_dir,
    _convert_raw_to_mzml,
    _create_temp_dir,
    _find_thermorawfileparser,
    _get_bundled_exe_path,
)


class TestGetBundledExePath:
    """Tests for _get_bundled_exe_path."""

    def test_returns_expected_path(self) -> None:
        result = _get_bundled_exe_path()
        assert result.name == "ThermoRawFileParser.exe"
        assert "tools" in str(result)


class TestFindThermorawfileparser:
    """Tests for _find_thermorawfileparser."""

    def test_raises_when_not_found(self) -> None:
        # Since ThermoRawFileParser is typically not installed in test environments,
        # this should raise FileNotFoundError unless the bundled exe exists
        bundled_path = _get_bundled_exe_path()
        if bundled_path.exists():
            # If bundled exists, it will be found
            result = _find_thermorawfileparser()
            assert result.name == "ThermoRawFileParser.exe"
        else:
            # Otherwise, should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                _find_thermorawfileparser()
            assert "ThermoRawFileParser not found" in str(exc_info.value)


class TestCreateTempDir:
    """Tests for _create_temp_dir."""

    def test_creates_directory(self) -> None:
        temp_dir = _create_temp_dir()
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert "thermo_" in temp_dir.name
        # Clean up
        temp_dir.rmdir()


class TestCleanupTempDir:
    """Tests for _cleanup_temp_dir."""

    def test_removes_directory(self) -> None:
        temp_dir = _create_temp_dir()
        assert temp_dir.exists()

        _cleanup_temp_dir(temp_dir)
        assert not temp_dir.exists()

    def test_handles_nonexistent_directory(self) -> None:
        # Should not raise for non-existent directory
        fake_dir = Path("/nonexistent/path/12345")
        _cleanup_temp_dir(fake_dir)  # Should not raise

    def test_removes_directory_with_contents(self) -> None:
        temp_dir = _create_temp_dir()
        # Create a file inside
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        assert temp_dir.exists()
        assert test_file.exists()

        _cleanup_temp_dir(temp_dir)
        assert not temp_dir.exists()


class TestFindThermorawfileparserBranches:
    """Tests for _find_thermorawfileparser covering all branches."""

    def test_finds_candidate_when_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test branch 57->56: candidate.exists() returns True.

        Creates a real executable file and verifies it is found.
        """
        import instrument_io._protocols.thermo as thermo_mod

        # Create a real executable file
        exe_path = tmp_path / "ThermoRawFileParser.exe"
        exe_path.touch()

        # Point bundled path to our real file
        monkeypatch.setattr(thermo_mod, "_get_bundled_exe_path", lambda: exe_path)

        result = _find_thermorawfileparser()
        assert result == exe_path
        assert result.exists()

    def test_finds_via_path_with_real_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test lines 61-63: shutil.which finds executable in PATH.

        Creates a real executable in a temp directory added to PATH.
        """
        import instrument_io._protocols.thermo as thermo_mod

        # Make bundled path not exist
        monkeypatch.setattr(thermo_mod, "_get_bundled_exe_path", lambda: Path("/nonexistent"))

        # Create a real executable in tmp_path
        if sys.platform == "win32":
            exe_name = "ThermoRawFileParser.exe"
            exe_path = tmp_path / exe_name
            exe_path.write_text("@echo off\n")
        else:
            exe_name = "ThermoRawFileParser"
            exe_path = tmp_path / exe_name
            exe_path.write_text("#!/bin/bash\n")
            exe_path.chmod(exe_path.stat().st_mode | stat.S_IEXEC)

        # Set PATH to include tmp_path (monkeypatch handles restoration)
        monkeypatch.setenv("PATH", str(tmp_path))

        result = _find_thermorawfileparser()
        # Windows may return different case, so compare case-insensitively
        assert result.name.lower() == exe_name.lower()

    def test_raises_when_not_found_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test lines 66-71: FileNotFoundError raised when not found.

        Ensures no candidate paths exist and PATH doesn't contain it.
        """
        import instrument_io._protocols.thermo as thermo_mod

        # Make bundled path not exist
        monkeypatch.setattr(thermo_mod, "_get_bundled_exe_path", lambda: Path("/nonexistent/path"))

        # Clear PATH to ensure shutil.which won't find it
        monkeypatch.setenv("PATH", str(tmp_path))  # Empty dir with no executables

        with pytest.raises(FileNotFoundError) as exc_info:
            _find_thermorawfileparser()

        assert "ThermoRawFileParser not found" in str(exc_info.value)
        assert "dotnet tool install" in str(exc_info.value)


class TestConvertRawToMzml:
    """Tests for _convert_raw_to_mzml using real subprocess execution."""

    def test_raises_when_subprocess_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test line 118: ThermoReadError raised when subprocess fails.

        Creates a real script that exits with non-zero code.
        """
        import instrument_io._protocols.thermo as thermo_mod

        raw_file = tmp_path / "test.raw"
        raw_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a real script that fails
        if sys.platform == "win32":
            script_path = tmp_path / "fail_parser.bat"
            script_path.write_text(
                "@echo off\necho Conversion failed: test error 1>&2\nexit /b 1\n"
            )
        else:
            script_path = tmp_path / "fail_parser.sh"
            script_path.write_text(
                "#!/bin/bash\necho 'Conversion failed: test error' >&2\nexit 1\n"
            )
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

        # Point to our failing script
        monkeypatch.setattr(thermo_mod, "_find_thermorawfileparser", lambda: script_path)

        with pytest.raises(ThermoReadError) as exc_info:
            _convert_raw_to_mzml(raw_file, output_dir)

        assert "ThermoRawFileParser failed" in str(exc_info.value)

    def test_raises_when_output_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test line 127: ThermoReadError raised when output file missing.

        Creates a real script that succeeds but doesn't create output.
        """
        import instrument_io._protocols.thermo as thermo_mod

        raw_file = tmp_path / "test.raw"
        raw_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a real script that succeeds but doesn't create output
        if sys.platform == "win32":
            script_path = tmp_path / "no_output_parser.bat"
            script_path.write_text("@echo off\nexit /b 0\n")
        else:
            script_path = tmp_path / "no_output_parser.sh"
            script_path.write_text("#!/bin/bash\nexit 0\n")
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

        # Point to our script
        monkeypatch.setattr(thermo_mod, "_find_thermorawfileparser", lambda: script_path)

        with pytest.raises(ThermoReadError) as exc_info:
            _convert_raw_to_mzml(raw_file, output_dir)

        assert "Expected output file not found" in str(exc_info.value)
        assert "test.mzML" in str(exc_info.value)

    def test_success_when_output_file_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test successful conversion when output file is created.

        Creates a real script that succeeds and creates the output file.
        """
        import instrument_io._protocols.thermo as thermo_mod

        raw_file = tmp_path / "test.raw"
        raw_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a real script that creates the output file
        if sys.platform == "win32":
            script_path = tmp_path / "success_parser.bat"
            # Windows batch: parse -o argument (4th positional) and create output
            script_content = f"""@echo off
echo. > "{output_dir}\\test.mzML"
exit /b 0
"""
            script_path.write_text(script_content)
        else:
            script_path = tmp_path / "success_parser.sh"
            script_content = f"""#!/bin/bash
touch "{output_dir}/test.mzML"
exit 0
"""
            script_path.write_text(script_content)
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

        # Point to our script
        monkeypatch.setattr(thermo_mod, "_find_thermorawfileparser", lambda: script_path)

        result = _convert_raw_to_mzml(raw_file, output_dir)

        assert result == output_dir / "test.mzML"
        assert result.exists()
