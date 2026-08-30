"""Dependency-injection hooks for the harness package.

Every non-pure operation the harness performs is reached through a
module-level symbol here, bound to its real implementation at import time.
Callers always invoke the hook directly — there is no ``if testing`` branch
anywhere, so the production and test code paths are byte-identical in shape.

Tests rebind a symbol to a fake and restore it afterwards. This module is
private to the package; consumers outside it must not import from here.

The contracts live in :mod:`rw_bot.harness._hook_protocols` and the
implementations in :mod:`rw_bot.harness._hook_defaults`; this module is the
binding between them, and the only name any caller needs.
"""

from __future__ import annotations

from rw_bot.harness._hook_defaults import (
    _copy_entry_impl,
    _count_cores_impl,
    _get_env_impl,
    _kill_tree_impl,
    _list_names_impl,
    _make_dirs_impl,
    _new_stamp_impl,
    _path_exists_impl,
    _read_argv_impl,
    _read_environment_impl,
    _read_executable_impl,
    _read_platform_impl,
    _read_text_lines_impl,
    _remove_path_impl,
    _resolve_root_impl,
    _run_capture_impl,
    _run_inherited_impl,
    _serve_forever_impl,
    _sleep_impl,
    _spawn_game_impl,
    _spawn_match_impl,
    _wait_for_port_impl,
    _write_line_impl,
    _write_text_lines_impl,
)
from rw_bot.harness._hook_protocols import (
    CopyEntryProto,
    CountCoresProto,
    GetEnvProto,
    KillTreeProto,
    ListNamesProto,
    MakeDirsProto,
    NewStampProto,
    PathExistsProto,
    ReadArgvProto,
    ReadEnvironmentProto,
    ReadExecutableProto,
    ReadPlatformProto,
    ReadTextLinesProto,
    RemovePathProto,
    ResolveRootProto,
    RunCaptureProto,
    RunInheritedProto,
    ServeForeverProto,
    SleepProto,
    SpawnedMatchProto,
    SpawnGameProto,
    SpawnMatchProto,
    WaitForPortProto,
    WriteLineProto,
    WriteTextLinesProto,
)

copy_entry: CopyEntryProto = _copy_entry_impl
count_cores: CountCoresProto = _count_cores_impl
get_env: GetEnvProto = _get_env_impl
kill_tree: KillTreeProto = _kill_tree_impl
list_names: ListNamesProto = _list_names_impl
new_stamp: NewStampProto = _new_stamp_impl
spawn_game: SpawnGameProto = _spawn_game_impl
spawn_match: SpawnMatchProto = _spawn_match_impl
make_dirs: MakeDirsProto = _make_dirs_impl
path_exists: PathExistsProto = _path_exists_impl
read_argv: ReadArgvProto = _read_argv_impl
read_environment: ReadEnvironmentProto = _read_environment_impl
read_executable: ReadExecutableProto = _read_executable_impl
read_platform: ReadPlatformProto = _read_platform_impl
read_text_lines: ReadTextLinesProto = _read_text_lines_impl
remove_path: RemovePathProto = _remove_path_impl
resolve_root: ResolveRootProto = _resolve_root_impl
run_capture: RunCaptureProto = _run_capture_impl
run_inherited: RunInheritedProto = _run_inherited_impl
sleep: SleepProto = _sleep_impl
serve_forever: ServeForeverProto = _serve_forever_impl
wait_for_port: WaitForPortProto = _wait_for_port_impl
write_line: WriteLineProto = _write_line_impl
write_text_lines: WriteTextLinesProto = _write_text_lines_impl


__all__ = [
    "CopyEntryProto",
    "CountCoresProto",
    "GetEnvProto",
    "KillTreeProto",
    "ListNamesProto",
    "MakeDirsProto",
    "NewStampProto",
    "PathExistsProto",
    "ReadArgvProto",
    "ReadEnvironmentProto",
    "ReadExecutableProto",
    "ReadPlatformProto",
    "ReadTextLinesProto",
    "RemovePathProto",
    "ResolveRootProto",
    "RunCaptureProto",
    "RunInheritedProto",
    "ServeForeverProto",
    "SleepProto",
    "SpawnGameProto",
    "SpawnMatchProto",
    "SpawnedMatchProto",
    "WaitForPortProto",
    "WriteLineProto",
    "WriteTextLinesProto",
    "copy_entry",
    "count_cores",
    "get_env",
    "kill_tree",
    "list_names",
    "make_dirs",
    "new_stamp",
    "path_exists",
    "read_argv",
    "read_environment",
    "read_executable",
    "read_platform",
    "read_text_lines",
    "remove_path",
    "resolve_root",
    "run_capture",
    "run_inherited",
    "serve_forever",
    "sleep",
    "spawn_game",
    "spawn_match",
    "wait_for_port",
    "write_line",
    "write_text_lines",
]
