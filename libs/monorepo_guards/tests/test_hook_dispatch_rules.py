"""Tests for the nullable-hook and hook-dispatch rules."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.hook_dispatch_rules import HookDispatchRule, NullableHookRule


def _write(path: Path, text: str) -> Path:
    """Write a file, creating parents.

    Args:
        path: File to write.
        text: Contents to write.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestNullableHookRule:
    """Tests for NullableHookRule."""

    def test_flags_optional_protocol_hook(self, tmp_path: Path) -> None:
        """A hook declared as '<Proto> | None = None' is flagged."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "from __future__ import annotations\n"
            "guard_find_monorepo_root: FindMonorepoRootProto | None = None\n",
        )

        violations = NullableHookRule().run([path])

        assert [v.kind for v in violations] == ["nullable-hook-declaration"]
        assert violations[0].line_no == 2

    def test_flags_optional_protocol_suffix_hook(self, tmp_path: Path) -> None:
        """The 'Protocol' suffix is recognised as well as 'Proto'."""
        path = _write(
            tmp_path / "_hooks.py",
            "runner: WorkerRunnerProtocol | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1

    def test_flags_optional_callable_hook(self, tmp_path: Path) -> None:
        """A Callable-annotated hook is recognised."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "from collections.abc import Callable\nread_file: Callable[[str], str] | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1

    def test_flags_optional_hook_written_none_first(self, tmp_path: Path) -> None:
        """The union is recognised regardless of operand order."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "loader: None | LoadOrchestratorProto = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1

    def test_flags_qualified_protocol_annotation(self, tmp_path: Path) -> None:
        """A dotted protocol annotation is recognised."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "loader: mod.LoadOrchestratorProto | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1

    def test_accepts_hook_bound_to_real_implementation(self, tmp_path: Path) -> None:
        """The sanctioned pattern is not flagged."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "def _real_is_dir(path: Path) -> bool:\n    return path.is_dir()\n\n\n"
            "is_dir: IsDirProtocol = _real_is_dir\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_optional_module_state(self, tmp_path: Path) -> None:
        """Optional non-hook state is allowed; only hook types are flagged."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "from pathlib import Path\n_SCRIPT_PATH: Path | None = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_optional_hook_with_a_real_default(self, tmp_path: Path) -> None:
        """Only a None default is flagged, not an optional-typed real binding."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "hook: SomeProto | None = _real_hook\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_ignores_files_that_are_not_hook_modules(self, tmp_path: Path) -> None:
        """Non-hook modules are not scanned."""
        path = _write(
            tmp_path / "service.py",
            "guard_find_monorepo_root: FindMonorepoRootProto | None = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_an_optional_callback_on_a_fake(self, tmp_path: Path) -> None:
        """Per-instance state in a fake is not a hook declaration.

        Hooks modules also hold the fakes tests inject, and a fake's optional
        callback is assigned through self, not declared as a module or class
        name -- which is how this rule first reported FakeKafkaConsumer.
        """
        path = _write(
            tmp_path / "_test_hooks.py",
            "from collections.abc import Callable\n\n\n"
            "class FakeConsumer:\n"
            "    def __init__(self) -> None:\n"
            "        self._on_poll: Callable[[], None] | None = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_flags_a_hook_declared_in_a_class_body(self, tmp_path: Path) -> None:
        """A Hooks container declares its hooks as class-body names."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "class Hooks:\n    load_model: ModelLoaderProto | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1


class TestHookDispatchRule:
    """Tests for HookDispatchRule."""

    def test_flags_check_then_return_dispatch(self, tmp_path: Path) -> None:
        """Checking a hook then returning its call is flagged."""
        path = _write(
            tmp_path / "guard.py",
            "def find(start):\n"
            "    if _test_hooks.guard_find is not None:\n"
            "        return _test_hooks.guard_find(start)\n"
            "    return _impl(start)\n",
        )

        violations = HookDispatchRule().run([path])

        assert [v.kind for v in violations] == ["hook-conditional-dispatch"]
        assert violations[0].line_no == 2

    def test_flags_check_then_call_dispatch(self, tmp_path: Path) -> None:
        """A bare call in the branch body is flagged too."""
        path = _write(
            tmp_path / "worker.py",
            "def run():\n    if hooks.on_start is not None:\n        hooks.on_start()\n",
        )

        assert len(HookDispatchRule().run([path])) == 1

    def test_flags_dispatch_through_nested_hook_module(self, tmp_path: Path) -> None:
        """A hooks container reached through a testing module is flagged."""
        path = _write(
            tmp_path / "redis.py",
            "def load():\n"
            "    if testing.hooks.load_module is not None:\n"
            "        return testing.hooks.load_module()\n"
            "    return real()\n",
        )

        assert len(HookDispatchRule().run([path])) == 1

    def test_accepts_direct_hook_call(self, tmp_path: Path) -> None:
        """The sanctioned pattern is not flagged."""
        path = _write(
            tmp_path / "guard.py",
            "def find(start):\n"
            "    if _test_hooks.is_dir(start):\n"
            "        return start\n"
            "    return None\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_none_check_on_non_hook_module(self, tmp_path: Path) -> None:
        """Ordinary optional values are not hooks."""
        path = _write(
            tmp_path / "service.py",
            "def run():\n"
            "    if config.timeout is not None:\n"
            "        return config.timeout()\n"
            "    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_none_check_that_calls_something_else(self, tmp_path: Path) -> None:
        """Only check-then-call-the-same-hook is the anti-pattern."""
        path = _write(
            tmp_path / "worker.py",
            "def run():\n"
            "    if hooks.value is not None:\n"
            "        return other(hooks.value)\n"
            "    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_is_none_comparison(self, tmp_path: Path) -> None:
        """An 'is None' guard clause is not dispatch."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run():\n    if hooks.value is None:\n        return 0\n    return 1\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_non_none_comparison(self, tmp_path: Path) -> None:
        """Comparing a hook to something other than None is not dispatch."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run():\n"
            "    if hooks.value is not sentinel:\n"
            "        return hooks.value()\n"
            "    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_multi_operand_comparison(self, tmp_path: Path) -> None:
        """A chained comparison is not the dispatch pattern."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run():\n"
            "    if a is not None is not b:\n"
            "        return hooks.value()\n"
            "    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_check_on_bare_name(self, tmp_path: Path) -> None:
        """A local variable, not reached through a hooks module, is allowed."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run(value):\n    if value is not None:\n        return value()\n    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_branch_body_that_is_not_a_call(self, tmp_path: Path) -> None:
        """Returning the hook itself without calling it is not dispatch."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run():\n    if hooks.value is not None:\n        x = 1\n    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_flags_or_ed_fallback_binding(self, tmp_path: Path) -> None:
        """Or-ing a hook with a fallback is the same dispatch, spelled shorter.

        DiscordBot's orchestrator wrote both of its live hooks this way, and
        the 'is not None' check alone did not see them.
        """
        path = _write(
            tmp_path / "orchestrator.py",
            "def sync(self):\n"
            "    sync_func = _test_hooks.sync_global_override or self._sync_global\n"
            "    return sync_func()\n",
        )

        violations = HookDispatchRule().run([path])

        assert [v.kind for v in violations] == ["hook-fallback-dispatch"]
        assert violations[0].line_no == 2

    def test_flags_annotated_or_ed_fallback_binding(self, tmp_path: Path) -> None:
        """The annotated form of the same binding is flagged."""
        path = _write(
            tmp_path / "orchestrator.py",
            "def start(self):\n    get_cog: GetCog = hooks.get_cog_override or self.bot.get_cog\n",
        )

        assert len(HookDispatchRule().run([path])) == 1

    def test_accepts_or_ed_hook_in_a_condition(self, tmp_path: Path) -> None:
        """Asking whether either is set is a question, not a choice of impl."""
        path = _write(
            tmp_path / "orchestrator.py",
            "def ready():\n    if hooks.override or default_ready:\n        return True\n"
            "    return False\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_or_ed_binding_that_names_no_hook(self, tmp_path: Path) -> None:
        """An ordinary or-ed default is not a hooks reference."""
        path = _write(
            tmp_path / "service.py",
            "def run(config):\n    timeout = config.timeout or 30\n    return timeout\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_or_ed_binding_rooted_in_a_call(self, tmp_path: Path) -> None:
        """An attribute chain rooted in a call is not a hooks reference."""
        path = _write(
            tmp_path / "service.py",
            "def run():\n    f = get_hooks().override or real\n    return f()\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_and_ed_binding(self, tmp_path: Path) -> None:
        """Only 'or' picks a fallback; 'and' does not."""
        path = _write(
            tmp_path / "service.py",
            "def run():\n    ok = hooks.enabled and hooks.ready\n    return ok\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_plain_hook_binding(self, tmp_path: Path) -> None:
        """Binding a hook to a name without a fallback is the sanctioned form."""
        path = _write(
            tmp_path / "service.py",
            "def run():\n    f = _test_hooks.sync_global\n    return f()\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_annotation_without_a_value(self, tmp_path: Path) -> None:
        """A bare annotation binds nothing, so it dispatches nothing."""
        path = _write(
            tmp_path / "service.py",
            "class C:\n    hook: SomeProto\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_flags_qualified_callable_annotation(self, tmp_path: Path) -> None:
        """A dotted Callable annotation is recognised."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "import typing\nread: typing.Callable[[str], str] | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1

    def test_accepts_unrecognised_annotation_form(self, tmp_path: Path) -> None:
        """An annotation that names no hook type is not flagged."""
        path = _write(
            tmp_path / "_test_hooks.py",
            'hook: "SomeProto" | None = None\n',
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_subscript_of_a_subscript(self, tmp_path: Path) -> None:
        """A subscript whose base is itself a subscript names no hook."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "hook: list[str][int] | None = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_subscript_of_a_non_callable(self, tmp_path: Path) -> None:
        """A subscripted non-Callable type is not a hook."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "values: list[str] | None = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_non_union_annotation(self, tmp_path: Path) -> None:
        """A hook annotated without a None union is not flagged here."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "hook: SomeProto = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_union_without_none(self, tmp_path: Path) -> None:
        """A union of two real types is not the nullable-hook pattern."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "hook: AProto | BProto = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_non_none_constant_default(self, tmp_path: Path) -> None:
        """Only a None default is the nullable-hook pattern."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "hook: SomeProto | None = 5\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_accepts_check_on_a_call_result(self, tmp_path: Path) -> None:
        """An attribute chain rooted in a call is not a hooks reference."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run():\n"
            "    if get_hooks().value is not None:\n"
            "        return get_hooks().value()\n"
            "    return 0\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_accepts_branch_returning_a_non_call(self, tmp_path: Path) -> None:
        """Returning a plain value from the branch is not dispatch."""
        path = _write(
            tmp_path / "hooks_user.py",
            "def run():\n    if hooks.value is not None:\n        return 0\n    return 1\n",
        )

        assert HookDispatchRule().run([path]) == []

    def test_flags_hook_typed_against_a_locally_defined_protocol(self, tmp_path: Path) -> None:
        """A hook protocol need not be named with a Proto suffix to be recognised.

        Containers such as Model-Trainer's finetuning strategies declare hooks
        against names like PeftModelCreator, which the suffix check alone missed.
        """
        path = _write(
            tmp_path / "_test_hooks.py",
            "from typing import Protocol\n\n\n"
            "class PeftModelCreator(Protocol):\n"
            "    def __call__(self) -> None: ...\n\n\n"
            "class Hooks:\n"
            "    create_peft_model: PeftModelCreator | None = None\n",
        )

        violations = NullableHookRule().run([path])

        assert [v.kind for v in violations] == ["nullable-hook-declaration"]

    def test_flags_hook_typed_against_a_qualified_local_protocol(self, tmp_path: Path) -> None:
        """The same holds when the annotation is written as a dotted name."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "from typing import Protocol\n\n\n"
            "class Runner(Protocol):\n"
            "    def __call__(self) -> None: ...\n\n\n"
            "hook: mod.Runner | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1

    def test_accepts_optional_state_typed_against_a_non_protocol_class(
        self, tmp_path: Path
    ) -> None:
        """A class that is not a Protocol is data, so an optional field is allowed."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "class Settings:\n    pass\n\n\nvalue: Settings | None = None\n",
        )

        assert NullableHookRule().run([path]) == []

    def test_flags_hook_typed_against_a_protocol_with_a_preceding_base(
        self, tmp_path: Path
    ) -> None:
        """A Protocol declared after another base is still recognised."""
        path = _write(
            tmp_path / "_test_hooks.py",
            "from typing import Generic, Protocol, TypeVar\n\n"
            "T = TypeVar('T')\n\n\n"
            "class Loader(Generic[T], Protocol):\n"
            "    def __call__(self) -> None: ...\n\n\n"
            "hook: Loader | None = None\n",
        )

        assert len(NullableHookRule().run([path])) == 1
