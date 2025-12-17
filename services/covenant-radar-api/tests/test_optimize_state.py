"""Tests for scripts/optimize/state.py - lifecycle state management."""

from __future__ import annotations

from scripts.optimize.state import (
    OptimizationState,
    get_state,
    is_interrupted,
    managed_execution,
)


class TestOptimizationState:
    """Tests for OptimizationState class."""

    def test_initial_state_not_interrupted(self) -> None:
        """Test initial state is not interrupted."""
        state = OptimizationState()
        assert state.interrupted is False

    def test_set_interrupted_marks_state(self) -> None:
        """Test set_interrupted marks state as interrupted."""
        state = OptimizationState()
        state.set_interrupted()
        assert state.interrupted is True

    def test_register_shutdown_callback_stores_callback(self) -> None:
        """Test register_shutdown_callback stores callback for shutdown."""
        state = OptimizationState()
        called = False

        def callback() -> None:
            nonlocal called
            called = True

        state.register_shutdown_callback(callback)
        state.start()
        state.stop()

        assert called is True

    def test_shutdown_callbacks_run_in_reverse_order(self) -> None:
        """Test shutdown callbacks run in reverse registration order."""
        state = OptimizationState()
        order: list[int] = []

        def callback1() -> None:
            order.append(1)

        def callback2() -> None:
            order.append(2)

        state.register_shutdown_callback(callback1)
        state.register_shutdown_callback(callback2)
        state.start()
        state.stop()

        assert order == [2, 1]

    def test_start_resets_interrupted_flag(self) -> None:
        """Test start resets interrupted flag."""
        state = OptimizationState()
        state.set_interrupted()
        assert state.interrupted is True

        state.start()
        assert state.interrupted is False
        state.stop()

    def test_stop_clears_shutdown_callbacks(self) -> None:
        """Test stop clears shutdown callbacks."""
        state = OptimizationState()

        def callback() -> None:
            pass

        state.register_shutdown_callback(callback)
        state.start()
        state.stop()

        # Callbacks should be cleared
        assert len(state._shutdown_callbacks) == 0

    def test_handle_sigint_sets_interrupted_flag(self) -> None:
        """Test _handle_sigint sets interrupted flag when called."""
        state = OptimizationState()
        assert state.interrupted is False

        # Simulate signal handler being called
        state._handle_sigint(2, None)

        assert state.interrupted is True

    def test_stop_restores_original_sigint_handler(self) -> None:
        """Test stop restores the original SIGINT handler."""
        import signal

        state = OptimizationState()

        # Capture original handler before start
        original_handler = signal.getsignal(signal.SIGINT)

        state.start()
        # After start, handler should be different (our custom handler)
        current_handler = signal.getsignal(signal.SIGINT)
        assert current_handler == state._handle_sigint

        state.stop()
        # After stop, handler should be restored
        restored_handler = signal.getsignal(signal.SIGINT)
        assert restored_handler == original_handler

    def test_stop_without_start_does_not_restore_handler(self) -> None:
        """Test stop without start doesn't try to restore None handler."""
        state = OptimizationState()
        # _original_sigint is None since start() was not called
        assert state._original_sigint is None

        # Should not raise - just runs shutdown callbacks
        state.stop()

        # Still None after stop
        assert state._original_sigint is None


class TestGetState:
    """Tests for get_state singleton function."""

    def test_returns_same_instance(self) -> None:
        """Test get_state returns same instance on multiple calls."""
        state1 = get_state()
        state2 = get_state()
        assert state1 is state2


class TestIsInterrupted:
    """Tests for is_interrupted function."""

    def test_returns_false_when_not_interrupted(self) -> None:
        """Test is_interrupted returns False when not interrupted."""
        state = get_state()
        state._interrupted = False
        result: bool = is_interrupted()
        assert result is False

    def test_returns_true_when_interrupted(self) -> None:
        """Test is_interrupted returns True when interrupted."""
        state = get_state()
        state._interrupted = True
        result: bool = is_interrupted()
        assert result is True
        # Reset for other tests
        state._interrupted = False


class TestManagedExecution:
    """Tests for managed_execution context manager."""

    def test_yields_state(self) -> None:
        """Test managed_execution yields same OptimizationState from get_state."""
        with managed_execution() as state:
            assert state is get_state()

    def test_starts_state_on_entry(self) -> None:
        """Test managed_execution starts state on entry."""
        state = get_state()
        state.set_interrupted()  # Set interrupted before

        with managed_execution():
            # Should be reset on start
            assert state.interrupted is False

    def test_stops_state_on_exit(self) -> None:
        """Test managed_execution stops state on normal exit."""
        called = False

        def callback() -> None:
            nonlocal called
            called = True

        state = get_state()
        state.register_shutdown_callback(callback)

        with managed_execution():
            pass

        assert called is True
