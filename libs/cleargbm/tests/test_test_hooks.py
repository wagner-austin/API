"""Tests for cleargbm._test_hooks module.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

from cleargbm._test_hooks import (
    RandomStateProtocol,
    _PythonRandomStateWrapper,
    get_random_state,
)


class TestPythonRandomStateWrapper:
    """Tests for _PythonRandomStateWrapper."""

    def test_permutation_returns_permuted_tuple(self) -> None:
        """permutation should return a tuple of the correct length."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.permutation(10)

        assert len(result) == 10
        # Should contain all integers 0-9
        assert set(result) == set(range(10))

    def test_choice_returns_correct_size(self) -> None:
        """choice should return tuple of requested size."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.choice(100, size=5, replace=False)

        assert len(result) == 5
        # All values should be in range
        assert all(0 <= v < 100 for v in result)

    def test_choice_with_replacement(self) -> None:
        """choice with replacement can have duplicates."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.choice(3, size=10, replace=True)

        assert len(result) == 10
        # All values should be in range
        assert all(0 <= v < 3 for v in result)

    def test_rand_1d_returns_floats_in_range(self) -> None:
        """rand_1d should return tuple of floats in [0, 1)."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.rand_1d(5)

        assert len(result) == 5
        assert all(0.0 <= v < 1.0 for v in result)

    def test_rand_2d_returns_nested_tuples(self) -> None:
        """rand_2d should return tuple of tuples."""
        wrapper = _PythonRandomStateWrapper(42)
        result = wrapper.rand_2d(3, 4)

        assert len(result) == 3
        for row in result:
            assert len(row) == 4
            assert all(0.0 <= v < 1.0 for v in row)

    def test_same_seed_gives_same_results(self) -> None:
        """Same seed should produce identical sequences."""
        wrapper1 = _PythonRandomStateWrapper(123)
        wrapper2 = _PythonRandomStateWrapper(123)

        result1 = wrapper1.permutation(10)
        result2 = wrapper2.permutation(10)

        assert result1 == result2

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds should (usually) produce different sequences."""
        wrapper1 = _PythonRandomStateWrapper(1)
        wrapper2 = _PythonRandomStateWrapper(2)

        result1 = wrapper1.rand_1d(100)
        result2 = wrapper2.rand_1d(100)

        # Very unlikely to be equal with different seeds
        assert result1 != result2


class TestGetRandomState:
    """Tests for get_random_state function."""

    def test_returns_random_state_protocol(self) -> None:
        """get_random_state should return something conforming to protocol."""
        rng = get_random_state(42)

        # Verify all protocol methods work by actually calling them
        perm = rng.permutation(5)
        assert len(perm) == 5

        choice_result = rng.choice(10, size=3, replace=False)
        assert len(choice_result) == 3

        rand_1d_result = rng.rand_1d(5)
        assert len(rand_1d_result) == 5

        rand_2d_result = rng.rand_2d(2, 3)
        assert len(rand_2d_result) == 2
        assert len(rand_2d_result[0]) == 3

    def test_default_factory_creates_python_wrapper(self) -> None:
        """Default factory should create _PythonRandomStateWrapper."""
        rng = get_random_state(42)

        # Check it works like Python random
        result = rng.rand_1d(10)
        assert len(result) == 10
        assert all(0.0 <= v < 1.0 for v in result)


class TestRandomStateProtocol:
    """Tests to verify protocol is correctly implemented."""

    def test_wrapper_implements_protocol(self) -> None:
        """_PythonRandomStateWrapper should implement RandomStateProtocol."""
        wrapper = _PythonRandomStateWrapper(42)

        # Type check at runtime via duck typing
        def accepts_protocol(rng: RandomStateProtocol) -> int:
            return len(rng.permutation(5))

        result = accepts_protocol(wrapper)
        assert result == 5
