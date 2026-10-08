"""Every test's fixtures, module-scoped ones included, see one torch thread.

``tests/conftest.py`` pins torch's intra-op pool in ``pytest_runtest_setup``
rather than in an autouse fixture, because a module-scoped fixture is set up
before any function-scoped one and so trained under whatever pool the test
before it had left. These tests run in order on one worker: the first leaves
the pool at four threads, and the module-scoped fixture the second requests
is created after that, so it reads one thread only if the hook ran before
pytest created it.
"""

from __future__ import annotations

import pytest
import torch

#: The module-scoped reading below must be made on the worker that ran the
#: test before it (tests/test_xdist_grouping.py says why modules group).
pytestmark = pytest.mark.xdist_group("test_torch_thread_pin.py")

_LEFT_BEHIND = 4


@pytest.fixture(name="threads_at_module_setup", scope="module")
def _threads_at_module_setup() -> int:
    """Read the pool size at the moment a module-scoped fixture is created.

    Returns:
        ``torch.get_num_threads()`` during this fixture's setup.
    """
    return torch.get_num_threads()


def test_a_test_leaves_a_wider_pool_behind() -> None:
    torch.set_num_threads(_LEFT_BEHIND)

    assert torch.get_num_threads() == _LEFT_BEHIND


def test_a_module_fixture_created_after_it_sees_one_thread(threads_at_module_setup: int) -> None:
    assert threads_at_module_setup == 1
