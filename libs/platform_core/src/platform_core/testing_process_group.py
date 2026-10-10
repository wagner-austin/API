"""Configurable group API for consumers testing process ownership failures."""


class FakeGroupApi:
    """A group API whose calls succeed or fail by configuration.

    Attributes:
        session: What ``setsid`` answers, -1 for a failed call.
        kill_result: What ``killpg`` answers, -1 for a failed call.
        code: What ``last_errno`` answers.
        calls: The method names, in order.
        signalled: Every ``(pgid, signal)`` pair ``killpg`` received.
    """

    def __init__(self, *, session: int = 1234, kill_result: int = 0, code: int = 0) -> None:
        """Configure the answers.

        Args:
            session: The session id, -1 for a failed ``setsid``.
            kill_result: The ``killpg`` result, -1 for a failed call.
            code: The errno code.
        """
        self.session = session
        self.kill_result = kill_result
        self.code = code
        self.calls: list[str] = []
        self.signalled: list[tuple[int, int]] = []

    def setsid(self) -> int:
        self.calls.append("setsid")
        return self.session

    def killpg(self, pgid: int, signal: int) -> int:
        self.calls.append("killpg")
        self.signalled.append((pgid, signal))
        return self.kill_result

    def last_errno(self) -> int:
        self.calls.append("errno")
        return self.code
