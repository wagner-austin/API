"""Configurable job API for consumers testing process ownership failures."""


class FakeJobApi:
    """A job API whose calls succeed or fail by configuration.

    Attributes:
        handle: What ``create_job_object`` answers.
        set_ok: What ``set_kill_on_close`` answers.
        assign_ok: What ``assign_current_process`` answers.
        error: What ``last_error`` answers.
        calls: The method names, in order.
    """

    def __init__(
        self, *, handle: int = 42, set_ok: bool = True, assign_ok: bool = True, error: int = 0
    ) -> None:
        """Configure the answers.

        Args:
            handle: The job handle, 0 for a failed creation.
            set_ok: Whether setting the limit succeeds.
            assign_ok: Whether assignment succeeds.
            error: The Win32 error code.
        """
        self.handle = handle
        self.set_ok = set_ok
        self.assign_ok = assign_ok
        self.error = error
        self.calls: list[str] = []
        self.handles: list[int] = []

    def create_job_object(self) -> int:
        self.calls.append("create")
        return self.handle

    def set_kill_on_close(self, handle: int) -> bool:
        self.calls.append("set")
        self.handles.append(handle)
        return self.set_ok

    def assign_current_process(self, handle: int) -> bool:
        self.calls.append("assign")
        self.handles.append(handle)
        return self.assign_ok

    def last_error(self) -> int:
        self.calls.append("error")
        return self.error
