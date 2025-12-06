from __future__ import annotations

from platform_discord.handwriting.runtime import DigitsRuntime, new_runtime, on_progress


def test_on_progress_branches_without_optional_metrics() -> None:
    rt: DigitsRuntime = new_runtime()
    # val_acc=None and time_s=None cover false branches
    act = on_progress(
        rt,
        user_id=1,
        request_id="r",
        epoch=1,
        total_epochs=2,
        val_acc=None,
        train_loss=None,
        time_s=None,
    )
    assert act["request_id"] == "r"
