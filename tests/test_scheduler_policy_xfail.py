from pathlib import Path

import pytest

from engine.sequence import Sequence
from tests._helpers import make_scheduler


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Policy decision pending: for non-chunked prefill, over-budget prompts could "
        "be rejected earlier at add_sequence() instead of failing later in schedule()."
    ),
)
def test_non_chunked_over_budget_prompt_rejected_at_admission(tmp_path: Path) -> None:
    scheduler = make_scheduler(
        tmp_path,
        chunked_prefill=False,
        max_num_batched_tokens=128,
    )
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        scheduler.add_sequence(Sequence([1] * 200))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Policy decision pending: when no sequence can make progress, scheduler should "
        "avoid the generic 'empty schedule' crash (return [] or raise a specific error)."
    ),
)
def test_unschedulable_step_avoids_generic_empty_schedule_error(tmp_path: Path) -> None:
    scheduler = make_scheduler(
        tmp_path,
        num_kvcache_blocks=1,
        max_num_batched_tokens=4096,
    )
    seq = Sequence([1] * 256)
    scheduler.add_sequence(seq)

    scheduled = scheduler.schedule()
    scheduler.postprocess(scheduled, token_ids=[2], seq_need_compute_logits=[0])

    try:
        result = scheduler.schedule()
    except RuntimeError as exc:
        assert "empty schedule" not in str(exc)
    else:
        assert result == []
