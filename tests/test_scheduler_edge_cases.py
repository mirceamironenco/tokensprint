from pathlib import Path

import pytest

from engine.sequence import Sequence
from tests._helpers import make_scheduler


def test_schedule_raises_when_running_seq_self_preempts(tmp_path: Path) -> None:
    """
    Repro:
    - first step prefill consumes the only block
    - postprocess appends one token
    - second step needs one additional block and preempts itself
    - schedule returns empty and raises
    """
    scheduler = make_scheduler(tmp_path, num_kvcache_blocks=1, max_num_batched_tokens=4096)
    seq = Sequence([1] * 256)
    scheduler.add_sequence(seq)

    step1 = scheduler.schedule()
    scheduler.postprocess(step1, token_ids=[2], seq_need_compute_logits=[0])

    with pytest.raises(RuntimeError, match="empty schedule"):
        scheduler.schedule()


def test_schedule_raises_when_non_chunked_prefill_exceeds_budget(
    tmp_path: Path,
) -> None:
    """
    Repro:
    - chunked_prefill disabled
    - waiting sequence prompt length > token budget
    - scheduler cannot schedule any sequence and raises
    """
    scheduler = make_scheduler(
        tmp_path,
        chunked_prefill=False,
        max_num_batched_tokens=128,
    )
    scheduler.add_sequence(Sequence([1] * 200))

    with pytest.raises(RuntimeError, match="empty schedule"):
        scheduler.schedule()


def test_schedule_progresses_with_chunked_prefill_enabled(tmp_path: Path) -> None:
    scheduler = make_scheduler(
        tmp_path,
        chunked_prefill=True,
        max_num_batched_tokens=128,
    )
    seq = Sequence([1] * 200)
    scheduler.add_sequence(seq)

    scheduled = scheduler.schedule()

    assert len(scheduled) == 1
    assert scheduled[0] is seq
    assert seq.num_new_tokens == 128


def test_schedule_progresses_after_decode_if_a_free_block_exists(
    tmp_path: Path,
) -> None:
    scheduler = make_scheduler(tmp_path, num_kvcache_blocks=2, max_num_batched_tokens=4096)
    seq = Sequence([1] * 256)
    scheduler.add_sequence(seq)

    step1 = scheduler.schedule()
    scheduler.postprocess(step1, token_ids=[2], seq_need_compute_logits=[0])

    step2 = scheduler.schedule()

    assert len(step2) == 1
    assert step2[0] is seq
    assert seq.num_new_tokens == 1
