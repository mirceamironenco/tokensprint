from pathlib import Path

from engine.sequence import SamplingParams, Sequence, SequenceStatus
from tests._helpers import make_scheduler


def assert_queue_invariants(scheduler) -> None:
    waiting_ids = {seq.seq_id for seq in scheduler.waiting}
    running_ids = {seq.seq_id for seq in scheduler.running}

    assert waiting_ids.isdisjoint(running_ids)
    assert all(seq.status == SequenceStatus.WAITING for seq in scheduler.waiting)
    assert all(seq.status == SequenceStatus.RUNNING for seq in scheduler.running)


def test_invariants_after_scheduling_waiting_sequence(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, max_num_batched_tokens=512, chunked_prefill=False)
    seq = Sequence([1] * 200)
    scheduler.add_sequence(seq)

    scheduled = scheduler.schedule()

    assert scheduled == [seq]
    assert seq.status == SequenceStatus.RUNNING
    assert seq in scheduler.running
    assert seq not in scheduler.waiting
    assert seq.num_cached_tokens == 0
    assert seq.num_new_tokens == 200
    assert seq.num_current_blocks == len(seq.block_table)
    assert_queue_invariants(scheduler)


def test_invariants_after_postprocess_for_unfinished_sequence(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, max_num_batched_tokens=512)
    seq = Sequence([1] * 32)
    scheduler.add_sequence(seq)

    scheduled = scheduler.schedule()
    prev_cached = seq.num_cached_tokens
    prev_new = seq.num_new_tokens
    prev_blocks = list(seq.block_table)
    prev_len = len(seq)

    scheduler.postprocess(scheduled, token_ids=[42], seq_need_compute_logits=[0])

    assert seq.status == SequenceStatus.RUNNING
    assert seq in scheduler.running
    assert seq not in scheduler.waiting
    assert seq.num_cached_tokens == prev_cached + prev_new
    assert seq.num_new_tokens == 0
    assert seq.block_table == prev_blocks
    assert len(seq) == prev_len + 1
    assert_queue_invariants(scheduler)


def test_invariants_after_postprocess_finished_sequence(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, max_num_batched_tokens=512, num_kvcache_blocks=8)
    seq = Sequence([1] * 16, SamplingParams(max_tokens=1))
    scheduler.add_sequence(seq)

    scheduled = scheduler.schedule()
    scheduler.postprocess(scheduled, token_ids=[99], seq_need_compute_logits=[0])

    assert seq.status == SequenceStatus.FINISHED
    assert seq not in scheduler.running
    assert seq not in scheduler.waiting
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert seq.num_new_tokens == 0
    assert len(scheduler.block_manager.used_block_ids) == 0
    assert_queue_invariants(scheduler)


def test_preempt_transition_keeps_queues_disjoint(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, num_kvcache_blocks=2, max_num_batched_tokens=512)
    seq_a = Sequence([1] * 256)
    seq_b = Sequence([2] * 256)
    scheduler.add_sequence(seq_a)
    scheduler.add_sequence(seq_b)

    scheduler.schedule()
    assert_queue_invariants(scheduler)

    preempted = scheduler.running.pop()
    scheduler.preempt(preempted)

    assert preempted.status == SequenceStatus.WAITING
    assert preempted in scheduler.waiting
    assert preempted not in scheduler.running
    assert preempted.block_table == []
    assert preempted.num_cached_tokens == 0
    assert preempted.num_new_tokens == 0
    assert_queue_invariants(scheduler)
