from __future__ import annotations

import logging
import time

from tqdm.auto import tqdm

from engine.generation.config import Config, SamplingParams
from engine.generation.model_runner import ModelRunner
from engine.generation.scheduler import Scheduler
from engine.model import Transformer
from engine.sequence import Sequence
from engine.tokenizer import PretrainedHFTokenizer

log = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        model: Transformer,
        config: Config,
        tokenizer: PretrainedHFTokenizer,
    ) -> None:
        self.model_runner = ModelRunner(config, rank=0, model=model)
        self.scheduler = Scheduler(config)
        self.tokenizer = tokenizer

        log.debug("Model architecture:\n%s", model)

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if not isinstance(token_ids, list):
            raise TypeError("Expected tokenizer.encode(...) to return list[int].")

        return token_ids

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def add_request(
        self, prompt: str | list[int], sampling_params: SamplingParams
    ) -> None:
        if isinstance(prompt, str):
            prompt = self._tokenize_prompt(prompt)

        sequence = Sequence(token_ids=prompt, sampling_params=sampling_params)

        self.scheduler.add_sequence(sequence)

    def step(self):
        seqs = self.scheduler.schedule()

        token_ids, seq_need_compute_logits = self.model_runner.run(seqs)

        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)

        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]

        num_total_tokens = sum(len(seq) for seq in seqs if seq.is_finished)

        return outputs, num_total_tokens

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[list[int]]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        if len(sampling_params) != len(prompts):
            raise ValueError(
                "sampling_params length must match prompts length, "
                f"got {len(sampling_params)} and {len(prompts)}."
            )

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        num_total_tokens = 0
        t = time.perf_counter()
        while not self.is_finished():
            output, num_step_tokens = self.step()
            num_total_tokens += num_step_tokens
            total_throughput = num_total_tokens / (time.perf_counter() - t)

            if use_tqdm:
                pbar.set_postfix(
                    {
                        "total_throughput": f"{int(total_throughput)}tok/s",
                    }
                )

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        return outputs
