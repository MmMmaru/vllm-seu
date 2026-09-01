# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Participant model wrapper for the DNDX benchmark."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


_VERBOSE_EN_INSTRUCTION = (
    "Solve this single-choice question. Your response must make one final choice "
    "among A/B/C/D clearly. You may include one short reason."
)
_VERBOSE_CN_INSTRUCTION = (
    "请完成这道单选题。请给出你认为正确的选项，并可附带一句简短理由。"
    "答案必须明确，且只能对应 A/B/C/D 中的一个选项。"
)
_ANSWER_MAX_TOKENS = 6


def _make_answer_only_prompt(prompt: str) -> str:
    prompt = prompt.replace(_VERBOSE_EN_INSTRUCTION, "Answer only: Answer: X (A/B/C/D).")
    return prompt.replace(_VERBOSE_CN_INSTRUCTION, "仅回答：答案：X（A/B/C/D）。")


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict[str, Any]


class VLMModel:
    """
    Default participant wrapper.

    `backend="dummy"` is for demo-only smoke tests.
    `backend="transformers"` uses a local Hugging Face model directory.
    `backend="vllm"` uses vLLM offline inference with local model weights.
    Participants can replace the internals while preserving `generate_with_metrics`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.backend = backend
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend_name = "dummy"

        if backend == "vllm":
            self._load_vllm_backend()
            self._backend_name = "vllm"
        elif backend in {"auto", "transformers"}:
            try:
                self._load_transformers_backend()
                self._backend_name = "transformers"
            except Exception as exc:
                if backend == "transformers":
                    raise
                self._load_dummy_backend(str(exc))
        else:
            self._load_dummy_backend("backend=dummy")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def shutdown(self) -> None:
        if self._backend_name != "vllm" or self._model is None:
            return
        self._model.shutdown()
        self._vllm_loop.run_until_complete(asyncio.sleep(0))
        self._vllm_loop.close()
        self._model = None

    def generate_with_metrics(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        if self._backend_name == "transformers":
            return self._generate_with_transformers(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )
        if self._backend_name == "vllm":
            return self._generate_with_vllm(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
                sample_id=sample_id,
            )
        return self._generate_with_dummy(
            prompt=prompt,
            choices=choices,
            generation_config=generation_config,
            sample_id=sample_id,
        )

    def _load_transformers_backend(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _load_vllm_backend(self) -> None:
        from transformers import AutoProcessor

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._vllm_loop = asyncio.new_event_loop()
        # The evaluator submits one image at a time. Capture the actual public
        # prompt/vision-token ranges so both the ViT and multimodal prefill can
        # replay CUDA graphs. Larger private-set inputs safely fall back to eager.
        compilation_config = {
            "cudagraph_capture_sizes": [
                1, 2, 4, 8, 16, 24, 32,
                128, 192, 256, 320, 384, 448, 512, 576,
            ],
            "cudagraph_mm_encoder": True,
            "encoder_cudagraph_token_budgets": [
                256, 384, 512, 640, 768, 896, 1024,
            ],
            "encoder_cudagraph_max_vision_items_per_batch": 1,
        }
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tokenizer=self.model_path,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=2048,
            max_num_seqs=16,
            gpu_memory_utilization=float(
                os.environ.get("ZSX_GPU_MEMORY_UTILIZATION", "0.82")
            ),
            enable_prefix_caching=False,
            enforce_eager=os.environ.get("ZSX_PPU_EAGER") == "1",
            compilation_config=compilation_config,
            skip_mm_profiling=True,
            limit_mm_per_prompt={"image": 1},
        )
        self._model = AsyncLLM.from_engine_args(engine_args)
        self._tokenizer = self._model.tokenizer
        self._vllm_request_index = 0

    def _load_dummy_backend(self, reason: str) -> None:
        self._dummy_reason = reason

    def _generate_with_transformers(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        import torch
        from transformers import TextIteratorStreamer

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "do_sample": generation_config.temperature > 0,
            "use_cache": True,
            "streamer": streamer,
        }

        output_holder: dict[str, Any] = {}

        def _run_generate() -> None:
            with torch.no_grad():
                output_holder["output_ids"] = self._model.generate(**generation_kwargs)

        worker = threading.Thread(target=_run_generate, daemon=True)
        start = time.perf_counter()
        worker.start()

        first_chunk_at = None
        chunks: list[str] = []
        for chunk in streamer:
            now = time.perf_counter()
            if first_chunk_at is None and chunk:
                first_chunk_at = now
            chunks.append(chunk)
        worker.join()
        end = time.perf_counter()

        output_ids = output_holder["output_ids"]
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        ttft = (first_chunk_at - start) if first_chunk_at is not None else (end - start)
        return GenerationResult(
            text=text,
            token_count=int(generated_ids.shape[0]),
            ttft_seconds=ttft,
            elapsed_seconds=end - start,
            meta={"backend": "transformers"},
        )

    def _generate_with_vllm(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        return self._vllm_loop.run_until_complete(
            self._generate_with_vllm_async(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
                sample_id=sample_id,
            )
        )

    async def _generate_with_vllm_async(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _make_answer_only_prompt(prompt)},
                ],
            }
        ]
        rendered_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=min(generation_config.max_new_tokens, _ANSWER_MAX_TOKENS),
            output_kind=RequestOutputKind.DELTA,
        )
        llm_input = {
            "prompt": rendered_prompt,
            "multi_modal_data": {"image": image},
        }

        start = time.perf_counter()
        first_token_at = None
        chunks: list[str] = []
        token_count = 0
        last_output = None
        request_id = f"eval-{sample_id}-{self._vllm_request_index}"
        self._vllm_request_index += 1
        async for output in self._model.generate(
            prompt=llm_input,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            now = time.perf_counter()
            last_output = output
            delta_tokens = sum(
                len(completion.token_ids) for completion in output.outputs
            )
            if delta_tokens:
                if first_token_at is None:
                    first_token_at = now
                token_count += delta_tokens
            for completion in output.outputs:
                if completion.text:
                    chunks.append(completion.text)
            if output.finished:
                break
        end = time.perf_counter()

        if first_token_at is None:
            raise RuntimeError("vLLM did not stream any output tokens.")
        prompt_tokens = 0
        if last_output is not None:
            prompt_tokens = len(last_output.prompt_token_ids or [])

        return GenerationResult(
            text="".join(chunks).strip(),
            token_count=token_count,
            ttft_seconds=first_token_at - start,
            elapsed_seconds=end - start,
            meta={
                "backend": "vllm",
                "ttft_source": "stream_delta",
                "prompt_tokens": prompt_tokens,
            },
        )

    def _generate_with_dummy(
        self,
        *,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        start = time.perf_counter()
        usable_choices = [
            key for key, value in choices.items() if (value or "").strip()
        ]
        picked = (
            usable_choices[hash(sample_id) % len(usable_choices)]
            if usable_choices
            else "A"
        )
        text = (
            f"Answer: {picked}\n"
            "Explanation: dummy backend selected a deterministic option "
            "for smoke testing."
        )
        token_count = max(1, min(generation_config.max_new_tokens, len(text.split())))
        end = time.perf_counter()
        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=max(end - start, 1e-4),
            elapsed_seconds=max(end - start, 2e-4),
            meta={
                "backend": "dummy",
                "reason": getattr(self, "_dummy_reason", "n/a"),
                "prompt_chars": len(prompt),
            },
        )
