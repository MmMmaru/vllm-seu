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


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


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

        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._vllm_loop = asyncio.new_event_loop()
        profile_request_index = int(os.environ.get("PROFILE_REQUEST_INDEX", "-1"))
        thinking_mode = os.environ.get("VLLM_THINKING_MODE", "0") == "1"
        spec_method = os.environ.get("VLLM_SPEC_METHOD", "ngram")
        if spec_method == "suffix":
            speculative_config = {
                "method": "suffix",
                "suffix_decoding_max_tree_depth": int(
                    os.environ.get("VLLM_SUFFIX_MAX_TREE_DEPTH", "24")
                ),
                "suffix_decoding_max_cached_requests": int(
                    os.environ.get("VLLM_SUFFIX_MAX_CACHED_REQUESTS", "10000")
                ),
                "suffix_decoding_max_spec_factor": float(
                    os.environ.get("VLLM_SUFFIX_MAX_SPEC_FACTOR", "1.0")
                ),
                "suffix_decoding_min_token_prob": float(
                    os.environ.get("VLLM_SUFFIX_MIN_TOKEN_PROB", "0.1")
                ),
            }
        elif spec_method == "none":
            speculative_config = None
        elif spec_method == "ngram":
            speculative_config = {
                "method": "ngram",
                "num_speculative_tokens": 31,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 5,
            }
        elif spec_method == "ngram_gpu":
            speculative_config = {
                "method": "ngram_gpu",
                "num_speculative_tokens": 31,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 5,
            }
        elif spec_method == "custom_class":
            speculative_config = {
                "method": "custom_class",
                "model": "eval.global_ngram_proposer.GlobalNgramProposer",
                "num_speculative_tokens": 32,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 5,
            }
        elif spec_method == "mtp":
            # Native MTP: reuse the MTP weights shipped inside the target
            # checkpoint; token count configurable via VLLM_SPEC_NUM_TOKENS.
            speculative_config = {
                "method": "mtp",
                "num_speculative_tokens": int(
                    os.environ.get("VLLM_SPEC_NUM_TOKENS", "2")
                ),
            }
            # Streaming MTP: return sampled tokens before drafting completes
            # (deferred draft thread), requires the streaming-MTP model runner.
            if os.environ.get("VLLM_MTP_STREAM_OUTPUT", "0") == "1":
                speculative_config["stream_output_before_drafting"] = True
        else:
            raise ValueError(
                "VLLM_SPEC_METHOD must be one of: "
                "custom_class, mtp, ngram, ngram_gpu, none, suffix"
            )
        profiler_config = ({"profiler": "cuda"})
        if os.environ.get("PROFILE_BACKEND", "cuda") == "torch":
            # PROFILE_TRACE_DIR: 支持外部指定，否则按 profile-D-H-M 自动生成
            # e.g. .temp/profile-31-14-30 (D=日, H=时, M=分)
            _default_profile_dir = os.environ.get("PROFILE_TRACE_DIR")
            if not _default_profile_dir:
                from datetime import datetime

                _default_profile_dir = (
                    f".temp/profile-{datetime.now().strftime('%d-%H-%M')}"
                )
            profiler_config=(
                {
                    "profiler": os.environ.get("PROFILE_BACKEND", "cuda"),
                    "torch_profiler_dir": _default_profile_dir,
                    "torch_profiler_with_stack": True,
                    # 抓 HGG 小核需要更细粒度：开启 shape 记录便于对照 x(1,2048) 触发条件
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_flops": False,
                }
                if profile_request_index >= 0
                else {}
            )
        
        basic_mode = os.environ.get("VLLM_BASIC", "0") == "1"
        if basic_mode:
            # Most-basic vLLM baseline: strip every optimization knob.
            # No speculative decoding, no cuda graph / compilation, eager only,
            # synchronous scheduling, no mm profiling shortcuts, no encoder
            # cudagraph, no custom kernel envs (must also be disabled by the
            # caller via VLLM_PPU_FUSED_*=0).
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tokenizer=self.model_path,
                trust_remote_code=True,
                dtype=os.environ.get("VLLM_DTYPE", "float16"),
                max_model_len=6144 if thinking_mode else 2048,
                max_num_seqs=1,
                gpu_memory_utilization=0.80,
                disable_log_stats=True,
                profiler_config={},
                speculative_config=None,
                enforce_eager=True,
                async_scheduling=False,
            )
        else:
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tokenizer=self.model_path,
                trust_remote_code=True,
                dtype=os.environ.get("VLLM_DTYPE", "float16"),
                max_model_len=6144 if thinking_mode else 2048,
                max_num_seqs=1,
                gpu_memory_utilization=0.80,
                disable_log_stats=True,
                profiler_config=profiler_config if profile_request_index >= 0 else {},
                speculative_config=speculative_config,
                # debug switch: VLLM_ENFORCE_EAGER=1 disables cudagraph entirely
                enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
                compilation_config={
                    "cudagraph_mm_encoder": True,
                    "encoder_cudagraph_token_budgets": [64, 128, 256, 384, 512, 640, 768, 896, 1024],
                    "cudagraph_capture_sizes": [1,2,4,8,16,24,32,40,48,56,64,128,256,384,448,512],
                    "max_cudagraph_capture_size": 512,
                    "cudagraph_mode": "FULL_AND_PIECEWISE"
                },
                skip_mm_profiling=True,
                limit_mm_per_prompt={"image": 1},
                async_scheduling=(
                    # unset -> None (engine derives the default, which is ON for
                    # ngram_gpu); "1" -> force on; "0" -> force off
                    {"1": True, "0": False}.get(
                        os.environ.get("VLLM_ASYNC_SCHEDULING", "")
                    )
                ),
            )
        self._model = AsyncLLM.from_engine_args(engine_args)
        self._tokenizer = self._model.tokenizer
        line_break_ids = self._tokenizer.encode("\n", add_special_tokens=False)
        paragraph_break_ids = self._tokenizer.encode("\n\n", add_special_tokens=False)
        if len(line_break_ids) != 1 or len(paragraph_break_ids) != 1:
            raise ValueError("vLLM stop separators must each encode to one token.")
        self._vllm_stop_token_ids = [line_break_ids[0], paragraph_break_ids[0]]
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

    def _resize_image_like_processor(self, image):
        """Apply the processor's smart_resize geometry before the timed window.

        The engine-side image processor re-applies smart_resize on the
        pre-resized image, which is then a no-op, so pixels and semantics
        are unchanged while the CPU resize cost leaves the timed path.
        """
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                smart_resize,
            )

            ip = getattr(self._processor, "image_processor", None)
            if ip is None or image is None:
                return image
            new_h, new_w = smart_resize(
                image.height,
                image.width,
                factor=ip.patch_size * ip.merge_size,
                min_pixels=ip.size["shortest_edge"],
                max_pixels=ip.size["longest_edge"],
            )
            if (new_h, new_w) == (image.height, image.width):
                return image
            return image.resize((new_w, new_h), ip.resample)
        except Exception:
            return image

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
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=os.environ.get("VLLM_THINKING_MODE", "0") == "1",
        )
        prompt_token_ids = self._tokenizer.encode(
            rendered_prompt,
            add_special_tokens=False,
        )
        # Basic mode keeps the image untouched (engine-side resize stays inside
        # the timed path); optimized mode pre-resizes to move CPU cost out.
        if os.environ.get("VLLM_BASIC", "0") != "1":
            image = self._resize_image_like_processor(image)
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_tokens=generation_config.max_new_tokens,
            stop_token_ids=(
                []
                if os.environ.get("VLLM_THINKING_MODE", "0") == "1"
                else self._vllm_stop_token_ids
            ),
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
        )
        llm_input = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"image": image},
        }
        profile_request_index = int(os.environ.get("PROFILE_REQUEST_INDEX", "-1"))
        request_index = self._vllm_request_index
        should_profile = request_index == profile_request_index
        if should_profile:
            await self._model.start_profile(profile_prefix=f"request-{request_index}")
        start = time.perf_counter()
        first_token_at = None
        generated_token_ids: list[int] = []
        token_count = 0
        last_output = None
        request_id = f"eval-{sample_id}-{request_index}"
        self._vllm_request_index += 1
        try:
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
                        generated_token_ids.extend(completion.token_ids)
                if output.finished:
                    break
            text = self._tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            end = time.perf_counter()
        finally:
            if should_profile:
                await self._model.stop_profile()

        if first_token_at is None:
            raise RuntimeError("vLLM did not stream any output tokens.")
        prompt_tokens = 0
        if last_output is not None:
            prompt_tokens = len(last_output.prompt_token_ids or [])

        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=first_token_at - start,
            elapsed_seconds=end - start,
            meta={
                "backend": "vllm",
                "ttft_source": "stream_delta",
                "prompt_tokens": prompt_tokens,
                "profile_request_index": profile_request_index,
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
