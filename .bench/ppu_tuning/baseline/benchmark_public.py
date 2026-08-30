#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public DNDX self-test benchmark for participants.

This script is for local development on public dev data only. Final ranking is
computed by the organizer with the private judge package.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Use this checkout's Python sources even when the shared virtual environment
# contains an editable vLLM installation pointing at another checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (
    os.environ.get("ZSX_USE_INSTALLED_VLLM") != "1"
    and str(PROJECT_ROOT) not in sys.path
):
    sys.path.insert(0, str(PROJECT_ROOT))

import pybase64 as base64
import regex as re
from evaluation_wrapper import GenerationConfig, VLMModel
from PIL import Image

ANSWER_RE = re.compile(
    r"(?:answer|答案|正确答案)\s*(?:is|为|是)?\s*[:：]?\s*\**\s*([ABCD])"
    r"|^\s*\**\s*([ABCD])(?:\**[\s\.\):：]|$)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Sample:
    sample_id: str
    language: str
    question: str
    hint: str
    choices: dict[str, str]
    answer: str
    image_b64: str
    category: str
    subcategory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNDX public self-test benchmark")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/mmbench/mmbench_dev_en.tsv",
        help="Path to a public MMBench TSV file",
    )
    parser.add_argument("--model-path", type=str, default="./Qwen3.5-2B")
    parser.add_argument("--output", type=str, default="result_public.json")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--backend", choices=["auto", "dummy", "transformers", "vllm"], default="vllm"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-samples", type=int, default=2)
    return parser.parse_args()


def load_mmbench_tsv(path: Path, limit: int | None = None) -> list[Sample]:
    language = "cn" if "_cn" in path.name.lower() else "en"
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            samples.append(
                Sample(
                    sample_id=str(row["index"]),
                    language=language,
                    question=(row.get("question") or "").strip(),
                    hint=(row.get("hint") or "").strip(),
                    choices={
                        key: (row.get(key) or "").strip()
                        for key in ["A", "B", "C", "D"]
                    },
                    answer=(row.get("answer") or "").strip().upper(),
                    image_b64=row["image"],
                    category=(row.get("category") or "").strip(),
                    subcategory=(row.get("l2-category") or "").strip(),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def build_prompt(sample: Sample) -> str:
    option_block = "\n".join(
        f"{key}. {value}" for key, value in sample.choices.items() if value.strip()
    )
    hint_block = f"Hint: {sample.hint}\n" if sample.hint else ""
    if sample.language == "cn":
        instruction = (
            "请完成这道单选题。"
            "请给出你认为正确的选项，并可附带一句简短理由。"
            "答案必须明确，且只能对应 A/B/C/D 中的一个选项。"
        )
    else:
        instruction = (
            "Solve this single-choice question."
            " Your response must make one final choice among A/B/C/D clearly."
            " You may include one short reason."
        )
    return f"{instruction}\n{hint_block}Question: {sample.question}\n{option_block}\n"


def fixed_generation_config() -> GenerationConfig:
    return GenerationConfig(max_new_tokens=64, temperature=0.0, top_p=1.0)


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    match = ANSWER_RE.search(text)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group.upper()
    return None


def compute_throughput(
    token_count: int, ttft_seconds: float, elapsed_seconds: float
) -> float:
    if token_count <= 0 or elapsed_seconds <= 0:
        return 0.0
    decode_window = max(elapsed_seconds - max(ttft_seconds, 0.0), 1e-6)
    effective_tokens = max(token_count - 1, 1)
    return effective_tokens / decode_window


def settle_runtime(model: VLMModel) -> None:
    # vLLM returns only after the streamed request is complete.  Sleeping here
    # adds fixed wall-clock overhead to every sample without synchronizing any
    # outstanding work.  Keep synchronization only for the Transformers path.
    if getattr(model, "backend_name", "") == "vllm":
        return
    torch_mod = getattr(model, "_torch", None)
    if torch_mod is None:
        return
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
            torch_mod.cuda.empty_cache()
            torch_mod.cuda.synchronize()
    except Exception:
        pass
    time.sleep(0.01)


def validate_public_result(
    text: str,
    parsed_answer: str | None,
    token_count: int,
    max_new_tokens: int,
) -> list[str]:
    errors: list[str] = []
    normalized = (text or "").strip()
    if not normalized:
        errors.append("empty_output")
    if parsed_answer not in {"A", "B", "C", "D"}:
        errors.append("missing_choice_answer")
    if token_count <= 0:
        errors.append("zero_generated_tokens")
    if token_count > max_new_tokens + 8:
        errors.append("token_count_exceeds_budget")
    if len(normalized) > 1200:
        errors.append("output_too_long")
    return errors


def run_benchmark(args: argparse.Namespace) -> dict:
    benchmark_start = time.perf_counter()
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass

    dataset_path = Path(args.dataset_path).resolve()
    if "/datasets/test/" in str(dataset_path):
        raise ValueError("benchmark_public.py only supports public dev datasets.")

    output_path = Path(args.output).resolve()
    samples = load_mmbench_tsv(dataset_path, limit=args.num_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    model = VLMModel(args.model_path, backend=args.backend, device=args.device)

    records = []
    ttfts_ms = []
    throughputs = []
    correct = 0
    validation_errors = 0

    try:
        generation_config = fixed_generation_config()
        for sample in samples[: min(args.warmup_samples, len(samples))]:
            settle_runtime(model)
            model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=generation_config,
                sample_id=sample.sample_id,
            )
            settle_runtime(model)

        for sample in samples:
            settle_runtime(model)
            config = generation_config
            result = model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=config,
                sample_id=sample.sample_id,
            )
            parsed_answer = extract_answer(result.text)
            errors = validate_public_result(
                result.text, parsed_answer, result.token_count, config.max_new_tokens
            )
            validation_errors += int(bool(errors))
            is_correct = parsed_answer == sample.answer
            correct += int(is_correct)

            ttft_ms = result.ttft_seconds * 1000.0
            throughput = compute_throughput(
                result.token_count, result.ttft_seconds, result.elapsed_seconds
            )
            if math.isfinite(ttft_ms) and ttft_ms > 0:
                ttfts_ms.append(ttft_ms)
            if math.isfinite(throughput) and throughput > 0:
                throughputs.append(throughput)

            records.append(
                {
                    "question_id": sample.sample_id,
                    "parsed_answer": parsed_answer,
                    "correct": is_correct,
                    "ttft_ms": round(ttft_ms, 3),
                    "throughput_tokens_per_sec": round(throughput, 3),
                    "token_count": result.token_count,
                    "validation_errors": errors,
                    "meta": result.meta,
                }
            )
            settle_runtime(model)
    finally:
        shutdown = getattr(model, "shutdown", None)
        if shutdown is not None:
            shutdown()

    elapsed = time.perf_counter() - benchmark_start
    payload = {
        "benchmark_version": "dndx_public_self_test",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "seed": args.seed,
        "backend": model.backend_name,
        "performance": {
            "avg_ttft_ms": (
                round(sum(ttfts_ms) / len(ttfts_ms), 3) if ttfts_ms else None
            ),
            "avg_throughput_tokens_per_sec": (
                round(sum(throughputs) / len(throughputs), 3) if throughputs else 0.0
            ),
        },
        "timing": {
            "benchmark_elapsed_seconds": round(elapsed, 3),
            "benchmark_elapsed_minutes": round(elapsed / 60.0, 3),
            "avg_seconds_per_sample": round(elapsed / len(samples), 3),
        },
        "accuracy": {
            "score": round(correct / len(samples), 6),
            "correct": correct,
            "total": len(samples),
        },
        "public_validation": {
            "passed": validation_errors == 0,
            "failed_samples": validation_errors,
        },
        "answers": records,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    payload = run_benchmark(parse_args())
    print(
        json.dumps(
            {
                "backend": payload["backend"],
                "sample_count": payload["sample_count"],
                "avg_ttft_ms": payload["performance"]["avg_ttft_ms"],
                "avg_throughput_tokens_per_sec": payload["performance"][
                    "avg_throughput_tokens_per_sec"
                ],
                "accuracy": payload["accuracy"]["score"],
                "public_validation_passed": payload["public_validation"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
