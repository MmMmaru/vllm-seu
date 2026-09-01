#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public DNDX self-test benchmark for participants.

This script is for local development on public dev data only. Final ranking is
computed by the organizer with the private judge package.
"""

from __future__ import annotations

import os
import sys

sys.path[0] = os.getcwd()

import argparse
import csv
import io
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pybase64 as base64
import regex as re
from PIL import Image

from eval.evaluation_wrapper import GenerationConfig, VLMModel

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
        default="/home/maru/project/datasets/mmbench/mmbench_dev_cn.tsv",
        help="Path to a public MMBench TSV file",
    )
    parser.add_argument(
        "--model-path", type=str, default="/home/maru/huggingface/Qwen3.5-2B-W8A16-noGDN"  # 
    )
    parser.add_argument("--output", type=str, default="eval/results/result_public_w8a16_cn.json")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--backend", choices=["auto", "dummy", "transformers", "vllm"], default="vllm"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument(
        "--spec-stats-path",
        type=str,
        default=".temp/spec_stats.jsonl",
        help="JSONL written by the speculative proposer (SPEC_STATS_PATH env); "
        "merged into per-sample meta when present",
    )
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
    if os.environ.get("VLLM_THINKING_MODE", "0") == "1":
        return GenerationConfig(
            max_new_tokens=int(os.environ.get("VLLM_THINKING_MAX_NEW_TOKENS", "4096")),
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
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


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile; None for empty input."""
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (pct / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def load_spec_stats(path: Path) -> dict[str, dict]:
    """Aggregate proposer JSONL step events into per-request stats.

    Keyed by request id (e.g. "eval-42-3") when the engine hook provides it,
    otherwise by request ordinal (warmup requests included, arrival order).
    Missing file yields an empty dict so non-instrumented configurations are
    silently skipped.
    """
    stats: dict[str, dict] = {}
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        key = event.get("req_id") or f"ordinal-{event['ordinal']}"
        rec = stats.setdefault(
            key,
            {
                "steps": 0,
                "in_request_hits": 0,
                "global_hits": 0,
                "no_draft_steps": 0,
                "proposed_tokens": 0,
                "accepted_tokens": 0,
                "acceptance_lengths": [],
                "proposer_cpu_ms": 0.0,
                "global_support": [],
            },
        )
        rec["steps"] += 1
        source = event.get("source")
        if source == "in_request":
            rec["in_request_hits"] += 1
        elif source == "global":
            rec["global_hits"] += 1
        else:
            rec["no_draft_steps"] += 1
        rec["proposed_tokens"] += int(event.get("draft_len") or 0)
        rec["proposer_cpu_ms"] += float(event.get("cpu_ms") or 0.0)
        accepted_prev = event.get("accepted_prev")
        if accepted_prev is not None:
            rec["accepted_tokens"] += int(accepted_prev)
            rec["acceptance_lengths"].append(int(accepted_prev))
        support = event.get("support")
        if support is not None:
            rec["global_support"].append(int(support))
    for rec in stats.values():
        rec["proposer_cpu_ms"] = round(rec["proposer_cpu_ms"], 3)
    return stats


def summarize_spec_stats(stats: dict[str, dict]) -> dict | None:
    """Build the run-level spec_decode aggregate for the result payload."""
    if not stats:
        return None
    total_proposed = sum(r["proposed_tokens"] for r in stats.values())
    total_accepted = sum(r["accepted_tokens"] for r in stats.values())
    histogram: dict[str, int] = {}
    cpu_ms: list[float] = []
    for rec in stats.values():
        for length in rec["acceptance_lengths"]:
            histogram[str(length)] = histogram.get(str(length), 0) + 1
        cpu_ms.append(rec["proposer_cpu_ms"])
    return {
        "requests": len(stats),
        "in_request_hits": sum(r["in_request_hits"] for r in stats.values()),
        "global_hits": sum(r["global_hits"] for r in stats.values()),
        "no_draft_steps": sum(r["no_draft_steps"] for r in stats.values()),
        "proposed_tokens": total_proposed,
        "accepted_tokens": total_accepted,
        "acceptance_rate": (
            round(total_accepted / total_proposed, 4) if total_proposed else None
        ),
        "acceptance_length_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
        "avg_proposer_cpu_ms_per_request": (
            round(sum(cpu_ms) / len(cpu_ms), 3) if cpu_ms else None
        ),
    }


def settle_runtime(model: VLMModel) -> None:
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
    spec_stats_path = Path(args.spec_stats_path)
    # Start each run with a fresh stats file; the worker-side proposer
    # (SPEC_STATS_PATH env) appends step events during generation.
    spec_stats_path.parent.mkdir(parents=True, exist_ok=True)
    spec_stats_path.unlink(missing_ok=True)
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
        for sample in samples[: min(args.warmup_samples, len(samples))]:
            settle_runtime(model)
            model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=fixed_generation_config(),
                sample_id=sample.sample_id,
            )
            settle_runtime(model)

        for sample in samples:
            settle_runtime(model)
            config = fixed_generation_config()
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
                    "response_text": result.text,
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
    warmup_count = min(args.warmup_samples, len(samples))
    spec_stats = load_spec_stats(spec_stats_path)
    if spec_stats:
        # Engine-side request ids are eval-{sample_id}-{index}-{suffix},
        # where index counts every generation call (warmup first) and the
        # suffix is appended by vLLM for uniqueness; fall back to ordinals.
        for idx, record in enumerate(records):
            prefix = f"eval-{record['question_id']}-{warmup_count + idx}-"
            record_spec = next(
                (v for k, v in spec_stats.items() if k.startswith(prefix)),
                None,
            ) or spec_stats.get(f"ordinal-{warmup_count + idx}")
            if record_spec is not None:
                record["meta"]["spec"] = record_spec
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
            "p50_ttft_ms": (
                round(v, 3) if (v := percentile(ttfts_ms, 50)) is not None else None
            ),
            "p95_ttft_ms": (
                round(v, 3) if (v := percentile(ttfts_ms, 95)) is not None else None
            ),
            "avg_throughput_tokens_per_sec": (
                round(sum(throughputs) / len(throughputs), 3) if throughputs else 0.0
            ),
            "p50_throughput_tokens_per_sec": (
                round(v, 3)
                if (v := percentile(throughputs, 50)) is not None
                else None
            ),
            "p95_throughput_tokens_per_sec": (
                round(v, 3)
                if (v := percentile(throughputs, 95)) is not None
                else None
            ),
        },
        "spec_decode": summarize_spec_stats(spec_stats),
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
