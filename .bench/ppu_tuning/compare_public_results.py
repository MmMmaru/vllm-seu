#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    return parser.parse_args()


def pct(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def main() -> None:
    ns = args()
    before = json.loads(Path(ns.before).read_text(encoding="utf-8"))
    after = json.loads(Path(ns.after).read_text(encoding="utf-8"))
    b_perf, a_perf = before["performance"], after["performance"]
    b_time, a_time = before["timing"], after["timing"]
    b_answers = {item["question_id"]: item["parsed_answer"] for item in before["answers"]}
    a_answers = {item["question_id"]: item["parsed_answer"] for item in after["answers"]}
    common = sorted(set(b_answers) & set(a_answers))
    matching = sum(b_answers[key] == a_answers[key] for key in common)
    report = {
        "before": {
            "performance": b_perf,
            "timing": b_time,
            "accuracy": before["accuracy"],
            "public_validation": before["public_validation"],
        },
        "after": {
            "performance": a_perf,
            "timing": a_time,
            "accuracy": after["accuracy"],
            "public_validation": after["public_validation"],
        },
        "change_percent": {
            "avg_ttft_ms": pct(b_perf["avg_ttft_ms"], a_perf["avg_ttft_ms"]),
            "avg_throughput_tokens_per_sec": pct(
                b_perf["avg_throughput_tokens_per_sec"],
                a_perf["avg_throughput_tokens_per_sec"],
            ),
            "benchmark_elapsed_seconds": pct(
                b_time["benchmark_elapsed_seconds"],
                a_time["benchmark_elapsed_seconds"],
            ),
        },
        "answer_agreement": {
            "matching": matching,
            "total": len(common),
            "rate": matching / len(common) if common else None,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
