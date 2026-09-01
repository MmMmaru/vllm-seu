# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from evaluation_wrapper import _make_answer_only_prompt


def test_answer_only_prompt_preserves_question_and_removes_reason_request() -> None:
    prompt = (
        "Solve this single-choice question. Your response must make one final choice "
        "among A/B/C/D clearly. You may include one short reason.\nQuestion: 2+2?"
    )

    optimized = _make_answer_only_prompt(prompt)

    assert optimized == "Answer only: Answer: X (A/B/C/D).\nQuestion: 2+2?"
