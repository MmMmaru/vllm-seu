# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-request n-gram speculative proposer for the DNDX benchmark.

Standard vLLM n-gram speculation only matches n-grams within the current
request. In this benchmark answers share short templates across requests
("Answer: X ..." / "答案是 X ……"), so a global occurrence table built from
every token stream seen so far yields extra draft hits, especially for
Chinese answers whose per-request matches are sparse.

Drafts come from an exact pure-Python port of the stock in-request KMP
match first; only when the current request has no match is the global
table consulted, with a stricter minimum n-gram length to avoid junk
short matches from unrelated samples. (The port avoids numba: nesting the
stock jitted proposer crashed the EngineCore worker with a segfault in
numba's OpenMP pool.)

Drafts are verified by the target model under greedy decoding, so output
tokens equal non-speculative decoding up to floating-point tie-breaks;
only speed changes. The table lives in the EngineCore worker process
memory and is rebuilt from scratch on every engine start.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from vllm.config import VllmConfig


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the leading run where both token lists agree."""
    n = min(len(a), len(b))
    for j in range(n):
        if a[j] != b[j]:
            return j
    return n


class SpecStatsRecorder:
    """Append-only per-step speculative decoding event recorder.

    Lives in the worker process alongside the proposer; events are streamed
    to a JSONL file so the benchmark process can aggregate them afterwards.
    Disabled entirely when `path` is None (submission default).
    """

    def __init__(self, path: str | None) -> None:
        self._fh = open(path, "a", encoding="utf-8") if path else None
        # Request ordinal, aligned with the wrapper's per-engine request index
        # (warmup requests included): incremented on every new-request event.
        self._ordinal = -1
        # Batch row -> draft returned for that row at the previous step.
        self._prev_drafts: dict[int, list[int]] = {}

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def on_step(
        self,
        *,
        row: int,
        req_id: str | None,
        is_new_request: bool,
        source: str,
        draft: list[int],
        sampled: list[int],
        cpu_ms: float,
        global_support: int | None,
    ) -> None:
        """Record one propose step as a JSONL event.

        The acceptance of the previous step's draft is derived here: the
        sampled tokens of this step begin with the accepted draft prefix
        (followed by the recovered/bonus token), so the common prefix length
        against the stashed draft is the exact acceptance length.
        """
        if self._fh is None:
            return
        if is_new_request:
            self._ordinal += 1
            self._prev_drafts.pop(row, None)
        prev = self._prev_drafts.get(row)
        accepted_prev = (
            _common_prefix_len(prev, sampled) if prev else None
        )
        self._prev_drafts[row] = draft
        event = {
            "ordinal": self._ordinal,
            "req_id": req_id,
            "row": row,
            "source": source,
            "draft_len": len(draft),
            "accepted_prev": accepted_prev,
            "cpu_ms": round(cpu_ms, 4),
            "support": global_support,
        }
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()


def _in_request_draft(
    ids: list[int],
    min_n: int,
    max_n: int,
    k: int,
) -> list[int]:
    """Pure-Python port of NgramProposer's KMP suffix match.

    Finds the longest n-gram (length within [min_n, max_n]) that matches a
    suffix of `ids` at an earlier position and returns up to k tokens that
    followed the earliest such occurrence. Mirrors the numba original,
    including its tie-breaking.
    """
    total = len(ids)
    if total < min_n or k <= 0:
        return []
    tokens = ids[::-1]
    # lps[i]: longest prefix (length <= max_n) that is a suffix of
    # tokens[:i+1], see the numba original for the derivation.
    lps = [0] * max_n
    longest = 0
    position = 0
    prev = 0
    i = 1
    while i < total:
        if tokens[prev] == tokens[i]:
            prev += 1
            if prev >= longest:
                longest = prev
                position = i
            if i < max_n:
                lps[i] = prev
            if prev == max_n:
                prev = lps[max_n - 1]
            i += 1
        elif prev != 0:
            prev = lps[prev - 1]
        else:
            i += 1
    if longest < min_n:
        return []
    start = total - 1 - position + longest
    return ids[start : start + k]


class GlobalNgramProposer:
    """In-request n-gram proposer with a cross-request fallback table."""

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        spec = vllm_config.speculative_config
        # Number of draft tokens to propose after a match.
        self.k = spec.num_speculative_tokens
        # N-gram length range used for matching, mirroring the ngram config.
        self.min_n = spec.prompt_lookup_min or 1
        self.max_n = spec.prompt_lookup_max or 5
        # Cross-request drafts must match a full-length n-gram: shorter
        # cross-request matches are mostly junk drafts that waste verify
        # compute (observed on en20), while 5-gram hits carry templates.
        self.global_min_n = self.max_n
        self.max_model_len = vllm_config.model_config.max_model_len
        # Global token stream: prompt + generated tokens of every request,
        # concatenated in arrival order.
        self._stream: list[int] = []
        # n-gram tuple -> ascending stream positions right after each
        # occurrence (i.e. index of the first following token).
        self._pos: dict[tuple[int, ...], list[int]] = {}
        # Batch row -> number of tokens of that row already ingested.
        self._ingested: dict[int, int] = {}
        # Safety valve for RAM on very large private sets.
        self._max_keys = 1_500_000
        # Optional per-step event recorder (SPEC_STATS_PATH env, worker side).
        self._stats = SpecStatsRecorder(os.environ.get("SPEC_STATS_PATH") or None)
        # Stats-only request identity hook: gpu_model_runner fills
        # current_req_ids before each propose call when wants_req_ids is set.
        # Used solely for per-request stats alignment; the drafting logic
        # below intentionally keeps its original total-drop heuristic so the
        # instrumented run behaves exactly like the submission default.
        self.wants_req_ids = True
        self.current_req_ids: list[str] = []
        self._row_req_ids: dict[int, str] = {}

    def load_model(self, *args, **kwargs):
        # No draft model to load.
        pass

    def _ingest(self, ids: list[int]) -> None:
        """Append newly arrived tokens of a batch row to the global stream."""
        base = len(self._stream)
        self._stream.extend(ids)
        if len(self._pos) >= self._max_keys:
            return
        for offset in range(len(ids)):
            end = base + offset  # stream index of the new token
            # Only n-gram lengths that _lookup can actually use are
            # indexed (currently just max_n); this keeps ingestion cheap
            # and the table small.
            for n in range(self.global_min_n, self.max_n + 1):
                start = end - n
                if start < 0:
                    break
                key = tuple(self._stream[start:end])
                self._pos.setdefault(key, []).append(end)

    def _lookup_impl(
        self, suffix: list[int], max_len: int
    ) -> tuple[list[int], int | None]:
        """Find a draft via the global table, also reporting support.

        Returns the draft plus the number of candidate positions recorded
        for the matched n-gram key (None when no key matched).
        """
        upper = min(self.max_n, len(suffix))
        for n in range(upper, self.global_min_n - 1, -1):
            positions = self._pos.get(tuple(suffix[-n:]))
            if not positions:
                continue
            # Prefer the most recent occurrence that has following tokens.
            for pos in reversed(positions[-8:]):
                if pos < len(self._stream):
                    draft = self._stream[pos : pos + max_len]
                    if draft:
                        return draft, len(positions)
        return [], None

    def _lookup(self, suffix: list[int], max_len: int) -> list[int]:
        """Find a draft for the given token suffix via the global table."""
        return self._lookup_impl(suffix, max_len)[0]

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings=None,
    ) -> list[list[int]]:
        """Propose draft tokens for each request in the batch.

        Args:
            sampled_token_ids: Newly sampled (accepted) tokens per request.
            num_tokens_no_spec: Prompt + accepted token counts per request.
            token_ids_cpu: Token id buffer of shape (batch, max_model_len).
            slot_mappings: Unused, kept for interface compatibility.

        Returns:
            A list of draft token id lists, one per request.
        """
        stats_on = self._stats.enabled
        started = time.perf_counter() if stats_on else 0.0
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            total = int(num_tokens_no_spec[i])
            if not sampled or total >= self.max_model_len:
                drafts.append([])
                continue
            prev = self._ingested.get(i, 0)
            is_new_request = i not in self._ingested or prev > total
            if prev > total:
                prev = 0  # row reused by a new request
            if prev < total:
                self._ingest(ids=token_ids_cpu[i, prev:total].tolist())
                self._ingested[i] = total
            ids = token_ids_cpu[i, :total].tolist()
            draft = _in_request_draft(ids, self.min_n, self.max_n, self.k)
            if stats_on:
                req_id = (
                    self.current_req_ids[i]
                    if i < len(self.current_req_ids)
                    else None
                )
                if req_id is not None:
                    # Exact boundary detection for stats alignment.
                    is_new_request = self._row_req_ids.get(i) != req_id
                    self._row_req_ids[i] = req_id
                source = "in_request" if draft else "none"
                support: int | None = None
                if not draft:
                    draft, support = self._lookup_impl(ids[-self.max_n :], self.k)
                    if draft:
                        source = "global"
            elif not draft:
                draft = self._lookup(ids[-self.max_n :], self.k)
            drafts.append(draft)
            if stats_on:
                self._stats.on_step(
                    row=i,
                    req_id=req_id,
                    is_new_request=is_new_request,
                    source=source,
                    draft=draft,
                    sampled=list(sampled),
                    cpu_ms=(time.perf_counter() - started) * 1000.0,
                    global_support=support,
                )
        return drafts
