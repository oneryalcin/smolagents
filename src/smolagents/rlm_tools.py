"""
RLM (Recursive Language Model) tools for sub-LLM calls.

These tools give a CodeAgent the ability to delegate semantic analysis
to sub-LLM calls while keeping the orchestration in Python code.

Note: LLMQueryBatchedTool calls model.generate() from multiple threads.
The Model implementation must be thread-safe for concurrent generate() calls.
Most HTTP-based clients (LiteLLM, OpenAI SDK) are thread-safe.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage
from smolagents.tools import Tool


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Limits for sub-LLM calls. None = unlimited."""

    max_llm_calls: int | None = None
    max_total_tokens: int | None = None

    def __post_init__(self):
        if self.max_llm_calls is not None and self.max_llm_calls <= 0:
            raise ValueError(f"max_llm_calls must be positive, got {self.max_llm_calls}")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError(f"max_total_tokens must be positive, got {self.max_total_tokens}")


class BudgetExceededError(Exception):
    """Raised when a budget limit is reached."""


class BudgetManager:
    """Thread-safe budget tracker for sub-LLM calls.

    Usage protocol in tools:
        budget.pre_call_check()              # reserve slot, raises if exhausted
        try:
            response = model.generate(...)
        except Exception:
            budget.release_call()            # return slot on failure
            raise
        budget.record_usage(response.token_usage)  # record tokens (never raises)

    pre_call_check atomically reserves a call slot. If generate() fails,
    release_call() returns the slot. Token limits are checked on the next
    pre_call_check, so the call that crosses a token threshold completes —
    the *next* call is blocked.
    """

    def __init__(self, budget: Budget):
        self.budget = budget
        self._lock = threading.Lock()
        self._llm_calls = 0
        self._total_tokens = 0

    def pre_call_check(self):
        """Reserve a call slot. Raises BudgetExceededError if budget exhausted.

        Atomically increments the call counter to reserve the slot. If generate()
        fails afterward, call release_call() to return the slot.
        """
        with self._lock:
            if self.budget.max_llm_calls is not None and self._llm_calls >= self.budget.max_llm_calls:
                raise BudgetExceededError(
                    f"Sub-LLM call budget exceeded: {self._llm_calls}/{self.budget.max_llm_calls} calls used"
                )
            if self.budget.max_total_tokens is not None and self._total_tokens >= self.budget.max_total_tokens:
                raise BudgetExceededError(
                    f"Sub-LLM token budget exceeded: {self._total_tokens:,}/{self.budget.max_total_tokens:,} tokens used"
                )
            self._llm_calls += 1

    def release_call(self):
        """Return a reserved call slot if generate() failed. Decrements call counter."""
        with self._lock:
            self._llm_calls -= 1

    def record_usage(self, token_usage: TokenUsage | None = None):
        """Record token usage after a successful call. Never raises."""
        if not token_usage:
            return
        with self._lock:
            self._total_tokens += token_usage.total_tokens

    @property
    def summary(self) -> str:
        with self._lock:
            calls = self._llm_calls
            tokens = self._total_tokens
        # Format outside lock
        calls_str = f"Sub-LLM calls: {calls}"
        if self.budget.max_llm_calls is not None:
            calls_str += f"/{self.budget.max_llm_calls}"
        tokens_str = f"Sub-LLM tokens: {tokens:,}"
        if self.budget.max_total_tokens is not None:
            tokens_str += f"/{self.budget.max_total_tokens:,}"
        return f"{calls_str} | {tokens_str}"

    def reset(self):
        with self._lock:
            self._llm_calls = 0
            self._total_tokens = 0


class LLMQueryTool(Tool):
    """Query a sub-LLM for semantic analysis of a chunk of text."""

    name = "llm_query"
    description = (
        "Query a language model for semantic analysis (classification, extraction, summarization). "
        "Use Python string ops (in, re, len) for pattern/counting tasks — they're free and instant."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The prompt including both instructions AND context to analyze.",
        }
    }
    output_type = "string"

    def __init__(self, model: Model, budget_manager: BudgetManager | None = None, rlm_logger=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.budget_manager = budget_manager
        self.rlm_logger = rlm_logger

    def forward(self, prompt: str) -> str:
        if self.budget_manager:
            self.budget_manager.pre_call_check()
        call_start = time.time()
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = self.model.generate(messages)
        except Exception:
            if self.budget_manager:
                self.budget_manager.release_call()
            raise
        call_end = time.time()
        if self.budget_manager:
            self.budget_manager.record_usage(response.token_usage)
        if self.rlm_logger:
            self.rlm_logger.emit_sub_llm(
                prompt=prompt, response=response.content or "",
                token_usage=response.token_usage,
                call_start=call_start, call_end=call_end,
            )
        return response.content or ""


class LLMQueryBatchedTool(Tool):
    """Query a sub-LLM with multiple prompts in parallel."""

    name = "llm_query_batched"
    description = (
        "Query a language model with multiple prompts in PARALLEL. "
        "Much faster than calling llm_query in a loop. Returns a list of responses "
        "in the same order as the input prompts."
    )
    inputs = {
        "prompts": {
            "type": "array",
            "description": "List of prompt strings to process in parallel.",
        }
    }
    output_type = "array"

    def __init__(self, model: Model, max_workers: int = 8, budget_manager: BudgetManager | None = None, rlm_logger=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.max_workers = max_workers
        self.budget_manager = budget_manager
        self.rlm_logger = rlm_logger

    def forward(self, prompts: list) -> list:
        if not prompts:
            return []

        n = len(prompts)

        def _query_one(prompt: str) -> str:
            # NOTE: keep budget + logging protocol in sync with LLMQueryTool.forward
            if self.budget_manager:
                self.budget_manager.pre_call_check()
            call_start = time.time()
            try:
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
                response = self.model.generate(messages)
            except Exception:
                if self.budget_manager:
                    self.budget_manager.release_call()
                raise
            call_end = time.time()
            if self.budget_manager:
                self.budget_manager.record_usage(response.token_usage)
            if self.rlm_logger:
                self.rlm_logger.emit_sub_llm(
                    prompt=prompt, response=response.content or "",
                    token_usage=response.token_usage,
                    call_start=call_start, call_end=call_end,
                )
            return response.content or ""

        results = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as executor:
            futures = {executor.submit(_query_one, p): i for i, p in enumerate(prompts)}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                # Cancel pending (not-yet-started) futures to avoid wasting API calls.
                # Already-running futures will complete — cancel() is a no-op for those.
                for f in futures:
                    f.cancel()
                raise

        return [results[i] for i in range(n)]
