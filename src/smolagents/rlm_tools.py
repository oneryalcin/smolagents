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


def _execute_sub_llm(
    model: Model, prompt: str, budget_manager: BudgetManager | None, rlm_logger,
    max_prompt_chars: int | None = None,
) -> str:
    """Execute a single sub-LLM call with budget tracking and logging.

    Shared by LLMQueryTool and LLMQueryBatchedTool to avoid protocol drift.
    Thread-safe: budget and logger handle their own locking.
    """
    if max_prompt_chars and len(prompt) > max_prompt_chars:
        raise ValueError(
            f"Prompt too long: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens). "
            f"Max allowed: {max_prompt_chars:,} chars (~{max_prompt_chars // 4:,} tokens). "
            f"Filter or chunk your data before calling llm_query."
        )
    if budget_manager:
        budget_manager.pre_call_check()
    call_start = time.time()
    try:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = model.generate(messages)
    except Exception:
        if budget_manager:
            budget_manager.release_call()
        raise
    call_end = time.time()
    if budget_manager:
        budget_manager.record_usage(response.token_usage)
    if rlm_logger:
        rlm_logger.emit_sub_llm(
            prompt=prompt, response=response.content or "",
            token_usage=response.token_usage,
            call_start=call_start, call_end=call_end,
        )
    return response.content or ""


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

    def __init__(self, model: Model, budget_manager: BudgetManager | None = None, rlm_logger=None, max_prompt_chars: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.budget_manager = budget_manager
        self.rlm_logger = rlm_logger
        self.max_prompt_chars = max_prompt_chars

    def forward(self, prompt: str) -> str:
        return _execute_sub_llm(self.model, prompt, self.budget_manager, self.rlm_logger, self.max_prompt_chars)


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

    def __init__(self, model: Model, max_workers: int = 8, budget_manager: BudgetManager | None = None, rlm_logger=None, max_prompt_chars: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.max_workers = max_workers
        self.budget_manager = budget_manager
        self.rlm_logger = rlm_logger
        self.max_prompt_chars = max_prompt_chars

    def forward(self, prompts: list) -> list:
        if not prompts:
            return []

        n = len(prompts)
        results = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as executor:
            futures = {
                executor.submit(_execute_sub_llm, self.model, p, self.budget_manager, self.rlm_logger, self.max_prompt_chars): i
                for i, p in enumerate(prompts)
            }
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


class RLMQueryTool(Tool):
    """Delegate a sub-task to a child RLM agent that has its own Python REPL.

    At each depth level, the child gets llm_query + llm_query_batched tools.
    If not at max depth, it also gets its own rlm_query for further recursion.

    Depth semantics (with max_depth=2):
        depth=0  →  spawns child CodeAgent, child gets rlm_query(depth=1)
        depth=1  →  spawns child CodeAgent, child has NO rlm_query (leaf agent)
        depth=2  →  leaf: falls back to flat _execute_sub_llm (no REPL)

    The BudgetManager is shared across all depths — one global cap on LLM calls.

    Note on context vs max_prompt_chars:
        The ``context`` argument is injected into the child's state as a Python
        variable — it does NOT go into an LLM prompt directly. max_prompt_chars
        guards llm_query prompts, not rlm_query context. The child is expected
        to chunk large context via code before calling llm_query.

    Note on concurrency:
        Each rlm_query call blocks until the child finishes (synchronous).
        A child's llm_query_batched uses up to max_workers threads. Thread
        multiplication (N children × M workers) is NOT possible because the
        LocalPythonExecutor's AST interpreter blocks ``concurrent.futures``
        imports — generated code cannot parallelize rlm_query calls. The
        maximum concurrent threads at any point is max_workers (from a single
        llm_query_batched call), and the shared BudgetManager caps total calls.
    """

    name = "rlm_query"
    description = (
        "Delegate a sub-task to a child RLM agent with its own Python REPL. "
        "The child can peek, grep, and call llm_query on its input. "
        "Use for complex sub-tasks that need code execution, not just a single LLM call. "
        "Pass the data as part of the context string — the child sees it as a variable."
    )
    inputs = {
        "task": {"type": "string", "description": "The sub-task instruction."},
        "context": {"type": "string", "description": "The data for the child to process."},
    }
    output_type = "string"

    def __init__(
        self,
        model: Model,
        sub_model: Model,
        depth: int,
        max_depth: int,
        budget_manager: BudgetManager | None,
        rlm_logger,
        max_prompt_chars: int | None,
        max_child_steps: int,
        max_workers: int,
        additional_authorized_imports: list[str] | None = None,
        **kwargs,
    ):
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        super().__init__(**kwargs)
        self.model = model          # orchestrator model (writes code)
        self.sub_model = sub_model  # sub-LLM model (flat calls)
        self.depth = depth
        self.max_depth = max_depth
        self.budget_manager = budget_manager
        self.rlm_logger = rlm_logger
        self.max_prompt_chars = max_prompt_chars
        self.max_child_steps = max_child_steps
        self.max_workers = max_workers
        self.additional_authorized_imports = additional_authorized_imports or []

    def forward(self, task: str, context: str) -> str:
        # Leaf depth: no REPL, just a flat LLM call with task + context as prompt
        if self.depth >= self.max_depth:
            return _execute_sub_llm(
                self.sub_model, f"{task}\n\n{context}",
                self.budget_manager, self.rlm_logger, self.max_prompt_chars,
            )

        # --- Lazy imports: only loaded when we actually spawn a child agent ---
        # Avoids circular dep (rlm_tools → agents → rlm) and keeps module-level
        # imports clean — these are only needed for the recursive path.
        from io import StringIO

        from rich.console import Console

        from smolagents.agents import CodeAgent
        from smolagents.monitoring import AgentLogger, LogLevel
        from smolagents.rlm import _build_rlm_instructions

        child_tools = self._build_child_tools()
        child_has_rlm = self.depth + 1 < self.max_depth
        instructions = _build_rlm_instructions(self.max_prompt_chars, recursive=child_has_rlm)

        # Suppress all rich.Live / console output from the child to avoid
        # interleaving with the parent's output.
        silent_logger = AgentLogger(level=LogLevel.OFF, console=Console(file=StringIO()))

        # If rlm_logger exists, register step callbacks so child orchestrator
        # steps appear in the JSONL log. Without this, only sub-LLM calls from
        # the child's tools are logged — the child's code/observations are invisible.
        child_callbacks = None
        if self.rlm_logger:
            from smolagents.memory import ActionStep, FinalAnswerStep
            from smolagents.rlm_logging import _format_usage, _truncate, _ts

            child_depth = self.depth + 1
            logger = self.rlm_logger

            def _log_child_step(memory_step, agent=None):
                logger.emit(
                    "execution_result",
                    depth=child_depth,
                    step=memory_step.step_number,
                    code=_truncate(memory_step.code_action),
                    output=_truncate(memory_step.observations),
                    has_error=memory_step.error is not None,
                    usage=_format_usage(memory_step.token_usage),
                    timestamps={
                        "llm_call_start": _ts(memory_step.timing.start_time),
                        "execution_end": _ts(memory_step.timing.end_time),
                    },
                )

            def _log_child_final(memory_step, agent=None):
                logger.emit("final_result", depth=child_depth,
                            result=_truncate(str(memory_step.output)))

            child_callbacks = {
                ActionStep: _log_child_step,
                FinalAnswerStep: _log_child_final,
            }

        child = CodeAgent(
            tools=child_tools,
            model=self.model,
            instructions=instructions,
            max_steps=self.max_child_steps,
            additional_authorized_imports=self.additional_authorized_imports,
            step_callbacks=child_callbacks,
            verbosity_level=LogLevel.OFF,
            logger=silent_logger,
        )
        try:
            # Inject context into the child's state dict before run().
            # run() calls send_variables(self.state) which copies it into the
            # executor's namespace — so `context` is available as a Python variable.
            child.state["context"] = context
            result = child.run(task=task)
            return str(result) if result is not None else ""
        finally:
            # CodeAgent.cleanup() releases executor resources (local executor is
            # lightweight, but remote executors hold Docker/e2b handles).
            child.cleanup()

    def _build_child_tools(self):
        """Build the tool set for a child agent at depth+1.

        Every child gets llm_query + llm_query_batched.
        Non-leaf children also get rlm_query for further recursion.
        """
        tools = [
            LLMQueryTool(
                model=self.sub_model,
                budget_manager=self.budget_manager,
                rlm_logger=self.rlm_logger,
                max_prompt_chars=self.max_prompt_chars,
            ),
            LLMQueryBatchedTool(
                model=self.sub_model,
                max_workers=self.max_workers,
                budget_manager=self.budget_manager,
                rlm_logger=self.rlm_logger,
                max_prompt_chars=self.max_prompt_chars,
            ),
        ]

        next_depth = self.depth + 1
        if next_depth < self.max_depth:
            tools.append(RLMQueryTool(
                model=self.model,
                sub_model=self.sub_model,
                depth=next_depth,
                max_depth=self.max_depth,
                budget_manager=self.budget_manager,
                rlm_logger=self.rlm_logger,
                max_prompt_chars=self.max_prompt_chars,
                max_child_steps=self.max_child_steps,
                max_workers=self.max_workers,
                additional_authorized_imports=self.additional_authorized_imports,
            ))

        return tools
