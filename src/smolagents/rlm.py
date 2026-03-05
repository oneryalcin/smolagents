"""
RLM (Recursive Language Model) agent for handling arbitrarily large contexts.

Instead of stuffing everything into the prompt, the agent gets a Python REPL
and sub-LLM tools. It writes code to chunk, grep, and map-reduce over data,
delegating semantic analysis to sub-LLM calls.

Usage:
    from smolagents import LiteLLMModel
    from smolagents.rlm import RLMAgent

    agent = RLMAgent(
        model=LiteLLMModel(model_id="gpt-4.1-mini"),
        sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
    )
    result = agent.run(
        task="How many entries mention 'Python'?",
        context=large_text,
    )
"""

from pathlib import Path
from typing import Any

from smolagents.agents import CodeAgent
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import Model
from smolagents.monitoring import LogLevel
from smolagents.rlm_tools import Budget, BudgetManager, LLMQueryBatchedTool, LLMQueryTool


# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------

def make_variable_info(name: str, value: Any, preview_chars: int = 1000) -> str:
    """Create a metadata summary of a variable: type, size, preview.

    Injected into the task prompt so the LLM knows what it's working with
    without seeing the full data.
    """
    if isinstance(value, str):
        lines = value.split("\n")
        preview = value[:preview_chars]
        truncated = len(value) > preview_chars
        return (
            f"Variable: `{name}`\n"
            f"Type: str | Length: {len(value):,} chars | Lines: {len(lines):,}\n"
            f"Preview (first {preview_chars} chars):\n```\n{preview}{'...' if truncated else ''}\n```"
        )

    if isinstance(value, (list, tuple)):
        preview = str(value[:10]) if len(value) > 10 else str(value)
        return (
            f"Variable: `{name}`\n"
            f"Type: {type(value).__name__} | Items: {len(value):,}\n"
            f"Preview (first 10): {preview}"
        )

    if isinstance(value, dict):
        keys = list(value.keys())[:10]
        return (
            f"Variable: `{name}`\n"
            f"Type: dict | Keys: {len(value):,}\n"
            f"Preview keys: {keys}"
        )

    str_val = str(value)[:preview_chars]
    return f"Variable: `{name}`\nType: {type(value).__name__}\nValue: {str_val}"


# ---------------------------------------------------------------------------
# RLM instructions injected into system prompt
# ---------------------------------------------------------------------------

RLM_INSTRUCTIONS = """
## RLM: How to Handle Large Context

You have `context` in your Python environment and two sub-LLM tools:
- `llm_query(prompt)` — single sub-LLM call for semantic analysis
- `llm_query_batched(prompts)` — parallel sub-LLM calls (list in, list out)

### Rules
1. **PEEK FIRST.** Always inspect before processing:
   ```python
   print(f"Length: {len(context):,} chars, Lines: {len(context.splitlines()):,}")
   print(context[:2000])
   ```

2. **GREP before LLM.** String/regex matching is free and instant:
   ```python
   import re
   matches = [l for l in context.splitlines() if 'keyword' in l.lower()]
   ```

3. **BATCH for semantic tasks.** Chunk the data, send chunks in parallel:
   ```python
   lines = context.splitlines()
   chunk_size = max(1, len(lines) // 10)
   chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
   prompts = [f"Classify these entries:\\n{chr(10).join(c)}" for c in chunks]
   results = llm_query_batched(prompts)
   ```

4. **VERIFY before final_answer.** Print and sanity-check:
   ```python
   print(f"Found {count} matches")
   print(sample[:5])
   final_answer(count)
   ```

### Don't waste LLM calls on:
- Counting → `len()`, `sum()`
- Pattern matching → `in`, `re.search()`
- Filtering → list comprehensions
- Sorting/grouping → `sorted()`, `itertools.groupby()`
"""


# ---------------------------------------------------------------------------
# RLMAgent
# ---------------------------------------------------------------------------

_SUPER_RUN_PARAMS = {"reset", "max_steps", "stream", "images", "return_full_result"}


class RLMAgent(CodeAgent):
    """CodeAgent extended with sub-LLM tools for large-context processing.

    Args:
        model: Main orchestrator model (writes Python code).
        sub_model: Model used for llm_query / llm_query_batched. Defaults to `model`.
        budget: Optional budget limits for sub-LLM calls (call count, tokens).
        max_output_length: Truncation limit for code output. Lower values force the
            LLM to write smarter code instead of reading raw output. Default 3000.
        max_workers: Max parallel threads for llm_query_batched. Default 8.
        tools: Additional tools beyond the RLM defaults.
        **kwargs: Passed to CodeAgent (max_steps, planning_interval, etc.)
    """

    def __init__(
        self,
        model: Model,
        sub_model: Model | None = None,
        budget: Budget | None = None,
        log_path: str | Path | None = None,
        max_output_length: int = 3000,
        max_workers: int = 8,
        tools: list | None = None,
        **kwargs,
    ):
        sub_model = sub_model or model
        self.budget_manager = BudgetManager(budget) if budget else None

        # Lazy import — zero cost when logging disabled
        self.rlm_logger = None
        if log_path:
            from smolagents.rlm_logging import RLMLogger
            self.rlm_logger = RLMLogger(log_path)

        rlm_tools = [
            LLMQueryTool(model=sub_model, budget_manager=self.budget_manager, rlm_logger=self.rlm_logger),
            LLMQueryBatchedTool(
                model=sub_model, max_workers=max_workers,
                budget_manager=self.budget_manager, rlm_logger=self.rlm_logger,
            ),
        ]
        all_tools = rlm_tools + (tools or [])

        # Inject RLM instructions into the system prompt via additional instructions
        base_instructions = kwargs.pop("instructions", "") or ""
        kwargs["instructions"] = base_instructions + RLM_INSTRUCTIONS

        super().__init__(
            tools=all_tools,
            model=model,
            max_output_length=max_output_length,
            **kwargs,
        )

        # Per-instance set of state keys to clear between runs
        self._rlm_state_keys: set[str] = set()

        # Register budget callback to inject summary into observations
        if self.budget_manager:
            self.step_callbacks.register(ActionStep, self._budget_callback)

        # Register logging callbacks
        if self.rlm_logger:
            self.step_callbacks.register(ActionStep, self._log_action_step)
            self.step_callbacks.register(FinalAnswerStep, self._log_final_answer)

    def _budget_callback(self, memory_step, agent=None):
        """Append budget summary to step observations so the LLM sees remaining budget."""
        summary = self.budget_manager.summary
        if memory_step.observations:
            memory_step.observations += f"\n[Budget] {summary}"
        else:
            memory_step.observations = f"[Budget] {summary}"

    def _log_action_step(self, memory_step, agent=None):
        """Emit execution_result JSONL event for each orchestrator step."""
        from smolagents.rlm_logging import _format_usage, _ts

        self.rlm_logger.emit(
            "execution_result",
            step=memory_step.step_number,
            code=memory_step.code_action,
            output=memory_step.observations,
            hasError=memory_step.error is not None,
            usage=_format_usage(memory_step.token_usage),
            timestamps={
                "llm_call_start": _ts(memory_step.timing.start_time),
                "execution_end": _ts(memory_step.timing.end_time),
            },
        )

    def _log_final_answer(self, memory_step, agent=None):
        """Emit final_result JSONL event."""
        self.rlm_logger.emit("final_result", result=str(memory_step.output))

    def run(self, task: str, context=None, show_metadata: bool = True, **kwargs):
        """Run the RLM agent on a task with optional large context.

        Args:
            task: The question or instruction.
            context: Large data to process (str, list, dict, etc.).
                Stored in agent state, not in the prompt.
            show_metadata: Log variable metadata.
            **kwargs: Passed to CodeAgent.run() (reset, max_steps, stream, images, etc.)
                Strings >1000 chars are auto-routed to agent state instead.
        """
        # Reset budget counters for this run
        if self.budget_manager:
            self.budget_manager.reset()

        # Clear state keys from previous run to prevent cross-run leakage
        for key in self._rlm_state_keys:
            self.state.pop(key, None)
        self._rlm_state_keys.clear()

        additional_args = {}
        variables_info: list[str] = []
        super_kwargs = {}

        # Route context into state with metadata preview
        if context is not None:
            self.state["context"] = context
            self._rlm_state_keys.add("context")
            info = make_variable_info("context", context)
            variables_info.append(info)
            if show_metadata:
                self.logger.log(f"\n{'='*60}\n{info}\n{'='*60}", level=LogLevel.INFO)

        # Separate super().run() kwargs from data kwargs
        for k, v in kwargs.items():
            if k in _SUPER_RUN_PARAMS:
                super_kwargs[k] = v
            elif isinstance(v, str) and len(v) > 1000:
                self.state[k] = v
                self._rlm_state_keys.add(k)
                info = make_variable_info(k, v, preview_chars=200)
                variables_info.append(info)
            else:
                additional_args[k] = v

        if variables_info:
            additional_args["variables_info"] = "\n\n".join(variables_info)

        if self.rlm_logger:
            self.rlm_logger.emit("agent_start", task=task[:500])
        try:
            result = super().run(task=task, additional_args=additional_args, **super_kwargs)
        finally:
            if self.rlm_logger:
                self.rlm_logger.emit("agent_end")
        return result
