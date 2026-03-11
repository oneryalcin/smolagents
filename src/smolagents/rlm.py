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
from smolagents.memory import ActionStep, ChatMessage, FinalAnswerStep, MessageRole
from smolagents.models import Model
from smolagents.monitoring import LogLevel
from smolagents.rlm_logging import RLMLogger, _format_usage, _truncate, _ts
from smolagents.rlm_tools import Budget, BudgetManager, LLMQueryBatchedTool, LLMQueryTool, RLMQueryTool


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
        est_tokens = len(value) // 4
        preview = value[:preview_chars]
        truncated = len(value) > preview_chars
        return (
            f"Variable: `{name}`\n"
            f"Type: str | Length: {len(value):,} chars (~{est_tokens:,} tokens) | Lines: {len(lines):,}\n"
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

_REASONING_TRACE_INSTRUCTIONS = """
### Reasoning Trace (verbose_reasoning mode)
Before each code block, structure your Thought section with these tags:

**[APPROACH]** — What is your plan for this step? What sub-question are you tackling?
**[TOOL_CHOICE]** — Which tool(s) will you use and WHY?
  - State what triggered your choice: a system prompt rule, a tool description, data you observed, or output from a prior step.
  - If you considered alternatives, say why you rejected them.
  - Examples:
    "Using llm_query (not rlm_query) — system prompt says 'use llm_query for simple single-shot classification'."
    "Using rlm_query — data has 96 docs, too complex for one LLM call. System prompt says 'use rlm_query for sub-investigations'."
    "Python only — system prompt rule 'Don't waste LLM calls on counting'."
**[EVIDENCE]** — What evidence from previous steps or data inspection supports this choice?

Keep each tag to 1-2 sentences. This is for debugging, not an essay.
"""


def _build_rlm_instructions(sub_model_max_chars: int | None, recursive: bool = False, verbose_reasoning: bool = False) -> str:
    """Build RLM instructions with optional sub-model limit awareness."""
    limit_section = ""
    if sub_model_max_chars:
        est_tokens = sub_model_max_chars // 4
        limit_section = f"""
### Sub-LLM Limits
- Each `llm_query` / `llm_query_batched` prompt accepts at most **{sub_model_max_chars:,} chars (~{est_tokens:,} tokens)**
- Exceeding this limit raises an error — you MUST filter or chunk first
- NEVER pass raw `context` to llm_query when context is larger than this limit
"""

    recursive_section = ""
    if recursive:
        recursive_section = """
### Recursive Sub-Tasks (`rlm_query`)
- `rlm_query(task, context)` — delegate a sub-task to a child agent with its own Python REPL
- The child can peek, grep, call llm_query — just like you
- **Use `rlm_query` when a sub-task needs multiple steps**: filtering, then LLM analysis, then aggregation
- **Use `rlm_query` for sub-investigations**: "find all docs about X and extract Y from them"
- **Always filter context before passing to rlm_query** — do NOT pass the full raw context. Use Python to extract the relevant subset first.
- Use `llm_query` (not `rlm_query`) for simple single-shot classification — it's cheaper
- Example — partition and delegate:
  ```python
  # Split docs into chunks, delegate analysis of each chunk to a child agent
  docs = context.split("=== Document")
  chunk_size = len(docs) // 3
  chunks = [docs[i:i+chunk_size] for i in range(0, len(docs), chunk_size)]
  findings = []
  for chunk in chunks:
      result = rlm_query("Find documents mentioning the target criteria and extract key facts", "=== Document".join(chunk))
      findings.append(result)
  ```
"""

    return f"""
## RLM: How to Handle Large Context

You have `context` in your Python environment containing data that has NO pre-computed labels.
You MUST explore it and use your sub-LLM tools to classify/analyze items when needed.

**Tools:**
- `llm_query(prompt)` — single sub-LLM call for semantic analysis
- `llm_query_batched(prompts)` — parallel sub-LLM calls (list in, list out); much faster than a loop
{limit_section}{recursive_section}
### Critical: Explore Before Answering
Your first step MUST be to inspect the context structure and plan your approach.
Do NOT jump to a final answer without first understanding the data format and size.
The data typically has no explicit labels — YOU must classify items using the sub-LLM tools.

### Strategy
1. **PEEK** — inspect structure, size, and format:
   ```python
   print(f"Length: {{len(context):,}} chars, Lines: {{len(context.splitlines()):,}}")
   print(context[:2000])
   ```

2. **DECOMPOSE** — break the task into sub-questions. If the task has multiple constraints or requires combining evidence from different parts of the data, identify each sub-question explicitly before writing code. **If one sub-question hits a dead end, move on to the others** — you can often solve the problem by approaching from a different angle. Never give up after exhausting only one line of inquiry.

3. **USE PYTHON FIRST** — string ops, regex, counting are free and instant:
   ```python
   import re
   from collections import Counter
   matches = [l for l in context.splitlines() if 'keyword' in l.lower()]
   dates = re.findall(r'Date: (\\w+ \\d+, \\d{{4}})', context)
   ```

4. **DELEGATE or CLASSIFY** — choose the right tool for each sub-question:
   - `llm_query` / `llm_query_batched` — single-shot semantic analysis (classification, extraction, summarization)
   - `rlm_query` — multi-step sub-investigations where the child needs to explore, filter, and reason over a chunk of data
   ```python
   # Single-shot: classify a document
   label = llm_query(f"Is this document about topic X? Answer yes/no.\\n{{doc_text}}")
   # Multi-step: investigate a sub-question across many documents
   result = rlm_query("Find which country had 250K-300K assaults in 2003", relevant_docs)
   ```

5. **CROSS-REFERENCE** — combine findings from different sub-questions. The answer often requires chaining: result A narrows the search space for question B. If you can't confirm every constraint, commit to the best candidate that satisfies the most constraints rather than giving up.

6. **PARSE CAREFULLY** — when counting labels, avoid substring collisions:
   ```python
   # WRONG: 'positive' in 'not positive' → True
   # RIGHT: check exact word or exclude negations
   for line in resp.strip().splitlines():
       w = line.strip().lower()
       if 'incorrect' in w:
           labels.append('incorrect')
       elif 'correct' in w:
           labels.append('correct')
   ```

7. **VERIFY** — print results and sanity-check before final_answer.

### Batching Rules
- **NEVER** send 1 item per LLM call when you have >10 items
- 1000 items = ~25 prompts of 40, NOT 1000 prompts of 1
- The sub-LLM is capable — don't under-use it. Feed substantial context per call.

### Don't waste LLM calls on:
- Counting → `len()`, `sum()`, `Counter()`
- Pattern matching → `in`, `re.search()`
- Filtering → list comprehensions
- Sorting/grouping → `sorted()`, `itertools.groupby()`

{"" if not verbose_reasoning else _REASONING_TRACE_INSTRUCTIONS}**IMPORTANT: Every response MUST contain a `<code>` block. Never respond with only text.**
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
        sub_model_max_chars: Max chars per sub-LLM prompt. Forces chunking for
            large inputs. Default 64000 (~16K tokens). Set None to disable.
        max_output_length: Truncation limit for code output. Default 3000.
        max_workers: Max parallel threads for llm_query_batched. Default 8.
        prompt_cache: Inject cache_control on system prompt for Anthropic models.
            Reduces input token cost by ~90% on multi-step runs. Default True.
        tools: Additional tools beyond the RLM defaults.
        recursive: Enable rlm_query tool for spawning child RLM agents with
            their own REPL. Each child can peek, grep, and call llm_query.
        max_depth: Maximum recursion depth for rlm_query. With max_depth=2
            (default), the root can spawn a child, and the child can spawn
            a leaf (flat LLM only). Must be >= 1.
        max_child_steps: Maximum CodeAgent steps for each child agent.
        verbose_reasoning: When True, injects instructions asking the LLM to
            emit structured [APPROACH], [TOOL_CHOICE], [EVIDENCE] tags in its
            Thought section. The full reasoning text is logged to JSONL for
            prompt engineering and debugging. Default False.
        **kwargs: Passed to CodeAgent (max_steps, planning_interval, etc.)
    """

    def __init__(
        self,
        model: Model,
        sub_model: Model | None = None,
        budget: Budget | None = None,
        log_path: str | Path | None = None,
        sub_model_max_chars: int | None = 64_000,
        max_output_length: int = 3000,
        max_workers: int = 8,
        prompt_cache: bool = True,
        tools: list | None = None,
        recursive: bool = False,
        max_depth: int = 2,
        max_child_steps: int = 10,
        verbose_reasoning: bool = False,
        **kwargs,
    ):
        sub_model = sub_model or model
        self.prompt_cache = prompt_cache
        self.budget_manager = BudgetManager(budget) if budget else None

        self.rlm_logger = RLMLogger(log_path) if log_path else None

        rlm_tools = [
            LLMQueryTool(
                model=sub_model, budget_manager=self.budget_manager,
                rlm_logger=self.rlm_logger, max_prompt_chars=sub_model_max_chars,
            ),
            LLMQueryBatchedTool(
                model=sub_model, max_workers=max_workers,
                budget_manager=self.budget_manager, rlm_logger=self.rlm_logger,
                max_prompt_chars=sub_model_max_chars,
            ),
        ]

        if recursive:
            # Capture parent's authorized imports so children inherit them.
            # The value also flows to CodeAgent via **kwargs — we just read it here.
            parent_imports = kwargs.get("additional_authorized_imports")
            rlm_tools.append(RLMQueryTool(
                model=model,
                sub_model=sub_model,
                depth=0,
                max_depth=max_depth,
                budget_manager=self.budget_manager,
                rlm_logger=self.rlm_logger,
                max_prompt_chars=sub_model_max_chars,
                max_child_steps=max_child_steps,
                max_workers=max_workers,
                additional_authorized_imports=parent_imports,
            ))

        all_tools = rlm_tools + (tools or [])

        # Inject RLM instructions into the system prompt via additional instructions
        base_instructions = kwargs.pop("instructions", "") or ""
        self.verbose_reasoning = verbose_reasoning
        kwargs["instructions"] = base_instructions + _build_rlm_instructions(
            sub_model_max_chars, recursive=recursive, verbose_reasoning=verbose_reasoning,
        )

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

    def write_memory_to_messages(self, summary_mode=False):
        """Override to inject cache_control on system prompt for Anthropic prompt caching.

        Adds {"cache_control": {"type": "ephemeral"}} to the system message content block.
        This tells Anthropic to cache the static prefix (system prompt + tool definitions),
        reducing input token cost by ~90% on subsequent steps. Non-Anthropic providers
        ignore the extra key. See: https://github.com/huggingface/smolagents/issues/2054
        """
        messages = super().write_memory_to_messages(summary_mode=summary_mode)
        if self.prompt_cache and messages and messages[0].role == MessageRole.SYSTEM:
            content = messages[0].content
            if isinstance(content, list) and content:
                # Add cache_control to the last content block in system message
                last_block = content[-1]
                if isinstance(last_block, dict) and "cache_control" not in last_block:
                    last_block["cache_control"] = {"type": "ephemeral"}
        return messages

    def _budget_callback(self, memory_step, agent=None):
        """Append budget summary to step observations so the LLM sees remaining budget."""
        summary = self.budget_manager.summary
        if memory_step.observations:
            memory_step.observations += f"\n[Budget] {summary}"
        else:
            memory_step.observations = f"[Budget] {summary}"

    def _log_action_step(self, memory_step, agent=None):
        """Emit execution_result JSONL event for each orchestrator step."""
        extra = {}
        if self.verbose_reasoning and memory_step.model_output:
            # Log the full Thought text (before <code>) for prompt engineering debug.
            # model_output contains "Thought: ... <code>...</code>" — we want the part before <code>.
            raw = memory_step.model_output if isinstance(memory_step.model_output, str) else str(memory_step.model_output)
            code_start = raw.find("<code>")
            reasoning = raw[:code_start].strip() if code_start != -1 else raw.strip()
            extra["reasoning"] = _truncate(reasoning, max_len=4000)
        self.rlm_logger.emit(
            "execution_result",
            step=memory_step.step_number,
            code=_truncate(memory_step.code_action),
            output=_truncate(memory_step.observations),
            has_error=memory_step.error is not None,
            usage=_format_usage(memory_step.token_usage),
            timestamps={
                "llm_call_start": _ts(memory_step.timing.start_time),
                "execution_end": _ts(memory_step.timing.end_time),
            },
            **extra,
        )

    def _log_final_answer(self, memory_step, agent=None):
        """Emit final_result JSONL event."""
        self.rlm_logger.emit("final_result", result=_truncate(str(memory_step.output)))

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
        success = False
        try:
            result = super().run(task=task, additional_args=additional_args, **super_kwargs)
            success = True
        finally:
            if self.rlm_logger:
                self.rlm_logger.emit("agent_end", success=success)
        return result

    def close(self):
        """Close the JSONL logger. Safe to call multiple times."""
        if self.rlm_logger:
            self.rlm_logger.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
