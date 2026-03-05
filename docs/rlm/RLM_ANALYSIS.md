# RLM Support for smolagents — Engineering Analysis

> **Status:** Pre-implementation analysis
> **Date:** 2026-03-04
> **Fork:** [oneryalcin/smolagents](https://github.com/oneryalcin/smolagents)
> **Upstream:** [huggingface/smolagents](https://github.com/huggingface/smolagents) @ `v1.25.0.dev0` (commit `5c684c1`)
> **Prior Art:** [avbiswas/fast-rlm](https://github.com/avbiswas/fast-rlm), [rlm_v2.py](https://gist.github.com/oneryalcin/70464f35727a24ab8eb23fdb9ff471ad)
> **Paper:** [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)

---

## Table of Contents

1. [What is RLM and Why Does smolagents Need It](#1-what-is-rlm-and-why-does-smolagents-need-it)
2. [Current State of smolagents Internals](#2-current-state-of-smolagents-internals)
3. [Prior Art: fast-rlm and rlm_v2](#3-prior-art-fast-rlm-and-rlm_v2)
4. [Gap Analysis: What's Missing](#4-gap-analysis-whats-missing)
5. [Design Decisions](#5-design-decisions)
6. [Implementation Plan](#6-implementation-plan)
7. [File-Level Change Map](#7-file-level-change-map)
8. [Open Questions](#8-open-questions)
9. [References](#9-references)

---

## 1. What is RLM and Why Does smolagents Need It

### The Problem

LLMs have finite context windows (128K–2M tokens). Real-world tasks often involve data far exceeding this — millions of log entries, entire codebases, multi-document corpora. Naive approaches (stuff everything in the prompt, or truncate) either fail or lose critical information.

### The RLM Solution

A **Recursive Language Model (RLM)** gives an LLM a code execution environment (REPL) and the ability to spawn sub-LLM calls on slices of the data. Instead of reading all the data, the agent writes Python code to:

1. **Peek** — inspect structure, length, format
2. **Grep** — use regex/string matching (free, fast)
3. **Partition + Map** — chunk data, send each chunk to a sub-LLM in parallel
4. **Reduce** — aggregate sub-LLM responses with code
5. **Recurse** — sub-LLMs can themselves chunk and delegate deeper

The key insight: **the sub-LLM responses are returned as Python variables**, not injected into the parent's prompt. This prevents context window explosion.

### Why smolagents

smolagents already has a `CodeAgent` (LLM writes Python, executed in a REPL with persistent state) and a `Tool` system (callable functions available in generated code). This is 80% of RLM infrastructure. What's missing is the remaining 20%: sub-LLM tools, budget controls, aggressive truncation, and structured logging.

---

## 2. Current State of smolagents Internals

### 2.1 CodeAgent Execution Loop

**File:** `src/smolagents/agents.py`

The core loop lives in `_run_stream()` (line 540):

```
while step_number <= max_steps and not final_answer:
    [optional planning step]        # Lines 550-567
    action_step = ActionStep(...)   # Line 571
    for output in _step_stream():   # Line 578
        yield output
```

Each `_step_stream()` call (line 1639) does:

1. **Build messages** — `write_memory_to_messages()` reconstructs full conversation from `AgentMemory` (line 1647)
2. **Call LLM** — `self.model.generate(input_messages, stop_sequences=...)` (line 1678)
3. **Parse code** — `parse_code_blobs(output_text, code_block_tags)` extracts Python from markdown (line 1709)
4. **Execute** — `self.python_executor(code_action)` returns `CodeOutput(output, logs, is_final_answer)` (line 1727)
5. **Truncate & observe** — `truncate_content(str(code_output.output))` capped at 20K chars, appended to `memory_step.observations` (line 1753)

**Token usage** is captured from the model response and stored in `memory_step.token_usage` (line 1698). The `Monitor` class aggregates this across steps (line 100-117 in `monitoring.py`), but only for the *direct* agent — not for sub-LLM tool calls.

### 2.2 Executor System

**File:** `src/smolagents/local_python_executor.py`

#### LocalPythonExecutor (line 1688)

- **State persistence:** `self.state = {"__name__": "__main__"}` — a dict that survives across `__call__` invocations. Variables assigned in step N are available in step N+1.
- **Variable injection:** `send_variables(variables)` updates `self.state` (line 1760). Called from `agent.run()` at line 491.
- **Tool injection:** `send_tools(tools)` merges agent tools + `BASE_PYTHON_TOOLS` (print, len, range, etc.) into `self.static_tools` (line 1763).
- **Code execution:** Uses a custom AST evaluator (`evaluate_python_code`, line 1583), NOT `exec()`. Walks the AST node-by-node, enforcing import restrictions and operation limits.
- **Print capture:** `print()` is overridden to append to a `PrintContainer` object in `state["_print_outputs"]` (line 903).

#### Key Constants

| Constant | Value | Location | What It Controls |
|---|---|---|---|
| `DEFAULT_MAX_LEN_OUTPUT` | 50,000 | `local_python_executor.py:57` | Max print output length in executor |
| `MAX_OPERATIONS` | 10,000,000 | `local_python_executor.py:58` | Infinite loop protection |
| `MAX_WHILE_ITERATIONS` | 1,000,000 | `local_python_executor.py:59` | While loop limit |
| `MAX_EXECUTION_TIME_SECONDS` | 30 | `local_python_executor.py:60` | Execution timeout |
| `MAX_LENGTH_TRUNCATE_CONTENT` | 20,000 | `utils.py:254` | Output truncation in `_step_stream()` |

#### Two-Level Truncation

1. **Print output** — truncated inside `evaluate_python_code()` to `max_print_outputs_length` (default 50K). This is configurable via `CodeAgent(max_print_outputs_length=N)`.
2. **Step observation** — truncated in `_step_stream()` at line 1753 to `MAX_LENGTH_TRUNCATE_CONTENT` (20K). This is a **module-level constant, NOT configurable per-agent**. This is a problem.

#### Remote Executors

**File:** `src/smolagents/remote_executors.py`

| Executor | Sandboxed | State Persistent | Managed Agents |
|---|---|---|---|
| `local` | No (AST-eval only) | Yes (in-memory dict) | Yes |
| `docker` | Yes (container) | Yes (Jupyter kernel) | **No** (raises Exception, line 1608) |
| `e2b` | Yes (microVM) | Partial | **No** |
| `wasm` | Yes (Pyodide/Deno) | No | **No** |
| `modal` | Yes (Modal sandbox) | Yes (Jupyter kernel) | **No** |
| `blaxel` | Yes (VM) | Yes (VM memory) | **No** |

**Critical blocker:** Line 1608-1609 in `agents.py`:
```python
if self.managed_agents:
    raise Exception("Managed agents are not yet supported with remote code execution.")
```

This means RLM-style sub-agent tools can't be used with any sandboxed executor — unless we bypass `managed_agents` entirely and use regular `Tool` subclasses instead (which is what `rlm_v2.py` does).

### 2.3 Tool System

**File:** `src/smolagents/tools.py`

Tools are Python classes inheriting from `Tool`:

```python
class Tool(BaseTool):
    name: str                          # Function name in generated code
    description: str                   # LLM sees this
    inputs: dict[str, dict]            # Parameter schema
    output_type: str                   # Return type hint

    def forward(self, **kwargs) -> Any:
        ...  # User implements this
```

Tools are injected into the executor's namespace via `send_tools()` (line 1763 in `local_python_executor.py`). When the LLM generates `result = my_tool(arg)`, the AST evaluator calls `my_tool.__call__(arg)` which delegates to `forward()`.

**Key insight for RLM:** A `Tool` subclass can do anything in `forward()` — including calling `model.generate()`. This is exactly how `rlm_v2.py`'s `LLMQueryTool` works. No need to touch `managed_agents` at all.

### 2.4 managed_agents — Why They Don't Work for RLM

**File:** `src/smolagents/agents.py`, line 369

Managed agents are full `CodeAgent`/`ToolCallingAgent` instances registered as callable tools. When the parent's code calls `sub_agent(task="...")`, the sub-agent runs a complete `agent.run()` cycle.

**Problems:**

1. **Memory reset on each call** — Sub-agent calls `run(reset=True)` by default, losing all state between invocations ([#1695](https://github.com/huggingface/smolagents/issues/1695))
2. **Two-level nesting broken** — If agent A manages agent B which manages agent C, agent B never delegates to C in practice ([#1061](https://github.com/huggingface/smolagents/issues/1061))
3. **Shared state bugs** — Parallel managed agent calls share state incorrectly ([#1781](https://github.com/huggingface/smolagents/issues/1781))
4. **No remote executor support** — Hard exception at line 1608
5. **String-only returns** — Managed agents return strings, not Python objects

**Decision:** We bypass `managed_agents` entirely. RLM sub-LLM calls are implemented as `Tool` subclasses. This avoids all five problems above.

### 2.5 Step Callbacks

**File:** `src/smolagents/memory.py`, line 280 (`CallbackRegistry`)

Callbacks fire after each step type:

```python
# Registration
agent.step_callbacks.register(ActionStep, my_callback)
agent.step_callbacks.register(PlanningStep, my_callback)
agent.step_callbacks.register(FinalAnswerStep, my_callback)

# Callback signature
def my_callback(memory_step: ActionStep, agent: MultiStepAgent) -> None:
    step.step_number       # int
    step.timing            # Timing(start_time, end_time, duration)
    step.token_usage       # TokenUsage(input_tokens, output_tokens, total_tokens) or None
    step.code_action       # str — the Python code
    step.observations      # str — execution output (truncated)
    step.action_output     # Any — raw output before truncation
    step.error             # AgentError or None
    step.is_final_answer   # bool
```

**Extension point:** Callbacks have full access to per-step data. We can add:
- Cumulative token/cost tracking
- Budget enforcement (raise before next step)
- JSONL structured logging

### 2.6 Model Interface & Token Data

**File:** `src/smolagents/models.py`

All models return `ChatMessage` with optional `token_usage: TokenUsage`:

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int  # computed: input + output
```

**No cost data.** Unlike fast-rlm (which reads `usage.cost` from OpenRouter), smolagents' `TokenUsage` has no cost field. Cost estimation must be done externally using model ID + token counts.

### 2.7 Agent State

`agent.state` is a `dict[str, Any]` (line 331 in `agents.py`). Populated via:
- `additional_args` in `run()` (line 472)
- `python_executor.send_variables()` (line 491)

Variables in `state` are directly accessible in generated code. Large context goes here (not in the prompt text).

---

## 3. Prior Art: fast-rlm and rlm_v2

### 3.1 fast-rlm

**Repo:** [avbiswas/fast-rlm](https://github.com/avbiswas/fast-rlm)
**Architecture:** Python facade → Deno orchestrator → Pyodide (WASM) sandbox

| Feature | Implementation | Where |
|---|---|---|
| True recursion | Sub-agents get their own Pyodide REPL + can call `llm_query()` | `src/subagents.ts:87-99` |
| FINAL() mechanism | Global function captures Python objects, not just strings | `src/subagents.ts:104-113` |
| Output truncation | Hard 2000-char cap, shows last N chars | `src/subagents.ts:47-61` |
| Dollar budget | `max_money_spent` checked after each LLM call | `src/subagents.ts:176-178` |
| Token budget | `max_prompt_tokens`, `max_completion_tokens` | `src/subagents.ts:180-185` |
| Depth limit | `max_depth` (default 3) | `src/subagents.ts:88` |
| Parallelism | LLM writes `asyncio.gather()`, Pyodide bridges to JS Promises | `src/prompt.ts:93-114` |
| JSONL logging | Pino logger, structured per-step events with run_id/parent_run_id | `src/logging.ts` |
| TUI viewer | React + @opentui, timeline view, tree view | `tui_log_viewer/src/index.tsx` |
| Config | YAML file (`rlm_config.yaml`) | `src/subagents.ts:22-45` |

**Strengths:** True recursion, Pyodide sandbox, cost controls, excellent logging/TUI.

**Weaknesses:**
- Requires Deno + Bun (heavy dependency chain)
- Pyodide boot ~10-20s per recursion level
- Context injected via `JSON.stringify` into Python source (fragile for edge cases)
- Prompt has contradictions (LEAF_AGENT still references `llm_query`, `FINAL` instructions conflict)
- Python facade is a subprocess wrapper (no programmatic integration)

### 3.2 rlm_v2.py

**Gist:** [oneryalcin/70464f35727a24ab8eb23fdb9ff471ad](https://gist.github.com/oneryalcin/70464f35727a24ab8eb23fdb9ff471ad)
**Architecture:** Pure Python, extends `smolagents.CodeAgent`

| Feature | Implementation | Where |
|---|---|---|
| Sub-LLM calls | `LLMQueryTool(Tool)` — calls `model.generate()` | Line 79-109 |
| Parallel batch | `LLMQueryBatchedTool(Tool)` — `ThreadPoolExecutor` | Line 112-164 |
| Call counter | `LLMCallCounter` — thread-safe, shared across tools | Line 41-72 |
| Metadata preview | `make_variable_info()` — shows type/size/preview | Line 171-216 |
| Context routing | Large strings → `agent.state`, small → `additional_args` | Line 463-474 |
| Strategy prompt | `RLM_INSTRUCTIONS` — peek, grep, partition+map, verify | Line 302-359 |
| Step callback | Appends `[RLM]` info to `memory_step.observations` | Line 276-295 |

**Strengths:** Pure Python, composable, ThreadPoolExecutor parallelism, clean metadata injection, no extra runtime dependencies.

**Weaknesses:**
- No dollar/token budgeting (only call count)
- No structured logging
- Sub-LLM calls return strings only (no Python objects)
- Uses smolagents' default 20K truncation (too permissive)
- `thread_safe=False` by default — batched tool runs serial unless explicitly opted in

### 3.3 Comparative Summary

| Capability | fast-rlm | rlm_v2 | smolagents (native) |
|---|---|---|---|
| Sub-LLM as tool | Pyodide-bridged recursion | `Tool.forward()` calls `model.generate()` | `managed_agents` (broken for nesting) |
| Sub-LLM gets REPL | Yes | No | Yes (but broken at depth>1) |
| Parallelism | `asyncio.gather` in Pyodide | `ThreadPoolExecutor` | None |
| Output truncation | 2K chars (configurable) | 20K (smolagents default) | 20K (hardcoded constant) |
| Token budget | Yes (prompt + completion) | No | No |
| Dollar budget | Yes (via OpenRouter `usage.cost`) | No | No |
| Call limit | Per-subagent step limit | Shared counter across tools | `max_steps` only |
| Structured logging | JSONL with Pino | Callback string append | Rich console only |
| TUI | React + Bun | None | None |
| Sandbox | Pyodide (WASM) | Local exec (AST eval) | Local / Docker / E2B / WASM |
| Dependencies | Python + Deno + Bun | Python + smolagents | Python + smolagents |
| Object returns | Yes (via `FINAL()`) | Strings only | Strings only |

---

## 4. Gap Analysis: What's Missing

### 4.1 Must Have (P0)

| # | Gap | Why Critical | Effort |
|---|---|---|---|
| G1 | **Configurable output truncation per-agent** | 20K default is way too permissive for RLM. LLM gets lazy, reads entire context via print. fast-rlm uses 2K. Currently a module-level constant in `utils.py:254`. | Small — pass `max_length` through from CodeAgent to `truncate_content()` call at line 1753 |
| G2 | **Token/cost budget across all LLM calls** | `Monitor` only tracks direct agent steps. Sub-LLM tool calls (`llm_query`, `llm_query_batched`) are invisible. 50 calls × 100K context = $$$. | Medium — `BudgetManager` class, injected into sub-LLM tools, checked in step callback |
| G3 | **`llm_query` and `llm_query_batched` as first-class tools** | These are the core RLM primitives. Currently only exist in rlm_v2.py gist, not in smolagents. | Medium — port from rlm_v2, enhance with budget integration and object returns |
| G4 | **RLM system prompt injection** | The LLM must be taught *how* to think about large data: peek first, grep before LLM calls, chunk+batch for semantic tasks. Without this, agents waste calls. | Small — port `RLM_INSTRUCTIONS` from rlm_v2, refine with fast-rlm examples |

### 4.2 Should Have (P1)

| # | Gap | Why Important | Effort |
|---|---|---|---|
| G5 | **Structured JSONL logging** | Essential for debugging recursive agent trees. fast-rlm's TUI is unusable without their log format. Console output is unreadable for deep trees. | Medium — step callback that emits JSONL with `run_id`, `parent_run_id`, `depth`, `step`, `code`, `output`, `usage` |
| G6 | **Object returns from sub-LLM calls** | Currently `llm_query` returns strings. fast-rlm returns Python objects via `FINAL()`. String parsing is error-prone. | Medium — add optional `output_schema` / JSON-mode to sub-LLM calls, parse into Python objects |
| G7 | **Variable metadata injection** | `make_variable_info()` from rlm_v2 is excellent — shows type/size/preview without stuffing data in prompt. Should be built into RLMAgent. | Small — port directly |

### 4.3 Nice to Have (P2)

| # | Gap | Worth It? | Effort |
|---|---|---|---|
| G8 | fast-rlm TUI compatibility | Yes — reuse their viewer instead of building our own | Medium — match their JSONL schema exactly |
| G9 | True recursive sub-agents (sub-agent gets REPL) | Debatable — adds complexity, Pyodide boot overhead per level. rlm_v2's flat model is faster for map-reduce. | Large — would need to spawn child CodeAgent instances with isolated executors |
| G10 | Sandbox support for RLM tools | Would need to lift the `managed_agents` + remote executor restriction, or serialize tool calls across sandbox boundary | Large — deep changes to remote executor protocol |

---

## 5. Design Decisions

### D1: Bypass managed_agents, use Tool subclasses

**Rationale:** `managed_agents` is broken for nesting (#1061), resets state (#1695), has shared-state bugs (#1781), and doesn't work with any sandboxed executor (line 1608). Tool subclasses work everywhere, are simpler, and don't have these bugs.

**Tradeoff:** Sub-LLM calls won't have their own REPL (no true recursion). For map-reduce workloads, this doesn't matter. For deep multi-level reasoning, it's a limitation.

### D2: ThreadPoolExecutor for parallelism, not asyncio

**Rationale:** smolagents' executor is synchronous — no async support in any executor backend. fast-rlm gets async via Pyodide's JS bridge, but we don't have that. `ThreadPoolExecutor` is the pragmatic choice. It works, it's simple, and the LLM doesn't need to write threading code.

**Tradeoff:** Thread safety depends on the model client. Most HTTP-based clients (LiteLLM, OpenAI SDK) are thread-safe. We add a `thread_safe` flag (default `True` for LiteLLM) with fallback to serial.

### D3: Two-tier truncation — aggressive for RLM, normal for other agents

**Rationale:** 2K truncation (fast-rlm default) is too aggressive for general CodeAgent use. But for RLM, it's exactly right — forces the LLM to write code instead of reading output. We make this configurable per-agent, not globally.

**Implementation:** Add `max_output_length` parameter to `CodeAgent.__init__()`. Pass it to `truncate_content()` at line 1753. Default stays at 20K for backward compat. `RLMAgent` sets it to 2000-5000.

### D4: Budget tracks calls + tokens + estimated cost

**Rationale:** Call count alone (rlm_v2) is insufficient — 50 calls with 100K context each is expensive. Token count alone (fast-rlm) doesn't map to cost without model pricing. We track all three: calls, tokens, and estimated USD (using a simple pricing table per model).

### D5: JSONL logging compatible with fast-rlm TUI

**Rationale:** Building a TUI is a large effort. fast-rlm already has one. If we emit compatible JSONL, users can run `fast-rlm-log <file> --tui` to view our agent's execution tree. Zero UI work for us.

---

## 6. Implementation Plan

### Phase 1: Core RLM Agent (P0 items)

#### 1.1 Configurable output truncation

**Files to change:**
- `src/smolagents/agents.py` — Add `max_output_length` param to `CodeAgent.__init__()`, pass to `truncate_content()` at line 1753
- No changes to `utils.py` — `truncate_content()` already accepts `max_length` param

**Diff sketch:**
```python
# agents.py, CodeAgent.__init__
def __init__(self, ..., max_output_length: int | None = None, ...):
    self.max_output_length = max_output_length  # None = use default 20K

# agents.py, _step_stream, line 1753
truncated_output = truncate_content(
    str(code_output.output),
    max_length=self.max_output_length or MAX_LENGTH_TRUNCATE_CONTENT,
)
```

#### 1.2 BudgetManager

**New file:** `src/smolagents/budget.py`

```python
@dataclass
class Budget:
    max_llm_calls: int = 50
    max_total_tokens: int | None = None     # None = unlimited
    max_cost_usd: float | None = None       # None = unlimited

class BudgetManager:
    """Thread-safe budget tracker shared across agent + sub-LLM tools."""

    def __init__(self, budget: Budget):
        ...

    def record_call(self, input_tokens: int, output_tokens: int, model_id: str) -> None:
        """Record a sub-LLM call. Raises BudgetExceededError if over limit."""
        ...

    def record_step(self, token_usage: TokenUsage) -> None:
        """Record an agent step (called from step callback)."""
        ...

    @property
    def summary(self) -> dict:
        """Current usage summary: calls, tokens, estimated cost."""
        ...
```

**Integration:**
- Injected into `LLMQueryTool` and `LLMQueryBatchedTool` via constructor
- Step callback calls `budget.record_step()` after each `ActionStep`
- Sub-LLM tools call `budget.record_call()` after each `model.generate()`

#### 1.3 RLM Tools

**New file:** `src/smolagents/rlm_tools.py`

Two tools, ported from rlm_v2 with enhancements:

```python
class LLMQueryTool(Tool):
    """Single sub-LLM call for semantic analysis."""
    name = "llm_query"

    def forward(self, prompt: str) -> str:
        self.budget.record_call(...)
        response = self.model.generate([ChatMessage(role=USER, content=prompt)])
        return response.content

class LLMQueryBatchedTool(Tool):
    """Parallel sub-LLM calls via ThreadPoolExecutor."""
    name = "llm_query_batched"

    def forward(self, prompts: list[str]) -> list[str]:
        self.budget.record_call(n=len(prompts), ...)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            ...
```

#### 1.4 RLMAgent

**New file:** `src/smolagents/rlm.py`

```python
class RLMAgent(CodeAgent):
    """CodeAgent with RLM capabilities for arbitrarily large contexts."""

    def __init__(
        self,
        model,
        sub_model=None,              # Cheaper model for sub-LLM calls
        budget: Budget | None = None, # Token/cost/call limits
        max_output_length: int = 2000, # Aggressive truncation
        ...
    ):
        # Create budget manager
        # Create LLMQueryTool + LLMQueryBatchedTool with shared budget
        # Inject RLM_INSTRUCTIONS into system prompt
        # Register budget step callback
        super().__init__(tools=[...], max_output_length=max_output_length, ...)

    def run(self, task: str, context: str | Any = None, ...):
        # Route large data to self.state (not prompt)
        # Generate variable metadata
        # Reset budget
        return super().run(task=task, additional_args={...})
```

#### 1.5 RLM System Prompt

**In `src/smolagents/rlm.py` or `src/smolagents/rlm_prompts.py`:**

Port `RLM_INSTRUCTIONS` from rlm_v2 + best examples from fast-rlm's `prompt.ts`. Key sections:

1. **READ METADATA FIRST** — variable info tells you type/size/preview
2. **PEEK** — `print(context[:2000])` before anything else
3. **GREP** — string/regex matching is free and fast
4. **PARTITION + MAP** — chunk + `llm_query_batched()` for semantic tasks
5. **VERIFY** — print results before `final_answer()`
6. **DON'T WASTE** — counting, filtering, pattern matching = Python, not LLM

### Phase 2: Observability (P1 items)

#### 2.1 JSONL Logger Callback

**New file:** `src/smolagents/rlm_logging.py`

Step callback that emits one JSONL line per event:

```json
{"event_type": "execution_result", "run_id": "...", "parent_run_id": null, "depth": 0, "step": 3, "code": "...", "output": "...", "usage": {"prompt_tokens": 1234, "completion_tokens": 567}, "timestamps": {"llm_call_start": "...", "execution_end": "..."}}
```

Schema matches fast-rlm's `src/logging.ts` so their TUI works with our logs.

#### 2.2 Object Returns from Sub-LLM

Enhance `LLMQueryTool` with optional `response_format`:

```python
result = llm_query("Extract entities as JSON list", response_format={"type": "json_object"})
# result is a parsed Python dict/list, not a string
```

Uses the model's JSON mode if available, falls back to string.

#### 2.3 Variable Metadata

Port `make_variable_info()` from rlm_v2 into `src/smolagents/rlm.py`. Called automatically in `RLMAgent.run()` for any large variable.

### Phase 3: Polish (P2 items)

- fast-rlm TUI compatibility testing
- Benchmark against fast-rlm on OolongBench
- Documentation + examples

---

## 7. File-Level Change Map

### Modified Files

| File | Change | Lines Affected |
|---|---|---|
| `src/smolagents/agents.py` | Add `max_output_length` param to `CodeAgent.__init__()` and `_step_stream()` | ~1530 (init), 1753 |
| `src/smolagents/__init__.py` | Export new RLM classes | Top-level imports |

### New Files

| File | Purpose |
|---|---|
| `src/smolagents/budget.py` | `Budget`, `BudgetManager`, `BudgetExceededError` |
| `src/smolagents/rlm_tools.py` | `LLMQueryTool`, `LLMQueryBatchedTool` |
| `src/smolagents/rlm.py` | `RLMAgent`, `RLM_INSTRUCTIONS`, `make_variable_info()` |
| `src/smolagents/rlm_logging.py` | `RLMJSONLLogger` callback + fast-rlm-compatible schema |
| `tests/test_rlm.py` | Tests for RLM tools, budget, agent |
| `examples/rlm_basic.py` | Basic RLM usage example |

### Files NOT Changed

| File | Why |
|---|---|
| `src/smolagents/local_python_executor.py` | No changes needed — `max_print_outputs_length` already configurable |
| `src/smolagents/remote_executors.py` | RLM tools are regular Tools, not managed_agents — no blocker |
| `src/smolagents/tools.py` | Base Tool class is sufficient |
| `src/smolagents/memory.py` | CallbackRegistry already supports what we need |

---

## 8. Implementation Roadmap

The plan below is sequenced so that each milestone produces something **testable**. No milestone depends on later ones. If we stop at any point, what we've built so far is usable.

### Milestone 0: Fork setup & smoke test — COMPLETE

> **Details:** [`docs/rlm/M0_fork_setup.md`](../smolagents-rlm/docs/rlm/M0_fork_setup.md)

**Goal:** Cloned fork builds, existing tests pass, we can run a vanilla `CodeAgent` end-to-end.

**Tasks:**
1. Clone `oneryalcin/smolagents`, create branch `feat/rlm`
2. `uv pip install -e ".[dev]"`, run `pytest tests/ -x -q` — establish baseline
3. Run a trivial CodeAgent script with `LiteLLMModel` to verify the dev loop works

**Blocker to resolve first:** Confirm which Python version, which test subset passes clean. smolagents has known flaky tests around remote executors — we skip those.

**Done when:** `pytest` green (or known-flaky-only failures), manual CodeAgent script returns an answer.

---

### Milestone 1: Minimal RLM — `llm_query` + truncation — COMPLETE

> **Details:** [`docs/rlm/M1_minimal_rlm.md`](../smolagents-rlm/docs/rlm/M1_minimal_rlm.md)

**Goal:** An `RLMAgent` that can chunk a large text and ask sub-LLM calls about each chunk. This is the **smallest useful thing**.

**What to build:**
1. `max_output_length` param on `CodeAgent` (~5 lines changed in `agents.py`)
2. `LLMQueryTool` — single sub-LLM call (port from rlm_v2, ~40 lines)
3. `LLMQueryBatchedTool` — parallel sub-LLM calls (port from rlm_v2, ~60 lines)
4. `RLMAgent(CodeAgent)` — wires up tools + truncation + `RLM_INSTRUCTIONS` prompt (~80 lines)
5. `make_variable_info()` — metadata preview (port from rlm_v2, ~50 lines)

**What NOT to build yet:** Budget, logging, object returns. Just get the loop working.

**Test script:**
```python
from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent

agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
    max_output_length=3000,
)

# Generate a large synthetic context
context = "\n".join(f"Entry {i}: The color is {'red' if i % 7 == 0 else 'blue'}" for i in range(5000))

result = agent.run(
    task="How many entries have the color red?",
    context=context,
)
print(result)  # Should be ~715
```

**What we learn:** Does the LLM actually write chunking code? Does it use `llm_query_batched`? Does truncation force good behavior? How many steps does it take?

**Done when:** The test script returns a correct (or close) answer, and the agent used Python code to chunk/count rather than trying to read the entire context.

---

### Milestone 2: Budget controls

> **Details:** [`docs/rlm/M2_budget.md`](../smolagents-rlm/docs/rlm/M2_budget.md)

**Goal:** Prevent runaway costs. Without this, we can't safely test on real workloads.

**What to build:**
1. `BudgetManager` class — thread-safe call + token counter (~80 lines)
2. Wire into `LLMQueryTool` and `LLMQueryBatchedTool` — `record_call()` after each `generate()`
3. Step callback that calls `record_step()` for the orchestrator LLM's own token usage
4. `BudgetExceededError` that the agent sees as a tool error (not a crash)

**Key design question to resolve:** When budget is exceeded mid-batch (e.g., 30 of 50 prompts complete), do we:
- (a) Return partial results + error message — **prefer this**, more useful
- (b) Raise immediately, losing all results

**Test:**
```python
agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    budget=Budget(max_llm_calls=5),  # Very tight
)
# Agent should hit the limit and gracefully use what it has
```

**Done when:** Budget limit triggers, agent sees the error in its REPL, and either uses partial results or falls back to Python-only approach.

---

### Milestone 3: Thread safety verification

> **Details:** [`docs/rlm/M3_thread_safety.md`](../smolagents-rlm/docs/rlm/M3_thread_safety.md)

**Goal:** Confirm `llm_query_batched` actually works in parallel without corruption.

**What to build:** Nothing new. This is a testing milestone.

**Tests:**
1. Run `llm_query_batched` with 20 prompts, `max_workers=8`, verify all 20 responses are distinct and correct
2. Run it 10 times in a row — no state leaks between runs
3. Verify `BudgetManager` counter is correct after parallel execution (race condition check)

**Blocker to resolve:** Is `LiteLLMModel.generate()` thread-safe? Test empirically:
```python
# Throwaway test
from concurrent.futures import ThreadPoolExecutor
model = LiteLLMModel(model_id="gpt-4.1-nano")
def call(i): return model.generate([ChatMessage(role="user", content=f"Say '{i}'")])
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(call, range(20)))
# Check: 20 distinct results, no exceptions
```

If NOT thread-safe: default `max_workers=1`, document the limitation, and investigate whether creating one `LiteLLMModel` instance per thread is viable.

**Done when:** Parallel batched calls work reliably, or we have a documented workaround.

---

### Milestone 4: Real-world benchmark

> **Details:** [`docs/rlm/M4_benchmark.md`](../smolagents-rlm/docs/rlm/M4_benchmark.md)

**Goal:** Run against a real long-context task and compare with fast-rlm.

**What to build:**
1. Port fast-rlm's `benchmarks/oolong_synth_benchmark.py` to use our `RLMAgent`
2. Run on 3-5 OolongBench examples (counting, timeline, user tasks)
3. Compare: correctness, total LLM calls, total tokens, wall-clock time

**What we learn:** Where does our agent fail? Is truncation too aggressive / not aggressive enough? Are the `RLM_INSTRUCTIONS` effective? Does the LLM actually use `llm_query_batched` or does it loop sequentially?

**Done when:** We have a results table comparing our fork vs fast-rlm on the same tasks.

---

### Milestone 5: JSONL logging + fast-rlm TUI

> **Details:** [`docs/rlm/M5_jsonl_logging.md`](../smolagents-rlm/docs/rlm/M5_jsonl_logging.md)

**Goal:** Structured logs so we can debug what the agent is doing.

**What to build:**
1. `RLMJSONLLogger` — step callback emitting one JSON line per event (~100 lines)
2. Match fast-rlm's schema: `run_id`, `parent_run_id`, `depth`, `step`, `code`, `output`, `usage`, `timestamps`
3. For sub-LLM calls: emit log events from inside the tools (not just from step callbacks)

**Test:** Run an RLM task, then `fast-rlm-log output.jsonl --tui` and verify the tree renders.

**Done when:** fast-rlm's TUI shows our agent's execution tree correctly.

---

### Milestone 6: Iterate on prompt & strategy

> **Details:** [`docs/rlm/M6_prompt_iteration.md`](../smolagents-rlm/docs/rlm/M6_prompt_iteration.md)

**Goal:** Tune `RLM_INSTRUCTIONS` based on what we learned in Milestones 1-5.

This is not code work — it's prompt engineering informed by real runs:
- Did the LLM peek before processing? If not, make the instruction more forceful.
- Did it use grep for pattern tasks? If not, add more examples.
- Did it batch or loop? If loop, emphasize `llm_query_batched` more.
- Did it verify before `final_answer`? If not, add a stronger nudge.

No milestone gate — this is continuous improvement.

---

### What's explicitly deferred

| Feature | Why Deferred |
|---|---|
| True recursive sub-agents (sub-agent gets REPL) | Complex, needs isolated executor per child. Flat model works for map-reduce. Revisit if benchmarks show we need depth>1 reasoning. |
| Remote executor support | Our tools are regular `Tool` subclasses so they work with local executor. Remote support needs `managed_agents` fixes upstream or a different serialization strategy. |
| Object returns from sub-LLM | Useful but not blocking. String returns work. Add JSON mode in a later PR. |
| Cost estimation (USD) | Needs pricing table. Token budget is sufficient for safety. Add cost tracking once we pick a pricing source (`litellm.model_cost` or manual). |
| Upstream PR to huggingface | Wait until we've battle-tested the fork. The `max_output_length` one-liner is PR-worthy immediately, but RLM module needs more validation. |

---

### Dependency graph

```
M0 (fork setup)
 └─► M1 (minimal RLM — llm_query + truncation)
      ├─► M2 (budget controls)
      │    └─► M3 (thread safety verification)
      │         └─► M4 (real-world benchmark)
      └─► M5 (JSONL logging)
           └─► M4 (real-world benchmark, enhanced with logs)
                └─► M6 (prompt iteration)
```

M1 is the critical path. Everything else branches from it. M2 and M5 can be done in parallel.

---

## 9. Open Questions

### Q1: Should sub-LLM calls count toward `max_steps`?

Currently `max_steps` limits REPL iterations. Sub-LLM tool calls happen *within* a single step. If the LLM writes `llm_query_batched(50_prompts)` in one step, that's 50 LLM calls but 1 step. The `BudgetManager` handles this via call/token limits, but should we also enforce at the step level?

**Tentative answer:** No. Budget handles it. Steps are for REPL iterations.

### Q2: Should we support true recursion (sub-agents with REPLs)?

fast-rlm's main differentiator is that sub-agents can write code. Our `llm_query` is a flat text-in/text-out call. For most map-reduce workloads, flat is better (faster, simpler, no Pyodide boot). But for tasks requiring multi-hop reasoning on sub-chunks, true recursion helps.

**Tentative answer:** Not in Phase 1. Phase 2 could add an optional `LLMAgentTool` that spawns a child `CodeAgent` with its own executor, but this is complex.

### Q3: Thread safety of LiteLLMModel

`LLMQueryBatchedTool` uses `ThreadPoolExecutor`. Is `LiteLLMModel.generate()` thread-safe? LiteLLM's underlying HTTP clients (httpx, requests) are generally thread-safe, but smolagents' wrapper may have state. Need to verify.

**Action:** Test empirically before defaulting `thread_safe=True`.

### Q4: Pricing table for cost estimation

`BudgetManager` needs a model_id → cost_per_token mapping for `max_cost_usd`. Options:
- Hardcode a table (stale quickly)
- Use `litellm.model_cost` (litellm has a built-in pricing table)
- Make it optional (skip cost tracking if no pricing data)

**Tentative answer:** Use `litellm.model_cost` if available, otherwise skip cost estimation. Token budget still works regardless.

### Q5: Upstream PR or permanent fork?

If the changes are clean and backward-compatible, we could upstream to huggingface/smolagents. The `max_output_length` change is a one-liner fix that would benefit everyone. The RLM module could be an optional extra.

**Tentative answer:** Keep as fork initially. Once battle-tested, propose upstream PR for the core changes (`max_output_length`, `budget.py`). RLM-specific code may stay in the fork or become a separate package.

---

## 9. References

### Code

| Resource | Location |
|---|---|
| smolagents upstream | [huggingface/smolagents](https://github.com/huggingface/smolagents) |
| Fork | [oneryalcin/smolagents](https://github.com/oneryalcin/smolagents) |
| fast-rlm | [avbiswas/fast-rlm](https://github.com/avbiswas/fast-rlm) |
| rlm_v2.py | [gist:70464f35727a24ab8eb23fdb9ff471ad](https://gist.github.com/oneryalcin/70464f35727a24ab8eb23fdb9ff471ad) |
| RLM paper | [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601) |
| fast-rlm local copy | `/tmp/fast-rlm/` |
| smolagents local copy | `/tmp/smolagents/` |

### Relevant smolagents Issues

| Issue | Title | Status | Relevance |
|---|---|---|---|
| [#1061](https://github.com/huggingface/smolagents/issues/1061) | Two-level managed agent hierarchy broken | Open | Why we bypass managed_agents |
| [#1695](https://github.com/huggingface/smolagents/issues/1695) | Sub-agent memory cleared on each call | Open | Why we bypass managed_agents |
| [#1781](https://github.com/huggingface/smolagents/issues/1781) | Parallel managed agents share state | Open | Why we bypass managed_agents |
| [#524](https://github.com/huggingface/smolagents/issues/524) | CodeAgent truncates input before tool execution | Open | Related to truncation design |
| [#960](https://github.com/huggingface/smolagents/issues/960) | Prompt fails to trigger managed agent | Open | managed_agents unreliable |
| [#1774](https://github.com/huggingface/smolagents/issues/1774) | Optimize for reasoning models | Open | Adjacent work |
| [#1883](https://github.com/huggingface/smolagents/issues/1883) | Lifecycle hooks for CodeAgent | Open | Could help with budget enforcement |
| [#1875](https://github.com/huggingface/smolagents/issues/1875) | Token count manager | Open | Related to budget tracking |

### Key File Quick Reference

| What | File | Line |
|---|---|---|
| CodeAgent execution loop | `agents.py` | 540-611 |
| Single step execution | `agents.py` | 1639-1765 |
| Output truncation call | `agents.py` | 1753 |
| `truncate_content()` | `utils.py` | 257-265 |
| `MAX_LENGTH_TRUNCATE_CONTENT` | `utils.py` | 254 |
| Executor creation | `agents.py` | 1598-1617 |
| managed_agents setup | `agents.py` | 369-387 |
| managed_agents + remote block | `agents.py` | 1608-1609 |
| `LocalPythonExecutor.__call__` | `local_python_executor.py` | 1747-1758 |
| Print capture override | `local_python_executor.py` | 903 |
| `DEFAULT_MAX_LEN_OUTPUT` | `local_python_executor.py` | 57 |
| `Monitor.update_metrics` | `monitoring.py` | 100-117 |
| `TokenUsage` dataclass | `monitoring.py` | 36-54 |
| `CallbackRegistry` | `memory.py` | 280-316 |
| `Tool.__call__` → `forward()` | `tools.py` | 231-246 |
| `send_tools()` | `local_python_executor.py` | 1763-1765 |
| `send_variables()` | `local_python_executor.py` | 1760-1761 |
| `agent.state` init | `agents.py` | 331 |
| `agent.run()` | `agents.py` | 436-538 |
