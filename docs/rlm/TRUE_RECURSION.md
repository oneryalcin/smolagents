# True RLM Recursion: Analysis & Implementation Guide

> **Status:** DEFERRED (implement after M4 if flat approach hits limits)
> **Branch:** `feat/rlm`
> **Depends on:** M4 benchmark data to justify complexity
> **References:**
> - [Alex Zhang's blog post](https://alexzhang13.github.io/blog/2025/rlm/)
> - [Alex Zhang's RLM repo](https://github.com/alexzhang13/rlm) (local: `/tmp/rlm/`)
> - [fast-rlm repo](https://github.com/avbiswas/fast-rlm) (local: `/tmp/fast-rlm/`)
> - [RLM paper](https://arxiv.org/abs/2512.24601)

---

## Why True Recursion Matters

Our current implementation is **flat map-reduce**: `llm_query(prompt) → string`. The sub-LLM gets a single prompt, returns a string. No REPL, no tools, no ability to recurse.

This works for ~90% of tasks (classify/extract/summarize per chunk). It fails for:

1. **10M+ token contexts** — chunks are still too large for a single sub-LLM call. The sub needs to re-chunk and delegate further.
2. **Multi-hop reasoning on a chunk** — sub needs REPL to grep, cross-reference, iterate.
3. **BrowseComp-Plus style** — 1000 docs, multi-hop associations. fast-rlm logs confirm depth=3 active in practice.

Alex Zhang's blog (Oct 2025):
> "In our experiments we only consider a recursive depth of 1... for most modern long context benchmarks, a recursive depth of 1 was sufficient. However, for future work... enabling larger recursive depth will naturally lead to stronger and more interesting systems."

Their BrowseComp-Plus results show `RLM(GPT-5)` is the only approach maintaining perfect performance at 1000-doc scale. The fast-rlm lex_fridman sample log confirms: depth=0 → 10 depth=1 → 5 depth=2 → 3 depth=3 (leaf).

---

## How Reference Implementations Do It

### Alex Zhang's `/tmp/rlm/`

Each RLM instance runs its own REPL loop via `exec()` in isolated `locals` dict:

```
RLM.completion(prompt)
  └─ _spawn_completion_context()  → LMHandler TCP server + LocalREPL
  └─ for i in range(max_iterations):
       LM responds with ```repl``` code blocks
       LocalREPL.execute_code(block)
         ├─ llm_query(...)   → flat LM call (no REPL)
         └─ rlm_query(...)   → RLM._subcall() → new child RLM(depth+1)
                               with its own REPL + LMHandler
```

Key files:
- `rlm/rlm.py` — `RLM` class, `_subcall()` method, depth tracking
- `rlm/environments/local_repl.py` — `LocalREPL`, `_rlm_query()`, `_llm_query()`
- `rlm/lm_handler.py` — TCP server for LLM calls, async batched via `asyncio.gather`

Depth mechanism:
```python
# rlm/rlm.py
def _subcall(self, prompt, model=None):
    next_depth = self.depth + 1
    if next_depth >= self.max_depth:
        # leaf: plain LM call, no REPL
        return client.completion(prompt)
    child = RLM(depth=next_depth, max_depth=self.max_depth, ...)
    return child.completion(prompt)
```

Budget propagation — remaining amounts passed to children:
```python
remaining_budget = self.max_budget - self._cumulative_cost
child = RLM(max_budget=remaining_budget, max_timeout=remaining_timeout, ...)
# After child completes:
self._cumulative_cost += result.usage_summary.total_cost
```

### fast-rlm `/tmp/fast-rlm/`

Each sub-agent gets a fresh Pyodide WASM runtime:

```
subagent(context, depth=0):
  loadPyodide()  // new Python runtime
  inject: context, FINAL(), llm_query = async () => subagent(ctx, depth+1)
  loop MAX_CALLS:
    LLM generates code → execute in pyodide
    if llm_query() called → subagent(ctx, depth+1) [recursive]
    if FINAL(x) called → return x
```

Key files:
- `src/subagents.ts:67-100` — `subagent()`, `llm_query` closure, depth guard
- `src/prompt.ts` — `SYSTEM_PROMPT` (non-leaf, includes llm_query docs) vs `LEAF_AGENT_SYSTEM_PROMPT` (no llm_query)
- `src/call_llm.ts:59` — prompt selection based on `is_leaf_agent`

Parallelism: LLM writes `asyncio.gather(*[llm_query(chunk) for chunk in chunks])` in Python. Pyodide bridges to JS Promises → concurrent sub-agent spawns.

Leaf enforcement: `subagent_depth >= MAX_DEPTH` → `llm_query()` throws. Leaf prompt omits all mention of `llm_query`.

---

## What We Need to Change (smolagents)

### Architecture: Spawn child CodeAgent in LLMQueryTool

The key insight: **CodeAgent instances are naturally isolated.** Each creates its own `LocalPythonExecutor` with its own `self.state` dict. Multiple instances can coexist without shared mutable state.

### Feasibility Assessment

| Resource | Thread-safe? | Notes |
|---|---|---|
| `Model.generate()` | Yes | HTTP clients are stateless per-call |
| `BudgetManager` | Yes | Uses `threading.Lock` |
| `LocalPythonExecutor` | N/A | Each agent gets its own instance |
| `agent.state` | N/A | Instance-level, not shared |
| `agent.memory` | N/A | Instance-level |
| `rich.Live` console | **No** | Must suppress in child agents |

### Implementation Sketch (~50 lines of changes)

#### 1. Add depth tracking to `LLMQueryTool` in `rlm_tools.py`

```python
class LLMQueryTool(Tool):
    def __init__(self, model, budget_manager=None,
                 recursive=False, max_depth=2, _current_depth=0,
                 max_child_steps=5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.budget_manager = budget_manager
        self.recursive = recursive
        self.max_depth = max_depth
        self._current_depth = _current_depth
        self.max_child_steps = max_child_steps

    def forward(self, prompt: str) -> str:
        if not self.recursive or self._current_depth >= self.max_depth:
            # Current flat behavior — single LM call
            return self._flat_query(prompt)

        # True recursion: spawn child CodeAgent with its own REPL
        child_tool = LLMQueryTool(
            model=self.model,
            budget_manager=self.budget_manager,  # shared, thread-safe
            recursive=True,
            max_depth=self.max_depth,
            _current_depth=self._current_depth + 1,
        )
        child_batched = LLMQueryBatchedTool(
            model=self.model,
            budget_manager=self.budget_manager,
        )
        child = CodeAgent(
            tools=[child_tool, child_batched],
            model=self.model,  # or a separate orchestrator model for children
            max_steps=self.max_child_steps,
            verbosity_level=LogLevel.ERROR,  # suppress rich.Live
        )
        result = child.run(task=prompt)
        return str(result)

    def _flat_query(self, prompt: str) -> str:
        # Existing budget + generate logic (current forward() body)
        ...
```

#### 2. Wire into `RLMAgent.__init__()` in `rlm.py`

```python
class RLMAgent(CodeAgent):
    def __init__(self, ..., recursive=False, max_depth=2, ...):
        rlm_tools = [
            LLMQueryTool(
                model=sub_model,
                budget_manager=self.budget_manager,
                recursive=recursive,
                max_depth=max_depth,
                _current_depth=0,
            ),
            LLMQueryBatchedTool(...),
        ]
```

#### 3. Prompt changes for leaf vs non-leaf

At `_current_depth == max_depth - 1` (leaf), the child CodeAgent should get instructions WITHOUT `llm_query` references — to prevent the model from trying to call it. fast-rlm does this with separate `LEAF_AGENT_SYSTEM_PROMPT`.

Options:
- (a) Conditionally omit RLM_INSTRUCTIONS when child has no recursive tools
- (b) Use a separate `LEAF_INSTRUCTIONS` prompt

### Gotchas to Handle

1. **`rich.Live` in child agents** — set `verbosity_level=LogLevel.ERROR` on child CodeAgent. This suppresses the interactive console that's not thread-safe.

2. **Unbounded recursion** — `max_depth` + shared `BudgetManager` provides double protection. Budget caps total LLM calls across ALL depths.

3. **Batched + recursive** — `LLMQueryBatchedTool` uses `ThreadPoolExecutor`. If each thread spawns a child CodeAgent (recursive mode), that's N concurrent agent loops. Safe (isolated executors) but heavy. Consider limiting `max_workers` when recursive.

4. **Child agent's model** — Alex Zhang's repo uses the same model for children's REPL orchestration. fast-rlm uses `sub_agent` model. We could use `sub_model` for the child's orchestrator too, or add a separate `recursive_orchestrator_model` param.

5. **Context passing** — The parent's REPL constructs the prompt string. The child only sees what the parent passes to `llm_query()`. This is correct — same as both reference implementations.

6. **Result type** — `child.run()` returns the `final_answer()` value. Cast to `str` for tool compatibility. For object returns, we'd need `final_answer()` to preserve types — separate concern.

---

## What NOT to Build

- **Pyodide/WASM sandbox per child** — fast-rlm's approach. Too heavy for smolagents. `LocalPythonExecutor` (AST-eval) is sufficient isolation.
- **TCP server per depth** — Alex Zhang's `LMHandler` pattern. Unnecessary when everything is in-process.
- **Custom event loop** — Both refs use async for parallelism at leaf level. Our `ThreadPoolExecutor` already handles this.

---

## Testing Plan

1. **Unit test: depth tracking** — verify `_current_depth` increments, leaf gets no recursive tool
2. **Integration test: two-pass** — orchestrator chunks, child re-chunks (depth=2)
3. **Budget across depths** — shared BudgetManager caps total calls regardless of depth
4. **Benchmark comparison** — same task with `recursive=False` vs `recursive=True`, measure quality delta

## Decision Gate

Run M4 benchmarks with flat approach first. If flat scores within 10% of fast-rlm on OOLONG, recursion is low priority. If flat fails on 10M+ scale or multi-hop tasks, implement recursion before M6.
