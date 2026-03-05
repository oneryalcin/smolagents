# M1: Minimal RLM — llm_query + truncation

> **Status:** COMPLETE
> **Branch:** `feat/rlm`
> **Depends on:** M0

## Goal

An `RLMAgent` that can chunk large text and ask sub-LLM calls about each chunk. Smallest useful thing.

## What to Build

### 1. `max_output_length` on CodeAgent

**File:** `src/smolagents/agents.py`
- Add param to `CodeAgent.__init__()` (~line 1530)
- Pass to `truncate_content()` at line 1753

~5 lines changed. Backward compatible (default `None` = existing 20K behavior).

### 2. `LLMQueryTool` and `LLMQueryBatchedTool`

**New file:** `src/smolagents/rlm_tools.py`

Port from rlm_v2 with minimal changes:
- `LLMQueryTool` — single `model.generate()` call, returns string
- `LLMQueryBatchedTool` — `ThreadPoolExecutor`, returns list of strings
- No budget integration yet (M2)
- `thread_safe` flag, default `False` (safe default, verify in M3)

### 3. `RLMAgent`

**New file:** `src/smolagents/rlm.py`

- Subclass `CodeAgent`
- Wires up `LLMQueryTool` + `LLMQueryBatchedTool`
- Sets `max_output_length=3000` by default
- Injects `RLM_INSTRUCTIONS` into system prompt
- `run(task, context=...)` routes large data to `self.state`
- `make_variable_info()` generates metadata previews

### 4. `RLM_INSTRUCTIONS` prompt

In `src/smolagents/rlm.py` (or separate file if large):
- Port from rlm_v2's `RLM_INSTRUCTIONS`
- Add parallelism example from fast-rlm's prompt
- Keep it concise — no contradictions

## What NOT to Build

- Budget (M2)
- JSONL logging (M5)
- Object returns (deferred)
- Tests beyond the manual smoke test

## Validation Script

```python
from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent

agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
)

context = "\n".join(f"Entry {i}: color={'red' if i%7==0 else 'blue'}" for i in range(5000))
result = agent.run(task="How many entries have the color red?", context=context)
print(result)  # Expect ~715
```

**Success criteria:** Agent uses Python chunking + `llm_query_batched` (or pure Python), returns correct answer, does NOT try to print the entire context.

## Implementation Notes

### Changes Made (2026-03-04)

**`src/smolagents/agents.py`** — 3 edits:
1. Added `max_output_length: int | None = None` param to `CodeAgent.__init__()` (line 1541)
2. Added `self.max_output_length = max_output_length` (line 1546)
3. Changed `truncate_content()` call at line 1756 to use `self.max_output_length or MAX_LENGTH_TRUNCATE_CONTENT`
4. Added `MAX_LENGTH_TRUNCATE_CONTENT` to imports from `.utils` (line 96)
5. Added docstring for the new param

**`src/smolagents/rlm_tools.py`** — new file, 83 lines:
- `LLMQueryTool` — calls `model.generate()`, returns `response.content`
- `LLMQueryBatchedTool` — `ThreadPoolExecutor(max_workers)`, `as_completed` pattern, preserves order
- Decided against `thread_safe` flag from rlm_v2 — defaulting to parallel (M3 will verify). Simpler API.

**`src/smolagents/rlm.py`** — new file, 164 lines:
- `make_variable_info()` — handles str, list, tuple, dict, fallback
- `RLM_INSTRUCTIONS` — 1335 chars, 4 rules: peek, grep, batch, verify
- `RLMAgent(CodeAgent)` — wires tools, sets truncation, injects prompt, routes context to state

**`examples/rlm_smoke_test.py`** — counting task (pure Python expected)
**`examples/rlm_semantic_test.py`** — classification task (llm_query_batched expected)

### Test Results

**Regression:** 628 passed, 0 failed (2 pre-existing excluded: `test_errors_logging`, `test_transformers_toolcalling`)

**Smoke test (counting):** Agent used pure Python `sum(1 for line in lines if "red" in line)`, 1 step, result=715 (correct)

**Semantic test (classification):** Agent used `llm_query_batched` with 12 parallel prompts, 2 steps, all 12 correct. Returned Python dict.

### Design Notes

- Dropped `thread_safe` flag from rlm_v2. All modern HTTP-based model clients are thread-safe for `generate()`. If we find one that isn't, we'll add it back. YAGNI for now.
- `max_output_length` defaults to `None` on `CodeAgent` (backward compat) but `3000` on `RLMAgent`. This means existing CodeAgent users see zero behavior change.
- `RLM_INSTRUCTIONS` is deliberately shorter than rlm_v2's version. Removed Strategy 5 (hierarchical summarization) and Strategy 6 (hybrid) — they added length without adding signal. The LLM figures these out from the 4 core rules.
- The `run()` method auto-routes strings >1000 chars to `self.state`. Threshold is conservative — avoids accidental prompt stuffing.

### Post-Review Fixes (2026-03-04)

Three independent reviewers (Carmack/Dijkstra/Simplifier personas) found 7 issues to fix immediately:

1. **CRITICAL: Stale `self.state` across runs** — `context` persists if second run omits it. Fixed: clear RLM-managed keys at start of `run()`.
2. **CRITICAL: `[ERROR]` sentinel in batched tool** — silently poisons results. Fixed: let exceptions propagate, framework handles them.
3. **CRITICAL: kwargs routing swallows `super().run()` params** — `max_steps`, `reset` etc. silently go to state. Fixed: explicit param forwarding.
4. **IMPORTANT: `response.content` can be `None`** — Fixed: return `""` if None.
5. **MINOR: Double braces in `RLM_INSTRUCTIONS`** — `{{` in non-f-string renders as literal `{{`. Fixed: single braces.
6. **MINOR: Unused `import threading`** — Removed.
7. **MINOR: `print()` instead of `self.logger`** — Fixed: use logger.
8. **MINOR: Docstring mentions "RLM agents"** in upstream CodeAgent — Fixed: made generic.

Second review pass caught 3 more:

9. **BUG: `_RLM_STATE_KEYS` was a module-level mutable set** — shared across all RLMAgent instances, causing cross-instance pollution. Fixed: per-instance `self._rlm_state_keys`.
10. **MINOR: `max_output_length <= 0` not validated** — Fixed: `ValueError` if non-positive.
11. **MINOR: `str | Any` type annotation meaningless** — Fixed: removed annotation on `context` param.

Deferred (low impact):
- Tool name collision undetected (upstream issue)
- `str(value)` unbounded in `make_variable_info` fallback (edge case)
- `make_variable_info` list/dict branches unreachable from kwargs routing (reachable via `context=`)
