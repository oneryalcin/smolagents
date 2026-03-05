# M2: Budget Controls

> **Status:** COMPLETE
> **Branch:** `feat/rlm`
> **Depends on:** M1

## Goal

Prevent runaway costs. Without this, real workloads are risky.

## What Was Built

### `Budget`, `BudgetManager`, `BudgetExceededError` in `src/smolagents/rlm_tools.py`

- `Budget(max_llm_calls=N, max_total_tokens=N)` — both optional, validated positive
- `BudgetManager` — thread-safe via `threading.Lock`
- Protocol: `pre_call_check()` → `generate()` → `record_usage()`, with `release_call()` on failure

### Wired into tools

- Both `LLMQueryTool` and `LLMQueryBatchedTool` accept optional `budget_manager`
- Atomic slot reservation: `pre_call_check` increments under lock, `release_call` rolls back on `generate()` failure
- Batched tool cancels pending futures on any exception (budget or otherwise)

### Step callback on `RLMAgent`

- `_budget_callback` appends `[Budget] Sub-LLM calls: 3/10 | Sub-LLM tokens: 4,521` to observations
- LLM sees remaining budget after each step and can adapt

### Tests: `tests/test_rlm.py` — 36 tests

Covers M1 + M2: budget enforcement, thread safety (20 threads racing for 10 slots), slot release on failed generate, budget exceeded during agent.run(), kwargs forwarding.

## Design Decisions (changed from original plan)

1. **No separate `budget.py` file** — Budget classes live in `rlm_tools.py` (tightly coupled, small).
2. **No `partial_results` on `BudgetExceededError`** — framework wraps tool exceptions into strings, so `.partial_results` was dead code. Removed. The budget summary in observations tells the LLM what happened.
3. **No `max_cost_usd`** — deferred. Needs pricing table (`litellm.model_cost`). Token budget is sufficient for safety.
4. **No orchestrator token tracking** — `Monitor` already does this. Budget only tracks sub-LLM calls.
5. **Atomic slot reservation** — `pre_call_check` reserves (check+increment), `release_call` returns slot on failure. Prevents both "failed call burns budget" and concurrent overshoot.
6. **Token limit is lazy** — checked on next `pre_call_check`, not during `record_usage`. The call that crosses the threshold completes; the next call is blocked. Unavoidable since token count is unknown before the call.
7. **`RLMAgent.run` uses `**kwargs` passthrough** — doesn't mirror parent signature. Forward-compatible if `CodeAgent.run` adds params.

## Known Limitations

- `future.cancel()` only cancels pending (not-yet-started) futures. Already-running threads complete. Overshoot bounded by `max_workers`.
- Token budget overshoot: with `max_workers=8`, up to 8 concurrent calls can cross the token threshold before the next `pre_call_check` catches it. This is a soft limit.

## Validation

```python
from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent
from smolagents.rlm_tools import Budget

agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
    budget=Budget(max_llm_calls=10, max_total_tokens=50000),
)
result = agent.run(task="Classify entries", context=large_text)
# Agent sees [Budget] in observations, adapts if limit approached
```
