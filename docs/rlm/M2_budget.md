# M2: Budget Controls

> **Status:** NOT STARTED
> **Branch:** `feat/rlm`
> **Depends on:** M1

## Goal

Prevent runaway costs. Without this, real workloads are risky.

## What to Build

### 1. `BudgetManager`

**New file:** `src/smolagents/budget.py`

- `Budget` dataclass: `max_llm_calls`, `max_total_tokens`, `max_cost_usd` (all optional)
- `BudgetManager`: thread-safe tracker, `record_call()`, `record_step()`, `.summary`
- `BudgetExceededError`: raised when limit hit

### 2. Wire into tools

- `LLMQueryTool.forward()` calls `budget.record_call()` after `generate()`
- `LLMQueryBatchedTool.forward()` calls `budget.record_call(n=len(prompts))`
- On budget exceeded mid-batch: return partial results + error string (not crash)

### 3. Step callback

- Register callback on `ActionStep` that calls `budget.record_step()` for orchestrator LLM's own token usage
- Inject budget summary into `memory_step.observations` so the LLM sees remaining budget

## Design Decision: Error Semantics in Batched Tool

M1 reviewers unanimously rejected the `"[ERROR] ..."` sentinel string pattern — it silently poisons result lists. M1 fix: let exceptions propagate (framework handles them). For M2, when budget is exceeded mid-batch:

- Let in-flight requests complete (already sent)
- Cancel pending futures
- **Raise `BudgetExceededError` with partial results attached** as an attribute
- The agent's error handler sees the exception, can access `.partial_results` if needed
- This is explicit failure, not silent corruption

## Validation

```python
agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    budget=Budget(max_llm_calls=5),
)
# Should hit limit, agent gracefully falls back
```

## Implementation Notes

*(to be filled during implementation)*
