# M3: Thread Safety Verification

> **Status:** NOT STARTED
> **Branch:** `feat/rlm`
> **Depends on:** M2

## Goal

Confirm `llm_query_batched` works in parallel without corruption.

## What to Build

Nothing new. This is a testing milestone.

## Tests to Run

### 1. Parallel correctness
Run `llm_query_batched` with 20 prompts, `max_workers=8`. Verify all 20 responses distinct and correct.

### 2. Repeated runs
Run 10 times in a row — no state leaks.

### 3. Budget counter under concurrency
After parallel batch of 20, verify `budget.summary["calls"] == 20` exactly.

### 4. LiteLLMModel thread safety
```python
from concurrent.futures import ThreadPoolExecutor
model = LiteLLMModel(model_id="gpt-4.1-nano")
def call(i): return model.generate([ChatMessage(role="user", content=f"Say '{i}'")])
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(call, range(20)))
# Expect: 20 distinct results, no exceptions
```

## If NOT Thread-Safe

- Default `max_workers=1`
- Investigate one `LiteLLMModel` per thread
- Document limitation

### 5. Verify `self.model` has no shared mutable state under concurrent access

All three M1 reviewers flagged that `LLMQueryBatchedTool` calls `self.model.generate()` from N threads simultaneously. The `Model` contract does not document thread-safety. Test: inspect model instance for mutable state (token counters, session objects) before/after concurrent calls.

## Implementation Notes

*(to be filled during implementation)*
