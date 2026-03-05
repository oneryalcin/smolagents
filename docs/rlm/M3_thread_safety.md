# M3: Thread Safety Verification

> **Status:** COMPLETE (absorbed into M2 tests; LiteLLMModel empirical test deferred to M4)
> **Branch:** `feat/rlm`
> **Depends on:** M2

## Goal

Confirm `llm_query_batched` works in parallel without corruption.

## What Was Tested (in M2)

| Test | Location | Result |
|------|----------|--------|
| Parallel correctness — order preserved across workers | `test_forward_preserves_order` (4 workers, 10 prompts) | Pass |
| Budget counter exact under 20-thread race | `test_thread_safety_exact_count` | Pass — atomic slot reservation prevents overshoot |
| State cleared between runs | `test_state_cleared_between_runs` | Pass |
| Budget resets between runs | `test_budget_resets_between_runs` | Pass |
| Failed generate releases slot | `test_failed_generate_releases_slot` | Pass |
| Budget exceeded during agent.run() | `test_budget_exceeded_during_run` | Pass |

## Deferred to M4

**LiteLLMModel empirical thread safety** — requires real API keys. Will be tested as part of M4 (real-world benchmark) which already needs live API calls. Test plan:

```python
from concurrent.futures import ThreadPoolExecutor
from smolagents import LiteLLMModel
from smolagents.models import ChatMessage, MessageRole

model = LiteLLMModel(model_id="gpt-4.1-nano")
def call(i):
    return model.generate([ChatMessage(role=MessageRole.USER, content=f"Say '{i}'")])

with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(call, range(20)))

# Verify: 20 distinct results, no exceptions, no shared-state corruption
assert len(results) == 20
assert len(set(r.content for r in results)) > 1
```

If NOT thread-safe: default `max_workers=1`, document limitation.
