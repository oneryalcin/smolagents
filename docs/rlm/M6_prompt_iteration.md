# M6: Prompt & Strategy Iteration

> **Status:** NOT STARTED
> **Branch:** `feat/rlm`
> **Depends on:** M4 (needs benchmark data)

## Goal

Tune `RLM_INSTRUCTIONS` based on observed agent behavior from M1-M5.

## This is Not Code Work

Prompt engineering informed by real runs. Questions to answer:

1. Did the LLM peek before processing? If not → make instruction more forceful
2. Did it grep for pattern tasks? If not → add more examples
3. Did it batch or loop? If loop → emphasize `llm_query_batched` harder
4. Did it verify before `final_answer`? If not → stronger nudge
5. Did it waste LLM calls on counting/filtering? If yes → more "DON'T WASTE" examples

## Method

- Collect JSONL logs from M4 benchmark runs
- Analyze: which strategies did the LLM choose?
- A/B test prompt variants on same tasks
- Track: calls used, correctness, time

## Implementation Notes

*(to be filled during implementation)*
