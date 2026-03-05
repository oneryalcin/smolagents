# M4: Real-World Benchmark

> **Status:** NOT STARTED
> **Branch:** `feat/rlm`
> **Depends on:** M3, M5 (logging helps but not required)

## Goal

Run against real long-context tasks, compare with fast-rlm.

## What to Build

1. Port fast-rlm's `benchmarks/oolong_synth_benchmark.py` to use `RLMAgent`
2. Run on 3-5 OolongBench examples (counting, timeline, user tasks)
3. Results table: correctness, total calls, total tokens, wall-clock time

## Benchmark Tasks

From `oolongbench/oolong-synth` dataset:
- `counting` — count occurrences (should use Python grep, minimal LLM)
- `timeline` — temporal ordering (needs semantic understanding, should use sub-LLM)
- `user` — user-specific queries (mixed strategy)

## Comparison Axes

| Metric | Our RLMAgent | fast-rlm |
|---|---|---|
| Correct answer | ? | ? |
| Total LLM calls | ? | ? |
| Total tokens | ? | ? |
| Wall-clock time | ? | ? |
| Cost (estimated) | ? | ? |

## What We Learn

- Where does our agent fail?
- Is truncation too aggressive / not enough?
- Are `RLM_INSTRUCTIONS` effective?
- Does LLM use `llm_query_batched` or loop sequentially?

## Implementation Notes

*(to be filled during implementation)*
