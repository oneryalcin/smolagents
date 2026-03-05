# M5: JSONL Logging

> **Status:** COMPLETE
> **Branch:** `feat/rlm`
> **Depends on:** M1

## Goal

Structured JSONL logs for debugging RLM agent runs. Opt-in via `log_path` parameter.

## What Was Built

### `RLMLogger` in `src/smolagents/rlm_logging.py`

- Thread-safe JSONL writer (`threading.Lock` around write+flush)
- `emit(event_type, **fields)` — single write path for all events
- `emit_sub_llm(prompt, response, token_usage, call_start, call_end)` — convenience for sub-LLM call events
- `_format_usage()` — converts smolagents `TokenUsage` to fast-rlm compatible dict
- `_ts()` — epoch float to ISO 8601
- Context manager support (`with RLMLogger(...) as logger:`)
- `mkdir(parents=True)` on path creation

### Wired into tools (`rlm_tools.py`)

- Both `LLMQueryTool` and `LLMQueryBatchedTool` accept optional `rlm_logger`
- Timing captured around `model.generate()` via `time.time()`
- Each sub-LLM call emits an `llm_call` event with own `run_id`, parent's `run_id` as `parent_run_id`

### Wired into agent (`rlm.py`)

- `log_path: str | Path | None = None` parameter on `RLMAgent.__init__()`
- Lazy import of `RLMLogger` (zero cost when disabled)
- Step callbacks: `_log_action_step` (ActionStep → `execution_result`), `_log_final_answer` (FinalAnswerStep → `final_result`)
- `run()` wrapped with `agent_start`/`agent_end` events (try/finally)

### Tests: `tests/test_rlm.py` — 53 tests (36 existing + 17 new)

New test classes: `TestRLMLogger` (6), `TestRLMLoggerHelpers` (4), `TestToolLogging` (3), `TestAgentLogging` (4)

## Event Schema

```
agent.run(task, context)
  ├─ emit("agent_start", task=...)
  ├─ Step 1: LLM generates code → code executes
  │   ├─ [during execution] llm_query → emit("llm_call", depth=1, own run_id)
  │   ├─ [during execution] llm_query_batched → emit("llm_call") × N
  │   └─ [step finalized] → emit("execution_result", step=1, code, output, usage)
  ├─ Step 2: ...
  ├─ emit("final_result", result=...)
  └─ emit("agent_end")
```

Each JSONL line has: `level=30`, `time` (ISO 8601), `run_id`, `parent_run_id`, `depth`, `event_type`, plus event-specific fields.

## fast-rlm TUI Compatibility

| fast-rlm event | Our event | Status |
|---|---|---|
| `agent_start` | `agent_start` | Compatible |
| `agent_end` | `agent_end` | Compatible |
| `execution_result` | `execution_result` | Compatible |
| `code_generated` | — | Not emitted (smolagents callbacks fire post-execution) |
| `final_result` | `final_result` | Compatible |
| — | `llm_call` | Extra (sub-LLM calls, TUI ignores unknown types) |

Tree rendering works via `run_id`/`parent_run_id` mapping.

## Design Decisions

1. **No stdlib logging** — smolagents uses it minimally. Our logger writes JSONL directly.
2. **Lazy import** — `RLMLogger` only imported inside `if log_path:` branch.
3. **Prompts/responses truncated to 2000 chars** in logs — full data in agent memory.
4. **run_id format** — `{epoch_ms}-{hex9}` matches fast-rlm's `generateRunId()`.
5. **No `step` field on sub-LLM events** — tools don't know which orchestrator step they're in. Tree structure suffices.
6. **flush() after every write** — crash-safe logging.

## Usage

```python
from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent

agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
    log_path="logs/run.jsonl",
)
result = agent.run(task="Classify entries", context=large_text)
# Inspect: cat logs/run.jsonl | python -m json.tool --json-lines
```
