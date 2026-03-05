# M5: JSONL Logging + fast-rlm TUI Compatibility

> **Status:** NOT STARTED
> **Branch:** `feat/rlm`
> **Depends on:** M1

## Goal

Structured logs for debugging. Compatible with fast-rlm's TUI viewer.

## What to Build

### 1. `RLMJSONLLogger`

**New file:** `src/smolagents/rlm_logging.py`

Step callback emitting JSONL matching fast-rlm schema:

```json
{
  "time": "2026-03-04T...",
  "run_id": "abc123",
  "parent_run_id": null,
  "depth": 0,
  "step": 3,
  "event_type": "execution_result",
  "code": "chunks = ...",
  "output": "[TRUNCATED]...",
  "usage": {"prompt_tokens": 1234, "completion_tokens": 567},
  "timestamps": {"llm_call_start": "...", "execution_end": "..."}
}
```

### 2. Sub-LLM call logging

Emit events from inside `LLMQueryTool` and `LLMQueryBatchedTool`, not just step callbacks. Each sub-call gets its own `run_id` with `parent_run_id` pointing to the parent agent.

### 3. fast-rlm TUI compatibility

**Schema must match:** `src/logging.ts` in fast-rlm.
Key fields: `event_type` ∈ {`agent_start`, `agent_end`, `execution_result`, `code_generated`, `final_result`}

**Test:** `fast-rlm-log output.jsonl --tui` renders our logs correctly.

## fast-rlm Log Schema Reference

From `fast-rlm/src/logging.ts` and `tui_log_viewer/src/index.tsx`:

```typescript
interface LogEntry {
  level: number;
  time: string;
  run_id: string;
  parent_run_id?: string;
  depth: number;
  step?: number;
  event_type: "execution_result" | "code_generated" | "final_result" | "agent_start" | "agent_end";
  code?: string;
  output?: string;
  hasError?: boolean;
  reasoning?: string;
  usage?: Usage;
  result?: unknown;
  timestamps?: StepTimestamps;
}
```

## Implementation Notes

*(to be filled during implementation)*
