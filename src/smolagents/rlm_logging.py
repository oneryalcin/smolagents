"""JSONL structured logging for RLMAgent runs.

Emits one JSON line per event to a file. Compatible with fast-rlm TUI schema.
Thread-safe: multiple threads may call emit() concurrently (batched tool use).

Usage:
    agent = RLMAgent(
        model=LiteLLMModel(model_id="gpt-4.1-mini"),
        sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
        log_path="logs/run.jsonl",
    )
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _generate_run_id() -> str:
    """Generate a unique run ID matching fast-rlm format: {epoch_ms}-{hex9}."""
    return f"{int(time.time() * 1000)}-{uuid4().hex[:9]}"


def _ts(t: float | None) -> str | None:
    """Convert epoch float to ISO 8601 UTC string."""
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _format_usage(token_usage: Any) -> dict | None:
    """Convert smolagents TokenUsage to fast-rlm compatible usage dict."""
    if not token_usage:
        return None
    return {
        "prompt_tokens": token_usage.input_tokens,
        "completion_tokens": token_usage.output_tokens,
        "total_tokens": token_usage.total_tokens,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost": None,
    }


class RLMLogger:
    """Writes structured JSONL log events for RLM agent runs.

    Args:
        path: Path to JSONL file (created/appended).
        run_id: Optional fixed run ID. Generated if not provided.
    """

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.run_id = run_id or _generate_run_id()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(
        self,
        event_type: str,
        *,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        depth: int = 0,
        **fields: Any,
    ) -> None:
        """Write one JSONL line. Thread-safe."""
        record = {
            "level": 30,
            "time": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id or self.run_id,
            "parent_run_id": parent_run_id,
            "depth": depth,
            "event_type": event_type,
            **fields,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def emit_sub_llm(
        self,
        *,
        prompt: str,
        response: str,
        token_usage: Any = None,
        call_start: float,
        call_end: float,
    ) -> None:
        """Emit a sub-LLM call event. Called from tool threads."""
        self.emit(
            "llm_call",
            run_id=_generate_run_id(),
            parent_run_id=self.run_id,
            depth=1,
            prompt=prompt[:2000],
            response=response[:2000],
            usage=_format_usage(token_usage),
            timestamps={
                "llm_call_start": _ts(call_start),
                "llm_call_end": _ts(call_end),
            },
        )

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
