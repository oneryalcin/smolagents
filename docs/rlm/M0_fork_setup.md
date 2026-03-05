# M0: Fork Setup & Smoke Test

> **Status:** COMPLETE
> **Date:** 2026-03-04
> **Branch:** `feat/rlm`

## What We Did

1. Forked `huggingface/smolagents` → [oneryalcin/smolagents](https://github.com/oneryalcin/smolagents)
2. Cloned to `/tmp/smolagents-rlm`, created branch `feat/rlm` from `5c684c1`
3. `uv sync --extra dev` — all deps installed cleanly
4. Ran core test suite: **413 passed, 2 skipped, 1 flaky** (passes on rerun)
   ```
   uv run pytest tests/test_local_python_executor.py tests/test_agents.py \
     tests/test_tools.py tests/test_memory.py tests/test_monitoring.py \
     tests/test_utils.py tests/test_import.py -x -q --timeout=60
   ```
5. Smoke-tested CodeAgent with real LLM:
   ```python
   model = LiteLLMModel(model_id='gpt-4.1-nano')
   agent = CodeAgent(model=model, tools=[], verbosity_level=0)
   result = agent.run('What is 2**10?')  # → 1024
   ```

## Test Baseline

| Suite | Passed | Skipped | Failed |
|---|---|---|---|
| `test_agents.py` | 24 | 0 | 1 (flaky, passes on rerun) |
| `test_local_python_executor.py` | 347 | 2 | 0 |
| `test_tools.py` | 16 | 0 | 0 |
| `test_memory.py` | 8 | 0 | 0 |
| `test_monitoring.py` | 5 | 0 | 0 |
| `test_utils.py` | 11 | 0 | 0 |
| `test_import.py` | 2 | 0 | 0 |

The flaky test is `test_transformers_toolcalling_agent` — HuggingFace local model with attention mask warning. Pre-existing, not our problem.

## Notes

- Python 3.12.10 on macOS Darwin 25.2.0
- smolagents v1.25.0.dev0
- `uv sync` is the right approach (not `uv pip install -e`), project has a proper `pyproject.toml` + `uv.lock`
- Tests we skip going forward: `test_remote_executors.py` (needs Docker/E2B), `test_all_docs.py` (slow), `test_gradio_ui.py` (needs browser)
