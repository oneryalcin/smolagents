"""Tests for RLM tools, budget, logging, and agent."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage
from smolagents.rlm import RLMAgent, make_variable_info
from smolagents.rlm_logging import RLMLogger, _format_usage, _truncate, _ts
from smolagents.rlm_tools import (
    Budget,
    BudgetExceededError,
    BudgetManager,
    LLMQueryBatchedTool,
    LLMQueryTool,
)


# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------


class FakeSubModel(Model):
    """Sub-LLM model that returns a canned response with token usage."""

    def __init__(self, response: str = "sub-response", tokens: tuple[int, int] = (10, 5)):
        super().__init__()
        self.response = response
        self.input_tokens, self.output_tokens = tokens
        self.call_count = 0
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs):
        with self._lock:
            self.call_count += 1
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=self.response,
            token_usage=TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class FakeSubModelNoTokens(Model):
    """Sub-LLM model that returns no token usage (some providers don't)."""

    def generate(self, messages, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")


class FakeFailingModel(Model):
    """Model that raises on generate() to test failed-call-doesn't-burn-budget."""

    def generate(self, messages, **kwargs):
        raise ConnectionError("API unavailable")


class FakeOrchestratorModel(Model):
    """Orchestrator that peeks at context then counts reds via Python."""

    def __init__(self):
        super().__init__()

    def generate(self, messages, stop_sequences=None, **kwargs):
        text = str(messages)
        if "Length:" in text:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content='<code>\ncount = sum(1 for line in context.splitlines() if "red" in line)\nfinal_answer(count)\n</code>',
                token_usage=TokenUsage(input_tokens=50, output_tokens=20),
            )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content='<code>\nprint(f"Length: {len(context):,} chars")\n</code>',
            token_usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


class FakeOrchestratorWithLLMQuery(Model):
    """Orchestrator that calls llm_query, used to test budget during agent.run()."""

    def __init__(self):
        super().__init__()
        self._step = 0

    def generate(self, messages, stop_sequences=None, **kwargs):
        self._step += 1
        text = str(messages)
        if "BudgetExceededError" in text:
            # Budget was hit — fall back to Python
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content='<code>\nfinal_answer("budget hit, used python")\n</code>',
                token_usage=TokenUsage(input_tokens=50, output_tokens=20),
            )
        if self._step == 1:
            # First step: call llm_query_batched with many prompts
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content='<code>\nresults = llm_query_batched(["classify: " + str(i) for i in range(20)])\nprint(len(results))\n</code>',
                token_usage=TokenUsage(input_tokens=100, output_tokens=30),
            )
        # Fallback
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content='<code>\nfinal_answer("done")\n</code>',
            token_usage=TokenUsage(input_tokens=50, output_tokens=10),
        )


# ---------------------------------------------------------------------------
# Budget dataclass validation
# ---------------------------------------------------------------------------


class TestBudget:
    def test_rejects_negative_calls(self):
        with pytest.raises(ValueError, match="max_llm_calls must be positive"):
            Budget(max_llm_calls=-1)

    def test_rejects_zero_tokens(self):
        with pytest.raises(ValueError, match="max_total_tokens must be positive"):
            Budget(max_total_tokens=0)

    def test_unlimited_is_valid(self):
        b = Budget()
        assert b.max_llm_calls is None
        assert b.max_total_tokens is None


# ---------------------------------------------------------------------------
# BudgetManager
# ---------------------------------------------------------------------------


class TestBudgetManager:
    def test_call_limit_enforced(self):
        bm = BudgetManager(Budget(max_llm_calls=3))
        for _ in range(3):
            bm.pre_call_check()
            bm.record_usage(TokenUsage(input_tokens=10, output_tokens=5))
        with pytest.raises(BudgetExceededError, match="3/3 calls"):
            bm.pre_call_check()

    def test_token_limit_blocks_next_call(self):
        """Token limit is checked on pre_call_check, not during record_usage."""
        bm = BudgetManager(Budget(max_total_tokens=100))
        bm.pre_call_check()
        bm.record_usage(TokenUsage(input_tokens=60, output_tokens=50))  # 110 > 100, but record_usage never raises
        # Next pre_call_check should catch it
        with pytest.raises(BudgetExceededError, match="token budget exceeded"):
            bm.pre_call_check()

    def test_unlimited_budget(self):
        bm = BudgetManager(Budget())
        for _ in range(200):
            bm.pre_call_check()
            bm.record_usage(TokenUsage(input_tokens=100, output_tokens=100))
        assert "200" in bm.summary
        assert "40,000" in bm.summary

    def test_reset_clears_counters(self):
        bm = BudgetManager(Budget(max_llm_calls=5))
        for _ in range(5):
            bm.pre_call_check()
            bm.record_usage()
        with pytest.raises(BudgetExceededError):
            bm.pre_call_check()
        bm.reset()
        bm.pre_call_check()  # should work after reset
        bm.record_usage()
        assert "calls: 1" in bm.summary

    def test_summary_with_limits(self):
        bm = BudgetManager(Budget(max_llm_calls=10, max_total_tokens=5000))
        bm.pre_call_check()
        bm.record_usage(TokenUsage(input_tokens=100, output_tokens=50))
        summary = bm.summary
        assert "1/10" in summary
        assert "150" in summary
        assert "5,000" in summary

    def test_summary_without_limits(self):
        bm = BudgetManager(Budget())
        bm.pre_call_check()
        bm.record_usage()
        summary = bm.summary
        assert "Sub-LLM calls: 1" in summary
        assert "/" not in summary.split("|")[0]

    def test_thread_safety_exact_count(self):
        """20 threads race for 10 call slots — exactly 10 must succeed."""
        bm = BudgetManager(Budget(max_llm_calls=10))
        succeeded = 0
        failed = 0
        lock = threading.Lock()

        def try_call():
            nonlocal succeeded, failed
            try:
                bm.pre_call_check()
                bm.record_usage(TokenUsage(input_tokens=10, output_tokens=5))
                with lock:
                    succeeded += 1
            except BudgetExceededError:
                with lock:
                    failed += 1

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(try_call) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        assert succeeded == 10
        assert failed == 10
        assert "10/10" in bm.summary
        assert "150" in bm.summary

    def test_none_token_usage_ignored(self):
        bm = BudgetManager(Budget(max_total_tokens=100))
        bm.record_usage(None)
        assert "tokens: 0" in bm.summary

    def test_failed_generate_releases_slot(self):
        """If generate() fails, release_call() returns the reserved slot."""
        bm = BudgetManager(Budget(max_llm_calls=2))
        # Reserve a slot, then release it (simulating failed generate)
        bm.pre_call_check()
        bm.release_call()
        # Both slots should still be available
        bm.pre_call_check()
        bm.record_usage()
        bm.pre_call_check()
        bm.record_usage()
        # Now we've used both slots
        with pytest.raises(BudgetExceededError):
            bm.pre_call_check()


# ---------------------------------------------------------------------------
# LLMQueryTool
# ---------------------------------------------------------------------------


class TestLLMQueryTool:
    def test_forward_returns_content(self):
        model = FakeSubModel(response="classified: positive")
        tool = LLMQueryTool(model=model)
        result = tool.forward("classify this")
        assert result == "classified: positive"
        assert model.call_count == 1

    def test_forward_with_budget(self):
        model = FakeSubModel()
        bm = BudgetManager(Budget(max_llm_calls=2))
        tool = LLMQueryTool(model=model, budget_manager=bm)
        tool.forward("p1")
        tool.forward("p2")
        with pytest.raises(BudgetExceededError):
            tool.forward("p3")
        assert model.call_count == 2

    def test_forward_without_budget_is_unlimited(self):
        model = FakeSubModel()
        tool = LLMQueryTool(model=model)
        for _ in range(50):
            tool.forward("prompt")
        assert model.call_count == 50

    def test_forward_records_token_usage(self):
        model = FakeSubModel(tokens=(100, 50))
        bm = BudgetManager(Budget())
        tool = LLMQueryTool(model=model, budget_manager=bm)
        tool.forward("prompt")
        assert "150" in bm.summary

    def test_forward_handles_none_content(self):
        model = FakeSubModel(response="x")
        tool = LLMQueryTool(model=model)
        original = model.generate

        def gen_none(messages, **kwargs):
            msg = original(messages, **kwargs)
            msg.content = None
            return msg

        model.generate = gen_none
        assert tool.forward("test") == ""

    def test_forward_handles_no_token_usage(self):
        model = FakeSubModelNoTokens()
        bm = BudgetManager(Budget(max_total_tokens=100))
        tool = LLMQueryTool(model=model, budget_manager=bm)
        tool.forward("test")
        assert "calls: 1" in bm.summary
        assert "tokens: 0" in bm.summary

    def test_failed_generate_doesnt_consume_budget(self):
        """Network error during generate() should not consume a budget slot."""
        model = FakeFailingModel()
        bm = BudgetManager(Budget(max_llm_calls=2))
        tool = LLMQueryTool(model=model, budget_manager=bm)
        with pytest.raises(ConnectionError):
            tool.forward("test")
        # Budget should still have 2 slots available
        assert "calls: 0" in bm.summary


# ---------------------------------------------------------------------------
# LLMQueryBatchedTool
# ---------------------------------------------------------------------------


class TestLLMQueryBatchedTool:
    def test_forward_preserves_order(self):
        model = FakeSubModel()
        original = model.generate

        def gen_echo(messages, **kwargs):
            msg = original(messages, **kwargs)
            msg.content = f"response-{messages[0].content}"
            return msg

        model.generate = gen_echo
        tool = LLMQueryBatchedTool(model=model, max_workers=4)
        prompts = [f"p{i}" for i in range(10)]
        results = tool.forward(prompts)
        assert len(results) == 10
        for i, r in enumerate(results):
            assert r == f"response-p{i}"

    def test_forward_empty_list(self):
        model = FakeSubModel()
        tool = LLMQueryBatchedTool(model=model)
        assert tool.forward([]) == []
        assert model.call_count == 0

    def test_forward_budget_exceeded_raises(self):
        """Budget exceeded mid-batch should raise BudgetExceededError."""
        model = FakeSubModel()
        bm = BudgetManager(Budget(max_llm_calls=3))
        tool = LLMQueryBatchedTool(model=model, max_workers=1, budget_manager=bm)
        with pytest.raises(BudgetExceededError):
            tool.forward([f"p{i}" for i in range(10)])
        assert model.call_count == 3

    def test_forward_records_all_token_usage(self):
        model = FakeSubModel(tokens=(10, 5))
        bm = BudgetManager(Budget())
        tool = LLMQueryBatchedTool(model=model, max_workers=4, budget_manager=bm)
        tool.forward([f"p{i}" for i in range(8)])
        assert "calls: 8" in bm.summary
        assert "120" in bm.summary  # 8 * 15


# ---------------------------------------------------------------------------
# make_variable_info
# ---------------------------------------------------------------------------


class TestMakeVariableInfo:
    def test_string_metadata(self):
        text = "line1\nline2\nline3"
        info = make_variable_info("ctx", text)
        assert "Variable: `ctx`" in info
        assert "str" in info
        assert "Lines: 3" in info
        assert "17 chars" in info

    def test_long_string_truncated(self):
        text = "x" * 5000
        info = make_variable_info("ctx", text, preview_chars=100)
        assert "..." in info
        assert len(info) < 500

    def test_list_metadata(self):
        info = make_variable_info("data", list(range(100)))
        assert "list" in info
        assert "Items: 100" in info
        assert "Preview (first 10)" in info

    def test_dict_metadata(self):
        info = make_variable_info("data", {f"k{i}": i for i in range(20)})
        assert "dict" in info
        assert "Keys: 20" in info


# ---------------------------------------------------------------------------
# RLMAgent integration
# ---------------------------------------------------------------------------


class TestRLMAgent:
    def test_creates_with_default_tools(self):
        model = FakeSubModel()
        agent = RLMAgent(model=model)
        tool_names = [t.name for t in agent.tools.values()]
        assert "llm_query" in tool_names
        assert "llm_query_batched" in tool_names

    def test_budget_wired_to_tools(self):
        model = FakeSubModel()
        agent = RLMAgent(model=model, budget=Budget(max_llm_calls=5))
        assert agent.budget_manager is not None
        for tool in agent.tools.values():
            if hasattr(tool, "budget_manager"):
                assert tool.budget_manager is agent.budget_manager

    def test_no_budget_by_default(self):
        model = FakeSubModel()
        agent = RLMAgent(model=model)
        assert agent.budget_manager is None

    def test_run_with_context_returns_answer(self):
        """End-to-end: orchestrator peeks then counts reds via Python."""
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3)
        big_text = "\n".join(f"Entry {i}: color={'red' if i%7==0 else 'blue'}" for i in range(100))
        result = agent.run(task="How many red?", context=big_text)
        assert result == 15
        assert agent.state["context"] == big_text

    def test_budget_resets_between_runs(self):
        orchestrator = FakeOrchestratorModel()
        agent = RLMAgent(model=orchestrator, budget=Budget(max_llm_calls=10), max_steps=3)
        agent.budget_manager.pre_call_check()
        agent.budget_manager.record_usage(TokenUsage(input_tokens=10, output_tokens=5))
        assert "calls: 1" in agent.budget_manager.summary
        # run() resets budget; FakeOrchestratorModel makes 0 sub-LLM calls
        agent.run(task="test", context="small context")
        assert "calls: 0" in agent.budget_manager.summary

    def test_budget_callback_injects_summary(self):
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, budget=Budget(max_llm_calls=50), max_steps=3)
        context = "\n".join(f"Entry {i}: color={'red' if i%7==0 else 'blue'}" for i in range(100))
        agent.run(task="How many red?", context=context)
        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        budget_found = any("[Budget]" in (s.observations or "") for s in action_steps)
        assert budget_found, "Budget summary should appear in step observations"

    def test_state_cleared_between_runs(self):
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3)
        agent.run(task="Count reds", context="Entry 0: color=red\nEntry 1: color=blue")
        assert "context" in agent.state
        agent.run(task="Simple task")
        assert "context" not in agent.state

    def test_budget_exceeded_during_run(self):
        """Agent hits budget during run — should see error and recover gracefully."""
        sub_model = FakeSubModel()
        orchestrator = FakeOrchestratorWithLLMQuery()
        agent = RLMAgent(
            model=orchestrator,
            sub_model=sub_model,
            budget=Budget(max_llm_calls=5),  # tight: orchestrator tries 20 batched calls
            max_steps=5,
        )
        result = agent.run(task="Classify entries", context="some data")
        # Agent should recover — either return a result or hit max_steps
        # The key assertion: no crash, agent handled BudgetExceededError
        assert result is not None

    def test_kwargs_forwarded_to_super(self):
        """reset, max_steps etc. should be forwarded to CodeAgent.run()."""
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=10)
        agent.run(task="test", context="data", reset=False, max_steps=2)
        # If max_steps wasn't forwarded, agent would use default 10
        # With max_steps=2 and FakeOrchestratorModel needing 2 steps, it should complete
        assert agent.step_number <= 3


# ---------------------------------------------------------------------------
# RLMLogger
# ---------------------------------------------------------------------------


def _read_events(path):
    """Read JSONL file and return list of parsed dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRLMLogger:
    def test_emit_writes_valid_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        logger = RLMLogger(path, run_id="test-run")
        logger.emit("agent_start", task="hello")
        logger.close()

        events = _read_events(path)
        assert len(events) == 1
        e = events[0]
        assert e["event_type"] == "agent_start"
        assert e["run_id"] == "test-run"
        assert e["level"] == 30
        assert e["depth"] == 0
        assert "time" in e
        assert e["task"] == "hello"

    def test_emit_thread_safety(self, tmp_path):
        """20 threads writing concurrently — all lines present and valid JSON."""
        path = tmp_path / "threaded.jsonl"
        logger = RLMLogger(path, run_id="ts-test")

        def write_event(i):
            logger.emit("execution_result", step=i)

        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(write_event, range(20)))
        logger.close()

        events = _read_events(path)
        assert len(events) == 20
        steps = sorted(e["step"] for e in events)
        assert steps == list(range(20))

    def test_emit_sub_llm_schema(self, tmp_path):
        """Sub-LLM events get their own run_id with parent_run_id pointing to agent."""
        path = tmp_path / "sub.jsonl"
        logger = RLMLogger(path, run_id="parent-123")
        now = time.time()
        logger.emit_sub_llm(
            prompt="classify this text",
            response="category A",
            token_usage=TokenUsage(input_tokens=100, output_tokens=20),
            call_start=now - 0.5,
            call_end=now,
        )
        logger.close()

        events = _read_events(path)
        assert len(events) == 1
        e = events[0]
        assert e["event_type"] == "llm_call"
        assert e["run_id"] != "parent-123"  # own run_id
        assert e["parent_run_id"] == "parent-123"
        assert e["depth"] == 1
        assert e["usage"]["prompt_tokens"] == 100
        assert e["usage"]["completion_tokens"] == 20
        assert e["usage"]["total_tokens"] == 120
        assert "llm_call_start" in e["timestamps"]
        assert "llm_call_end" in e["timestamps"]

    def test_sub_llm_truncates_long_prompts(self, tmp_path):
        path = tmp_path / "trunc.jsonl"
        logger = RLMLogger(path, run_id="trunc-test")
        logger.emit_sub_llm(
            prompt="x" * 5000,
            response="y" * 5000,
            token_usage=None,
            call_start=time.time(),
            call_end=time.time(),
        )
        logger.close()

        events = _read_events(path)
        assert len(events[0]["prompt"]) == 2000
        assert len(events[0]["response"]) == 2000

    def test_context_manager(self, tmp_path):
        path = tmp_path / "ctx.jsonl"
        with RLMLogger(path, run_id="ctx-test") as logger:
            logger.emit("agent_start")
        # File should be closed, readable
        events = _read_events(path)
        assert len(events) == 1

    def test_mkdir_creates_parents(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "log.jsonl"
        logger = RLMLogger(path, run_id="dir-test")
        logger.emit("agent_start")
        logger.close()
        assert path.exists()

    def test_emit_after_close_is_silent(self, tmp_path):
        """Writes after close() are silently dropped, not crashes."""
        path = tmp_path / "closed.jsonl"
        logger = RLMLogger(path, run_id="closed-test")
        logger.emit("agent_start")
        logger.close()
        logger.emit("should_be_dropped")  # must not raise
        events = _read_events(path)
        assert len(events) == 1
        assert events[0]["event_type"] == "agent_start"

    def test_double_close_is_safe(self, tmp_path):
        path = tmp_path / "dbl.jsonl"
        logger = RLMLogger(path, run_id="dbl-test")
        logger.close()
        logger.close()  # must not raise


class TestRLMLoggerHelpers:
    def test_format_usage_with_tokens(self):
        usage = _format_usage(TokenUsage(input_tokens=100, output_tokens=50))
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["cost"] is None
        assert usage["cached_tokens"] is None  # unknown, not 0
        assert usage["reasoning_tokens"] is None

    def test_format_usage_none(self):
        assert _format_usage(None) is None

    def test_format_usage_non_token_usage_type(self):
        """Non-TokenUsage truthy value should return None, not crash."""
        assert _format_usage("not a token usage") is None
        assert _format_usage(42) is None

    def test_ts_converts_epoch(self):
        ts = _ts(0.0)
        assert ts == "1970-01-01T00:00:00+00:00"

    def test_ts_none(self):
        assert _ts(None) is None

    def test_truncate_long_string(self):
        assert len(_truncate("x" * 5000)) == 2000

    def test_truncate_short_string(self):
        assert _truncate("short") == "short"

    def test_truncate_none(self):
        assert _truncate(None) is None


class TestToolLogging:
    def test_llm_query_emits_log(self, tmp_path):
        """LLMQueryTool with logger emits llm_call event."""
        path = tmp_path / "tool.jsonl"
        logger = RLMLogger(path, run_id="tool-test")
        model = FakeSubModel(response="classified", tokens=(50, 10))
        tool = LLMQueryTool(model=model, rlm_logger=logger)

        result = tool.forward("classify this")
        logger.close()

        assert result == "classified"
        events = _read_events(path)
        assert len(events) == 1
        e = events[0]
        assert e["event_type"] == "llm_call"
        assert e["parent_run_id"] == "tool-test"
        assert e["usage"]["prompt_tokens"] == 50

    def test_llm_query_batched_emits_per_prompt(self, tmp_path):
        """Batched tool emits one llm_call per prompt."""
        path = tmp_path / "batched.jsonl"
        logger = RLMLogger(path, run_id="batch-test")
        model = FakeSubModel()
        tool = LLMQueryBatchedTool(model=model, max_workers=4, rlm_logger=logger)

        tool.forward(["p1", "p2", "p3"])
        logger.close()

        events = _read_events(path)
        assert len(events) == 3
        assert all(e["event_type"] == "llm_call" for e in events)
        assert all(e["parent_run_id"] == "batch-test" for e in events)
        # Each has unique run_id
        run_ids = [e["run_id"] for e in events]
        assert len(set(run_ids)) == 3

    def test_no_logger_no_overhead(self):
        """Tools with rlm_logger=None work unchanged."""
        model = FakeSubModel()
        tool = LLMQueryTool(model=model)
        assert tool.forward("test") == "sub-response"


class TestAgentLogging:
    def test_agent_lifecycle_events(self, tmp_path):
        """Agent run emits agent_start → execution_result(s) → final_result → agent_end."""
        path = tmp_path / "agent.jsonl"
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3, log_path=str(path))
        context = "\n".join(f"Entry {i}: color={'red' if i%7==0 else 'blue'}" for i in range(10))

        agent.run(task="Count reds", context=context)
        agent.close()

        events = _read_events(path)
        event_types = [e["event_type"] for e in events]
        assert event_types[0] == "agent_start"
        assert event_types[-1] == "agent_end"
        assert "execution_result" in event_types
        assert "final_result" in event_types
        # final_result comes before agent_end
        final_idx = event_types.index("final_result")
        end_idx = event_types.index("agent_end")
        assert final_idx < end_idx
        # All agent events share the same run_id
        agent_run_id = events[0]["run_id"]
        for e in events:
            if e["event_type"] != "llm_call":
                assert e["run_id"] == agent_run_id

    def test_agent_end_has_success_flag(self, tmp_path):
        path = tmp_path / "success.jsonl"
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3, log_path=str(path))
        agent.run(task="Count reds", context="Entry 0: red")
        agent.close()

        events = _read_events(path)
        end = next(e for e in events if e["event_type"] == "agent_end")
        assert end["success"] is True

    def test_agent_start_contains_task(self, tmp_path):
        path = tmp_path / "task.jsonl"
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3, log_path=str(path))
        agent.run(task="Find red entries", context="Entry 0: red")
        agent.close()

        events = _read_events(path)
        start = next(e for e in events if e["event_type"] == "agent_start")
        assert "Find red entries" in start["task"]

    def test_no_log_path_means_no_logger(self):
        model = FakeSubModel()
        agent = RLMAgent(model=model)
        assert agent.rlm_logger is None

    def test_sub_llm_calls_logged_during_run(self, tmp_path):
        """When agent calls llm_query_batched, sub-LLM events appear in log."""
        path = tmp_path / "sub_calls.jsonl"
        sub_model = FakeSubModel()
        orchestrator = FakeOrchestratorWithLLMQuery()
        agent = RLMAgent(
            model=orchestrator, sub_model=sub_model,
            budget=Budget(max_llm_calls=5),
            log_path=str(path), max_steps=5,
        )
        agent.run(task="Classify", context="data")
        agent.close()

        events = _read_events(path)
        llm_calls = [e for e in events if e["event_type"] == "llm_call"]
        assert len(llm_calls) == 5  # budget limited to 5
        assert all(e["depth"] == 1 for e in llm_calls)

    def test_execution_result_uses_snake_case(self, tmp_path):
        """Verify has_error field uses snake_case, not camelCase."""
        path = tmp_path / "snake.jsonl"
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3, log_path=str(path))
        agent.run(task="Count", context="Entry 0: red")
        agent.close()

        events = _read_events(path)
        exec_events = [e for e in events if e["event_type"] == "execution_result"]
        assert len(exec_events) > 0
        for e in exec_events:
            assert "has_error" in e
            assert "hasError" not in e

    def test_multi_run_logging(self, tmp_path):
        """Logger survives across multiple run() calls."""
        path = tmp_path / "multi.jsonl"
        model = FakeOrchestratorModel()
        agent = RLMAgent(model=model, max_steps=3, log_path=str(path))
        agent.run(task="Run 1", context="Entry 0: red")
        agent.run(task="Run 2", context="Entry 0: blue")
        agent.close()

        events = _read_events(path)
        starts = [e for e in events if e["event_type"] == "agent_start"]
        ends = [e for e in events if e["event_type"] == "agent_end"]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0]["task"] == "Run 1"
        assert starts[1]["task"] == "Run 2"

    def test_context_manager(self, tmp_path):
        """RLMAgent as context manager closes logger on exit."""
        path = tmp_path / "ctx.jsonl"
        model = FakeOrchestratorModel()
        with RLMAgent(model=model, max_steps=3, log_path=str(path)) as agent:
            agent.run(task="test", context="Entry 0: red")
        # Logger should be closed, file readable
        events = _read_events(path)
        assert any(e["event_type"] == "agent_start" for e in events)
