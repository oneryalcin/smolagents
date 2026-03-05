"""
RLM (Recursive Language Model) tools for sub-LLM calls.

These tools give a CodeAgent the ability to delegate semantic analysis
to sub-LLM calls while keeping the orchestration in Python code.

Note: LLMQueryBatchedTool calls model.generate() from multiple threads.
The Model implementation must be thread-safe for concurrent generate() calls.
Most HTTP-based clients (LiteLLM, OpenAI SDK) are thread-safe.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.tools import Tool


class LLMQueryTool(Tool):
    """Query a sub-LLM for semantic analysis of a chunk of text."""

    name = "llm_query"
    description = (
        "Query a language model for semantic analysis (classification, extraction, summarization). "
        "Use Python string ops (in, re, len) for pattern/counting tasks — they're free and instant."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The prompt including both instructions AND context to analyze.",
        }
    }
    output_type = "string"

    def __init__(self, model: Model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, prompt: str) -> str:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = self.model.generate(messages)
        return response.content or ""


class LLMQueryBatchedTool(Tool):
    """Query a sub-LLM with multiple prompts in parallel."""

    name = "llm_query_batched"
    description = (
        "Query a language model with multiple prompts in PARALLEL. "
        "Much faster than calling llm_query in a loop. Returns a list of responses "
        "in the same order as the input prompts."
    )
    inputs = {
        "prompts": {
            "type": "array",
            "description": "List of prompt strings to process in parallel.",
        }
    }
    output_type = "array"

    def __init__(self, model: Model, max_workers: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.max_workers = max_workers

    def forward(self, prompts: list) -> list:
        if not prompts:
            return []

        n = len(prompts)

        def _query_one(prompt: str) -> str:
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = self.model.generate(messages)
            return response.content or ""

        results = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as executor:
            futures = {executor.submit(_query_one, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        return [results[i] for i in range(n)]
