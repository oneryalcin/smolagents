"""M1 smoke test: RLMAgent on a synthetic counting task."""

from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent

agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
    max_output_length=3000,
    verbosity_level=2,
)

# 5000 entries, every 7th is "red" → floor(4999/7)+1 = 715
context = "\n".join(
    f"Entry {i}: The color is {'red' if i % 7 == 0 else 'blue'}" for i in range(5000)
)

result = agent.run(
    task="How many entries have the color red? Use Python string operations — do NOT use llm_query for this.",
    context=context,
)

print(f"\nResult: {result}")
expected = sum(1 for i in range(5000) if i % 7 == 0)
print(f"Expected: {expected}")
print(f"Correct: {result == expected or str(result) == str(expected)}")
