"""M1 semantic test: RLMAgent uses llm_query_batched for classification."""

from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent

agent = RLMAgent(
    model=LiteLLMModel(model_id="gpt-4.1-mini"),
    sub_model=LiteLLMModel(model_id="gpt-4.1-nano"),
    max_output_length=3000,
    verbosity_level=2,
    max_steps=10,
)

# Mix of positive and negative reviews — needs semantic understanding
reviews = [
    "The food was absolutely divine, best meal I've had in years",
    "Terrible service, waited 45 minutes and the waiter was rude",
    "Not bad, but nothing special. Average experience overall",
    "I would give this place zero stars if I could. Disgusting",
    "Wonderful atmosphere and the dessert was to die for",
    "The pasta was overcooked and the sauce tasted like ketchup",
    "Perfect date night spot. Romantic lighting and great wine list",
    "Found a hair in my soup. Manager didn't even apologize",
    "Exceeded all expectations! The chef came out to greet us personally",
    "Mediocre at best. Overpriced for what you get",
    "The sushi was so fresh, you could taste the ocean",
    "Parking was a nightmare and the hostess was condescending",
]

context = "\n".join(f"Review {i+1}: {r}" for i, r in enumerate(reviews))

result = agent.run(
    task=(
        "Classify each review as POSITIVE, NEGATIVE, or NEUTRAL. "
        "Return a Python dict mapping review number (int) to sentiment (str). "
        "Use llm_query_batched to classify them."
    ),
    context=context,
)
print(f"\nResult: {result}")
print(f"Type: {type(result)}")
