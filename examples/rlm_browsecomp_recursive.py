"""Run RLMAgent on BrowseComp-Plus with recursive=True.

Tests whether the agent uses rlm_query to delegate doc analysis to child agents.
Builds context from gold + negative docs for a given query.

Usage:
    uv run examples/rlm_browsecomp_recursive.py 296
    uv run examples/rlm_browsecomp_recursive.py 296 --model anthropic/claude-sonnet-4-20250514
    uv run examples/rlm_browsecomp_recursive.py 296 --no-recursive  # baseline comparison
"""

import argparse
import json
import random

from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent
from smolagents.rlm_tools import Budget

# --- Config ---
parser = argparse.ArgumentParser()
parser.add_argument("idx", type=int, nargs="?", default=296)
parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514")
parser.add_argument("--sub-model", default="gemini/gemini-2.0-flash-lite")
parser.add_argument("--no-recursive", action="store_true", help="Disable recursion (baseline)")
parser.add_argument("--max-depth", type=int, default=2)
parser.add_argument("--max-steps", type=int, default=20)
parser.add_argument("--max-child-steps", type=int, default=10)
parser.add_argument("--budget", type=int, default=None, help="Max sub-LLM calls")
parser.add_argument("--verbose-reasoning", action="store_true", help="Log structured reasoning traces")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

IDX = args.idx
RECURSIVE = not args.no_recursive
LOG_PATH = f"/tmp/rlm_browsecomp_idx{IDX}_{'recursive' if RECURSIVE else 'flat'}.jsonl"

# --- Load query ---
# Decrypted JSONL has gold_docs/negative_docs as dicts with {docid, text, url}
queries_path = "/tmp/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"
with open(queries_path) as f:
    queries = [json.loads(line) for line in f]

query = queries[IDX]
print(f"=== BrowseComp-Plus idx={IDX} ===")
print(f"Query: {query['query'][:200]}...")
print(f"Answer: {query['answer']}")
print(f"Gold docs: {len(query['gold_docs'])} | Neg docs: {len(query['negative_docs'])}")

# --- Build context from gold + negative docs (text is inline) ---
all_docs = query["gold_docs"] + query["negative_docs"]
docs = [f"=== Document {d['docid']} ===\n{d['text']}" for d in all_docs]

random.seed(args.seed)
random.shuffle(docs)
context = "\n\n".join(docs)

print(f"Context: {len(docs)} docs, {len(context):,} chars (~{len(context)//4:,} tokens)")
print(f"Model: {args.model} | Sub-model: {args.sub_model}")
print(f"Recursive: {RECURSIVE} | max_depth: {args.max_depth}")
print(f"Log: {LOG_PATH}")
print("=" * 60)

# --- Run ---
budget = Budget(max_llm_calls=args.budget) if args.budget else None

with RLMAgent(
    model=LiteLLMModel(model_id=args.model),
    sub_model=LiteLLMModel(model_id=args.sub_model),
    log_path=LOG_PATH,
    max_steps=args.max_steps,
    budget=budget,
    recursive=RECURSIVE,
    max_depth=args.max_depth,
    max_child_steps=args.max_child_steps,
    verbose_reasoning=args.verbose_reasoning,
    verbosity_level=2,
) as agent:
    result = agent.run(task=query["query"], context=context)

print(f"\n{'=' * 60}")
print(f"Result:   {result}")
print(f"Expected: {query['answer']}")
correct = str(result).lower().strip() in query["answer"].lower()
print(f"Correct:  {correct}")

# --- Summarize JSONL trace ---
print(f"\n--- Trace Summary ({LOG_PATH}) ---")
with open(LOG_PATH) as f:
    events = [json.loads(line) for line in f]

rlm_calls = sum(1 for e in events if e["event_type"] == "execution_result"
                and e.get("code") and "rlm_query" in e.get("code", ""))
llm_calls = sum(1 for e in events if e["event_type"] == "llm_call")
child_steps = sum(1 for e in events if e["event_type"] == "execution_result" and e.get("depth", 0) > 0)
root_steps = sum(1 for e in events if e["event_type"] == "execution_result" and e.get("depth", 0) == 0)

print(f"  Root steps: {root_steps}")
print(f"  Child steps: {child_steps}")
print(f"  rlm_query calls (in code): {rlm_calls}")
print(f"  Sub-LLM calls: {llm_calls}")
print(f"  Total events: {len(events)}")
