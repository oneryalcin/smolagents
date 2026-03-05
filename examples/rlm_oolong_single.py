"""Run RLMAgent on a single OolongBench question with full JSONL tracing."""

import argparse
import json
from datasets import load_dataset
from smolagents import LiteLLMModel
from smolagents.rlm import RLMAgent

# --- Config ---
parser = argparse.ArgumentParser()
parser.add_argument("idx", type=int, nargs="?", default=0)
parser.add_argument("--model", default="gpt-4.1-mini")
parser.add_argument("--sub-model", default="gpt-4.1-nano")
parser.add_argument("--unlabeled", action="store_true")
args = parser.parse_args()

IDX = args.idx
LOG_PATH = f"/tmp/rlm_oolong_idx{IDX}.jsonl"
MODEL = args.model
SUB_MODEL = args.sub_model
USE_UNLABELED = args.unlabeled

# --- Load dataset ---
ds = load_dataset("oolongbench/oolong-synth", split="test")
ex = ds[IDX]

context_key = "context_window_text" if USE_UNLABELED else "context_window_text_with_labels"
context = ex[context_key]

print(f"=== OolongBench idx={IDX} ===")
print(f"Task group: {ex['task_group']} | task: {ex['task']} | dataset: {ex['dataset']}")
print(f"Context: {len(context):,} chars ({context_key})")
print(f"Model: {MODEL} | Sub-model: {SUB_MODEL}")
print(f"Question: {ex['question']}")
print(f"Expected: {ex['answer']}")
print(f"Log: {LOG_PATH}")
print("=" * 40)

# --- Run ---
with RLMAgent(
    model=LiteLLMModel(model_id=MODEL),
    sub_model=LiteLLMModel(model_id=SUB_MODEL),
    log_path=LOG_PATH,
    max_steps=15,
    verbosity_level=2,
) as agent:
    result = agent.run(
        task=ex["question"],
        context=context,
    )

print(f"\n{'=' * 40}")
print(f"Result:   {result}")
print(f"Expected: {ex['answer']}")

# --- Print JSONL trace summary ---
print(f"\n--- JSONL Trace ({LOG_PATH}) ---")
with open(LOG_PATH) as f:
    for line in f:
        evt = json.loads(line)
        t = evt["event_type"]
        if t == "agent_start":
            print(f"  [{t}] task={evt.get('task', '')[:80]}")
        elif t == "execution_result":
            code = (evt.get("code") or "")[:200]
            out = (evt.get("output") or "")[:200]
            print(f"  [{t}] step={evt.get('step')} err={evt.get('has_error')}")
            print(f"    code: {code}")
            print(f"    output: {out}")
        elif t == "llm_call":
            prompt = (evt.get("prompt") or "")[:100]
            resp = (evt.get("response") or "")[:100]
            print(f"  [{t}] prompt={prompt}")
            print(f"    response={resp}")
        elif t == "final_result":
            print(f"  [{t}] result={evt.get('result')}")
        elif t == "agent_end":
            print(f"  [{t}] success={evt.get('success')}")
