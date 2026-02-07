# Phase 1 — Recursive Agent Skeleton (RLM-backed)

This phase introduces the first working DSPy module that wires together planning,
execution, and synthesis while delegating leaf execution to `dspy.predict.rlm.RLM`.

## What’s Implemented
- `dspy_deepagents.recursive_agent.RecursiveAgent`:
  - Depth/budget controls via `RecursionConfig`.
  - Planner → recursive calls → synthesizer flow.
  - Leaf execution uses `RLM("context, query -> output", max_iterations=10)`.
  - Minimal trace capturing plan/execute/synthesize events.

## What’s Still Missing
The remaining phases (tooling, memory/trace, reviewer loops, and evaluation harness)
are implemented in the Phase 2–6 notes and code. See `docs/phase-2-roles.md` through
`docs/phase-6-evaluation.md` for summaries.

## Usage Sketch
```python
import dspy
from dspy_deepagents import RecursiveAgent, RecursionConfig

agent = RecursiveAgent(config=RecursionConfig(max_depth=2, budget=5))
result = agent(task="Summarize the report", context="...", memory="")
print(result.result)
```

## Next Step
Extend the leaf execution path to support tool routing and implement explicit memory/trace types.
