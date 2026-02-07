# dspy-deepagents

## Goal
Implement Deep Agents (https://github.com/langchain-ai/deepagents) natively in DSPy with **RLM-style recursive sub-agents**, leveraging the existing `dspy.rlm` module for recursion and runtime control.

## Key Assumption
DSPy already provides RLM via `dspy.predict.rlm.RLM`, so recursion orchestration should **wrap or extend** it rather than reimplementing recursion primitives.

## Conceptual Mapping
| Deep Agents Concept | DSPy Native Construct | RLM Alignment |
|---|---|---|
| Recursive sub-agents | `dspy.rlm` orchestration + DSPy `Module` hierarchy | Recursive decomposition layers |
| Agent state + memory | `Prediction` + explicit memory store | Persistent recursive context |
| Tool use | DSPy `Signature` + tool wrappers | Actions at recursion leaves |
| Reflection | DSPy `Module` for self-critique | Recursive refinement |

## Implementation Plan

### Phase 0 — Discovery & Scoping
1. Review `dspy.rlm` API surface and any recursion helpers.
2. Identify how `dspy.rlm` expects sub-agent invocation, budgets, and stop conditions.
3. Map Deep Agents features to `dspy.rlm` capabilities and gaps.

See [docs/phase-0-discovery.md](docs/phase-0-discovery.md) for the detailed Phase 0 checklist and expected outputs.

### Phase 1 — Core Recursive Agent Skeleton (RLM-backed)
1. Build a `RecursiveAgent` DSPy `Module` that **uses `dspy.rlm` for recursion control** (depth, budget, halting).
2. Define structured I/O via `Signature`s:
   - `PlanSignature` (decompose into child tasks)
   - `ExecuteSignature` (solve leaf tasks / tool calls)
   - `SynthesizeSignature` (aggregate child outputs)
3. Enforce explicit stop criteria (depth cap, budget, confidence threshold).

### Phase 2 — Agent Roles as DSPy Modules
Implement the Deep Agents role topology as composable DSPy modules:
- **PlannerAgent**: creates child tasks.
- **ExecutorAgent**: handles atomic tasks and tool calls.
- **ReviewerAgent**: validates outputs and confidence.
- **SynthesizerAgent**: merges child results.

Each role should be usable stand-alone or via `RecursiveAgent`.

### Phase 3 — Memory & Trace
1. Define a lightweight memory structure (per-node scratch + persistent history).
2. Thread memory through `dspy.rlm` recursion context.
3. Persist a structured trace (depth, node id, task, outputs, confidence).

### Phase 4 — Tooling (Leaf-Only Actions)
1. Build a tool registry and selector signature.
2. Constrain tool calls to **leaf recursion nodes**.
3. Record tool use in trace for explainability.

### Phase 5 — Reflection & Refinement
1. Add a reviewer/reflection loop for low-confidence outputs.
2. Allow re-entry into `dspy.rlm` recursion if critique indicates missing coverage.

### Phase 6 — Evaluation Harness
1. Define a small suite of hierarchical tasks (multi-step + tool-optional).
2. Compare flat DSPy baseline vs. recursive agent.
3. Add metrics: depth usage, correctness, cost, tool call rate.

## Design Principles
- **RLM as the backbone**: recursion must be driven by `dspy.rlm` rather than bespoke recursion.
- **Composable DSPy modules**: keep roles modular and testable.
- **Strict I/O schemas**: reduce prompt drift in recursion.
- **Traceability**: each node should be inspectable and reproducible.

## Next Steps
1. Inspect `dspy.rlm` API and draft the minimal `RecursiveAgent` skeleton.
2. Stub role modules and signatures.
3. Create a first evaluation example (manual test) to validate recursion flow.
