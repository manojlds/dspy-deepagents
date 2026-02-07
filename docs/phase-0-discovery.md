# Phase 0 — Discovery & Scoping

This phase focuses on **understanding DSPy's existing `dspy.rlm` module** so we can implement Deep Agents recursion by extending/leveraging those primitives rather than re-implementing them.

## Goals
- Identify the public API surface for `dspy.rlm` (classes, functions, configs).
- Understand how recursion depth, budgets, and halting conditions are expressed.
- Map Deep Agents concepts to `dspy.rlm` constructs and note gaps.

## Checklist
- [x] Locate `dspy.rlm` source within the DSPy package.
- [x] Document the primary entry points (modules, classes, functions).
- [x] Record recursion control parameters (depth, budget, termination signals).
- [x] Note how state/memory is threaded through recursion.
- [x] Capture any utilities or hooks for tracing.
- [x] Summarize integration points for Deep Agents roles (planner, executor, reviewer, synthesizer).

## Findings

### API Surface & Source Locations
- `dspy.predict.rlm.RLM` is the primary entry point (exported via `dspy.predict`).
- Source file: `dspy/predict/rlm.py` in the DSPy package installation.
- The module is implemented as a DSPy `Module` and uses a sandboxed REPL to drive iterative reasoning and sub-LLM calls.

### Control Parameters & Halting
- `max_iterations`: limits the number of REPL interaction loops.
- `max_llm_calls`: caps `llm_query`/`llm_query_batched` usage to bound recursion-like expansion.
- `max_output_chars`: bounds REPL output size.
- Halting is explicit via the `SUBMIT(...)` function inside the REPL; if a submit never happens, iteration can continue until max limits are hit.

### State & Memory
- RLM uses a persistent REPL state across iterations within a single `forward()` call.
- State is represented by interpreter history (e.g., variables, prints) rather than an explicit memory object.

### Tools & Extensions
- Built-in REPL tools: `llm_query(prompt)` and `llm_query_batched(prompts)`.
- User tools can be registered; validation enforces naming and signature constraints.

### Tracing & Observability
- `verbose` logging is available, but there is no first-class recursion trace tree.
- Any deeper traceability must be layered on top (e.g., storing node-level summaries in a custom memory object).

## Mapping to Deep Agents & Gaps
| Deep Agents Feature | `dspy.rlm` Capability | Gap/Needed Work |
|---|---|---|
| Recursive sub-agents | Sub-LLM calls via `llm_query` | No explicit agent tree or role modules |
| Budgeted recursion | `max_iterations`, `max_llm_calls` | No depth-based recursion budget |
| Agent roles (planner/executor/reviewer) | None directly | Must define DSPy `Module` roles on top |
| Memory / state | REPL variable state | Need explicit shared memory + trace |
| Tooling | REPL + user tools | Need tool routing + leaf-only action guardrails |

## Decision Notes
- Use `dspy.predict.rlm.RLM` as the recursion backbone to handle iterative exploration and batched sub-LLM calls.
- Build a `RecursiveAgent` wrapper that *externally* manages depth/budget counters, roles, and memory/trace, while delegating inner calls to RLM.
- Implement tool routing and reviewer loops outside RLM to maintain explicit Deep Agents structure.

## Expected Outputs
- A short summary of the `dspy.rlm` API with links to the relevant source files.
- A mapping table of Deep Agents features to `dspy.rlm` features (with gaps).
- A decision note on what must be built on top of `dspy.rlm`.

## Next Step After Completion
Proceed to Phase 1 and implement a minimal `RecursiveAgent` skeleton that calls into `dspy.rlm` for recursion control.
