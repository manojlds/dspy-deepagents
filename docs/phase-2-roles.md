# Phase 2 — Agent Roles as DSPy Modules

This phase implements the Deep Agents role topology as composable DSPy modules.

## What’s Implemented
- `PlannerAgent`, `ExecutorAgent`, `ReviewerAgent`, and `SynthesizerAgent` in
  `dspy_deepagents/roles.py`.
- Role-specific signatures with explicit I/O fields.
- `ExecutorAgent` defaults to RLM-backed execution while preserving signature-based
  execution when a predictor is supplied.

## Notes
- Each role can be instantiated independently or supplied to `RecursiveAgent`.
- Signatures are exported at the package root for customization.
