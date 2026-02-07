# Phase 4 — Tooling (Leaf-Only Actions)

This phase introduces leaf-only tool routing.

## What’s Implemented
- `Tool` and `ToolRegistry` in `dspy_deepagents/tools.py`.
- `ToolSelectSignature` and `ToolSelectorAgent` for tool selection (RLM-backed by
  default).
- `RecursiveAgent` performs tool selection and invocation only at leaf nodes and
  records tool usage in the trace and memory.
