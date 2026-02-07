# Phase 3 — Memory & Trace

This phase adds explicit memory and structured tracing.

## What’s Implemented
- `RecursiveMemory` and `MemoryEntry` in `dspy_deepagents/memory.py`.
- `TraceEvent` in `dspy_deepagents/trace.py` with serialization via `to_dict()`.
- `RecursiveAgent` now threads a shared `RecursiveMemory` and returns serialized
  memory plus a structured trace in the prediction output.
