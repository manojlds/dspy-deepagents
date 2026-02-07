# Phase 5 — Reflection & Refinement

This phase adds a reviewer/reflection loop for low-confidence outputs.

## What’s Implemented
- `ReviewerAgent` with `ReviewSignature` for critique and retry decisions.
- Configurable thresholds and round limits in `RecursionConfig`.
- `RecursiveAgent` optionally refines low-confidence outputs based on reviewer feedback
  and records review/refine events in the trace.
