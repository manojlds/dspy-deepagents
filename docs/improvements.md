# Improvements: dspy-deepagents

Analysis comparing this implementation against the RLM paper (arXiv:2512.24601),
LangChain deepagents (github.com/langchain-ai/deepagents), and DSPy RLM API.

## What's Good

- Core architectural decision is correct -- RLM as the agent backbone, not a
  wrapper around RLM per-role
- Clean, minimal code (~400 lines for the full library)
- Proper context isolation via RLM's fresh sandbox per sub-agent
- Well-documented design rationale in `review-and-plan.md`
- Path traversal protection on workspace
- The four-pillar mapping (system prompts, planning, delegation, workspace) is
  complete and idiomatic DSPy
- The paper's three key design choices (Algorithm 1 vs Algorithm 2) are all
  satisfied:
  1. Prompt as symbolic REPL variable (not in LLM context window)
  2. Symbolic recursion (code can invoke sub-LLM calls programmatically)
  3. Response via REPL variable (`SUBMIT()`, not autoregressive generation)

---

## Critical Issues

### 1. Child sub-agents don't inherit `extra_tools`, `signature`, or `sub_lm`

**File:** `dspy_deepagents/tools.py:131-137`

When `SubAgentDelegator.delegate()` creates a child, it only passes `workspace`,
`max_depth`, `current_depth`, `max_iterations`, and `max_llm_calls`. It does NOT
forward `extra_tools`, `include_review`, `signature`, or `sub_lm`.

This means:
- A research agent with `wikipedia_summary` as an extra tool **cannot** delegate
  to a sub-agent that also has `wikipedia_summary`
- Sub-agents always use `DeepAgentSignature` even if the parent uses
  `ResearchAgentSignature`
- Sub-agents never get a `sub_lm` even if the parent has one

**Impact:** The deep research example is effectively broken -- the parent
delegates subtopics to sub-agents, but those sub-agents can't use
`wikipedia_summary`.

**Status:** Fixed.

### 2. No async support / no parallel sub-agent execution

The paper explicitly notes that "asynchronous sub-calls can potentially
significantly reduce the runtime and inference cost of RLMs." RLM has
`aforward()` for async. LangChain deepagents supports parallel sub-agent
execution. The current implementation is entirely synchronous.

**Status:** Open.

### 3. Hard-coded child budgets

Child agents always get `max_iterations=30, max_llm_calls=40` regardless of the
parent's budget. A parent with `max_iterations=100` allocates children the same
budget as one with `max_iterations=20`.

**Status:** Open.

---

## Important Improvements

### 4. No DSPy optimization integration (Phase 3 from plan)

`review-and-plan.md` Phase 3 calls for `MIPROv2` optimization of Signature
docstrings. This is the **unique DSPy advantage** over LangChain and is
completely unimplemented:
- No metric function definitions
- No training data / examples
- No optimizer instantiation
- No optimized agent export

**Status:** Open.

### 5. Deep research example is too basic

LangChain's deep research example has Tavily web search (real search + URL
fetching + markdown conversion), parallel sub-agent delegation, strategic
reflection via `think_tool`, hard limits on search counts, structured output to
files, and a 3-phase workflow.

The dspy-deepagents example only uses Wikipedia REST API summaries with
sequential delegation. It should be upgraded with:
- Real web search (Tavily or similar)
- Parallel delegation
- Structured research workflow with file-based artifacts
- Use `ResearchAgentSignature` for sub-agents

**Status:** Open.

### 6. `ResearchAgentSignature` is defined but never used

Exported in `__init__.py` but never referenced by `build_deep_agent()` or any
example. The deep research example should use it for sub-agents.

**Status:** Open.

### 7. No error handling in `TodoStore.write_todos`

`json.loads(todos_json)` can throw `json.JSONDecodeError` if the LLM passes
invalid JSON (common). Should catch and return a helpful error message.

**Status:** Open.

### 8. Missing `edit_file` and `grep` in Workspace

LangChain deepagents provides `read_file`, `write_file`, `edit_file`, `ls`,
`glob`, `grep`, `execute`. The current `Workspace` only has `write_file`,
`read_file`, `list_files`. For research agents that iteratively refine
documents, `edit_file` and `grep` are useful.

**Status:** Open.

### 9. `max_output_chars` and `verbose` not exposed in factory

RLM accepts `max_output_chars=100_000` and `verbose=False`. Neither is exposed
in `build_deep_agent()`. For agents processing large outputs or debugging,
these matter.

**Status:** Open.

---

## Minor Issues

### 10. Path traversal check uses string prefix matching

```python
if not str(target).startswith(str(self.root.resolve())):
```

`/workspace_evil` would pass a check for root `/workspace`. Should use
`Path.is_relative_to()` (Python 3.9+).

**Status:** Open.

### 11. Tests only verify construction, never execution

All tests are structural. No integration tests that actually run an agent with
an LLM. Acknowledged in `review-and-plan.md` Phase 5 but not implemented.

**Status:** Open.

### 12. `.tools` property returns `dict`, tests treat as list

RLM's `.tools` property returns `dict[str, Tool]`. The tests use
`len(agent.tools) >= 6` which works but the semantic assumption is fragile.
Should verify specific tool names exist.

**Status:** Open.

### 13. No typing validation on TodoStore entries

`write_todos` accepts any JSON -- `[1, 2, 3]` or `{"foo": "bar"}` would be
silently accepted. Should validate the expected schema.

**Status:** Open.

---

## Comparison Summary

| Feature | RLM Paper | LangChain deepagents | dspy-deepagents |
|---------|-----------|---------------------|-----------------|
| Prompt as REPL variable | Core design | N/A (not RLM) | Correct |
| Symbolic recursion | Core design | N/A | Correct |
| Context isolation | Fresh REPL per sub-call | Isolated LangGraph sub-agents | Correct |
| Sub-agent tool inheritance | Implicit (same setup) | Configurable per sub-agent | **Broken (fixed)** |
| Async/parallel sub-calls | Recommended | Supported | Missing |
| Prompt optimization | N/A | None | Planned but missing |
| Workspace tools | N/A | Full (7 tools) | Basic (3 tools) |
| Deep research example | BrowseComp+ benchmark | Full (Tavily + reflection) | Basic (Wikipedia) |
| Production readiness | Research prototype | Production (9.1k stars) | Prototype |
