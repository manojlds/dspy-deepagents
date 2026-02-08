# dspy-deepagents

DSPy-native Deep Agents with **RLM-backed orchestration** — an equivalent of
[LangChain Deep Agents](https://github.com/langchain-ai/deepagents) built on
DSPy's Recursive Language Model (RLM) primitive.

## Core Insight

RLM's REPL-based iteration loop **is** the agent loop.  Instead of building a
separate orchestration layer and using RLM as a predictor inside it, the agent
itself is an RLM.  All four Deep Agents pillars map directly to RLM primitives:

| Deep Agents Pillar | RLM Primitive |
|--------------------|---------------|
| Agent loop / decision-making | REPL iteration loop (LLM writes code each step) |
| System prompt | RLM Signature docstring + action instructions |
| Planning (todo list) | `write_todos()` / `read_todos()` tool functions |
| Sub-agent delegation | `delegate()` tool spawning child RLM (fresh sandbox = isolation) |
| Filesystem workspace | `write_file()` / `read_file()` / `list_files()` tools |
| Context management | Context lives in REPL variables — never in the prompt |
| Halting | `SUBMIT(result=...)` when the agent decides it's done |
| Budget control | `max_iterations` + `max_llm_calls` |

## Quick Start

```python
import dspy
from dspy_deepagents import build_deep_agent

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

agent = build_deep_agent(
    sub_lm=dspy.LM("openai/gpt-4o-mini"),  # cheaper model for sub-queries
)

result = agent(task="Research the four pillars of deep agents and write a report")
print(result.result)
```

## Architecture

```
DeepAgent = RLM(DeepAgentSignature, tools=[...], max_iterations=50)
│
│  REPL sandbox (Deno/Pyodide WASM)
│  ├── task, context     → Python str variables
│  │
│  ├── write_todos()     → planning tool
│  ├── read_todos()      → planning tool
│  ├── write_file()      → workspace tool
│  ├── read_file()       → workspace tool
│  ├── list_files()      → workspace tool
│  ├── delegate()        → spawns child RLM (isolated sandbox)
│  │
│  ├── llm_query()       → built-in: semantic reasoning
│  ├── llm_query_batched() → built-in: parallel queries
│  └── SUBMIT(result=...)  → built-in: halt and return
│
└── sub_lm → cheaper model for llm_query() calls
```

## API

### `build_deep_agent(...) -> RLM`

Factory function that wires the four pillar tools into a single RLM agent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `type[Signature]` | `DeepAgentSignature` | Agent's system prompt / IO contract |
| `workspace` | `Workspace` | auto temp dir | Shared filesystem for artifacts |
| `max_depth` | `int` | `3` | Maximum sub-agent nesting depth |
| `current_depth` | `int` | `0` | Current nesting level |
| `max_iterations` | `int` | `50` | REPL iteration budget |
| `max_llm_calls` | `int` | `80` | Sub-LLM call budget |
| `sub_lm` | `dspy.LM` | `None` | Cheaper model for `llm_query()` |
| `extra_tools` | `list[Callable]` | `None` | Domain-specific tool functions |
| `include_review` | `bool` | `False` | Add `review_draft` cross-agent review tool |

### Signatures

- **`DeepAgentSignature`** — General-purpose agent with detailed workflow instructions
- **`ResearchAgentSignature`** — Research-focused sub-agent
- **`ReviewSignature`** — Critical reviewer for cross-agent review

### Tool Classes

- **`TodoStore`** — Planning tool (`write_todos` / `read_todos`)
- **`Workspace`** — Shared filesystem (`write_file` / `read_file` / `list_files`)
- **`SubAgentDelegator`** — Sub-agent spawning with context isolation
- **`make_review_tool()`** — Factory for independent reviewer tool

## Examples

- `examples/basic_recursive_agent.py` — Simplest deep agent usage
- `examples/tool_recursive_agent.py` — Custom domain tools
- `examples/review_recursive_agent.py` — Cross-agent review
- `examples/deep_research_agent.py` — Research workflow with Wikipedia
- `examples/eval_harness.py` — Multi-task evaluation

## Design Principles

- **RLM as the backbone**: the agent loop is RLM's REPL loop, not custom Python recursion
- **Context isolation**: each sub-agent gets a fresh sandbox; only `SUBMIT()` return values cross boundaries
- **Workspace as shared memory**: files persist across agents; REPL state does not
- **DSPy-native optimization**: Signature docstrings are optimizable via `MIPROv2`
- **Minimal code**: ~150 lines for the complete agent (vs. ~330 in the previous architecture)
