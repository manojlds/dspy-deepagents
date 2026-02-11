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
| Filesystem | `list_dir()` / `grep()` / `glob_search()` / `read_file_lines()` / `stat()` / `replace_lines()` + workspace `write_file()` / `read_file()` |
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

### Exploring a codebase

Every deep agent gets filesystem tools by default.  Point `root` at a
codebase to let the agent explore it:

```python
agent = build_deep_agent(
    root="/path/to/repo",
    max_iterations=40,
    max_llm_calls=60,
    max_depth=2,
)

result = agent(
    task="Analyze the architecture and design patterns of this codebase.",
)
print(result.result)
```

## Architecture

```
DeepAgent = RLM(DeepAgentSignature, tools=[...], max_iterations=50)
│
│  REPL sandbox (Deno/Pyodide WASM)
│  ├── task, context        → Python str variables
│  │
│  ├── Filesystem tools (rooted at root or workspace):
│  │   ├── list_dir()       → paginated directory listing
│  │   ├── glob_search()    → file pattern matching
│  │   ├── grep()           → bounded text search with context
│  │   ├── read_file_lines()→ line-range file reading (numbered)
│  │   ├── stat()           → file/dir metadata (size, lines)
│  │   └── replace_lines()  → surgical line-level edits
│  │
│  ├── write_todos()        → planning tool
│  ├── read_todos()         → planning tool
│  ├── write_file()         → workspace tool
│  ├── read_file()          → workspace tool
│  ├── list_files()         → workspace tool
│  ├── delegate()           → spawns child RLM (isolated sandbox)
│  │
│  ├── llm_query()          → built-in: semantic reasoning
│  ├── llm_query_batched()  → built-in: parallel queries
│  └── SUBMIT(result=...)   → built-in: halt and return
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
| `root` | `str \| Path` | workspace root | Root directory for filesystem tools |
| `max_depth` | `int` | `3` | Maximum sub-agent nesting depth |
| `current_depth` | `int` | `0` | Current nesting level |
| `max_iterations` | `int` | `50` | REPL iteration budget |
| `max_llm_calls` | `int` | `80` | Sub-LLM call budget |
| `sub_lm` | `dspy.LM` | `None` | Cheaper model for `llm_query()` |
| `extra_tools` | `list[Callable]` | `None` | Domain-specific tool functions |
| `include_review` | `bool` | `False` | Add `review_draft` cross-agent review tool |

### Signatures

- **`DeepAgentSignature`** — General-purpose agent with filesystem, planning, delegation, and workspace tools
- **`ResearchAgentSignature`** — Research-focused sub-agent
- **`ReviewSignature`** — Critical reviewer for cross-agent review

### Tool Classes

- **`FilesystemTools`** — Paginated filesystem middleware (`list_dir` / `glob_search` / `grep` / `read_file_lines` / `stat` / `replace_lines`)
- **`TodoStore`** — Planning tool (`write_todos` / `read_todos`)
- **`Workspace`** — Shared filesystem for artifacts (`write_file` / `read_file` / `list_files`)
- **`SubAgentDelegator`** — Sub-agent spawning with context isolation
- **`make_review_tool()`** — Factory for independent reviewer tool

## Examples

- `examples/basic_recursive_agent.py` — Simplest deep agent usage
- `examples/tool_recursive_agent.py` — Custom domain tools
- `examples/review_recursive_agent.py` — Cross-agent review
- `examples/deep_research_agent.py` — Research workflow with Wikipedia
- `examples/deep_coding_agent.py` — Codebase exploration and architecture analysis
- `examples/eval_harness.py` — Multi-task evaluation

## Design Principles

- **RLM as the backbone**: the agent loop is RLM's REPL loop, not custom Python recursion
- **Filesystem as a core pillar**: every agent gets paginated, bounded file tools by default
- **Context isolation**: each sub-agent gets a fresh sandbox; only `SUBMIT()` return values cross boundaries
- **Workspace as shared memory**: files persist across agents; REPL state does not
- **DSPy-native optimization**: Signature docstrings are optimizable via `MIPROv2`
