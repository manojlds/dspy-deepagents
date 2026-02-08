# Critique & Reimplementation Plan: dspy-deepagents

## Part 1: Critique of the Current Implementation

### 1. RLM Used at the Wrong Level

The most significant issue: `dspy.RLM` is used as a drop-in replacement for
`dspy.Predict` on every individual agent role, rather than as the overall agent
orchestration backbone.

**What RLM actually does:** RLM stores inputs as Python variables in a sandboxed
REPL (Deno/Pyodide WASM). The LLM writes Python code to iteratively explore and
process data — calling `llm_query()` for semantic analysis of chunks, calling
registered tool functions, and calling `SUBMIT(...)` when done. Context lives in
REPL variables, not in the prompt. The prompt stays small; the data stays in the
interpreter's memory.

**How the current code uses it:** Every agent role wraps its Signature in RLM:

```python
# roles.py:82 — PlannerAgent
self.predictor = rlm or RLM(PlanSignature, max_iterations=10)

# roles.py:101 — ExecutorAgent
self.predictor = rlm or RLM(ExecuteSignature, max_iterations=10)

# ... same for Reviewer, Synthesizer, ToolSelector
```

This creates 5+ separate REPL sandboxes per recursion node, each processing short
strings that don't need the context-as-variable pattern. Meanwhile, the *outer*
orchestration — where context actually accumulates across steps — uses plain Python
recursion (`_run_node()`) with no RLM at all.

**The fix is architectural:** RLM should be the agent loop itself. One RLM per agent.
The REPL iterations *are* the agent steps. Tools (planning, filesystem, delegation)
are registered as RLM tool functions. Context lives in REPL variables, never exploding
the prompt. Sub-agents are nested RLM calls — each gets a fresh sandbox, giving
context isolation for free.

### 2. Misunderstanding of What LangChain Deep Agents Actually Is

LangChain's `deepagents` is built around **four architectural pillars**, not recursive
task decomposition:

| Pillar | LangChain deepagents | Current dspy-deepagents |
|--------|---------------------|------------------------|
| **Detailed system prompts** | First-class component with tool-usage instructions and few-shot examples | Missing — Signatures have 1-line descriptions |
| **Planning tool** | Todo-list (`write_todos`/`read_todos`) as a context-engineering "no-op" | Replaced with rigid hierarchical decomposition tree |
| **Sub-agent spawning** | Isolated sub-agents with **context isolation** | Shared `RecursiveMemory` across all nodes |
| **Filesystem access** | Shared workspace (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`) | Missing entirely |

The current implementation reduces all four pillars to a rigid
plan-execute-synthesize-review tree, missing the core insight: deep agents are
*context management strategies*, not hierarchical task planners.

### 3. No Sub-Agent Context Isolation

LangChain's deep agents spawn sub-agents with **isolated context windows**. The parent
delegates a focused task, the sub-agent works without seeing (or polluting) the parent's
full context, and only the result flows back up.

The current implementation does the opposite — a single `RecursiveMemory` object is
shared across every node at every depth:

```python
# recursive_agent.py:67
state = state or RecursiveMemory()
result, confidence = self._run_node(task=task, ..., state=state, ...)
```

Every planner, executor, reviewer, and synthesizer reads and writes to the same log.
There is no isolation and no mechanism to prevent context from growing unboundedly.

With RLM-native architecture, context isolation comes for free: each sub-agent is a
fresh `RLM` call with its own sandboxed REPL. The parent's REPL variables are invisible
to the child. Only the `SUBMIT()` return value crosses the boundary.

### 4. Memory Is an Append-Only Log

`RecursiveMemory` is a list with `.serialize(max_entries=12)` that dumps the last 12
entries as a flat string. Problems:

- **No relevance** — last 12 entries may be from unrelated subtasks
- **No scoping** — a leaf at depth 3 sees depth-0 planner entries mixed with depth-2
  executor entries from a different subtree
- **No workspace** — agents can't write intermediate artifacts

With RLM-native architecture, "memory" is just Python variables in the REPL — dicts,
lists, files on the sandbox's virtual filesystem. The LLM accesses what it needs via
code, not by reading a serialized string.

### 5. Self-Assessed Confidence Is Unreliable

The review loop is driven by LLM-generated confidence scores:

```python
if confidence >= self.config.review_threshold:
    return draft, confidence
```

LLMs are poorly calibrated at self-assessment. A wrong answer with 0.95 confidence
skips review. A correct answer with 0.5 wastes tokens on unnecessary refinement.

### 6. No DSPy Optimization

The point of DSPy is programmatic prompt/pipeline optimization via `BootstrapFewShot`,
`MIPROv2`, etc. The current implementation uses none:

- No training data or examples
- No optimizer instantiated
- No metric functions defined
- `eval_harness.py` prints results with no ground truth or scoring

### 7. Rigid Tree vs. Dynamic Loop

The agent follows a fixed structure at every level:

```
PlannerAgent → [child₁, child₂, ...] → SynthesizerAgent → ReviewerAgent
```

It cannot skip planning, re-plan based on results, or dynamically decide its next
action. LangChain's deep agents run a dynamic loop where the agent chooses what to
do next at each step.

### 8. Budget Mechanism Is Flawed

```python
for sub_task in sub_tasks:
    child_result, child_confidence = self._run_node(
        ..., budget=budget - 1, ...  # same value for every sibling
    )
```

Each sibling gets `budget - 1`, not a globally-decremented counter. Three siblings
each think they have the same remaining budget.

### 9. Tool System Is Too Simplistic

Tools are `Callable[[str], str]` — no structured parameters, no JSON schemas, no error
handling, no timeouts. RLM's native tool system already supports typed function
signatures with automatic validation — the custom `ToolRegistry` reinvents a weaker
version of what RLM provides out of the box.

### 10. Tests Don't Test Anything Meaningful

Static mocks bypass all DSPy/RLM functionality:

```python
class StaticPlanner(PlannerAgent):
    def forward(self, task, context, memory):
        return dspy.Prediction(sub_tasks=["child-task"])
```

These test control flow with hardcoded returns, not actual agent behavior.

---

## Part 2: Reimplementation Plan — RLM-Native Deep Agents

### Core Insight: RLM IS the Agent Loop

The key realization: RLM's REPL-based iteration loop *is* the agent loop. Instead of
building a separate orchestration layer and using RLM as a predictor inside it, the
agent itself should be an RLM. All four deep agents pillars map to RLM primitives:

| Deep Agents Pillar | RLM Primitive |
|--------------------|---------------|
| Agent loop / decision-making | REPL iteration loop (LLM writes code each step) |
| System prompt | RLM Signature docstring + action instructions |
| Planning (todo list) | Tool function (`write_todos`, `read_todos`) registered with RLM |
| Sub-agent delegation | Tool function that spawns a child RLM (fresh sandbox = isolation) |
| Filesystem workspace | Tool functions (`write_file`, `read_file`, etc.) registered with RLM |
| Context management | Context lives in REPL variables — never in the prompt |
| Halting | `SUBMIT(result=...)` when the agent decides it's done |
| Budget control | `max_iterations` + `max_llm_calls` |

### Architecture Overview

```
DeepAgent = RLM(DeepAgentSignature, tools=[...], max_iterations=50)
│
│  REPL sandbox (Deno/Pyodide WASM)
│  ├── task          → Python str variable (never re-injected into prompt)
│  ├── results       → Python dict accumulating findings
│  ├── todo_list     → Python list the LLM manages via code
│  │
│  ├── write_todos()   → registered tool function
│  ├── read_todos()    → registered tool function
│  ├── write_file()    → registered tool function
│  ├── read_file()     → registered tool function
│  ├── edit_file()     → registered tool function
│  ├── list_files()    → registered tool function
│  ├── delegate()      → registered tool: spawns child RLM (isolated sandbox)
│  ├── web_search()    → registered tool function (optional)
│  │
│  ├── llm_query()          → built-in: semantic reasoning on chunks
│  ├── llm_query_batched()  → built-in: parallel semantic queries
│  └── SUBMIT(result=...)   → built-in: halt and return
│
└── sub_lm → cheaper model for llm_query() calls
```

The LLM writes Python code at each iteration to inspect state, call tools, query
sub-LLMs, and build up its answer. Context never enters the prompt — it's in variables.
When done, the LLM calls `SUBMIT(result=final_answer)`.

Sub-agent delegation creates a **child RLM** with its own fresh sandbox. The parent's
REPL state is invisible to the child. Only the child's `SUBMIT()` return value flows
back to the parent. Context isolation is a structural guarantee, not a convention.

---

### Phase 0: The DeepAgent Signature

Define the agent's input/output contract as a DSPy Signature. The docstring becomes
RLM's system prompt — this is where Pillar 1 (detailed system prompts) lives.

```python
class DeepAgentSignature(dspy.Signature):
    """You are a deep research and execution agent. You work iteratively by
    writing Python code to accomplish your task. You have access to a set of
    tools and a shared workspace.

    WORKFLOW:
    1. Start by understanding the task. Call read_todos() to check for existing plans.
    2. Create a plan using write_todos() to break the task into steps.
    3. For each step, either:
       - Execute it directly using available tools and llm_query()
       - Delegate it to a sub-agent using delegate(task, context) for deep work
    4. Store intermediate results in the workspace using write_file()
    5. When all steps are complete, synthesize findings and call SUBMIT(result=...)

    TOOL USAGE:
    - write_todos(todos_json): Create/update your plan. Pass a JSON string of
      [{"content": "step description", "status": "pending|done"}]
    - read_todos(): Read current plan state
    - delegate(task, context=""): Spawn an isolated sub-agent for focused work.
      The sub-agent cannot see your state — pass what it needs via context.
      Returns the sub-agent's final result as a string.
    - write_file(path, content): Write to the shared workspace
    - read_file(path): Read from the shared workspace
    - list_files(): List workspace contents
    - llm_query(prompt): Ask a question requiring reasoning (built-in)
    - llm_query_batched(prompts): Ask multiple questions in parallel (built-in)

    IMPORTANT:
    - Use write_file() for large intermediate results instead of keeping them
      in variables (workspace persists across sub-agents too)
    - Use delegate() for tasks requiring deep focus — sub-agents get a clean
      context so they can go deep without being overwhelmed
    - Always print() results so you can see what happened before deciding
      the next step
    - Call SUBMIT(result=your_final_answer) when done
    """

    task: str = dspy.InputField(desc="The task to accomplish")
    context: str = dspy.InputField(
        desc="Optional initial context or background information",
        default="",
    )
    result: str = dspy.OutputField(desc="The final result of the task")
```

This single Signature replaces `PlanSignature`, `ExecuteSignature`,
`SynthesizeSignature`, `ReviewSignature`, and `ToolSelectSignature`. The LLM
decides when to plan, execute, synthesize, and review by writing code — not by
being routed through a fixed pipeline.

---

### Phase 1: Tool Functions (The Four Pillars)

Each pillar is implemented as one or more tool functions registered with RLM.
Tool functions must be regular Python callables with type hints — RLM extracts
their signatures automatically and validates names against reserved builtins
(`llm_query`, `llm_query_batched`, `SUBMIT`, `print`).

#### 1.1 Planning Tool (Pillar 2)

```python
import json

class TodoStore:
    """Stateful todo list — one instance per agent."""

    def __init__(self):
        self._todos: list[dict] = []

    def write_todos(self, todos_json: str) -> str:
        """Create or replace the todo list.

        Args:
            todos_json: JSON string of [{"content": "...", "status": "pending|done"}]
        """
        self._todos = json.loads(todos_json)
        return f"Updated {len(self._todos)} todos"

    def read_todos(self) -> str:
        """Read the current todo list."""
        if not self._todos:
            return "No todos yet."
        return json.dumps(self._todos, indent=2)
```

Note: RLM tools are callables, and they can be bound methods. We create a fresh
`TodoStore` per agent invocation.

#### 1.2 Filesystem Workspace (Pillar 4)

```python
from pathlib import Path

class Workspace:
    """Shared filesystem rooted at a temp directory."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the workspace."""
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        target = self.root / path
        if not target.exists():
            return f"Error: {path} not found"
        return target.read_text()

    def list_files(self) -> str:
        """List all files in the workspace."""
        files = sorted(p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file())
        if not files:
            return "Workspace is empty."
        return "\n".join(str(f) for f in files)
```

The workspace is shared across parent and sub-agents (same `root` directory),
while REPL state is isolated. This means sub-agents can read files written by the
parent and write results back — without sharing prompt context.

#### 1.3 Sub-Agent Delegation (Pillar 3)

```python
class SubAgentDelegator:
    """Spawns child RLM agents with isolated REPL sandboxes."""

    def __init__(self, workspace: Workspace, max_depth: int = 3, current_depth: int = 0):
        self.workspace = workspace
        self.max_depth = max_depth
        self.current_depth = current_depth

    def delegate(self, task: str, context: str = "") -> str:
        """Spawn an isolated sub-agent for a focused task.

        The sub-agent gets its own REPL sandbox (context isolation).
        It shares the workspace filesystem but not your variables.
        Returns the sub-agent's final result.

        Args:
            task: What the sub-agent should accomplish
            context: Optional context to pass (the sub-agent can't see your state)
        """
        if self.current_depth >= self.max_depth:
            return f"Error: max sub-agent depth ({self.max_depth}) reached. Solve directly."

        # Build child tools — same workspace, incremented depth
        child = build_deep_agent(
            workspace=self.workspace,
            max_depth=self.max_depth,
            current_depth=self.current_depth + 1,
            max_iterations=30,  # sub-agents get fewer iterations
            max_llm_calls=40,
        )

        # Execute in fresh sandbox — parent state is invisible
        result = child(task=task, context=context)
        return result.result
```

Key guarantee: `child(task=task, context=context)` creates a **new RLM forward()
call** with a **fresh PythonInterpreter**. The parent's REPL variables, history,
and state are structurally inaccessible to the child. Context isolation is enforced
by RLM's architecture, not by convention.

#### 1.4 System Prompt (Pillar 1)

The system prompt lives in the `DeepAgentSignature` docstring (see Phase 0). RLM
combines this with its own action instructions template, which tells the LLM about
available tools, the REPL environment, and the iteration protocol.

RLM's built-in action instructions already include:
```
You have access to a Python REPL environment. Write Python code and it will be executed.
Available:
- Variables: {inputs} (your input data)
- llm_query(prompt) — query a sub-LLM for semantic analysis
- llm_query_batched(prompts) — query multiple prompts concurrently
- print() — ALWAYS print to see results
- SUBMIT({output_fields}) — submit final output when done
- Additional tools: {registered_tool_descriptions}
```

The Signature docstring extends this with agent-specific workflow instructions.

---

### Phase 2: The `build_deep_agent` Factory

Wire everything together into a single RLM-backed agent:

```python
import dspy
from dspy.predict import RLM
from pathlib import Path
import tempfile


def build_deep_agent(
    workspace: Workspace | None = None,
    max_depth: int = 3,
    current_depth: int = 0,
    max_iterations: int = 50,
    max_llm_calls: int = 80,
    sub_lm: dspy.LM | None = None,
    extra_tools: list | None = None,
) -> RLM:
    """Build a deep agent as an RLM with the four pillars as tools.

    Args:
        workspace: Shared filesystem. Created as a temp dir if None.
        max_depth: Maximum sub-agent nesting depth.
        current_depth: Current nesting level (0 = root agent).
        max_iterations: REPL iteration budget (= agent step budget).
        max_llm_calls: Sub-LLM call budget for llm_query().
        sub_lm: Optional cheaper model for llm_query() calls.
        extra_tools: Additional domain-specific tool functions.

    Returns:
        An RLM module ready to call with (task=..., context=...).
    """
    if workspace is None:
        workspace = Workspace(Path(tempfile.mkdtemp(prefix="deepagent_")))

    # Pillar 2: Planning
    todo_store = TodoStore()

    # Pillar 3: Sub-agent delegation
    delegator = SubAgentDelegator(
        workspace=workspace,
        max_depth=max_depth,
        current_depth=current_depth,
    )

    # Collect all tool functions
    tools = [
        todo_store.write_todos,
        todo_store.read_todos,
        workspace.write_file,
        workspace.read_file,
        workspace.list_files,
        delegator.delegate,
    ]
    if extra_tools:
        tools.extend(extra_tools)

    return RLM(
        DeepAgentSignature,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        tools=tools,
        sub_lm=sub_lm,
    )
```

Usage:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

agent = build_deep_agent(
    sub_lm=dspy.LM("openai/gpt-4o-mini"),  # cheaper model for sub-queries
)

result = agent(task="Research the four pillars of deep agents and write a report")
print(result.result)
```

What happens at runtime:
1. RLM creates a Deno/Pyodide sandbox
2. The `task` and `context` strings are injected as Python variables
3. The LLM sees the Signature docstring (workflow instructions) + RLM's action template
4. The LLM writes code: `todos = read_todos()`, `write_file("notes.md", ...)`,
   `sub_result = delegate("research pillar 1", "focus on system prompts")`, etc.
5. Each iteration: LLM writes code → sandbox executes → LLM sees output → next step
6. Sub-agent calls create fresh sandboxes (context isolation)
7. LLM calls `SUBMIT(result="Final report: ...")` when done
8. Returns `dspy.Prediction(result="Final report: ...", trajectory=[...])`

**Lines of code:** ~150 for the complete deep agent, down from ~330 in the current
implementation, with strictly more capability.

---

### Phase 3: DSPy Optimization

With RLM as the agent backbone, DSPy optimization applies at three levels:

#### 3.1 Signature-Level Optimization

The `DeepAgentSignature` docstring is the system prompt. DSPy's `MIPROv2` optimizer
can rewrite it to improve agent behavior:

```python
from dspy.teleprompt import MIPROv2

def agent_metric(example, prediction, trace=None):
    """Score agent output quality and efficiency."""
    # Correctness: does the output address the task?
    correct = dspy.evaluate.SemanticF1()(example.result, prediction.result)

    # Efficiency: trajectory length relative to budget
    steps = len(prediction.get("trajectory", []))
    efficiency = max(0, 1.0 - steps / 50)

    return 0.7 * correct + 0.3 * efficiency

trainset = [
    dspy.Example(
        task="Summarize the key arguments in this debate: ...",
        result="The debate centers on three arguments: ...",
    ).with_inputs("task"),
    # ... 10-20 examples covering different task types
]

optimizer = MIPROv2(metric=agent_metric, num_candidates=7)

optimized_agent = optimizer.compile(
    build_deep_agent(),
    trainset=trainset,
)
```

This optimizes the Signature's docstring (the agent's behavioral instructions) based
on what actually produces good outcomes on the training tasks.

#### 3.2 Sub-LLM Strategy

Use `sub_lm` to control cost. The main LLM (e.g., GPT-4o, Claude Sonnet) drives the
REPL orchestration. The `sub_lm` (e.g., GPT-4o-mini, Claude Haiku) handles
`llm_query()` calls for semantic analysis within the REPL:

```python
agent = build_deep_agent(
    sub_lm=dspy.LM("openai/gpt-4o-mini"),  # 10-20x cheaper for sub-queries
    max_llm_calls=100,  # can afford more calls with cheaper model
)
```

#### 3.3 Domain-Specific Sub-Agent Signatures

For specialized sub-agents (research, coding, review), define separate Signatures
with their own docstrings and optimize each independently:

```python
class ResearchAgentSignature(dspy.Signature):
    """You are a research sub-agent. Your job is to deeply investigate a topic
    using web_search() and llm_query(), then write your findings to the workspace.
    ...detailed research-specific instructions...
    """
    task: str = dspy.InputField()
    context: str = dspy.InputField(default="")
    result: str = dspy.OutputField()

def build_research_agent(workspace, ...):
    return RLM(ResearchAgentSignature, tools=[...], ...)
```

Each specialized Signature can be independently optimized with `MIPROv2` on
domain-specific training examples.

---

### Phase 4: Review Without Self-Assessed Confidence

Replace the current confidence-threshold review loop with two approaches:

#### 4.1 Agent-Driven Review (via the REPL)

The agent can review its own work by writing code that uses `llm_query()` with
a critical prompt. This happens naturally inside the REPL — no separate
ReviewerAgent module needed:

```python
# Inside the REPL, the LLM might write:
draft = "My research found that..."
review = llm_query(f"Critically review this for accuracy and completeness: {draft}")
print(review)  # LLM sees the critique, decides whether to revise
```

This is more natural than a scalar confidence score because:
- The critique is textual and specific
- The agent decides whether/how to revise based on the critique content
- It's part of the agent's normal REPL flow, not a separate pipeline stage

#### 4.2 Cross-Agent Review (via delegation)

For higher-stakes review, delegate to a review sub-agent with a review-specific
Signature:

```python
class ReviewSignature(dspy.Signature):
    """You are a critical reviewer. Evaluate the draft against the original task.
    Identify specific issues: missing information, factual errors, logical gaps,
    structural problems. Be specific and actionable.
    """
    task: str = dspy.InputField(desc="The original task")
    draft: str = dspy.InputField(desc="The draft to review")
    result: str = dspy.OutputField(desc="Specific issues and suggestions, or 'PASS' if acceptable")

# Registered as a tool:
def review_draft(task: str, draft: str) -> str:
    """Have an independent reviewer evaluate a draft."""
    reviewer = RLM(ReviewSignature, max_iterations=10)
    result = reviewer(task=task, draft=draft)
    return result.result
```

This gives context isolation (reviewer can't see the agent's reasoning that
produced the draft) and avoids self-confirmation bias.

---

### Phase 5: Testing Strategy

#### 5.1 Unit Tests (No LLM Required)

Test tool functions independently:

```python
def test_workspace_write_read():
    ws = Workspace(Path(tmp_dir))
    ws.write_file("notes.md", "hello")
    assert ws.read_file("notes.md") == "hello"

def test_todo_store_roundtrip():
    store = TodoStore()
    store.write_todos('[{"content": "step 1", "status": "pending"}]')
    result = store.read_todos()
    assert "step 1" in result

def test_delegator_depth_limit():
    delegator = SubAgentDelegator(workspace=..., max_depth=2, current_depth=2)
    result = delegator.delegate("do something")
    assert "Error" in result  # at max depth
```

#### 5.2 Integration Tests (With LLM)

Test actual agent behavior end-to-end:

```python
def test_agent_uses_planning():
    agent = build_deep_agent(max_iterations=20)
    result = agent(task="List the 3 primary colors")
    assert "red" in result.result.lower()
    # Check trajectory shows tool use
    assert any("write_todos" in str(step) for step in result.trajectory)

def test_agent_delegates_subtasks():
    agent = build_deep_agent(max_iterations=30, max_depth=2)
    result = agent(task="Compare Python and Rust for systems programming")
    assert "python" in result.result.lower()
    assert "rust" in result.result.lower()
    assert any("delegate" in str(step) for step in result.trajectory)
```

#### 5.3 Benchmarks

Compare against baselines:

```python
BENCHMARKS = [
    {"task": "...", "criteria": [...], "expected_tools": [...]},
    ...
]

BASELINES = {
    "flat_cot": dspy.ChainOfThought("task -> result"),
    "react": dspy.ReAct("task -> result", tools=[...]),
    "deep_agent": build_deep_agent(),
    "deep_agent_optimized": optimized_agent,  # after MIPROv2
}
```

---

### Phase 6: Evaluation Harness

```python
import dspy

def evaluate_deep_agent(agent, tasks, metric_fn):
    """Run evaluation across a set of tasks."""
    results = []
    for task_spec in tasks:
        prediction = agent(task=task_spec["task"])
        score = metric_fn(task_spec, prediction)
        results.append({
            "task": task_spec["task"],
            "score": score,
            "steps": len(prediction.get("trajectory", [])),
            "result_preview": prediction.result[:200],
        })
    return {
        "mean_score": sum(r["score"] for r in results) / len(results),
        "mean_steps": sum(r["steps"] for r in results) / len(results),
        "results": results,
    }
```

---

## Summary: Current vs. Proposed

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Agent backbone** | Custom Python recursion tree calling RLM per-step | RLM IS the agent (REPL loop = agent loop) |
| **Tool system** | Custom `ToolRegistry` + `ToolSelectorAgent` | RLM's native tool registration (typed callables) |
| **Planning** | `PlannerAgent` decomposes into rigid subtask tree | `write_todos`/`read_todos` tool functions in REPL |
| **Sub-agents** | `_run_node()` with shared `RecursiveMemory` | `delegate()` tool spawning child RLM (isolated sandbox) |
| **Context isolation** | None — single shared memory | Structural — each RLM gets its own sandbox |
| **Context management** | Serialize last 12 log entries | Context lives in REPL variables (never in prompt) |
| **Filesystem** | None | `write_file`/`read_file` tools on shared workspace |
| **System prompt** | 1-line Signature descriptions | Detailed docstring with workflow instructions |
| **Review** | Self-assessed confidence score | `llm_query()` critique or `delegate()` to reviewer |
| **DSPy optimization** | None | `MIPROv2` on Signature docstrings, `sub_lm` for cost |
| **Lines of code** | ~330 (recursive_agent + roles + memory + trace + tools) | ~150 (factory + tools + signatures) |
| **REPL sandboxes per task** | 5+ per recursion node (one per role) | 1 per agent (+ 1 per sub-agent) |
| **Cost** | Hundreds of LLM calls for simple tasks | Bounded by `max_iterations` + `max_llm_calls` per agent |

## Implementation Priority

1. **Phase 0** — `DeepAgentSignature` with detailed workflow docstring
2. **Phase 1** — Tool functions: `TodoStore`, `Workspace`, `SubAgentDelegator`
3. **Phase 2** — `build_deep_agent()` factory wiring tools into RLM
4. **Phase 3** — DSPy optimization with `MIPROv2` + `sub_lm` strategy
5. **Phase 4** — Review via `llm_query()` critique and cross-agent review
6. **Phase 5** — Testing: unit tests for tools, integration tests with LLM
7. **Phase 6** — Evaluation harness with baselines and benchmarks
