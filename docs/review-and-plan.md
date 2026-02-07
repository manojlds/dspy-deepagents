# Critique & Reimplementation Plan: dspy-deepagents

## Part 1: Critique of the Current Implementation

### 1. Fundamental Misunderstanding of RLM's Purpose

The most significant issue is that `dspy.RLM` is used as a drop-in replacement for
`dspy.Predict` across every agent role, which misunderstands what RLM is designed for.

**What RLM actually does:** RLM (Recursive Language Model) solves the *context-as-variable*
problem. It stores large inputs as Python variables in a sandboxed REPL, then has the LLM
write code to iteratively explore and process that data — calling `llm_query()` on chunks
as needed. It was designed for tasks like processing 200k-token documents where stuffing
everything into a prompt causes "context rot."

**How the current code uses it:** Every agent role wraps its Signature in RLM:

```python
# roles.py:82 — PlannerAgent
self.predictor = rlm or RLM(PlanSignature, max_iterations=10)

# roles.py:101 — ExecutorAgent
self.predictor = rlm or RLM(ExecuteSignature, max_iterations=10)

# ... same pattern for Reviewer, Synthesizer, ToolSelector
```

This means that to decompose a task into subtasks, the PlannerAgent:
1. Spins up a sandboxed Python REPL
2. Has the LLM generate Python code to explore the (short) task string
3. Potentially makes up to 50 sub-LLM calls (`max_llm_calls` default)
4. Iterates up to 10 times in the REPL
5. All to produce a `list[str]` of subtasks

This is using a sledgehammer to hang a picture frame. `dspy.ChainOfThought` or even
`dspy.Predict` would be appropriate for generating subtask lists from short prompts.
The REPL/code-generation machinery of RLM is entirely wasted here because none of
the inputs are large enough to benefit from the context-as-variable pattern.

**Cost implication:** With 5 agent roles each using RLM with `max_iterations=10` and
`max_llm_calls=50`, a single depth-2 recursion with 3 children could trigger hundreds
of LLM calls for what should be straightforward structured generation tasks.

### 2. Misunderstanding of What LangChain Deep Agents Actually Is

LangChain's `deepagents` is built around **four architectural pillars**, not recursive
task decomposition:

| Pillar | LangChain deepagents | Current dspy-deepagents |
|--------|---------------------|------------------------|
| **Detailed system prompts** | First-class architectural component with extensive tool-usage instructions and few-shot examples | Missing entirely — Signatures have minimal 1-line descriptions |
| **Planning tool** | A todo-list (`write_todos`/`read_todos`) used as a context-engineering "no-op" to keep the agent on track | Replaced with hierarchical task decomposition tree — a fundamentally different concept |
| **Sub-agent spawning** | Isolated sub-agents with **context isolation** — they "go deep" without polluting the parent's context window | Shared `RecursiveMemory` across all nodes — the opposite of context isolation |
| **Filesystem access** | Shared workspace (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`) for collaboration across context limits | Missing entirely |

The current implementation reduces all four pillars to a single rigid
plan-execute-synthesize-review tree. This misses the core insight of deep agents:
they are *context management strategies*, not hierarchical task planners.

### 3. No Sub-Agent Context Isolation

In LangChain's deep agents, sub-agents are spawned with **isolated context windows**.
The parent delegates a focused task, the sub-agent works on it without seeing (or
polluting) the parent's full context, and only the result flows back up. This is
critical for long-running tasks where context accumulates beyond window limits.

The current implementation does the opposite — `RecursiveMemory` is a single shared
object passed to every node:

```python
# recursive_agent.py:67 — single shared state
state = state or RecursiveMemory()
result, confidence = self._run_node(
    task=task, ..., state=state, ...  # same state everywhere
)
```

Every planner, executor, reviewer, and synthesizer at every depth level reads and
writes to the same memory log. There is no isolation, no scoping, and no mechanism
to prevent context from growing unboundedly through the recursion tree.

### 4. Memory System Is an Append-Only Log, Not a Memory

`RecursiveMemory` is a list of entries with `.serialize(max_entries=12)` that dumps the
last 12 entries as a string. This has several problems:

- **No relevance-based retrieval** — the last 12 entries may be from completely
  unrelated subtasks
- **No hierarchical scoping** — a leaf node at depth 3 sees planner entries from
  depth 0 mixed with executor entries from depth 2 in a different subtree
- **No summarization** — context grows linearly with execution, eventually exceeding
  useful prompt length even with the 12-entry cap
- **No workspace** — agents can't write intermediate artifacts, only append log lines

### 5. Self-Assessed Confidence Is Unreliable

The review/refinement loop is driven entirely by LLM-generated confidence scores:

```python
# recursive_agent.py:249
if confidence >= self.config.review_threshold:
    return draft, confidence
```

LLMs are notoriously poorly calibrated at self-assessment. A model that generates
a wrong answer with 0.95 confidence will skip review entirely. A model that generates
a correct answer with 0.5 confidence will waste tokens on unnecessary refinement.
There is no external validation, no ground-truth comparison, and no mechanism to
improve calibration over time.

### 6. No DSPy Optimization

The entire value proposition of DSPy is that you can **optimize** prompts and pipelines
programmatically — using optimizers like `BootstrapFewShot`, `MIPROv2`, `BootstrapFinetune`,
etc. The current implementation uses none of them:

- No training data or examples for any Signature
- No optimizer is instantiated or run
- No metric functions defined for evaluating agent output quality
- The `eval_harness.py` just prints results — no ground truth, no scoring, no comparison

This means the agent is running with zero-shot prompts on every Signature, relying
entirely on the LLM's innate capabilities. The recursive decomposition adds cost and
complexity but never learns to decompose, execute, or review better.

### 7. Rigid Recursion Tree vs. Dynamic Agent Loop

LangChain's deep agents run as a **dynamic agent loop** — the agent decides what to do
next (plan, call a tool, spawn a sub-agent, write to filesystem) based on the current
state. The current implementation is a **static recursion tree**:

```
PlannerAgent → [child₁, child₂, child₃] → SynthesizerAgent → ReviewerAgent
```

This structure is fixed at design time. The agent cannot:
- Decide to skip planning and directly execute
- Spawn additional sub-agents based on intermediate results
- Iteratively refine its plan based on child outcomes
- Use filesystem to accumulate work products across steps

### 8. Budget Mechanism Is Flawed

Budget is decremented per child and shared across siblings:

```python
# recursive_agent.py:118-128
for sub_task in sub_tasks:
    child_result, child_confidence = self._run_node(
        ..., budget=budget - 1, ...  # same decremented budget for each child
    )
```

This doesn't actually track global budget consumption. Each sibling gets `budget - 1`,
meaning 3 siblings each think they have the same remaining budget. The budget is a
per-branch limit, not a global resource constraint.

### 9. Tool System Is Too Simplistic

Tools are `Callable[[str], str]` — string in, string out:

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[[str], str]
```

No structured parameters, no JSON schemas, no error handling, no async support, no
timeout mechanism. Compare with LangChain's deep agents which include file operations,
shell access, web search, and sandboxed execution environments.

### 10. Tests Don't Test Anything Meaningful

The test suite uses static mocks that bypass all DSPy/RLM functionality:

```python
class StaticPlanner(PlannerAgent):
    def forward(self, task, context, memory):
        return dspy.Prediction(sub_tasks=["child-task"])
```

These tests verify that the `RecursiveAgent` control flow works with hardcoded returns.
They don't test:
- That DSPy Signatures produce valid structured output
- That RLM generates useful decompositions
- That the review loop improves output quality
- Any integration with actual LLM calls
- Edge cases (empty plans, malformed confidence, tool errors)

---

## Part 2: Reimplementation Plan — Proper RLM Usage with DSPy

### Design Philosophy

The reimplementation should align with both LangChain's deep agents architecture
(the four pillars) AND DSPy's strengths (programmatic optimization, typed signatures,
modular composition). RLM should be used **where it actually helps** — processing
large accumulated context — not as a generic predictor.

### Architecture Overview

```
DeepAgent (main loop)
├── SystemPrompt (detailed, optimizable via DSPy)
├── PlanningTool (todo-list for self-tracking)
├── SubAgentTool (spawns isolated child agents)
├── FileSystemWorkspace (shared artifact store)
├── ToolRegistry (extensible tool system)
├── ContextManager (summarization + RLM for large contexts)
└── Optimizer (DSPy optimizer for prompt/pipeline improvement)
```

### Phase 0: Foundation — Correct DSPy Primitives

**Goal:** Establish the right DSPy primitives for each concern, using RLM only where
context length demands it.

#### 0.1 Agent Core Signature

```python
class AgentStep(dspy.Signature):
    """Decide the next action given the current state.

    You are a deep research agent. Given your current task, the work done so far
    (context), your todo list, and available tools, decide what to do next.

    Output exactly one action: plan, execute, delegate, write_file, or finish.
    """
    task: str = dspy.InputField(desc="The overall task to accomplish")
    context: str = dspy.InputField(desc="Summary of work done so far")
    todos: str = dspy.InputField(desc="Current todo list state")
    available_tools: str = dspy.InputField(desc="Tools and their descriptions")
    workspace_listing: str = dspy.InputField(desc="Files in the shared workspace")

    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning about what to do next")
    action: str = dspy.OutputField(desc="One of: plan, execute, delegate, write_file, finish")
    action_input: str = dspy.OutputField(desc="JSON input for the chosen action")
```

Use `dspy.ChainOfThought(AgentStep)` — NOT RLM. This is a short-context decision
that benefits from chain-of-thought reasoning, not REPL-based code execution.

#### 0.2 Where RLM IS Appropriate

RLM should be used in exactly two places:

1. **Context Summarization** — When accumulated context exceeds a threshold (e.g.,
   50k tokens), use RLM to process it iteratively:

   ```python
   class SummarizeContext(dspy.Signature):
       """Summarize a large body of accumulated agent context."""
       context: str = dspy.InputField(desc="Full accumulated context")
       task: str = dspy.InputField(desc="The task, for relevance filtering")
       summary: str = dspy.OutputField(desc="Concise summary preserving key findings")

   # RLM makes sense here — context can be 100k+ tokens
   summarizer = RLM(SummarizeContext, max_iterations=15, max_llm_calls=20)
   ```

2. **Deep Research Synthesis** — When a sub-agent has collected large amounts of
   research material (multiple documents, web pages, etc.) and needs to synthesize:

   ```python
   class SynthesizeResearch(dspy.Signature):
       """Synthesize research findings from multiple sources into a coherent report."""
       research_materials: str = dspy.InputField(desc="Collected research documents")
       query: str = dspy.InputField(desc="The research question")
       report: str = dspy.OutputField(desc="Synthesized research report with citations")

   synthesizer = RLM(SynthesizeResearch, max_iterations=20, max_llm_calls=30)
   ```

Everything else should use `dspy.ChainOfThought` or `dspy.Predict`.

### Phase 1: The Four Pillars

#### Pillar 1 — System Prompts as First-Class Components

```python
class DeepAgentConfig:
    """Configuration for a DeepAgent including its system prompt."""

    system_prompt: str  # Detailed instructions, tool usage examples, few-shot demos
    max_steps: int = 50  # Maximum agent loop iterations
    context_window_budget: int = 100_000  # Token budget before summarization
    max_sub_agent_depth: int = 3  # Nesting limit for sub-agents
```

System prompts should be:
- Stored as separate text files, not inline strings
- Include explicit tool usage instructions with examples
- Optimizable via DSPy's `MIPROv2` or manual iteration
- Different for different agent "personalities" (researcher, coder, reviewer)

#### Pillar 2 — Planning as a Todo Tool

Instead of a PlannerAgent that decomposes tasks into a tree, implement planning as
a **tool the agent can choose to invoke**:

```python
class TodoList:
    """Agent-managed todo list for tracking progress."""

    def write_todos(self, todos: list[dict]) -> str:
        """Write/update the todo list. Each item has: content, status."""
        ...

    def read_todos(self) -> str:
        """Read the current todo list state."""
        ...
```

The agent decides when to plan, when to update its plan, and when to deviate. This
is fundamentally different from a rigid decomposition tree — it's a *context
engineering* strategy that keeps the agent focused during long-running tasks.

#### Pillar 3 — Sub-Agent Spawning with Context Isolation

```python
class SubAgentTool:
    """Spawn an isolated sub-agent for a focused task."""

    def delegate(self, task: str, context: str = "") -> str:
        """
        Spawn a sub-agent with:
        - Its own context window (isolation from parent)
        - Access to the shared filesystem (for passing large artifacts)
        - A focused task description
        - A subset of tools appropriate for the subtask

        Returns only the sub-agent's final result (not its full context).
        """
        sub_agent = DeepAgent(
            config=self.sub_agent_config,
            workspace=self.shared_workspace,  # shared filesystem
            tools=self.sub_agent_tools,
            depth=self.current_depth + 1,
        )
        result = sub_agent.run(task=task, context=context)
        return result.final_output  # only the output crosses the boundary
```

Key difference from current implementation: the sub-agent gets a **fresh context
window**. It doesn't see the parent's full conversation history. Only the task
description and any explicitly shared context cross the boundary. This is what
enables deep agents to handle long-running complex tasks without context overflow.

#### Pillar 4 — Filesystem Workspace

```python
class Workspace:
    """Shared filesystem for agent collaboration."""

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    def write_file(self, path: str, content: str) -> str: ...
    def read_file(self, path: str) -> str: ...
    def edit_file(self, path: str, old: str, new: str) -> str: ...
    def list_files(self, pattern: str = "**/*") -> list[str]: ...
    def grep(self, pattern: str, path: str = ".") -> str: ...
```

The workspace serves two critical functions:
1. **Persistent artifacts** — agents write intermediate results to files rather than
   keeping them in context
2. **Cross-agent communication** — parent and sub-agents share a workspace, so
   sub-agents can write results that the parent reads back

### Phase 2: The Agent Loop

Replace the rigid recursion tree with a dynamic agent loop:

```python
class DeepAgent(dspy.Module):
    def __init__(self, config, workspace, tools, depth=0):
        super().__init__()
        self.config = config
        self.workspace = workspace
        self.tools = tools
        self.depth = depth

        # Core decision module — NOT RLM
        self.decide = dspy.ChainOfThought(AgentStep)

        # Context summarizer — YES RLM (for large contexts)
        self.summarize = RLM(SummarizeContext, max_iterations=15)

        # Todo list
        self.todos = TodoList()

        # Conversation history
        self.history = []

    def forward(self, task: str, context: str = "") -> dspy.Prediction:
        self.history.append({"role": "user", "content": task})

        for step in range(self.config.max_steps):
            # Manage context — summarize if too long
            current_context = self._build_context()
            if self._estimate_tokens(current_context) > self.config.context_window_budget:
                current_context = self.summarize(
                    context=current_context,
                    task=task
                ).summary

            # Decide next action
            decision = self.decide(
                task=task,
                context=current_context,
                todos=self.todos.read_todos(),
                available_tools=self._describe_tools(),
                workspace_listing=self.workspace.list_files(),
            )

            # Execute the chosen action
            if decision.action == "finish":
                return dspy.Prediction(
                    result=decision.action_input,
                    steps=step + 1,
                    trace=self.history,
                )

            result = self._execute_action(decision.action, decision.action_input)
            self.history.append({
                "role": "assistant",
                "action": decision.action,
                "reasoning": decision.reasoning,
                "result": result,
            })

        return dspy.Prediction(result=self._force_finish(), steps=self.config.max_steps)
```

This is fundamentally different from the current tree recursion:
- The agent **decides dynamically** what to do at each step
- It can plan, then execute, then re-plan based on results
- Sub-agent delegation is just another tool, not the structural backbone
- Context is managed explicitly with summarization when needed

### Phase 3: DSPy Optimization Integration

This is the phase that makes this actually *DSPy-native* rather than just "code that
imports dspy."

#### 3.1 Define Metrics

```python
def agent_metric(example, prediction, trace=None):
    """Evaluate agent output quality."""
    # Task-specific correctness (varies by task type)
    correct = evaluate_correctness(example.expected, prediction.result)

    # Efficiency: fewer steps is better
    efficiency = 1.0 - (prediction.steps / example.max_steps)

    # Cost: fewer LLM calls is better (from trace)
    cost_score = evaluate_cost(trace)

    return 0.6 * correct + 0.2 * efficiency + 0.2 * cost_score
```

#### 3.2 Create Training Examples

```python
trainset = [
    dspy.Example(
        task="Research the impact of RLHF on LLM alignment",
        expected="A report covering reward modeling, PPO training, ...",
        max_steps=20,
    ).with_inputs("task"),
    dspy.Example(
        task="Write a Python function to parse CSV files with error handling",
        expected="def parse_csv(path): ...",
        max_steps=10,
    ).with_inputs("task"),
    # ... more examples covering different task types
]
```

#### 3.3 Optimize the Pipeline

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=agent_metric,
    num_candidates=10,
    max_bootstrapped_demos=3,
)

optimized_agent = optimizer.compile(
    DeepAgent(config=default_config, workspace=workspace, tools=tools),
    trainset=trainset,
)
```

This optimizes:
- The system prompt (via instruction optimization)
- Few-shot examples for the `AgentStep` signature
- Potentially the summarization prompts
- The overall pipeline behavior

#### 3.4 Use BootstrapFewShot for Sub-Components

Individual signatures can also be optimized independently:

```python
from dspy.teleprompt import BootstrapFewShot

# Optimize the planning tool's output quality
plan_optimizer = BootstrapFewShot(metric=plan_quality_metric, max_rounds=3)
optimized_planner = plan_optimizer.compile(
    dspy.ChainOfThought(PlanSignature),
    trainset=planning_examples,
)
```

### Phase 4: Proper Tool System

#### 4.1 Structured Tool Definitions

```python
@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "integer", "boolean", "object"
    description: str
    required: bool = True

@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    func: Callable[..., str]
    timeout_seconds: float = 30.0

    def schema(self) -> dict:
        """Return JSON Schema for this tool's parameters."""
        ...
```

#### 4.2 Built-in Tools for Deep Agent Capability

```python
DEFAULT_TOOLS = [
    # Planning
    Tool(name="write_todos", description="Create or update the task plan", ...),
    Tool(name="read_todos", description="Read the current plan state", ...),

    # Filesystem
    Tool(name="write_file", description="Write content to workspace file", ...),
    Tool(name="read_file", description="Read a workspace file", ...),
    Tool(name="edit_file", description="Make targeted edits to a file", ...),
    Tool(name="list_files", description="List files in workspace", ...),

    # Sub-agent
    Tool(name="delegate", description="Spawn a sub-agent for a focused task", ...),

    # Research (optional)
    Tool(name="web_search", description="Search the web", ...),
    Tool(name="web_fetch", description="Fetch and extract content from URL", ...),
]
```

### Phase 5: Context Management with RLM

This is where RLM genuinely adds value — managing context that grows beyond
what fits in a prompt.

```python
class ContextManager:
    """Manages agent context using RLM for summarization when needed."""

    def __init__(self, budget_tokens: int = 100_000):
        self.budget = budget_tokens
        self.full_history = []  # Complete history
        self.summary = ""  # Running summary
        self.summarizer = RLM(
            SummarizeContext,
            max_iterations=15,
            max_llm_calls=20,
        )

    def add(self, entry: dict):
        self.full_history.append(entry)

    def get_context(self, task: str) -> str:
        """Return context that fits within budget.

        If full history exceeds budget, use RLM to summarize older
        entries and keep recent entries verbatim.
        """
        recent = self.full_history[-10:]  # Always keep last 10 entries
        recent_text = self._format(recent)

        if self._estimate_tokens(recent_text) > self.budget:
            # Even recent entries are too long — summarize everything
            return self.summarizer(
                context=self._format(self.full_history),
                task=task,
            ).summary

        remaining_budget = self.budget - self._estimate_tokens(recent_text)
        older = self.full_history[:-10]

        if older and self._estimate_tokens(self._format(older)) > remaining_budget:
            # Summarize older entries with RLM
            self.summary = self.summarizer(
                context=self._format(older),
                task=task,
            ).summary

        return f"{self.summary}\n\n---\nRecent:\n{recent_text}"
```

### Phase 6: Review via External Validation, Not Self-Assessment

Replace the self-assessed confidence mechanism with actual validation:

```python
class ReviewTool:
    """Review agent output using external validation strategies."""

    def __init__(self):
        # Use a SEPARATE model instance for review to avoid self-confirmation bias
        self.reviewer = dspy.ChainOfThought(ReviewSignature)

    def review(self, task: str, output: str, criteria: list[str]) -> dict:
        """Review output against specific criteria.

        Instead of asking "how confident are you?", check specific properties:
        - Does the output address all parts of the task?
        - Are claims supported by evidence/citations?
        - Is the output well-structured and coherent?
        - Are there factual inconsistencies?
        """
        ...

class ReviewSignature(dspy.Signature):
    """Review an output against specific quality criteria."""
    task: str = dspy.InputField()
    output: str = dspy.InputField()
    criteria: list[str] = dspy.InputField()

    issues: list[str] = dspy.OutputField(desc="Specific issues found, or empty list")
    passes: bool = dspy.OutputField(desc="Whether the output meets all criteria")
    suggestions: list[str] = dspy.OutputField(desc="Specific actionable improvements")
```

Key improvements:
- Criteria-based review rather than scalar confidence
- Specific, actionable feedback rather than a number
- Can use a different model for review (cross-model validation)
- Integrable with DSPy's optimization (train the reviewer too)

### Phase 7: Evaluation and Benchmarking

#### 7.1 Proper Evaluation Suite

```python
class DeepAgentBenchmark:
    """Multi-task benchmark with ground truth and metrics."""

    tasks = [
        {
            "task": "Research RLHF and write a 500-word summary with citations",
            "criteria": ["mentions reward modeling", "mentions PPO", "has citations",
                        "500+ words"],
            "tools_needed": ["web_search", "write_file"],
            "expected_delegation": True,  # Should use sub-agents
        },
        {
            "task": "Debug this Python function: ...",
            "criteria": ["identifies the bug", "provides fix", "fix is correct"],
            "tools_needed": ["write_file", "edit_file"],
            "expected_delegation": False,
        },
        # ... more tasks
    ]

    def evaluate(self, agent: DeepAgent) -> dict:
        results = []
        for task_spec in self.tasks:
            prediction = agent(task=task_spec["task"])
            score = self._score(prediction, task_spec)
            results.append(score)
        return self._aggregate(results)
```

#### 7.2 Baselines for Comparison

- **Flat baseline:** Single `dspy.ChainOfThought` call (no agents, no tools)
- **CoT + Tools:** `dspy.ReAct` with tools but no sub-agents
- **Deep Agent (no optimization):** The full pipeline with zero-shot prompts
- **Deep Agent (optimized):** After DSPy optimization with `MIPROv2`

This lets us measure the actual value added by each component.

---

## Summary of Critical Changes

| Current Implementation | Proposed Fix |
|----------------------|--------------|
| RLM used for every agent role | RLM only for large-context summarization and synthesis |
| Rigid plan-execute-synthesize tree | Dynamic agent loop with tool-based actions |
| Shared `RecursiveMemory` across all nodes | Context isolation per sub-agent, shared filesystem |
| No filesystem workspace | Full workspace with file read/write/edit |
| Planning = hierarchical decomposition | Planning = todo-list tool for self-tracking |
| Self-assessed confidence scores | Criteria-based review with specific feedback |
| No DSPy optimization | `MIPROv2` / `BootstrapFewShot` for pipeline optimization |
| Minimal 1-line Signature descriptions | Detailed system prompts as first-class components |
| `Callable[[str], str]` tools | Structured tools with JSON schemas and timeouts |
| Static mock tests only | Integration tests + benchmark suite with ground truth |
| Append-only memory log | Context manager with RLM-powered summarization |

## Implementation Priority

1. **Phase 0** (Foundation) — Get the primitives right: `ChainOfThought` for decisions,
   RLM only for large context
2. **Phase 1** (Four Pillars) — System prompts, todo tool, sub-agents, filesystem
3. **Phase 2** (Agent Loop) — Dynamic loop replacing static tree
4. **Phase 3** (Optimization) — DSPy optimizers with real metrics and training data
5. **Phase 4** (Tools) — Structured tool system with built-in deep agent tools
6. **Phase 5** (Context Management) — RLM-powered summarization for long-running tasks
7. **Phase 6** (Review) — Criteria-based external validation
8. **Phase 7** (Evaluation) — Benchmarks with baselines and ground truth
