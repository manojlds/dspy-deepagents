"""Tool functions implementing the four Deep Agents pillars.

Each class provides stateful tool functions that are registered with RLM.
RLM extracts function signatures and docstrings automatically for the LLM.

Pillar 2 (Planning): TodoStore -- write_todos / read_todos
Pillar 3 (Sub-agent delegation): SubAgentDelegator -- delegate
Pillar 4 (Filesystem workspace): Workspace -- write_file / read_file / list_files
Pillar 1 (System prompt): Lives in Signature docstrings (see signature.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class TodoStore:
    """Stateful todo list -- one instance per agent invocation."""

    def __init__(self) -> None:
        self._todos: list[dict[str, str]] = []

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


class Workspace:
    """Shared filesystem rooted at a directory.

    The workspace is shared between parent and sub-agents (same root),
    while REPL state is isolated per agent.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the workspace.

        Args:
            path: Relative path within the workspace.
            content: Text content to write.
        """
        target = (self.root / path).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            return f"Error: path '{path}' escapes workspace"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def read_file(self, path: str) -> str:
        """Read a file from the workspace.

        Args:
            path: Relative path within the workspace.
        """
        target = (self.root / path).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            return f"Error: path '{path}' escapes workspace"
        if not target.exists():
            return f"Error: {path} not found"
        return target.read_text()

    def list_files(self) -> str:
        """List all files in the workspace."""
        files = sorted(
            p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()
        )
        if not files:
            return "Workspace is empty."
        return "\n".join(str(f) for f in files)


class SubAgentDelegator:
    """Spawns child RLM agents with isolated REPL sandboxes.

    Each child gets the same shared workspace but a fresh REPL sandbox,
    ensuring context isolation while allowing file-based communication.
    """

    def __init__(
        self,
        workspace: Workspace,
        max_depth: int = 3,
        current_depth: int = 0,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.max_depth = max_depth
        self.current_depth = current_depth
        # Circular import avoidance: factory is injected by build_deep_agent
        self._agent_factory = agent_factory

    def delegate(self, task: str, context: str = "") -> str:
        """Spawn an isolated sub-agent for a focused task.

        The sub-agent gets its own REPL sandbox (context isolation).
        It shares the workspace filesystem but not your variables.
        Returns the sub-agent's final result.

        Args:
            task: What the sub-agent should accomplish.
            context: Optional context to pass (the sub-agent cannot see your state).
        """
        if self.current_depth >= self.max_depth:
            return (
                f"Error: max sub-agent depth ({self.max_depth}) reached. "
                "Solve directly instead of delegating."
            )
        if self._agent_factory is None:
            return "Error: sub-agent delegation is not configured."

        child = self._agent_factory(
            workspace=self.workspace,
            max_depth=self.max_depth,
            current_depth=self.current_depth + 1,
            max_iterations=30,
            max_llm_calls=40,
        )
        result = child(task=task, context=context)
        return result.result


def make_review_tool() -> Callable[[str, str], str]:
    """Create a cross-agent review tool function.

    Returns a callable that spawns an independent RLM reviewer with
    context isolation to avoid self-confirmation bias.
    """
    from dspy.predict import RLM

    from dspy_deepagents.signature import ReviewSignature

    def review_draft(task: str, draft: str) -> str:
        """Have an independent reviewer evaluate a draft.

        The reviewer cannot see the agent's reasoning, only the task and draft.

        Args:
            task: The original task description.
            draft: The draft output to review.
        """
        reviewer = RLM(ReviewSignature, max_iterations=10)
        result = reviewer(task=task, draft=draft)
        return result.result

    return review_draft
