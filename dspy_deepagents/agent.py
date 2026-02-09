"""RLM-native Deep Agent factory.

The core insight: RLM's REPL-based iteration loop IS the agent loop.  Instead
of building a separate orchestration layer and using RLM as a predictor inside
it, the agent itself is an RLM.  All four Deep Agents pillars map to RLM
primitives:

    Agent loop         -> REPL iteration loop (LLM writes code each step)
    System prompt      -> RLM Signature docstring + action instructions
    Planning (todo)    -> write_todos / read_todos tool functions
    Sub-agent spawn    -> delegate() tool spawning child RLM (fresh sandbox)
    Filesystem         -> write_file / read_file / list_files tools
    Context mgmt       -> Context lives in REPL variables, never in the prompt
    Halting            -> SUBMIT(result=...) when done
    Budget control     -> max_iterations + max_llm_calls
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import dspy
from dspy.predict import RLM

from dspy_deepagents.signature import DeepAgentSignature
from dspy_deepagents.tools import SubAgentDelegator, TodoStore, Workspace


def build_deep_agent(
    signature: type[dspy.Signature] | None = None,
    workspace: Workspace | None = None,
    max_depth: int = 3,
    current_depth: int = 0,
    max_iterations: int = 50,
    max_llm_calls: int = 80,
    sub_lm: dspy.LM | None = None,
    extra_tools: list[Callable[..., object]] | None = None,
    include_review: bool = False,
) -> RLM:
    """Build a deep agent as an RLM with the four pillars as tools.

    Args:
        signature: DSPy Signature class for the agent.  Defaults to
            ``DeepAgentSignature``.  The docstring becomes the agent's
            system prompt (Pillar 1).
        workspace: Shared filesystem.  Created as a temp dir if *None*.
        max_depth: Maximum sub-agent nesting depth.
        current_depth: Current nesting level (0 = root agent).
        max_iterations: REPL iteration budget (= agent step budget).
        max_llm_calls: Sub-LLM call budget for ``llm_query()``.
        sub_lm: Optional cheaper model for ``llm_query()`` calls.
        extra_tools: Additional domain-specific tool callables.
        include_review: If *True*, include a ``review_draft`` tool that
            spawns an independent reviewer sub-agent.

    Returns:
        An RLM module ready to call with ``(task=..., context=...)``.
    """
    sig = signature or DeepAgentSignature

    if workspace is None:
        workspace = Workspace(Path(tempfile.mkdtemp(prefix="deepagent_")))

    # Pillar 2: Planning
    todo_store = TodoStore()

    # Pillar 3: Sub-agent delegation
    delegator = SubAgentDelegator(
        workspace=workspace,
        max_depth=max_depth,
        current_depth=current_depth,
        agent_factory=build_deep_agent,
    )

    # Collect all tool functions
    tools: list[Callable[..., object]] = [
        todo_store.write_todos,
        todo_store.read_todos,
        workspace.write_file,
        workspace.read_file,
        workspace.list_files,
        delegator.delegate,
    ]

    # Phase 4: Optional cross-agent review
    if include_review:
        from dspy_deepagents.tools import make_review_tool

        tools.append(make_review_tool())

    if extra_tools:
        tools.extend(extra_tools)

    return RLM(
        sig,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        tools=tools,
        sub_lm=sub_lm,
    )
