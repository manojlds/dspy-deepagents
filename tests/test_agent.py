"""Tests for the build_deep_agent factory and agent construction.

These tests verify that the factory correctly wires tools into RLM
without requiring an actual LLM call.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dspy.predict import RLM

from dspy_deepagents.agent import build_deep_agent
from dspy_deepagents.signature import DeepAgentSignature, ResearchAgentSignature
from dspy_deepagents.tools import Workspace


def test_build_deep_agent_returns_rlm() -> None:
    agent = build_deep_agent()
    assert isinstance(agent, RLM)


def test_build_deep_agent_uses_default_signature() -> None:
    agent = build_deep_agent()
    assert agent.signature is DeepAgentSignature


def test_build_deep_agent_custom_signature() -> None:
    agent = build_deep_agent(signature=ResearchAgentSignature)
    assert agent.signature is ResearchAgentSignature


def test_build_deep_agent_has_all_pillar_tools() -> None:
    agent = build_deep_agent()
    assert agent.tools is not None
    assert len(agent.tools) >= 12


def test_build_deep_agent_shared_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp))
        agent = build_deep_agent(workspace=ws)
        assert agent.tools is not None
        assert len(agent.tools) >= 12


def test_build_deep_agent_with_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = build_deep_agent(root=tmp)
        assert agent.tools is not None
        assert len(agent.tools) >= 12


def test_build_deep_agent_with_extra_tools() -> None:
    def word_count(text: str) -> str:
        """Count words in text."""
        return str(len(text.split()))

    agent = build_deep_agent(extra_tools=[word_count])
    assert agent.tools is not None
    assert len(agent.tools) >= 13


def test_build_deep_agent_with_review_tool() -> None:
    agent = build_deep_agent(include_review=True)
    assert agent.tools is not None
    assert len(agent.tools) >= 13


def test_build_deep_agent_respects_iterations() -> None:
    agent = build_deep_agent(max_iterations=25, max_llm_calls=40)
    assert agent.max_iterations == 25
    assert agent.max_llm_calls == 40
