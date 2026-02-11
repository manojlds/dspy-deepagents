"""Tests for the build_coding_agent backward-compat wrapper."""

from __future__ import annotations

import tempfile

from dspy.predict import RLM

from dspy_deepagents.agent import build_coding_agent, build_deep_agent
from dspy_deepagents.signature import DeepAgentSignature


def test_build_coding_agent_returns_rlm() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = build_coding_agent(codebase_root=tmp)
        assert isinstance(agent, RLM)


def test_build_coding_agent_same_as_build_deep_agent_with_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        coding = build_coding_agent(codebase_root=tmp)
        deep = build_deep_agent(root=tmp)
        assert coding.tools is not None
        assert deep.tools is not None
        assert len(coding.tools) == len(deep.tools)


def test_build_coding_agent_default_signature() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = build_coding_agent(codebase_root=tmp)
        assert agent.signature is DeepAgentSignature


def test_build_coding_agent_has_all_tools() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = build_coding_agent(codebase_root=tmp)
        assert agent.tools is not None
        assert len(agent.tools) >= 12


def test_build_coding_agent_with_extra_tools() -> None:
    def line_count(text: str) -> str:
        """Count lines in text."""
        return str(len(text.splitlines()))

    with tempfile.TemporaryDirectory() as tmp:
        agent = build_coding_agent(codebase_root=tmp, extra_tools=[line_count])
        assert agent.tools is not None
        assert len(agent.tools) >= 13


def test_build_coding_agent_respects_iterations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = build_coding_agent(
            codebase_root=tmp, max_iterations=25, max_llm_calls=40
        )
        assert agent.max_iterations == 25
        assert agent.max_llm_calls == 40
