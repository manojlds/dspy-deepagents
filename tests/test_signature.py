"""Tests for Deep Agent signatures."""

from __future__ import annotations

import dspy

from dspy_deepagents.signature import (
    CodingAgentSignature,
    DeepAgentSignature,
    ResearchAgentSignature,
    ReviewSignature,
)


def test_deep_agent_signature_has_task_input() -> None:
    fields = DeepAgentSignature.input_fields
    assert "task" in fields


def test_deep_agent_signature_has_context_input() -> None:
    fields = DeepAgentSignature.input_fields
    assert "context" in fields


def test_deep_agent_signature_has_result_output() -> None:
    fields = DeepAgentSignature.output_fields
    assert "result" in fields


def test_deep_agent_signature_has_detailed_docstring() -> None:
    doc = DeepAgentSignature.__doc__
    assert doc is not None
    assert "WORKFLOW" in doc
    assert "FILESYSTEM TOOLS" in doc
    assert "list_dir" in doc
    assert "grep" in doc
    assert "read_file_lines" in doc
    assert "write_todos" in doc
    assert "delegate" in doc
    assert "SUBMIT" in doc


def test_review_signature_fields() -> None:
    assert "task" in ReviewSignature.input_fields
    assert "draft" in ReviewSignature.input_fields
    assert "result" in ReviewSignature.output_fields


def test_research_signature_fields() -> None:
    assert "task" in ResearchAgentSignature.input_fields
    assert "context" in ResearchAgentSignature.input_fields
    assert "result" in ResearchAgentSignature.output_fields


def test_signatures_are_dspy_signatures() -> None:
    assert issubclass(DeepAgentSignature, dspy.Signature)
    assert issubclass(ReviewSignature, dspy.Signature)
    assert issubclass(ResearchAgentSignature, dspy.Signature)


def test_coding_agent_signature_is_alias() -> None:
    assert CodingAgentSignature is DeepAgentSignature
