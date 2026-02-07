"""DSPy-native Deep Agents scaffolding."""

from dspy_deepagents.memory import RecursiveMemory
from dspy_deepagents.recursive_agent import RecursionConfig, RecursiveAgent
from dspy_deepagents.roles import (
    ExecuteSignature,
    ExecutorAgent,
    PlannerAgent,
    PlanSignature,
    ReviewerAgent,
    ReviewSignature,
    SynthesizerAgent,
    SynthesizeSignature,
)
from dspy_deepagents.tools import Tool, ToolRegistry

__all__ = [
    "PlanSignature",
    "ExecuteSignature",
    "ReviewSignature",
    "SynthesizeSignature",
    "RecursionConfig",
    "RecursiveAgent",
    "RecursiveMemory",
    "PlannerAgent",
    "ExecutorAgent",
    "ReviewerAgent",
    "SynthesizerAgent",
    "Tool",
    "ToolRegistry",
]
